# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
import wandb


class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))



    def score_embeds(self, inputs_embeds, tgts, batch_size=4, validation=False):
        """
        Lightning-optimized scoring using BARTScore with automatic gradient handling
        
        Implementation steps:
        1. Tokenize targets to get input IDs and attention masks
        2. Compute effective target length from attention mask
        3. Create encoder attention mask
        4. Forward pass with inputs_embeds
        5. Compute scores using LogSoftmax + NLLLoss
        6. Validate scores and handle errors
        """
        # BARTScore expected ranges
        MIN_SCORE = -3.0
        MAX_SCORE = 3.0
        OUTLIER_THRESHOLD = 5.0
        all_scores = []  # Store all scores for final histogram
        score_list = []
        
        batch_metrics = {
            'nan_count': 0,
            'outlier_count': 0,
            'processed_batches': 0
        }
        
        
        for i in range(0, len(inputs_embeds), batch_size):
            batch_embeds = inputs_embeds[i:i+batch_size]
            tgt_list = tgts[i:i+batch_size]

            try:
                # Lightning handles gradient context automatically
                with torch.set_grad_enabled(not validation):
                    # 1. Tokenization (always no_grad)
                    with torch.no_grad():
                        encoded_tgt = self.tokenizer(
                            tgt_list,
                            max_length=self.max_length,
                            truncation=True,
                            padding=True,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        tgt_tokens = encoded_tgt['input_ids']
                        tgt_mask = encoded_tgt['attention_mask']
                        # Compute the effective length (number of non-pad tokens) for each sentence.
                        tgt_len = tgt_mask.sum(dim=1)

                    # 2. Encoder mask creation
                    encoder_mask = torch.ones(
                        batch_embeds.size(0), batch_embeds.size(1),
                        dtype=torch.long,
                        device=self.device
                    )

                    # 3. Forward pass using 'inputs_embeds' instead of 'input_ids'.
                    output = self.model(
                        inputs_embeds=batch_embeds,
                        attention_mask=encoder_mask,
                        labels=tgt_tokens
                    )

                    # ======================================================
                    # 4. Score Calculation Pipeline
                    # ======================================================
                    
                    # 4a. Extract and flatten logits for all tokens in batch
                    #     Shape: [batch_size * seq_len, vocab_size]
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    
                    # 4b. Calculate negative log likelihood (NLL) loss:
                    #     1. Apply log-softmax to convert logits to log-probabilities
                    #     2. Compute NLL loss using ground truth tokens
                    #     Output shape: [batch_size * seq_len]
                    token_nll_loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    
                    # 4c. Reshape NLL losses to per-sequence format
                    #     Shape: [batch_size, seq_len]
                    sequence_nll_loss = token_nll_loss.view(tgt_tokens.shape[0], -1)
                    
                    # 4d. Compute normalized sequence-level loss:
                    #     1. Apply attention mask to ignore padding tokens
                    #     2. Sum losses across each sequence
                    #     3. Normalize by actual sequence length
                    #     Shape: [batch_size]
                    normalized_loss = (sequence_nll_loss * tgt_mask).sum(dim=1) / tgt_len
                    
                    # 5. Convert to final quality scores (higher = better):
                    #     Invert sign to transform loss to score metric
                    #     Shape: [batch_size] 
                    bart_scores = -normalized_loss

                    # Store scores for final histogram
                    scores_cpu = bart_scores.detach().cpu()
                    all_scores.extend(scores_cpu.tolist())
                    batch_metrics['processed_batches'] += 1
                    
                    # 5. Score validation
                    # NaN detection
                    if torch.isnan(scores_cpu).any():
                        nan_count = torch.isnan(scores_cpu).sum().item()
                        batch_metrics['nan_count'] += nan_count
                        self._safe_wandb_log({
                            "warnings/nan_in_batch": nan_count,
                            "warnings/batch_idx": i // batch_size
                        })

                    # Outlier detection
                    outliers = (scores_cpu < MIN_SCORE) | (scores_cpu > MAX_SCORE)
                    if outliers.any():
                        outlier_count = outliers.sum().item()
                        batch_metrics['outlier_count'] += outlier_count
                        self._safe_wandb_log({
                            "warnings/outlier_count": outlier_count,
                            "warnings/batch_idx": i // batch_size
                        })

                    score_list.append(bart_scores)

                    # Batch-level metrics
                    self._safe_wandb_log({
                        "batch/scores_mean": float(scores_cpu.mean()),
                        "batch/scores_std": float(scores_cpu.std()),
                        "batch/batch_idx": i // batch_size
                    })

            except Exception as e:
                self._safe_wandb_log({
                    "errors/exception": str(e),
                    "errors/batch_idx": i // batch_size
                })
                print(f"Error processing batch {i}: {str(e)}")
                if validation:
                    raise RuntimeError(f"Validation failed on batch {i}") from e
                continue

        # Final processing and logging
        scores_tensor = torch.cat(score_list) if score_list else torch.tensor([], device=self.device)
        
        if len(all_scores) > 0:
            try:
                # Convert to numpy array for W&B
                scores_np = np.array(all_scores)
                
                # Log final metrics
                self._safe_wandb_log({
                    "final/scores_mean": float(scores_np.mean()),
                    "final/scores_std": float(scores_np.std()),
                    "final/scores_min": float(scores_np.min()),
                    "final/scores_max": float(scores_np.max()),
                    "diagnostics/total_batches": batch_metrics['processed_batches'],
                    "diagnostics/total_nans": batch_metrics['nan_count'],
                    "diagnostics/total_outliers": batch_metrics['outlier_count']
                }, commit=False)
                
                # Guaranteed histogram logging
                self._safe_wandb_log({
                    "final/scores_dist": wandb.Histogram(scores_np)
                }, commit=True)
                print(f"Successfully logged histogram with {len(scores_np)} values")
                
            except Exception as e:
                print(f" Final logging failed: {str(e)}")
                # Fallback logging
                self._safe_wandb_log({
                    "final/scores_fallback": wandb.plot.line_series(
                        xs=np.arange(len(all_scores)),
                        ys=[all_scores],
                        keys=["scores"],
                        title="Score Distribution",
                        xname="Index")
                })
        
        return scores_tensor

    def _safe_wandb_log(self, metrics_dict, commit=True):
        """Minimal wandb logging without requiring self.logger"""
        if hasattr(self, '_wandb'):  # Check for injected wandb reference
            try:
                self._wandb().log(metrics_dict, commit=commit)
            except Exception as e:
                print(f"⚠ W&B fallback failed: {str(e)}")
                        

    # def score_embeds(self, inputs_embeds, tgts, batch_size=4, validation=False):
    #     """Score embeddings with gradient control - Preserves original logic with gradient flow"""
    #     print(f"\n[Scorer] Input embeddings grad: {inputs_embeds.requires_grad}")
    #     print(f"[Scorer] Mode: {'VALIDATION' if validation else 'TRAINING'}")
        
    #     if validation:
    #         with torch.no_grad():
    #             return self._score_embeds_impl(inputs_embeds, tgts, batch_size)
    #     else:
    #         # Ensure gradient flow for training
    #         return self._score_embeds_impl(inputs_embeds, tgts, batch_size)

    # def _score_embeds_impl(self, inputs_embeds, tgts, batch_size):
    #     """Original implementation with gradient flow preserved"""
    #     score_list = []
        
    #     for i in range(0, len(inputs_embeds), batch_size):
    #         batch_embeds = inputs_embeds[i:i+batch_size]
    #         tgt_list = tgts[i:i+batch_size]

    #         # Tokenization remains in no_grad
    #         with torch.no_grad():
    #             encoded_tgt = self.tokenizer(
    #                 tgt_list,
    #                 max_length=self.max_length,
    #                 truncation=True,
    #                 padding=True,
    #                 return_tensors='pt'
    #             )
    #             tgt_tokens = encoded_tgt['input_ids'].to(self.device)
    #             tgt_mask = encoded_tgt['attention_mask'].to(self.device)
    #             tgt_len = tgt_mask.sum(dim=1).to(self.device)

    #         encoder_mask = torch.ones(
    #             batch_embeds.size(0), batch_embeds.size(1),
    #             dtype=torch.long, 
    #             device=self.device
    #         )

    #         # Forward pass with gradient tracking
    #         output = self.model(
    #             inputs_embeds=batch_embeds,
    #             attention_mask=encoder_mask,
    #             labels=tgt_tokens
    #         )

    #         # === Only Modified Section ===
    #         logits = output.logits
    #         log_probs = self.lsm(logits.view(-1, logits.size(-1)))
    #         nll_loss = -log_probs.gather(1, tgt_tokens.view(-1, 1)).squeeze(1)
    #         scores = (nll_loss.view(tgt_tokens.shape) * tgt_mask).sum(dim=1) / tgt_len
    #         # === End Modified Section ===

    #         score_list.append(-scores)  # Keep as tensor

    #     # Return concatenated tensor instead of Python list
    #     return torch.cat(score_list) if score_list else torch.tensor([], device=self.device)




    # def score_embeds(self, inputs_embeds, tgts, batch_size=4):
    #     """
    #     Score a batch of examples using embeddings as input instead of token IDs.

    #     This method mirrors the behavior of the original 'score' method:
    #     1. Tokenize the target texts to get both input IDs and attention masks.
    #     2. Compute the effective target length (tgt_len) from the attention mask.
    #     3. Perform a forward pass using 'inputs_embeds' and the tokenized target (labels).
    #     4. Compute the loss manually from the logits (via LogSoftmax + NLLLoss).
    #     5. Normalize the loss by tgt_len and invert it (multiply by -1) to produce a BARTScore.

    #     """
    #     score_list = []  # Holds the final BARTScore for each example.

    #     # If 'inputs_embeds' is a Python list, len() is fine.
    #     # If it's a torch.Tensor, consider using 'inputs_embeds.size(0)' instead.
    #     for i in range(0, len(inputs_embeds), batch_size):
    #         # Extract the current batch of embeddings and corresponding target strings.
    #         batch_embeds = inputs_embeds[i: i + batch_size]
    #         tgt_list = tgts[i: i + batch_size]

    #         try:
    #             with torch.no_grad():
    #                 # 1. Tokenize the target texts to obtain input IDs and attention masks.
    #                 encoded_tgt = self.tokenizer(
    #                     tgt_list,
    #                     max_length=self.max_length,
    #                     truncation=True,
    #                     padding=True,
    #                     return_tensors='pt'
    #                 )
    #                 tgt_tokens = encoded_tgt['input_ids'].to(self.device)
    #                 tgt_mask = encoded_tgt['attention_mask'].to(self.device)

    #                 # 2. Compute the effective length (number of non-pad tokens) for each sentence.
    #                 tgt_len = tgt_mask.sum(dim=1).to(self.device)

    #                 # 3. New code : Create an encoder attention mask that matches the soft embeddings' sequence length.
    #                 #  This ensures no mismatch between batch_embeds and the provided attention mask.
    #                 encoder_mask = torch.ones(
    #                 batch_embeds.size(0), batch_embeds.size(1),
    #                 dtype=torch.long, device=self.device
    #                 )

    #                 # 3. Forward pass using 'inputs_embeds' instead of 'input_ids'.
    #                 #    If you have a suitable source mask, replace 'attention_mask=None' with it.
    #                 output = self.model(
    #                     inputs_embeds=batch_embeds.to(self.device),# Soft embeddings of shape [4, 90, 1024]
    #                     #attention_mask=tgt_mask, 
    #                     attention_mask=encoder_mask,# Encoder mask of shape [4, 90]
    #                     labels=tgt_tokens
    #                 )

    #                 # 4a. extract the logits, one per token
    #                 logits = output.logits.view(-1, self.model.config.vocab_size)

    #                 # 4b. Applies the LogSoftmax operator ("lsm") to the logits to turn them into logprobs,
    #                 #     and then applies loss_fct to turn the logprobs into the NLL los
    #                 loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))


    #                 # 4c. Reshape back to [batch_size, seq_len].
    #                 loss = loss.view(tgt_tokens.shape[0], -1)

    #                 # 4d. Adds them all up to get the total NLL of the whole ground-truth reference, normalised by its length
    #                 loss = loss.sum(dim=1) / tgt_len

    #                 # 5. Changes sign to turn the losses into scores (the higher, the better)
    #                 curr_score_list = [-x.item() for x in loss]

    #                 # Append these scores to the overall list.
    #                 score_list += curr_score_list

    #         except RuntimeError:
    #             # Print traceback and the target list if there's a runtime error.
    #             traceback.print_exc()
    #             print(f"Error processing target: {tgt_list}")
    #             exit(0)

    #     # Return the final list of BARTScores for all examples.
    #     return score_list
