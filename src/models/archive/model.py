# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model.py
import csv
import logging
import os
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.metrics import MetricsEvaluator
import pandas as pd
import wandb

# Initialize a logger for debugging and output control
logger = logging.getLogger(__name__)


class FlanT5FineTuner(pl.LightningModule):
    """
    A PyTorch Lightning module for fine-tuning the Flan-T5 model using policy gradient reinforcement learning.
    Supports both Maximum Likelihood Estimation (MLE) and Policy Gradient (PG) training modes.
    """

    def __init__(self, model_name, model_dir, file_label=""):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()
        # Save only essential hyperparameters
        self.save_hyperparameters('model_name')

        # Store model_dir and file_label as instance variables
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load T5 model and tokenizer with configurations specified in CONFIG
        config = T5Config.from_pretrained(
            model_name,
            output_attentions=CONFIG["output_attentions"]
        )
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Set unique file paths using file_label to prevent overwriting
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Initialize buffers for validation
        self.epoch_validation_details = []  # Storage for each validation epoch
        self.epoch_scores = []  # Validation scores buffer

        # Initialize buffers for testing
        self.epoch_test_details = []  # Storage for each test epoch
        self.epoch_test_scores = []  # Test scores buffer

        # Initialize MetricsEvaluator to handle custom scoring for rewards
        self.metrics_evaluator = MetricsEvaluator()

        # This attribute will be set in main.py to toggle between MLE and PG modes
        #self.use_policy_gradient = False
        # Ensure only valid training modes are set
        assert self.training_mode in ["mle", "pg", "dto"], "ERROR: Invalid training_mode!"

        self.epoch_scores = []  # Initialize the list to store scores

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Handles different forward behavior depending on training mode.

        - MLE: Uses standard forward pass with labels for loss calculation.
        - PG: Generates tokens with sampling for reinforcement learning.
        - DTO: Computes soft token embeddings instead of hard token predictions.
        """
        if labels is not None:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if self.training_mode == "pg":
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                do_sample=True,
                temperature=1.5,
                output_scores=True,
                return_dict_in_generate=True
            )
            return outputs.sequences, outputs.scores

        elif self.training_mode == "dto":
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_probs = torch.softmax(outputs.logits, dim=-1)
            expected_embeds = self.expected_embeddings(token_probs)
            return expected_embeds

        else:
            raise ValueError(f"ERROR: Unknown training mode {self.training_mode}")

    def expected_embeddings(self, token_probs):
        """
        Compute soft token embeddings using probability-weighted average over the model's embedding matrix.

        Args:
        - token_probs: Probability distribution over vocabulary (batch, seq_len, vocab_size)

        Returns:
        - expected_embeds: Soft token embeddings (batch, seq_len, hidden_dim)
        """

        # Retrieve the embedding matrix (vocab_size, hidden_dim).
        embedding_matrix = self.model.get_input_embeddings().weight

        # Compute weighted sum of embeddings per token based on probability distribution.
        expected_embeds = torch.einsum('bsv,vh->bsh', token_probs, embedding_matrix)

        # Sanity check to ensure embedding sequence length matches token_probs sequence length.
        assert expected_embeds.shape[1] == token_probs.shape[1], "Embedding sequence length mismatch!"

        return expected_embeds  # Return soft token embeddings.

    def differentiable_metric_loss(self, generated_embeds, references, metric="bert"):
        """
        Computes differentiable loss based on soft embeddings and a selected evaluation metric.

        Args:
            generated_embeds (Tensor): Soft token embeddings from the model.
            references (List[str]): List of ground-truth reference texts.
            metric (str): Metric to use ("bert", "bart", "rouge", "bleu").

        Returns:
            Tensor: Loss value computed based on the chosen metric.
        """

        if metric == "bert":
            # Use BERTScore as a differentiable loss function.
            _, _, f1_scores = self.metrics_evaluator.bert_scorer.score(
                generated_embeds, references, use_soft_embeddings=True
            )
            # Ensure valid output and prevent NaN values.
            return 1 - f1_scores.mean() if f1_scores is not None else torch.tensor(1.0, device=self.device)

        elif metric == "bart":
            # Compute BARTScore and use it as a loss.
            scores = self.metrics_evaluator.bart_scorer.score(generated_embeds, references)
            return -torch.tensor(scores, device=self.device).mean()

        elif metric == "rouge":
            # Compute ROUGE-L and use (1 - score) as a loss function.
            rouge_scores = self.metrics_evaluator.rouge.get_scores(generated_embeds, references, avg=True)
            return 1 - torch.tensor(rouge_scores["rouge-l"]["f"], device=self.device)

        elif metric == "bleu":
            # Compute BLEU score and normalize.
            references = [[ref] for ref in references]
            bleu_score = self.metrics_evaluator.sacre_bleu.corpus_score(generated_embeds, references).score
            return 1 - torch.tensor(bleu_score, device=self.device) / 100.0

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def apply_vocab_masking(self, logits):
        """
        Masks logits for tokens beyond the vocabulary size of the tokenizer.
        Handles both 2D and 3D tensors for compatibility with generated logits.
        """
        vocab_size = self.tokenizer.vocab_size

        # Check if logits is 2D (batch_size, vocab_size) or 3D (batch_size, sequence_length, vocab_size)
        if logits.dim() == 2:
            # Mask for 2D logits (each decoding step in generate)
            masked_logits = logits.clone()
            masked_logits[:, vocab_size:] = -float('inf')
        elif logits.dim() == 3:
            # Mask for 3D logits (entire sequence logits from forward pass)
            masked_logits = logits.clone()
            masked_logits[:, :, vocab_size:] = -float('inf')
        else:
            raise ValueError(f"Unexpected logits dimension: expected 2 or 3, got {logits.dim()}")

        return masked_logits

    def custom_loss(self, outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        """
        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(
            -1)  # Flatten weights to match sequence length [batch_size * seq_length]

        # Compute the standard loss function without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat

        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()

        return mean_weighted_loss

    def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards, baseline):
        """
        Calculates policy gradient loss based on generated tokens and rewards.
        Handles the case where BART scores are negative by flipping the sign of rewards.
        """
        # Stack logits along the sequence dimension and apply log softmax
        logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
        logits = self.apply_vocab_masking(logits)  # Apply masking to stacked logits

        # Gather the log probabilities for the generated tokens
        labels_for_indexing = generated_tokens[:, 1:].contiguous()
        token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

        # Create a mask to ignore padding tokens
        padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * padding_mask.float()

        # Sum log probabilities across the sequence dimension
        sequence_log_prob_sum = token_log_probs.sum(dim=1)

        # Handle special case for BART (negative rewards)
        if CONFIG.get("reward_metric") == "bart":
            rewards = rewards + 4  # add a Baseline to move to the positif size but you keep the magnitude
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Normalize BART rewards
        return -(rewards * sequence_log_prob_sum).mean()

    def training_step(self, batch, batch_idx):
        """
        Processes a batch during the training phase. Routes to PG or MLE logic based on mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        if self.use_policy_gradient:
            return self._training_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self._training_step_dif(batch, input_ids, attention_mask)
        else:
            return self._training_step_mle(batch, input_ids, attention_mask, labels)

    def _training_step_dif(self, batch, input_ids, attention_mask):
        """
        Handles training logic for Differentiable Training Objectives (DTO) mode.
        Uses soft token embeddings and optimizes differentiable metric-based loss.

        Args:
            batch (dict): Batch containing input sequences and reference edits.
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Mask for input tokens.

        Returns:
            Tensor: Computed differentiable loss.
        """

        # Compute soft token embeddings instead of discrete tokens.
        expected_embeds = self.forward(input_ids, attention_mask)

        # Retrieve reference edited endings.
        references = batch["edited_ending"]

        # Select the differentiable metric for optimization.
        metric_type = CONFIG.get("reward_metric", "bart")

        # Compute differentiable loss using soft embeddings.
        dif_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        # Log the computed loss.
        self.log('training_differentiable_loss', dif_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        logger.info(f'[TRAIN] DTO Loss: {dif_loss}')

        return dif_loss

    def _training_step_pg(self, batch, input_ids, attention_mask):
        """
        Training-specific logic for Policy Gradient (PG) mode.
        """
        # Forward pass
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Get ground-truth references
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate rewards
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()
            rewards = rewards - dynamic_baseline

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

            # Calculate PG loss
        pg_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=dynamic_baseline)

        # Logging
        self.log('training_pg_loss', pg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('training_pg_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('training_pg_baseline', dynamic_baseline, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('training_pg_delta_m1_mean', delta_m1.mean().item(), on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

        logger.info(
            f'[TRAIN] PG Loss: {pg_loss}, Baseline: {dynamic_baseline}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_loss

    def _training_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Training-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        # Forward pass
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        masked_logits = self.apply_vocab_masking(outputs.logits)  # Apply masking to logits

        # Calculate MLE loss
        if CONFIG['use_custom_loss'] and 'differential_weights' in batch:
            mle_train_loss = self.custom_loss(masked_logits, batch['labels'], batch['differential_weights'])
        else:
            mle_train_loss = outputs.loss

        # Log MLE training loss
        self.log('training_mle_loss', mle_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Decode and log scores for generated texts
        generated_texts = self.tokenizer.batch_decode(masked_logits.argmax(-1), skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_mean = scores.mean()
        self.log('training_mle_score_mean', score_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mle_train_loss  # Return MLE loss for optimization

    def validation_step(self, batch, batch_idx):
        """
        Processes a batch during the validation phase. Routes to PG or MLE logic based on mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        print(f"Validation Step: Processing batch {batch_idx}")

        if self.use_policy_gradient:
            return self._validation_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self._validation_step_dif(batch, input_ids, attention_mask)
        else:
            return self._validation_step_mle(batch, input_ids, attention_mask, labels)

    def _validation_step_dif(self, batch, input_ids, attention_mask):
        """
        Validation-specific logic for Differentiable Training (DTO) mode.
        Uses soft token embeddings and evaluates loss based on a differentiable metric.

        Args:
            batch (dict): Batch containing input sequences and reference edits.
            input_ids (Tensor): Input token IDs.
            attention_mask (Tensor): Mask for input tokens.

        Returns:
            Tensor: Computed validation loss.
        """

        # Compute soft token embeddings.
        expected_embeds = self.forward(input_ids, attention_mask)

        # Retrieve reference edited endings.
        references = batch["edited_ending"]

        # Select differentiable metric type.
        metric_type = CONFIG.get("reward_metric", "bart")

        # Compute validation loss using differentiable metric.
        dif_val_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        # Log validation loss (detached to prevent gradient tracking).
        self.log('validation_differentiable_loss', dif_val_loss.detach(), on_epoch=True, prog_bar=True, logger=True)

        # Save validation details for debugging.
        self.epoch_validation_details.append({
            'Epoch': self.current_epoch,
            'Premise': batch['premise'][0],
            'Initial': batch['initial'][0],
            'Counterfactual': batch['counterfactual'][0],
            'Original Ending': batch['original_ending'][0],
            'Edited Ending': batch['edited_ending'][0],
        })

        logger.info(f'[VALIDATION] Epoch {self.current_epoch} | DTO Loss: {dif_val_loss}')

        return dif_val_loss

    def _validation_step_pg(self, batch, input_ids, attention_mask):
        """
        Validation-specific logic for Policy Gradient (PG) mode.
        """
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG validation loss (baseline = 0.0, since no updates occur)
        pg_val_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log validation metrics
        self.log('validation_pg_loss', pg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('validation_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

        # Save validation details
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        logger.info(
            f'[VALIDATION] Epoch {self.current_epoch} | PG Loss: {pg_val_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_val_loss

    def _validation_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Validation-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_val_loss = outputs.loss

        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Calculate sentence-level scores
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        self.epoch_scores.extend(scores.tolist())  # Save validation scores for the dataset

        # Log MLE loss
        self.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Save validation details
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })
        return mle_val_loss

    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        print("Validation Epoch End")
        if self.epoch_validation_details:
            print(f"Saving {len(self.epoch_validation_details)} validation details to {self.val_csv_file_path}.")
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)

        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            print(f"Overall validation score: {overall_val_score}")
            self.log("overall_score", overall_val_score, prog_bar=True, logger=True)

        # Clear buffers for next validation run
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()

    def test_step(self, batch, batch_idx):
        """
        Processes a batch during the testing phase. Routes to PG or MLE logic based on mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        if self.use_policy_gradient:
            return self._test_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self._test_step_dif(batch, input_ids, attention_mask)
        else:
            return self._test_step_mle(batch, input_ids, attention_mask, labels)

    def _test_step_pg(self, batch, input_ids, attention_mask):
        """
        Test-specific logic for Policy Gradient (PG) mode.
        """
        generated_tokens, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Compute scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG test loss (baseline = 0.0, since no updates occur)
        pg_test_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log test metrics
        self.log('test_pg_loss', pg_test_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.log('test_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.log('test_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

        # Save test details
        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        logger.info(
            f'[TEST] Epoch {self.current_epoch} | PG Loss: {pg_test_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_test_loss

    def _test_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Test-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_test_loss = outputs.loss

        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        return mle_test_loss

    def _test_step_dif(self, batch, input_ids, attention_mask):
        """
        Test-specific logic for Differentiable Training Objectives (DTO) mode.
        Computes the test loss based on soft token embeddings and a differentiable text similarity metric.
        """

        # Compute soft token embeddings instead of discrete tokens.
        expected_embeds = self.forward(input_ids, attention_mask)

        # Retrieve reference edited endings.
        references = batch["edited_ending"]

        # Select differentiable metric for evaluation (e.g., BERTScore, ROUGE, BLEU, BARTScore).
        metric_type = CONFIG.get("reward_metric", "bart")

        # Compute test loss using the selected differentiable metric.
        dif_test_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        # Log test loss (ensuring detached tensor to avoid gradient tracking).
        self.log('test_differentiable_loss', dif_test_loss, on_epoch=True, prog_bar=True, logger=True)

        # Save test details
        self.epoch_test_details.append({
            'Epoch': self.current_epoch,
            'Premise': batch['premise'][0],
            'Initial': batch['initial'][0],
            'Counterfactual': batch['counterfactual'][0],
            'Original Ending': batch['original_ending'][0],
            'Edited Ending': batch['edited_ending'][0],
        })

        logger.info(f'[TEST] Epoch {self.current_epoch} | DTO Loss: {dif_test_loss}')

        return dif_test_loss

    def on_test_epoch_end(self):
        """
        Finalize and save test results at the end of the test epoch.
        """
        print("Test Epoch End")
        if self.epoch_test_details:
            print(f"Saving {len(self.epoch_test_details)} test details to {self.test_csv_file_path}.")
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        if self.epoch_test_scores:
            overall_test_score = torch.tensor(self.epoch_test_scores).mean().item()
            print(f"Overall test score: {overall_test_score}")
            self.log("test_overall_score", overall_test_score, prog_bar=True, logger=True)

        # Clear buffers for next test run
        self.epoch_test_details.clear()
        self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        """
        Writes the details to the specified CSV file.
        """
        print(f"Writing {len(details)} entries to {csv_file_path}.")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())
            if not file_exists:
                writer.writeheader()
            for detail in details:
                detail['Epoch'] = epoch
            writer.writerows(details)

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
