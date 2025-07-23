import csv
import logging
import os
import wandb
import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.dto.utils.config import CONFIG
from src.dto.utils.metrics import MetricsEvaluator
import pandas as pd

logger = logging.getLogger(__name__)

class BartFineTuner(pl.LightningModule):
    """
    A DTO-only model for Differentiable Training Objectives.
    It always produces soft embeddings and uses a differentiable loss.
    """
    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters()
        # 1. Basic setup
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # 2. Determine target device
        self.target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 3. Load main model components
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        # 4. Initialize metrics evaluator
        self.metrics_evaluator = MetricsEvaluator()
        self.metrics_evaluator.bart_scorer._wandb = lambda: wandb  # Direct wandb access
        
        
        # Paths for CSV logging.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # 5. Store original embeddings
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.clone()

        # 6. Gumbel-Softmax setup
        self.use_gumbel = CONFIG["use_gumbel"]
        self.temperature = CONFIG["gumbel_temperature"]
        self.gumbel_hard = CONFIG["gumbel_hard"]
        self.anneal_rate = CONFIG["gumbel_anneal_rate"] if CONFIG["gumbel_annealing"] else None
        self.min_temp = CONFIG["gumbel_min_temp"]

        # 7. Move everything to target device
        self.to(self.target_device)
        self.model = self.model.to(self.target_device)
        self.metrics_evaluator.bart_scorer.model = self.metrics_evaluator.bart_scorer.model.to(self.target_device)
        self.bart_scorer_og_embed = self.bart_scorer_og_embed.to(self.target_device)
        if hasattr(self.tokenizer, 'device'):
            self.tokenizer.device = self.target_device

        # Buffers for logging details.
        self.epoch_validation_details = []
        self.epoch_scores = []
        self.epoch_test_details = []
        self.epoch_test_scores = []
        
        logger.info("Model initialized successfully")
        
    def train(self, mode=True):
        """
        Override the train method to ensure `bart_scorer` remains in evaluation mode.
        """
        super().train(mode)
        self.metrics_evaluator.bart_scorer.model.eval()
        return self

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # ===== 1. Forward Pass =====
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # ===== 2. Device Safety =====
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        device = logits.device  # Ensure we use the same device for all tensors


        # ===== 4. Gumbel-Softmax =====
        
        probs = F.gumbel_softmax(
            logits,
            tau=CONFIG["gumbel_temperature"],  
            hard=CONFIG["gumbel_hard"],       
            dim=-1
        )

        
        # ===== 5. Embedding Projection =====
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.to(device)
        expected_embeddings = torch.matmul(probs, embedding_matrix)

        return expected_embeddings

    def dto_loss_embeds(self, expected_embeddings, edited_endings, original_endings, validation=False):
        """Optimized DTO loss using BARTScore's embedding scoring"""
        device = expected_embeddings.device
        
        # Calculate scores for both endings
        score_edited = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings,
            edited_endings,
            batch_size=CONFIG["batch_size"],
            validation=validation
        )
        score_original = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings,
            original_endings,
            batch_size=CONFIG["batch_size"],
            validation=validation
        )


        final_loss = -score_edited.mean()

        return final_loss

    def training_step(self, batch, batch_idx):
        # Forward pass
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get expected embeddings
        expected_embeddings = self.forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=None
        )
        
        # Loss calculation
        dto_loss_val = self.dto_loss_embeds(
            expected_embeddings,
            [str(e) for e in batch['edited_ending']],
            [str(o) for o in batch['original_ending']]
        )
        
        # Ensure loss is scalar and has gradient
        dto_loss_val = dto_loss_val.mean() if dto_loss_val.dim() > 0 else dto_loss_val
        dto_loss_val.requires_grad_(True)

        # Distribution analysis
        stats = self.analyze_distributions(outputs.logits)

        # Primary metric logging (explicit for reliability)
        self.log("train/dto_loss", 
                dto_loss_val,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch['input_ids']))
        

        # Manual W&B sync every 10 steps (optional backup)
        if batch_idx % 50 == 0:
            wandb.log({
                "train/dto_loss_debug": dto_loss_val.item(),
                "step": self.global_step
            }, commit=False)
        
        return dto_loss_val   

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # CRITICAL - no gradients during validation
            # Forward pass
            expected_embeddings = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            dto_loss = self.dto_loss_embeds(
                expected_embeddings,
                [str(e) for e in batch['edited_ending']],
                [str(o) for o in batch['original_ending']],
                validation=True
            )   
                   
        # Log metrics
        self.log('val/dto_loss', dto_loss, prog_bar=True)
        
        return dto_loss.item()   # Return scalar value

    def test_step(self, batch, batch_idx):
        with torch.no_grad():  # No gradients needed for testing
            # 1. Generate text
            generated_tokens = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=CONFIG["max_gen_length"],
                num_beams=CONFIG.get("num_beams", 5),
                early_stopping=True
            )
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Convert all text inputs to strings
            edited_endings = [str(e) for e in batch['edited_ending']]
            counterfactuals = [str(c) for c in batch['counterfactual']]
            original_endings = [str(o) for o in batch['original_ending']]

            # 2. Compute all text metrics
            text_metrics = {}
            
            # BARTScore metrics
            bart_metrics = self.metrics_evaluator.calculate_bart_similarity(
                generated_texts=generated_texts,
                edited_endings=edited_endings,
                counterfactuals=counterfactuals,
                original_endings=original_endings
            )
            text_metrics.update({f'bart/{k}': v for k, v in bart_metrics.items()})
            
            # BERTScore metrics (if enabled)
            if self.metrics_evaluator.bert_scorer is not None:
                bert_metrics = self.metrics_evaluator.calculate_bert_similarity(
                    generated_texts=generated_texts,
                    edited_endings=edited_endings,
                    counterfactuals=counterfactuals,
                    original_endings=original_endings
                )
                text_metrics.update({f'bert/{k}': v for k, v in bert_metrics.items()})
            
            # BLEU metrics (if enabled)
            if self.metrics_evaluator.sacre_bleu is not None:
                bleu_metrics = self.metrics_evaluator.calculate_bleu_similarity(
                    generated_texts=generated_texts,
                    edited_endings=edited_endings,
                    counterfactuals=counterfactuals,
                    original_endings=original_endings
                )
                text_metrics.update({f'bleu/{k}': v for k, v in bleu_metrics.items()})
            
            # ROUGE metrics
            rouge_metrics = self.metrics_evaluator.calculate_rouge_similarity(
                generated_texts=generated_texts,
                edited_endings=edited_endings,
                counterfactuals=counterfactuals,
                original_endings=original_endings
            )
            text_metrics.update({f'rouge/{k}': v for k, v in rouge_metrics.items()})

            # 3. Calculate delta metrics for all metric types
            delta_metrics = {}
            
            # BARTScore Delta
            if all(k in bart_metrics for k in ['pred_edited_score', 'pred_original_score']):
                delta_metrics['bart_delta'] = (
                    2 * bart_metrics['pred_edited_score'] - 
                    bart_metrics['pred_original_score']
                )
            
            # BERTScore Delta (F1)
            if self.metrics_evaluator.bert_scorer is not None:
                if all(k in bert_metrics for k in ['prediction_edited_f1', 'prediction_original_f1']):
                    delta_metrics['bert_delta'] = (
                        2 * bert_metrics['prediction_edited_f1'] - 
                        bert_metrics['prediction_original_f1']
                    )
            
            # BLEU Delta
            if self.metrics_evaluator.sacre_bleu is not None:
                if all(k in bleu_metrics for k in ['prediction_edited', 'prediction_original']):
                    delta_metrics['bleu_delta'] = (
                        2 * bleu_metrics['prediction_edited'] - 
                        bleu_metrics['prediction_original']
                    )
            
            # ROUGE Delta (using ROUGE-L F1)
            if 'prediction_edited_rouge-l_f' in rouge_metrics and 'prediction_original_rouge-l_f' in rouge_metrics:
                delta_metrics['rouge_delta'] = (
                    2 * rouge_metrics['prediction_edited_rouge-l_f'] - 
                    rouge_metrics['prediction_original_rouge-l_f']
                )

            # Add all delta metrics to the main metrics dictionary
            text_metrics.update({f'delta/{k}': v for k, v in delta_metrics.items()})

            # 4. Log all metrics
            self.log_dict(
                {f'test/{k}': v for k, v in text_metrics.items()},
                prog_bar=True,
                logger=True,
                batch_size=len(batch['input_ids']),
                sync_dist=True
            )

            # 5. Store test details for epoch-end processing
            self.epoch_test_details.extend({
                'generated': gen,
                'edited': edit,
                'counterfactual': cf,
                'original': orig,
                'input_ids': ids.tolist() if torch.is_tensor(ids) else ids
            } for gen, edit, cf, orig, ids in zip(
                generated_texts,
                edited_endings,
                counterfactuals,
                original_endings,
                batch['input_ids']
            ))
        
        return None

    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        if self.epoch_validation_details:
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)
        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            self.log("validation_overall_score", overall_val_score, prog_bar=True, logger=True)
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()

    def on_test_epoch_end(self):
        """Save complete metrics including deltas to CSV"""
        if not hasattr(self, 'epoch_test_details') or not self.epoch_test_details:
            return

        # 1. Keep original text logging
        self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        # 2. Calculate ALL metrics (like in test_step)
        all_texts = [d['generated'] for d in self.epoch_test_details]
        all_edited = [d['edited'] for d in self.epoch_test_details]
        all_cf = [d['counterfactual'] for d in self.epoch_test_details]
        all_original = [d['original'] for d in self.epoch_test_details]

        # Initialize metrics dictionary
        all_metrics = {}

        # Calculate all metric types
        bart_metrics = self.metrics_evaluator.calculate_bart_similarity(all_texts, all_edited, all_cf, all_original)
        all_metrics.update({f'bart/{k}': v for k, v in bart_metrics.items()})

        if self.metrics_evaluator.bert_scorer is not None:
            bert_metrics = self.metrics_evaluator.calculate_bert_similarity(all_texts, all_edited, all_cf, all_original)
            all_metrics.update({f'bert/{k}': v for k, v in bert_metrics.items()})

        if self.metrics_evaluator.sacre_bleu is not None:
            bleu_metrics = self.metrics_evaluator.calculate_bleu_similarity(all_texts, all_edited, all_cf, all_original)
            all_metrics.update({f'bleu/{k}': v for k, v in bleu_metrics.items()})

        rouge_metrics = self.metrics_evaluator.calculate_rouge_similarity(all_texts, all_edited, all_cf, all_original)
        all_metrics.update({f'rouge/{k}': v for k, v in rouge_metrics.items()})

        # Calculate deltas (same logic as test_step)
        deltas = {}
        if all(k in bart_metrics for k in ['pred_edited_score', 'pred_original_score']):
            deltas['bart_delta'] = 2*bart_metrics['pred_edited_score'] - bart_metrics['pred_original_score']
        
        if (self.metrics_evaluator.bert_scorer and 
            all(k in bert_metrics for k in ['prediction_edited_f1', 'prediction_original_f1'])):
            deltas['bert_delta'] = 2*bert_metrics['prediction_edited_f1'] - bert_metrics['prediction_original_f1']
        
        if (self.metrics_evaluator.sacre_bleu and 
            all(k in bleu_metrics for k in ['prediction_edited', 'prediction_original'])):
            deltas['bleu_delta'] = 2*bleu_metrics['prediction_edited'] - bleu_metrics['prediction_original']
        
        if all(k in rouge_metrics for k in ['prediction_edited_rouge-l_f', 'prediction_original_rouge-l_f']):
            deltas['rouge_delta'] = 2*rouge_metrics['prediction_edited_rouge-l_f'] - rouge_metrics['prediction_original_rouge-l_f']
        
        all_metrics.update({f'delta/{k}': v for k, v in deltas.items()})

        # 3. Save complete metrics to CSV
        metrics_csv_path = self.model_dir / f"test_metrics_epoch_{self.current_epoch}.csv"
        pd.DataFrame([
            {'Metric': k, 'Score': v} 
            for k, v in all_metrics.items()
        ]).to_csv(metrics_csv_path, index=False)

        # 4. Cleanup
        self.epoch_test_details.clear()
        if hasattr(self, 'epoch_test_scores'):
            self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        file_exists = os.path.isfile(csv_file_path)
        fieldnames = details[0].keys() if details else []
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for detail in details:
                if epoch is not None:
                    detail['Epoch'] = epoch
            writer.writerows(details)
        logger.info(f"Successfully wrote {len(details)} entries to {csv_file_path}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Temperature annealing - only active when enabled in config"""
        if self.use_gumbel and CONFIG["gumbel_annealing"]:
            new_temp = max(self.temperature * CONFIG["gumbel_anneal_rate"], 
                        CONFIG["gumbel_min_temp"])
            self.temperature = new_temp
            if batch_idx % CONFIG["gumbel_log_freq"] == 0:
                self.log('gumbel/temperature', self.temperature)
 
    def analyze_distributions(self, logits, probs=None):
        """Analyze the shape of output distributions
        
        Args:
            logits: Raw model outputs (before softmax)
            probs: Pre-computed probabilities (optional)
        """
        if probs is None:
            probs = torch.softmax(logits, dim=-1)
        
        # Entropy calculation (higher = more uncertain)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Top-k probability mass
        top1_prob = probs.topk(1).values.mean()
        top5_prob = probs.topk(5).values.sum(dim=-1).mean()
        
        # Additional useful metrics
        max_prob = probs.max(dim=-1).values.mean()  
        min_prob = probs.min(dim=-1).values.mean()  
        
        return {
            'entropy_mean': entropy.mean(),
            'entropy_std': entropy.std(),
            'top1_prob': top1_prob,
            'top5_prob': top5_prob,
            'max_prob': max_prob,
            'min_prob': min_prob
        }