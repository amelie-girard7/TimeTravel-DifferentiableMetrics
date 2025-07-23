import csv
import logging
import os
import wandb
import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.cpo.utils.config import CONFIG
from src.cpo.utils.metrics import MetricsEvaluator
import pandas as pd

logger = logging.getLogger(__name__)

class BartFineTuner(pl.LightningModule):
    """
    Key Mathematical Components:
    1. For each input story x, we have:
       - Preferred sequence (y_w): Human-edited counterfactual ending
       - Dispreferred sequence (y_l): Original ending to discourage
    2. Loss: L_CPO = -E[logσ(β(log p_w - log p_l)) + log p_w]
       where β controls preference sharpness
    """
    
    def __init__(self, model_name, model_dir, file_label=""):
        super().__init__()
        self.save_hyperparameters()
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

        # Buffers for logging details.
        self.epoch_validation_details = []
        self.epoch_scores = []
        self.epoch_test_details = []
        self.epoch_test_scores = []
       
        logger.info("Model initialized successfully")

    def tokenize_original_endings(self, text_endings):
        """Tokenize original endings on-the-fly"""
        return self.tokenizer(
            text_endings,
            padding='longest',
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt"
        ).input_ids.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        """Standard forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def compute_sequence_log_prob(self, input_ids, attention_mask, target_ids):
        """Now explicitly uses forward()"""
        outputs = self.forward(input_ids, attention_mask, target_ids)
        return -outputs.loss * (target_ids != self.tokenizer.pad_token_id).sum(dim=-1)
       
    def cpo_loss(self, batch):
        """Modified CPO loss that handles original ending tokenization internally"""
        # Tokenize original endings (yₗ)
        original_ids = self.tokenize_original_endings(batch['original_ending'])
        
        # Compute log p_w for edited endings (y_w)
        log_p_w = self.compute_sequence_log_prob(
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels']  # Edited ending (pre-tokenized)
        )
        
        log_p_l = self.compute_sequence_log_prob(
            batch['input_ids'],
            batch['attention_mask'],
            original_ids
        )
        
        beta = CONFIG.get("beta", 1.0)
        lamda = CONFIG.get("lamda", 1.0)
        
        delta = log_p_w - log_p_l
        return -(F.logsigmoid(beta * delta) + lamda * log_p_w).mean()

    def training_step(self, batch, batch_idx): 

        # Loss calculation
        cpo_loss = self.cpo_loss(batch)
        
        # Primary metric logging (explicit for reliability)
        self.log("train/cpo_loss", 
                cpo_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch['input_ids']))


        # Manual W&B sync every 10 steps (optional backup)
        if batch_idx % 50 == 0:
            wandb.log({
                "train/cpo_loss_debug": cpo_loss.item(),
                "step": self.global_step
            }, commit=False)
        
        return cpo_loss   

    def validation_step(self, batch, batch_idx):

        loss = self.cpo_loss(batch)
        self.log('val/cpo_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
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