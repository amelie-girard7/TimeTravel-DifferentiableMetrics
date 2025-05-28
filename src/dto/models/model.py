# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/dto/models/model.py
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
        # print(f"\n=== Device Configuration ===")
        # print(f"Target computation device: {self.target_device}")

        # 3. Load main model components
        # print("\n=== Main Model Loading ===")
        # print("Loading BART model and tokenizer...")
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        # 4. Initialize metrics evaluator
        # print("\n=== Metrics Evaluator Setup ===")
        # print("Initializing MetricsEvaluator...")
        self.metrics_evaluator = MetricsEvaluator()
        self.metrics_evaluator.bart_scorer._wandb = lambda: wandb  # Direct wandb access
        
        
        # Paths for CSV logging.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # 5. Store original embeddings
        # print("\n=== Embedding Configuration ===")
        # print("Storing original BART scorer embeddings...")
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.clone()

        # 6. Gumbel-Softmax setup
        # print("\n=== Gumbel-Softmax Configuration ===")
        self.use_gumbel = CONFIG["use_gumbel"]
        self.temperature = CONFIG["gumbel_temperature"]
        self.gumbel_hard = CONFIG["gumbel_hard"]
        self.anneal_rate = CONFIG["gumbel_anneal_rate"] if CONFIG["gumbel_annealing"] else None
        self.min_temp = CONFIG["gumbel_min_temp"]

        # 7. Move everything to target device
        # print("\n=== Device Synchronization ===")
        # print("Moving all components to target device...")
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
        
        # print("\n[Parameter Status]")
        # frozen_main = sum(1 for p in self.model.parameters() if not p.requires_grad)
        # total_main = sum(1 for _ in self.model.parameters())
        # frozen_scorer = sum(1 for p in self.metrics_evaluator.bart_scorer.model.parameters() if not p.requires_grad)
        # total_scorer = sum(1 for _ in self.metrics_evaluator.bart_scorer.model.parameters())
        # print(f"Main model - Frozen: {frozen_main}/{total_main}")
        # print(f"Scorer model - Frozen: {frozen_scorer}/{total_scorer}")
       
        # print("\n=== Initialization Complete ===")
        logger.info("Model initialized successfully")
        
    def train(self, mode=True):
        """
        Override the train method to ensure `bart_scorer` remains in evaluation mode.
        """
        super().train(mode)
        self.metrics_evaluator.bart_scorer.model.eval()
        # print(">> Setting BART scorer to eval()")
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

        # ===== 3. Debug Prints =====
        # print("\n>> Forward Pass")
        # print(f"Logits: {logits.shape} (device: {device})")
        # print(f"Range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # ===== 4. Gumbel-Softmax =====
        
        probs = F.gumbel_softmax(
            logits,
            tau=CONFIG["gumbel_temperature"],  
            hard=CONFIG["gumbel_hard"],       
            dim=-1
        )
        # print(f"Gumbel Probs: μ={probs.mean().item():.4f} σ={probs.std().item():.4f}")
        # print(f"Sum check (should be ~1.0): {probs.sum(dim=-1).mean().item():.4f}")
            
        
        # ===== 5. Embedding Projection =====
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.to(device)
        expected_embeddings = torch.matmul(probs, embedding_matrix)

        # print("\n>> Output Embeddings")
        # print(f"Shape: {expected_embeddings.shape}")
        # print(f"Grad: {'ON' if expected_embeddings.requires_grad else 'OFF'}")
        # print(f"Stats: μ={expected_embeddings.mean().item():.4f} (Δmax={torch.abs(embedding_matrix - self.bart_scorer_og_embed.to(device)).max().item():.4f})")

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

        # Numerical stability
        # score_edited = torch.clamp(score_edited, -5.0, 5.0)
        # score_original = torch.clamp(score_original, -5.0, 5.0)
        # delta = torch.clamp(score_edited - score_original, -10.0, 10.0)
        
        # Reward calculation
        delta = score_edited - score_original
        rewards = score_edited + delta
        # baseline = rewards.mean()
        #centered_rewards = rewards - baseline
        final_loss = -rewards.mean()
        return final_loss

    def training_step(self, batch, batch_idx):
        # Initialize metrics
        #total_grad_norm = 0.0  # Initialize here to ensure it always exists
        
        # Training step header
        # print(f"\n=== Training Step {batch_idx} === [Epoch {self.current_epoch}]")
        # print(f"Device: {self.device} | Batch Size: {len(batch['input_ids'])}")
        
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
        
        # Embedding stats
        # print(f"\n[Embeddings] μ={expected_embeddings.mean().item():.4f} σ={expected_embeddings.std().item():.4f}")
        # print(f"[Embeddings] Range: [{expected_embeddings.min().item():.4f}, {expected_embeddings.max().item():.4f}]")

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
        # print(f"[Distrib] Entropy: {stats['entropy_mean']:.4f} | Top1: {stats['top1_prob']:.4f}")

        # Primary metric logging (explicit for reliability)
        self.log("train/dto_loss", 
                dto_loss_val,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=len(batch['input_ids']))
        
        # Secondary metrics (grouped for efficiency)
        self.log_dict({
            'train/embed_mean': expected_embeddings.mean(),
            'train/embed_std': expected_embeddings.std(),
            'train/entropy': stats['entropy_mean'],
            #'train/grad_norm': torch.tensor(total_grad_norm, device=self.device),
        }, logger=True)
        
        # Gumbel-specific metrics
        if self.use_gumbel:
            self.log_dict({
                'gumbel/temperature': torch.tensor(self.temperature, device=self.device),
                'gumbel/top1_prob': stats['top1_prob'],
                'gumbel/top5_prob': stats['top5_prob'],
            }, logger=True)

        # Manual W&B sync every 10 steps (optional backup)
        if batch_idx % 10 == 0:
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
            
            # print(f"Validation mode - expected_embeddings requires_grad: {expected_embeddings.requires_grad}")

            # Add validation=True flag
            # dto_loss = self.dto_loss_embeds(
            #     expected_embeddings, 
            #     batch['edited_ending'],
            #     validation=True
            # )

            dto_loss = self.dto_loss_embeds(
                expected_embeddings,
                [str(e) for e in batch['edited_ending']],
                [str(o) for o in batch['original_ending']],
                validation=True
            )   

            # Debug print 2 - check score gradient status
            # print(f"Validation mode - dto_loss requires_grad: {dto_loss.requires_grad}")
                   
        # Log metrics
        self.log('val/dto_loss', dto_loss, prog_bar=True)
        
        return dto_loss.item()   # Return scalar value

    def test_step(self, batch, batch_idx):
        with torch.no_grad():  # No gradients needed for testing
            # 1. Generate text
            generated_tokens = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=CONFIG["max_gen_length"]
            )
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # 2. Compute text metrics 
            text_metrics = self.metrics_evaluator.calculate_bart_similarity(
                generated_texts=generated_texts,
                edited_endings=[str(e) for e in batch['edited_ending']],
                counterfactuals=[str(c) for c in batch['counterfactual']],
                initials=[str(i) for i in batch['initial']],
                original_endings=[str(o) for o in batch['original_ending']]
            )

            # 3. Log only the metrics you care about
            self.log_dict(
                {f'test/{k}': v for k, v in text_metrics.items()},
                prog_bar=True
            )

            # 4. Store minimal test details if needed
            self.epoch_test_details.extend({
                'generated': gen,
                'edited': edit,
                'counterfactual': cf,
                'original': orig
            } for gen, edit, cf, orig in zip(
                generated_texts,
                batch['edited_ending'],
                batch['counterfactual'],
                batch['original_ending']
            ))
        return None

    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        print(">>Validation Epoch End")
        if self.epoch_validation_details:
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)
        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            self.log("validation_overall_score", overall_val_score, prog_bar=True, logger=True)
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()

    def on_test_epoch_end(self):
        if self.epoch_test_details:
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)
        if self.epoch_test_scores:
            overall_test_score = torch.tensor(self.epoch_test_scores).mean().item()
            self.log("test_overall_score", overall_test_score, prog_bar=True, logger=True)
        self.epoch_test_details.clear()
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
        max_prob = probs.max(dim=-1).values.mean()  # Average max probability
        min_prob = probs.min(dim=-1).values.mean()  # Average min probability
        
        return {
            'entropy_mean': entropy.mean(),
            'entropy_std': entropy.std(),
            'top1_prob': top1_prob,
            'top5_prob': top5_prob,
            'max_prob': max_prob,
            'min_prob': min_prob
        }