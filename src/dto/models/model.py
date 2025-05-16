# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/dto/models/model.py
import csv
import logging
import os
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
        print(f"\n=== Device Configuration ===")
        print(f"Target computation device: {self.target_device}")

        # 3. Load main model components
        print("\n=== Main Model Loading ===")
        print("Loading BART model and tokenizer...")
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        # 4. Initialize metrics evaluator
        print("\n=== Metrics Evaluator Setup ===")
        print("Initializing MetricsEvaluator...")
        self.metrics_evaluator = MetricsEvaluator()
        
        # Paths for CSV logging.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # 5. Store original embeddings
        print("\n=== Embedding Configuration ===")
        print("Storing original BART scorer embeddings...")
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.clone()

        # 6. Gumbel-Softmax setup
        print("\n=== Gumbel-Softmax Configuration ===")
        self.use_gumbel = CONFIG["use_gumbel"]
        self.temperature = CONFIG["gumbel_temperature"]
        self.gumbel_hard = CONFIG["gumbel_hard"]
        self.anneal_rate = CONFIG["gumbel_anneal_rate"] if CONFIG["gumbel_annealing"] else None
        self.min_temp = CONFIG["gumbel_min_temp"]

        # 7. Move everything to target device
        print("\n=== Device Synchronization ===")
        print("Moving all components to target device...")
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

        # 10. Verification and Debug Output
        # print("\n=== System Verification ===")
        # print("[Component Devices]")
        # print(f"Main model: {next(self.model.parameters()).device}")
        # print(f"Scorer model: {next(self.metrics_evaluator.bart_scorer.model.parameters()).device}")
        # print(f"Embeddings: {self.bart_scorer_og_embed.device}")
        # print(f"Tokenizer: {getattr(self.tokenizer, 'device', 'N/A')}")
        
        # print("\n[Training Modes]")
        # print(f"Main model training: {self.model.training}")
        # print(f"Scorer model training: {self.metrics_evaluator.bart_scorer.model.training}")
        
        print("\n[Parameter Status]")
        frozen_main = sum(1 for p in self.model.parameters() if not p.requires_grad)
        total_main = sum(1 for _ in self.model.parameters())
        frozen_scorer = sum(1 for p in self.metrics_evaluator.bart_scorer.model.parameters() if not p.requires_grad)
        total_scorer = sum(1 for _ in self.metrics_evaluator.bart_scorer.model.parameters())
        print(f"Main model - Frozen: {frozen_main}/{total_main}")
        print(f"Scorer model - Frozen: {frozen_scorer}/{total_scorer}")
        
        # print("\n[Gumbel Parameters]")
        # print(f"Use Gumbel: {self.use_gumbel}")
        # print(f"Temperature: {self.temperature}")
        # print(f"Hard: {self.gumbel_hard}")
        # print(f"Annealing: {'Enabled' if CONFIG['gumbel_annealing'] else 'Disabled'}")
        # if CONFIG["gumbel_annealing"]:
        #     print(f"  Rate: {self.anneal_rate}")
        #     print(f"  Min temp: {self.min_temp}")
        
        # print("\n[Logging Paths]")
        # print(f"Validation log: {self.val_csv_file_path}")
        # print(f"Test log: {self.test_csv_file_path}")
        # print(f"Directory exists: {self.model_dir.exists()}")
        
        print("\n=== Initialization Complete ===")
        logger.info("Model initialized successfully")
        
    def train(self, mode=True):
        """
        Override the train method to ensure `bart_scorer` remains in evaluation mode.
        """
        super().train(mode)
        self.metrics_evaluator.bart_scorer.model.eval()
        print(">> Setting BART scorer to eval()")
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
        print("\n>> Forward Pass")
        print(f"Logits: {logits.shape} (device: {device})")
        print(f"Range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # ===== 4. Gumbel-Softmax =====
        probs = self.gumbel_softmax(logits)
        print(f"Probs: μ={probs.mean().item():.4f} σ={probs.std().item():.4f}")
            
        
        # ===== 4. Gumbel-Softmax =====
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.to(device)
        expected_embeddings = torch.matmul(probs, embedding_matrix)

        print("\n>> Output Embeddings")
        print(f"Shape: {expected_embeddings.shape}")
        print(f"Grad: {'ON' if expected_embeddings.requires_grad else 'OFF'}")
        print(f"Stats: μ={expected_embeddings.mean().item():.4f} (Δmax={torch.abs(embedding_matrix - self.bart_scorer_og_embed.to(device)).max().item():.4f})")

        return expected_embeddings

    def dto_loss_embeds(self, expected_embeddings, edited_endings, validation=False):
        """Compute DTO loss with gradient preservation and debugging"""
        print(f"\n=== DTO Loss ({'VALIDATION' if validation else 'TRAINING'}) ===")
        
        # ===== 1. Input Validation =====
        # print("\n=== DTO Loss Debug ===")
        # print(f"Input embeddings shape: {expected_embeddings.shape}")
        # print(f"Input embeddings requires_grad: {expected_embeddings.requires_grad}")
        # print(f"Input embeddings device: {expected_embeddings.device}")
        
        # ===== 2. Gradient Preservation =====
        if validation:
            # For validation, ensure no gradients
            with torch.no_grad():
                expected_embeddings = expected_embeddings.detach()
                print("Validation mode - gradients disabled")
        else:
            # For training, ensure gradients
            if not expected_embeddings.requires_grad:
                print("Training mode - enabling gradients for embeddings")
                expected_embeddings = expected_embeddings.requires_grad_(True)
        
        # ===== 3. BART Scorer Verification =====
        # print("\n--- BART Scorer Status ---")
        # frozen_params = sum(1 for p in self.metrics_evaluator.bart_scorer.model.parameters() if not p.requires_grad)
        # total_params = sum(1 for _ in self.metrics_evaluator.bart_scorer.model.parameters())
        # print(f"Frozen params: {frozen_params}/{total_params}")
        # print(f"Scorer model training mode: {self.metrics_evaluator.bart_scorer.model.training}")
        
        # ===== 4. Score Calculation =====
        print("\n--- Calculating Scores ---")
        try:
            score_tensor = self.metrics_evaluator.calculate_score_embeds(
                expected_embeddings, 
                edited_endings
            )
            
            # Debug score tensor
            print(f"Score tensor shape: {score_tensor.shape}")
            print(f"Score tensor requires_grad: {score_tensor.requires_grad}")
            print(f"Scores (first 5): {score_tensor[:5].detach().cpu().numpy()}")
            print(f"Mean score: {score_tensor.mean().item():.4f}")
            
        except Exception as e:
            print(f"!! ERROR in score calculation: {str(e)}")
            raise

        # ===== 5. Loss Computation =====
        print("\n--- Loss Computation ---")
        loss = -score_tensor.mean()
        
        # ===== 6. Final Checks =====
        print("\n--- Final Verification ---")
        print(f"Loss value: {loss.item():.4f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss device: {loss.device}")
        
        # Verify gradient chain
        if loss.grad_fn is None:
            print("!! CRITICAL WARNING: Loss has no gradient function !!")
        else:
            print("Valid gradient chain detected")
        
        return loss

    def dto_loss_embeds(self, expected_embeddings, edited_endings, validation=False):
        """Compute DTO loss with proper gradient handling"""
        if validation:
            # For validation, ensure no gradients
            with torch.no_grad():
                expected_embeddings = expected_embeddings.detach()
        else:
            # For training, ensure gradients
            if not expected_embeddings.requires_grad:
                expected_embeddings = expected_embeddings.requires_grad_(True)
        
        # Tokenize targets
        encoded_tgt = self.metrics_evaluator.bart_scorer.tokenizer(
            edited_endings,
            max_length=self.metrics_evaluator.bart_scorer.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(expected_embeddings.device)
        
        # Forward pass
        output = self.metrics_evaluator.bart_scorer.model(
            inputs_embeds=expected_embeddings,
            attention_mask=torch.ones_like(expected_embeddings[..., 0]),
            labels=encoded_tgt['input_ids']
        )
        
        return -output.loss  # Negative because we want to maximize the score  

    def training_step(self, batch, batch_idx):
        # Enhanced batch logging
        print(f"\n=== Training Step {batch_idx} ===")
        # print(f"Batch keys: {batch.keys()}")
        # print(f"Input IDs shape: {batch['input_ids'].shape}")
        # print(f"Sample edited ending: {batch['edited_ending'][0]}")

        # Forward pass
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        outputs = self.model(  # Store the full outputs to access logits later
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
        )

        # Get expected embeddings (this handles Gumbel-Softmax internally)
        expected_embeddings = self.forward(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        labels=None)
        
        # Loss calculation
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        dto_loss_val = self.dto_loss_embeds(expected_embeddings, edited_endings)

        # Gradient Debugging
        if batch_idx % 10 == 0:
            print(f"\n--- Gradient Report ---")
            total_grad_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    print(f"{name:60} grad_norm: {grad_norm:.6f}")
                else:
                    print(f"{name:60} NO GRADIENT")
            
            print(f"Total Gradient Norm: {total_grad_norm:.4f}")
            print(f"DTO Loss: {dto_loss_val.item():.4f} (requires_grad={dto_loss_val.requires_grad})")

        # Get distribution statistics from the raw outputs
        stats = self.analyze_distributions(outputs.logits)
        
        # Organized metrics - single dictionary for all logging
        metrics = {
            'train/dto_loss': dto_loss_val,
            'train/embed_mean': expected_embeddings.mean().detach().cpu(),
            'train/embed_std': expected_embeddings.std().detach().cpu(),
            'train/entropy': stats['entropy_mean'],
        }
        
        # Only add Gumbel metrics if enabled
        if self.use_gumbel:
            metrics.update({
                'gumbel/temperature': torch.tensor(self.temperature, device=self.device),
                'gumbel/top1_prob': stats['top1_prob'],
            })
            
        # Single logging call
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar={'train/dto_loss': True},
            logger=True
        )
        
        return dto_loss_val

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():  # CRITICAL - no gradients during validation
            # Forward pass
            expected_embeddings = self.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            print(f"Validation mode - expected_embeddings requires_grad: {expected_embeddings.requires_grad}")

            # Add validation=True flag
            dto_loss = self.dto_loss_embeds(
                expected_embeddings, 
                batch['edited_ending'],
                validation=True
            )
            
            # Debug print 2 - check score gradient status
            print(f"Validation mode - dto_loss requires_grad: {dto_loss.requires_grad}")
                   
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

    def gumbel_softmax(self, logits, temperature=None, hard=None):
        """
        Apply Gumbel-Softmax to log probabilities.
        
        Args:
            logits: Model output logits
            temperature: Optional override of config temperature
            hard: Optional override of config hard setting
        """
        if temperature is None:
            temperature = self.temperature
        if hard is None:
            hard = self.gumbel_hard
            
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20))
        y = logits + gumbel_noise.to(logits.device)
        
        # Apply softmax with temperature
        probs = F.softmax(y / temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            _, indices = probs.max(dim=-1)
            probs_hard = torch.zeros_like(probs).scatter_(-1, indices.unsqueeze(-1), 1.0)
            probs = (probs_hard - probs).detach() + probs
            
        return probs
    
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
    
  