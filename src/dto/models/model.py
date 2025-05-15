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
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load the main generation model and tokenizer.
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Paths for CSV logging.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Buffers for logging details.
        self.epoch_validation_details = []
        self.epoch_scores = []
        self.epoch_test_details = []
        self.epoch_test_scores = []
        self.metrics_evaluator = MetricsEvaluator()

        # Used in the print
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.clone()

        # Gumbel-Softmax parameters from config
        self.use_gumbel = CONFIG["use_gumbel"]
        self.temperature = CONFIG["gumbel_temperature"]
        self.gumbel_hard = CONFIG["gumbel_hard"]
        self.anneal_rate = CONFIG["gumbel_anneal_rate"] if CONFIG["gumbel_annealing"] else None
        self.min_temp = CONFIG["gumbel_min_temp"]

        logger.info("Initializing DTO mode...")

        logger.info(f"Model initialized: {model_name}")

    def train(self, mode=True):
        """
        Override the train method to ensure `bart_scorer` remains in evaluation mode.
        """
        super().train(mode)
        self.metrics_evaluator.bart_scorer.model.eval()
        print(">> Setting BART scorer to eval()")
        return self

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        
        print(">> Forward pass in DTO mode")
        # Add input validation
        print(f">> Input IDs shape: {input_ids.shape}")
        print(f">> Attention mask shape: {attention_mask.shape}")
        print(f">> Input IDs sample: {input_ids[0][:10]}...")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Add logging for model outputs
        print(f">> Output logits shape: {outputs.logits.shape}")
        print(f">> Logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")

        logits = outputs.logits  # [batch, seq_len, vocab_size]
        device = logits.device  # Ensure we use the same device for all tensors


        if self.use_gumbel:
            print(f">> Applying Gumbel-Softmax with temperature={self.temperature:.4f}")
            probs = self.gumbel_softmax(logits)
            
            # Log temperature and distribution stats
            stats = self.analyze_distributions(logits, probs)
        else:
            probs = torch.softmax(logits, dim=-1)
    

        # Get the embedding matrix and explicitly move it to the logits' device
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.to(device)

        # Enhanced embedding checks
        print(">> BART Scorer embeddings stats:")
        print(f"  - Shape: {embedding_matrix.shape}")
        print(f"  - Requires grad: {embedding_matrix.requires_grad}")
        print(f"  - Mean: {embedding_matrix.mean().item():.4f}")
        print(f"  - Std: {embedding_matrix.std().item():.4f}")
        
        # Compare with original embeddings
        diff = torch.abs(embedding_matrix - self.bart_scorer_og_embed.to(device))
        print(f">> Max diff from original: {diff.max().item():.4f}")
        print(f">> Mean diff from original: {diff.mean().item():.4f}")

        expected_embeddings = torch.matmul(probs, embedding_matrix)
        print(">> Expected embeddings stats:")
        print(f"  - Shape: {expected_embeddings.shape}")
        print(f"  - Mean: {expected_embeddings.mean().item():.4f}")
        print(f"  - Std: {expected_embeddings.std().item():.4f}")
        print(f"  - Min: {expected_embeddings.min().item():.4f}")
        print(f"  - Max: {expected_embeddings.max().item():.4f}")

        return expected_embeddings

    # def dto_loss_embeds(self, expected_embeddings, edited_endings):
    #     print(">> Computing DTO loss from expected embeddings")
    #     # Enhanced logging (NO functional changes)
    #     print(f">> Expected embeddings device: {expected_embeddings.device}")
    #     print(f">> Expected embeddings stats - mean: {expected_embeddings.mean().item():.4f}, "
    #         f"std: {expected_embeddings.std().item():.4f}")
    #     print(f">> Sample edited ending: {edited_endings[0][:100]}{'...' if len(edited_endings[0]) > 100 else ''}")

    #     # Original parameter check (unchanged)
    #     for param in self.metrics_evaluator.bart_scorer.model.parameters():
    #         if param.requires_grad:
    #             raise ValueError("BART Scorer model is not in eval mode")
            
    #     # Additional verification logging
    #     frozen_params = sum(1 for p in self.metrics_evaluator.bart_scorer.model.parameters() if not p.requires_grad)
    #     total_params = sum(1 for _ in self.metrics_evaluator.bart_scorer.model.parameters())
    #     print(f">> BART Scorer verification - Frozen params: {frozen_params}/{total_params}")

    #     # Original score calculation (unchanged)
    #     score_tensor = self.metrics_evaluator.calculate_score_embeds(expected_embeddings, edited_endings)

    #      # Additional score statistics logging
    #     print(f">> Score tensor stats - mean: {score_tensor.mean().item():.4f}, "
    #         f"std: {score_tensor.std().item():.4f}, "
    #         f"min: {score_tensor.min().item():.4f}, "
    #         f"max: {score_tensor.max().item():.4f}")

    #     # Original loss calculation (unchanged)
    #     loss = -score_tensor.mean()
    #     print(f">> DTO loss computed: {loss.item():.4f}")

    #     # Original gradient check (unchanged)
    #     if not loss.requires_grad:
    #         print("!!! Warning: Loss tensor requires_grad=False, forcing to True !!!")
    #         loss.requires_grad = True
    #     return loss

    def dto_loss_embeds(self, expected_embeddings, edited_endings):
        """Compute DTO loss with gradient preservation and debugging"""
        
        # ===== 1. Input Validation =====
        print("\n=== DTO Loss Debug ===")
        print(f"Input embeddings shape: {expected_embeddings.shape}")
        print(f"Input embeddings requires_grad: {expected_embeddings.requires_grad}")
        print(f"Input embeddings device: {expected_embeddings.device}")
        
        # ===== 2. Gradient Preservation =====
        if not expected_embeddings.requires_grad:
            print("!! WARNING: Input embeddings don't require grad - cloning with requires_grad=True !!")
            expected_embeddings = expected_embeddings.clone().requires_grad_(True)
        
        # ===== 3. BART Scorer Verification =====
        print("\n--- BART Scorer Status ---")
        frozen_params = sum(1 for p in self.metrics_evaluator.bart_scorer.model.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.metrics_evaluator.bart_scorer.model.parameters())
        print(f"Frozen params: {frozen_params}/{total_params}")
        print(f"Scorer model training mode: {self.metrics_evaluator.bart_scorer.model.training}")
        
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
        
        # Force gradient retention if needed (shouldn't be necessary with proper flow)
        if not loss.requires_grad:
            print("!! WARNING: Loss requires_grad=False - creating new tensor with gradients !!")
            loss = loss.clone().requires_grad_(True)
        
        # ===== 6. Final Checks =====
        print("\n--- Final Verification ---")
        print(f"Loss value: {loss.item():.4f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        print(f"Loss device: {loss.device}")
        
        # Verify gradient chain
        if loss.grad_fn is None:
            print("!! CRITICAL WARNING: Loss has no gradient function !!")
            print("Full grad_fn chain:")
            current = expected_embeddings.grad_fn
            while current is not None:
                print(f"  - {str(current)}")
                current = current.next_functions[0][0] if current.next_functions else None
        else:
            print("Youpiii Valid gradient chain detected")
        
        return loss

    def training_step(self, batch, batch_idx):
        # Enhanced batch logging
        print(f"\n=== Training Step {batch_idx} ===")
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Sample edited ending: {batch['edited_ending'][0]}")

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
        # Unpack batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        premises = batch['premise']
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        originals = [str(oe) for oe in batch['original_ending']]

        # 1. Differentiable Forward Pass (for both tracks)
        expected_embeddings = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 2. DTO Loss (same as training)
        dto_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)

        # 3. Embedding-based Metrics
        embed_edited = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, edited_endings
        ).mean()

        # 4. Text Generation
        with torch.no_grad():
            # Generate hard tokens
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG["max_gen_length"],
                num_beams=4, # Better quality than greedy
                early_stopping=True  # For generation only
            )
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

        # 5. Text-based Metrics
        text_metrics = self.metrics_evaluator.calculate_bart_similarity(
            generated_texts=generated_texts,
            edited_endings=edited_endings,
            counterfactuals=counterfactuals,
            initials=initials,
            original_endings=originals
        )

        # ===== COMBINED LOGGING =====
        self.log_dict({
            # Soft track metrics
            'val/dto_loss': dto_loss,
            'val/embed_edited': embed_edited,
            # Hard track metrics
            **{f'val/text_{k}': v for k, v in text_metrics.items()}
        }, prog_bar=True)

        # Store details including both soft and hard results
        for i in range(len(generated_texts)):
            self.epoch_validation_details.append({
                'epoch': self.current_epoch,
                # Hard track info
                'premise': premises[i],
                'generated': generated_texts[i],
                'edited': edited_endings[i],
                'counterfactual': counterfactuals[i],
                'original': originals[i],
                # Soft track info
                # 'soft_embed_mean': expected_embeddings[i].mean().item(),
                # 'soft_embed_std': expected_embeddings[i].std().item(),
                'dto_loss': dto_loss.item(),
                **{f'metric_{k}': v for k, v in text_metrics.items()}
            })

        return dto_loss  # <= Return only dto_loss so early stopping can track it

    def test_step(self, batch, batch_idx):
        # Unpack batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        premises = batch['premise']
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        originals = [str(oe) for oe in batch['original_ending']]

        # 1. Differentiable Forward Pass (for both tracks)
        expected_embeddings = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 2. DTO Loss (same as training)
        dto_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)

        # 3. Embedding-based Metrics
        embed_edited = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, edited_endings
        ).mean()

        # 4. Text Generation
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG["max_gen_length"]
            )
            generated_texts = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )

        # 5. Text-based Metrics
        text_metrics = self.metrics_evaluator.calculate_bart_similarity(
            generated_texts=generated_texts,
            edited_endings=edited_endings,
            counterfactuals=counterfactuals,
            initials=initials,
            original_endings=originals
        )

        # ===== COMBINED LOGGING =====
        self.log_dict({
            # Soft track metrics
            # 'test/dto_loss': dto_loss,
            # 'test/embed_edited': embed_edited,
            # Hard track metrics
            **{f'test/text_{k}': v for k, v in text_metrics.items()}
        }, prog_bar=True)

        # Store details including both soft and hard results
        for i in range(len(generated_texts)):
            self.epoch_test_details.append({
                'epoch': self.current_epoch,
                # Hard track info
                'premise': premises[i],
                'generated': generated_texts[i],
                'edited': edited_endings[i],
                'counterfactual': counterfactuals[i],
                'original': originals[i],
                # Soft track info
                # 'soft_embed_mean': expected_embeddings[i].mean().item(),
                # 'soft_embed_std': expected_embeddings[i].std().item(),
                # 'dto_loss': dto_loss.item(),
                **{f'metric_{k}': v for k, v in text_metrics.items()}
            })

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