# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/dto/models/model.py
import csv
import logging
import os
import torch
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

        # TODO: Temporary variable for debugging (delete after checking successfull)
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.clone()

        logger.info("Initializing DTO mode...")
        # Load BART-based scoring model
        # self.bart_scorer = BartForConditionalGeneration.from_pretrained(CONFIG["bart_scorer_checkpoint"])
        # self.bart_scorer_tokenizer = BartTokenizer.from_pretrained(CONFIG["bart_scorer_checkpoint"])
        # self.bart_scorer.eval()
        # # Freeze BART scorer so it does not update during training
        # for param in self.bart_scorer.parameters():
        #     param.requires_grad = False

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        device = logits.device  # Ensure we use the same device for all tensors
        probs = torch.softmax(logits, dim=-1)

        # Get the embedding matrix and explicitly move it to the logits' device
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.to(device)

        # TODO: Temporary prints statement for debugging if paramters are frozen (delete after checking successfull)
        print(">> BART Scorer embeddings requires grad: ", embedding_matrix.requires_grad)
        # Another print statement to check if embeddings are the same as the original model before training starts
        print(">> BART Scorer embeddings are the same as the original model: ", torch.equal(embedding_matrix, self.bart_scorer_og_embed.to(device)))

        expected_embeddings = torch.matmul(probs, embedding_matrix)
        print(">> Computed expected embeddings in DTO mode")
        return expected_embeddings

    def dto_loss_embeds(self, expected_embeddings, edited_endings):
        print(">> Computing DTO loss from expected embeddings")
        # TODO: Check if the BART scorer is in eval mode (delete after checking successfull)
        for param in self.metrics_evaluator.bart_scorer.model.parameters():
            if param.requires_grad:
                raise ValueError("BART Scorer model is not in eval mode")

        score_tensor = self.metrics_evaluator.calculate_score_embeds(expected_embeddings, edited_endings)
        loss = -score_tensor.mean()
        print(f">> DTO loss computed: {loss.item():.4f}")
        if not loss.requires_grad:
            loss.requires_grad = True
        return loss
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        expected_embeddings = self.forward(input_ids=input_ids, 
                                        attention_mask=attention_mask, 
                                        labels=None)
        
        # Loss calculation
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        dto_loss_val = self.dto_loss_embeds(expected_embeddings, edited_endings)
        
        # Organized metrics
        metrics = {
            'train/dto_loss': dto_loss_val,
            'train/embed_mean': expected_embeddings.mean().detach().cpu(),
            'train/embed_std': expected_embeddings.std().detach().cpu(),
            'train/embed_min': expected_embeddings.min().detach().cpu(),
            'train/embed_max': expected_embeddings.max().detach().cpu()
        }
        
        # Strategic logging
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar={'train/dto_loss': True},  # Only show loss in progress bar
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

        # ===== SOFT VALIDATION TRACK =====
        # 2. DTO Loss (same as training)
        dto_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)

        # 3. Embedding-based Metrics
        embed_edited = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, edited_endings
        ).mean()
        embed_cf = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, counterfactuals
        ).mean()

        # ===== HARD VALIDATION TRACK =====
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
            'val/dto_loss': dto_loss,
            'val/embed_edited': embed_edited,
            'val/embed_cf': embed_cf,
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
                'soft_embed_mean': expected_embeddings[i].mean().item(),
                'soft_embed_std': expected_embeddings[i].std().item(),
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

        # ===== SOFT TEST TRACK =====
        # 2. DTO Loss (same as training)
        dto_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)

        # 3. Embedding-based Metrics
        embed_edited = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, edited_endings
        ).mean()
        embed_cf = self.metrics_evaluator.calculate_score_embeds(
            expected_embeddings, counterfactuals
        ).mean()

        # ===== HARD TEST TRACK =====
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
            'test/dto_loss': dto_loss,
            'test/embed_edited': embed_edited,
            'test/embed_cf': embed_cf,
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
                'soft_embed_mean': expected_embeddings[i].mean().item(),
                'soft_embed_std': expected_embeddings[i].std().item(),
                'dto_loss': dto_loss.item(),
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


