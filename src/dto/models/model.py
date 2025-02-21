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

class FlanT5FineTuner(pl.LightningModule):
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
        self.bart_scorer_og_embed = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight.deepcopy()

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
        probs = torch.softmax(logits, dim=-1)
        embedding_matrix = self.metrics_evaluator.bart_scorer.model.get_input_embeddings().weight
        # TODO: Temporary prints statement for debugging if paramters are frozen (delete after checking successfull)
        print(">> BART Scorer embeddings requires grad: ",
              embedding_matrix.requires_grad)
        # Another print statement to check if embeddings are the same as the original model before training starts
        print(">> BART Scorer embeddings are the same as the original model: ",
              torch.equal(embedding_matrix, self.bart_scorer_og_embed))

        expected_embeddings = torch.matmul(probs, embedding_matrix)
        print(">> Computed expected embeddings in DTO mode")
        return expected_embeddings

    def dto_loss_embeds(self, expected_embeddings, edited_endings):
        print(">> Computing DTO loss from expected embeddings")
        # TODO: Check if the BART scorer is in eval mode (delete after checking successfull)
        for param in slef.metrics_evaluator.bart_scorer.model.parameters():
            if param.requires_grad:
                raise ValueError("BART Scorer model is not in eval mode")

        score_tensor = self.metrics_evaluator.calculate_score_embeds(expected_embeddings, edited_endings)
        loss = -score_tensor.mean()
        print(f">> DTO loss computed: {loss.item():.4f}")
        if not loss.requires_grad:
            loss.requires_grad = True
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        dto_loss_val = self.dto_loss_embeds(expected_embeddings, edited_endings)
        self.log('training_dto_loss', dto_loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dto_loss_val

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # TODO: Replace commented code below for the generate function
        # expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        # print(expected_embeddings.size())
        #
        # edited_endings = [str(ee) for ee in batch['edited_ending']]
        # dto_val_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)
        generated_texts = self.tokenizer.batch_decode(
            expected_embeddings.argmax(dim=-1), skip_special_tokens=True
        )
        
        # Retrieve additional reference texts for BARTScore comparisons.
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        bart_scores = self.metrics_evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, [], original_endings, logger
        )

        for metric_name, score in bart_scores.items():
            self.log(metric_name, score, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_dto_loss', dto_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Store validation details for later CSV logging.
        for i in range(len(generated_texts)):
            val_entry = {
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Edited Ending': edited_endings[i],
                'Counterfactual': counterfactuals[i],
                'Initial': initials[i],
                'Original Ending': original_endings[i],
                'Generated Text': generated_texts[i]
            }
            self.epoch_validation_details.append(val_entry)
        return dto_val_loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # TODO: Replace commented code below for the generate function
        # expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        #
        # edited_endings = [str(ee) for ee in batch['edited_ending']]
        # dto_test_loss = self.dto_loss_embeds(expected_embeddings, edited_endings)

        generated_texts = self.tokenizer.batch_decode(
            expected_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Retrieve additional reference texts for computing BARTScore similarities.
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        bart_scores = self.metrics_evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, [], original_endings, logger
        )

        for metric_name, score in bart_scores.items():
            self.log(metric_name, score, on_epoch=True, prog_bar=True, logger=True)

        self.log('test_dto_loss', dto_test_loss, on_epoch=True, prog_bar=True, logger=True)
        
        for i in range(len(generated_texts)):
            test_entry = {
                'Epoch': self.current_epoch,
                'Premise': batch.get('premise', [''])[i],
                'Edited Ending': edited_endings[i],
                'Counterfactual': counterfactuals[i],
                'Initial': initials[i],
                'Original Ending': original_endings[i],
                'Generated Text': generated_texts[i]
            }
            test_entry.update(bart_scores)
            self.epoch_test_details.append(test_entry)
        return dto_test_loss

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

