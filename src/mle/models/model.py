import csv
import logging
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.mle.utils.config import CONFIG
from src.mle.utils.metrics import MetricsEvaluator

# Initialize a logger for debugging and output control
logger = logging.getLogger(__name__)

class FlanT5FineTuner(pl.LightningModule):
    """
    Fine-tunes a pre-trained BART model for counterfactual story generation.
    This version supports only **Maximum Likelihood Estimation (MLE)** training.
    """

    def __init__(self, model_name, model_dir, file_label=""):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Store essential attributes
        self.model_dir = Path(model_dir)
        self.file_label = file_label

        # Load pre-trained BART model and tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        # Define paths to save validation and test details.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Buffers to store evaluation details and scores
        self.epoch_validation_details = []
        self.epoch_scores = []
        self.epoch_test_details = []
        self.epoch_test_scores = []

        # Initialize MetricsEvaluator for BARTScore calculations
        self.metrics_evaluator = MetricsEvaluator()

        logger.info(f"Model initialized: {model_name} (MLE mode)")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Standard forward pass using **MLE loss**.

        If `labels` are provided, computes cross-entropy loss for training.
        Otherwise, performs model inference.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=False
        )
        return outputs

    def training_step(self, batch, batch_idx):
        """
        Processes a batch during training using **MLE loss**.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        
        # Forward pass
        outputs = self.forward(input_ids, attention_mask, labels)
        mle_train_loss = outputs.loss

        # Log MLE training loss
        self.log('training_mle_loss', mle_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Decode generated texts for logging
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Compute BARTScore for generated texts vs. ground-truth edited endings
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        self.log('training_mle_score_mean', scores.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mle_train_loss

    def validation_step(self, batch, batch_idx):
        """
        Processes a batch during validation using **MLE loss**.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Forward pass
        outputs = self.forward(input_ids, attention_mask, labels)
        mle_val_loss = outputs.loss

        # Log validation loss
        self.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Compute BARTScore
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        self.epoch_scores.extend(scores.tolist())

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

    def test_step(self, batch, batch_idx):
        """
        Processes a batch during testing using **MLE loss**.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        # Forward pass
        outputs = self.forward(input_ids, attention_mask, labels)
        mle_test_loss = outputs.loss

        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

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

        return mle_test_loss

    def on_validation_epoch_end(self):
        """
        Finalizes and saves validation results at the end of each epoch.
        """
        if self.epoch_validation_details:
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)

        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            self.log("validation_overall_score", overall_val_score, prog_bar=True, logger=True)

        self.epoch_validation_details.clear()
        self.epoch_scores.clear()

    def on_test_epoch_end(self):
        """
        Finalizes and saves test results at the end of the test epoch.
        """
        if self.epoch_test_details:
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        self.epoch_test_details.clear()
        self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        """
        Writes the details to a CSV file.
        """
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=details[0].keys())

            if not file_exists:
                writer.writeheader()

            for detail in details:
                if epoch is not None:
                    detail['Epoch'] = epoch
            writer.writerows(details)

    def configure_optimizers(self):
        """
        Configures the optimizer.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])
