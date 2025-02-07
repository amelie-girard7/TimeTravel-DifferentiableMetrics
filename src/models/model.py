# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model.py
import csv
import logging
import os
import torch
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.metrics import MetricsEvaluator


# Import PG and MLE-specific functions
from src.models.model_pg import PGTrainer
from src.models.model_mle import MLETrainer
from src.models.model_dto import DTOTrainer

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

        # Set unique file paths using `file_label` to prevent overwriting
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Initialize buffers for validation


        # Initialize buffers for testing
        self.epoch_test_details = []  # Storage for each test epoch
        self.epoch_test_scores = []  # Test scores buffer
        self.epoch_validation_details = []  # Storage for each validation epoch
        self.epoch_scores = []  # Validation scores buffer

        # Initialize MetricsEvaluator to handle custom scoring for rewards
        self.metrics_evaluator = MetricsEvaluator()

        # This attribute will be set in main.py to toggle between MLE and PG modes
        self.use_policy_gradient = False

        # Initialize trainers
        self.pg_trainer = PGTrainer(self)
        self.mle_trainer = MLETrainer(self)
        self.dto_trainer = DTOTrainer(self)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the T5 model.
        If labels are provided, calculates the loss; otherwise, returns generated tokens and logits.
        """
        if labels is not None:
            # MLE training mode with labels for loss calculation
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=False
            )
            return outputs
        elif self.use_policy_gradient:
            # PG mode generates tokens without labels
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=CONFIG['max_gen_length'],
                # num_beams=1,  # Greedy decoding
                do_sample=True,  # Enable sampling
                #top_k=50,  # Use Top-K sampling
                #top_p=0.95,  # Use nucleus sampling
                temperature=0.7,  # Controls randomness; lower values make outputs more deterministic (try the values 1 , 1.5 , 0.7)
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_tokens = outputs.sequences
            logits = outputs.scores
            return generated_tokens, logits

        elif self.use_differentiable_metrics:
            # DTO mode: Compute soft embeddings instead of selecting discrete tokens.
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Convert raw logits to probability distributions using softmax.
            token_probs = torch.softmax(outputs.logits, dim=-1)  # Shape: (batch, seq_len, vocab_size)

            # Compute soft token embeddings using the probability-weighted sum.
            expected_embeds = self.expected_embeddings(token_probs)

            return expected_embeds  # Return soft embeddings instead of discrete tokens

        else:
            raise ValueError("Invalid training mode: Set either 'use_policy_gradient' or 'use_differentiable_metrics'.")

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

    def training_step(self, batch, batch_idx):
        """
        Processes a batch during the training phase. Routes to PG or MLE logic based on mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']

        if self.use_policy_gradient:
            return self.pg_trainer.training_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self.dto_trainer.training_step_dto(batch, input_ids, attention_mask)
        else:
            return self.mle_trainer.training_step_mle(batch, input_ids, attention_mask, labels)

    def validation_step(self, batch, batch_idx):
        """
        Processes a batch during the validation phase. Routes to PG or MLE logic based on mode.
        """
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        print(f"Validation Step: Processing batch {batch_idx}")

        if self.use_policy_gradient:
            return self.pg_trainer.validation_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self.dto_trainer.validation_step_dto(batch, input_ids, attention_mask)
        else:
            return self.mle_trainer.validation_step_mle(batch, input_ids, attention_mask, labels)

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
            return self.pg_trainer.test_step_pg(batch, input_ids, attention_mask)
        elif self.use_differentiable_metrics:
            return self.dto_trainer.test_step_dto(batch, input_ids, attention_mask)
        else:
            return self.mle_trainer.test_step_mle(batch, input_ids, attention_mask, labels)

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

