# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model.py
import csv
import logging
import os
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import pytorch_lightning as pl
from pathlib import Path
from src.utils.config import CONFIG
from src.utils.metrics import MetricsEvaluator

# Initialize a logger for debugging and output control
logger = logging.getLogger(__name__)


class FlanT5FineTuner(pl.LightningModule):
    """
    This module supports two training modes:
    
    1. Maximum Likelihood Estimation (MLE) mode:
       - Uses standard cross-entropy loss on discrete token outputs.
       
    2. Differentiable Training Objectives (DTO) mode:
       - Instead of discrete token selection (via argmax), the model outputs a soft probability
         distribution over the vocabulary.
       - The soft probabilities are used to compute an expected (weighted sum) embedding,
         preserving uncertainty and enabling smooth gradient flow.
       - BARTScore (a semantic similarity metric) is then applied as a differentiable loss.
    """

    def __init__(self, model_name, model_dir, file_label="", use_differentiable_training=False):
        """
        Initializes the fine-tuner with the specified model and tokenizer.
        """
        super().__init__()

        # Save hyperparameters (ensures availability when loading from a checkpoint)
        self.save_hyperparameters()

        # Store essential attributes
        self.model_dir = Path(model_dir)
        self.file_label = file_label
        self.use_differentiable_training = use_differentiable_training  # Enables DTO mode if True.

        # # Load pre-trained BART model for conditional generation and its tokenizer.
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

        # Define CSV paths to save validation and test details.
        self.val_csv_file_path = self.model_dir / f"validation_details{self.file_label}.csv"
        self.test_csv_file_path = self.model_dir / f"test_details{self.file_label}.csv"

        # Buffers to store details and scores for each epoch.
        self.epoch_validation_details = [] # For detailed validation logs.
        self.epoch_scores = []             # For aggregated validation scores.
        self.epoch_test_details = []       # For detailed test logs.
        self.epoch_test_scores = []        # For aggregated test scores.

        # Initialize the MetricsEvaluator to compute BARTScore.
        self.metrics_evaluator = MetricsEvaluator()

        # If DTO mode is enabled, initialize a separate frozen BART model for computing the BARTScore loss.
        if self.use_differentiable_training:
            logger.info("Initializing Differentiable Training Objectives (DTO) mode...")
            self.bart_scorer = BartForConditionalGeneration.from_pretrained(CONFIG["bart_scorer_checkpoint"])
            self.bart_scorer_tokenizer = BartTokenizer.from_pretrained(CONFIG["bart_scorer_checkpoint"])
            self.bart_scorer.eval()  # Freeze the BARTScore model.
            for param in self.bart_scorer.parameters():
                param.requires_grad = False  

        logger.info(f"Model initialized: {model_name} | DTO Mode: {self.use_differentiable_training}")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        In MLE mode (if labels are provided):
          - The model computes standard cross-entropy loss over discrete token outputs.

        In DTO mode (if labels are not provided):
          - The model generates raw logits which are converted into a soft probability
            distribution over the vocabulary.
          - These probabilities are then used to compute the expected embedding:

        Returns:
            outputs (dict or Tensor): In MLE mode, a dictionary with loss and logits; in DTO mode, the soft embeddings.
        """
        if labels is not None:
            # MLE Training Mode
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=False
            )
            return outputs
        else:
            # DTO Mode: Generate token probability distributions
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            logits = outputs.logits  # Raw output logits
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probability distributions

            # Compute expected embeddings as a weighted sum of BART embeddings
            embedding_matrix = self.model.get_input_embeddings().weight  # BART embedding matrix
            expected_embeddings = torch.matmul(probs, embedding_matrix)  # [batch, seq_len, embedding_dim]

            return expected_embeddings  # Soft token embeddings

    def dto_loss(self, generated_texts, edited_endings):
        """
        Computes the Differentiable Training Objective (DTO) loss as the negative of the BARTScore 
        between the generated text and the reference edited ending.
        Returns:
            loss (Tensor): A scalar differentiable loss value computed as the negative mean BARTScore.
        """
        # Compute BARTScore solely between generated texts and the edited endings.
        score_tensor = self.metrics_evaluator.calculate_score(generated_texts, edited_endings)
        
        # The loss is defined as the negative mean BARTScore,
        # so that maximizing semantic similarity (i.e., BARTScore) minimizes the loss.
        loss = -score_tensor.mean()
        
        # Ensure the loss is differentiable.
        if not loss.requires_grad:
            loss.requires_grad = True
            
        return loss

    def training_step(self, batch, batch_idx):
        """
        Processes a batch during the training phase. Routes to PG or MLE logic based on mode.
        """
        if self.use_differentiable_training:
            return self._training_step_dto(batch)
        else:
            return self._training_step_mle(batch, batch['input_ids'], batch['attention_mask'], batch['labels'])

    def _training_step_dto(self, batch):
        """
        Training step for Differentiable Training Objectives (DTO).
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # Generate soft embeddings (DTO mode)
        expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Decode generated embeddings into text
        generated_texts = self.tokenizer.batch_decode(
            expected_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Retrieve all required reference texts from the batch
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Compute DTO loss using all BART similarity comparisons
        dto_loss = self.dto_loss(generated_texts, edited_endings)

        # Log the overall loss
        self.log('training_dto_loss', dto_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dto_loss

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
        print(f"Validation Step: Processing batch {batch_idx}")

        if self.use_differentiable_training:
            return self._validation_step_dto(batch)
        else:
            return self._validation_step_mle(batch, batch['input_ids'], batch['attention_mask'], batch['labels'])

    def _validation_step_dto(self, batch):
        """
        Validation step for Differentiable Training Objectives (DTO).
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # Generate soft embeddings (DTO mode)
        expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Decode generated embeddings into text
        generated_texts = self.tokenizer.batch_decode(
            expected_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Retrieve reference texts
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Pass an empty list for premises if not available.
        bart_scores = self.metrics_evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, [], original_endings, logger
        )

        # Compute DTO validation loss using all BART similarity comparisons
        dto_val_loss = self.dto_loss(generated_texts, edited_endings)

        # Log each individual BART similarity score separately
        for metric_name, score in bart_scores.items():
            self.log(metric_name, score, on_epoch=True, prog_bar=True, logger=True)

        # Log validation loss
        self.log('validation_dto_loss', dto_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Store validation details for CSV logging
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
            # Add BART scores to each row
            #val_entry.update(bart_scores)
            self.epoch_validation_details.append(val_entry)

        return dto_val_loss

    def _validation_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Validation-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        # Forward pass through the model
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Compute MLE loss
        mle_val_loss = outputs.loss

        # Decode generated texts from the model
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )

        # Extract ground-truth edited endings
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Compute BARTScore for generated texts vs. ground-truth edited endings
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()

        # Store batch-level scores
        self.epoch_scores.extend(scores.tolist())

        # Log validation loss
        self.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Save details for CSV logging
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
        Processes a batch during the testing phase. Routes to PG or MLE logic based on mode.
        """

        if self.use_differentiable_training:
            return self._test_step_dto(batch)
        else:
            return self._test_step_mle(batch, batch['input_ids'], batch['attention_mask'], batch['labels'])

    def _test_step_dto(self, batch):
        """
        Test step for Differentiable Training Objectives (DTO).
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

        # Generate soft embeddings (DTO mode)
        expected_embeddings = self.forward(input_ids=input_ids, attention_mask=attention_mask)

        # Decode generated embeddings into text
        generated_texts = self.tokenizer.batch_decode(
            expected_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Retrieve reference texts
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        counterfactuals = [str(cf) for cf in batch['counterfactual']]
        initials = [str(init) for init in batch['initial']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Pass an empty list for premises if not available.
        bart_scores = self.metrics_evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, [], original_endings, logger
        )

        # Compute DTO loss using all BART similarity comparisons
        dto_test_loss = self.dto_loss(generated_texts, edited_endings)

        # Log each individual BART similarity score separately
        for metric_name, score in bart_scores.items():
            self.log(metric_name, score, on_epoch=True, prog_bar=True, logger=True)

        # Log test loss
        self.log('test_dto_loss', dto_test_loss, on_epoch=True, prog_bar=True, logger=True)

        # Store details for CSV logging
        for i in range(len(generated_texts)):
            test_entry = {
                'Epoch': self.current_epoch,
                'Premise': batch['premise'][i],
                'Edited Ending': edited_endings[i],
                'Counterfactual': counterfactuals[i],
                'Initial': initials[i],
                'Original Ending': original_endings[i],
                'Generated Text': generated_texts[i]
            }
            # Add BART scores to each row
            test_entry.update(bart_scores)
            self.epoch_test_details.append(test_entry)

        return dto_test_loss

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

    def on_validation_epoch_end(self):
        """
        Finalize and save validation results at the end of the validation epoch.
        """
        print("Validation Epoch End")

        if self.epoch_validation_details:
            print(f"Saving {len(self.epoch_validation_details)} validation details to {self.val_csv_file_path}.")
            self.log_to_csv(self.val_csv_file_path, self.epoch_validation_details, epoch=self.current_epoch)

        # Compute overall validation score from epoch scores
        if self.epoch_scores:
            overall_val_score = torch.tensor(self.epoch_scores).mean().item()
            print(f"Overall validation score: {overall_val_score}")
            self.log("validation_overall_score", overall_val_score, prog_bar=True, logger=True)

        # Clear buffers for the next validation run
        self.epoch_validation_details.clear()
        self.epoch_scores.clear()

    def on_test_epoch_end(self):
        """
        Finalize and save test results at the end of the test epoch.
        """
        print("Test Epoch End")

        if self.epoch_test_details:
            print(f"Saving {len(self.epoch_test_details)} test details to {self.test_csv_file_path}.")
            self.log_to_csv(self.test_csv_file_path, self.epoch_test_details, epoch=self.current_epoch)

        # Compute overall test score from epoch scores
        if self.epoch_test_scores:
            overall_test_score = torch.tensor(self.epoch_test_scores).mean().item()
            print(f"Overall test score: {overall_test_score}")
            self.log("test_overall_score", overall_test_score, prog_bar=True, logger=True)

        # Clear buffers for the next test run
        self.epoch_test_details.clear()
        self.epoch_test_scores.clear()

    def log_to_csv(self, csv_file_path, details, epoch=None):
        """
        Writes the details, including all saved metrics, to the specified CSV file.
        """
        print(f"Writing {len(details)} entries to {csv_file_path}.")
        file_exists = os.path.isfile(csv_file_path)

        # Ensure all keys are present
        fieldnames = details[0].keys() if details else []

        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header if the file does not exist
            if not file_exists:
                writer.writeheader()

            # Add epoch number to each row if applicable
            for detail in details:
                if epoch is not None:
                    detail['Epoch'] = epoch

            writer.writerows(details)

        print(f"Successfully wrote {len(details)} entries to {csv_file_path}.")

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=CONFIG["learning_rate"])

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
