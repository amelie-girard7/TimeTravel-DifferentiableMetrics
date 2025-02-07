# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model_mle.py
import logging
import torch.nn.functional as F
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)


class MLETrainer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.metrics_evaluator = model.metrics_evaluator

    def training_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Training-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        # Forward pass
        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        masked_logits = self.model.apply_vocab_masking(outputs.logits)  # Apply masking to logits

        # Calculate MLE loss
        if CONFIG['use_custom_loss'] and 'differential_weights' in batch:
            mle_train_loss = self.custom_loss(masked_logits, batch['labels'], batch['differential_weights'])
        else:
            mle_train_loss = outputs.loss

        # Log MLE training loss
        self.model.log('training_mle_loss', mle_train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Decode and log scores for generated texts
        generated_texts = self.tokenizer.batch_decode(masked_logits.argmax(-1), skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_mean = scores.mean()
        self.model.log('training_mle_score_mean', score_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return mle_train_loss  # Return MLE loss for optimization

    def validation_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Validation-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_val_loss = outputs.loss

        # Decode generated texts
        generated_texts = self.tokenizer.batch_decode(
            self.model.model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        # Calculate sentence-level scores
        scores = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        self.model.epoch_scores.extend(scores.tolist())  # Save validation scores for the dataset

        # Log MLE loss
        self.model.log('validation_mle_loss', mle_val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Save validation details
        for i in range(len(generated_texts)):
            self.model.epoch_validation_details.append({
                'Epoch': self.model.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })
        return mle_val_loss

    def test_step_mle(self, batch, input_ids, attention_mask, labels):
        """
        Test-specific logic for Maximum Likelihood Estimation (MLE) mode.
        """
        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        mle_test_loss = outputs.loss

        generated_texts = self.tokenizer.batch_decode(
            self.model.model.generate(input_ids=input_ids, attention_mask=attention_mask,max_length=CONFIG['max_gen_length']),
            skip_special_tokens=True
        )
        edited_endings = [str(ee) for ee in batch['edited_ending']]

        for i in range(len(generated_texts)):
            self.model.epoch_test_details.append({
                'Epoch': self.model.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': edited_endings[i],
                'Generated Text': generated_texts[i]
            })

        return mle_test_loss

    def custom_loss(self, outputs, targets, differential_weights):
        """
        Custom loss function that applies differential weights to the calculation.
        """
        logits_flat = outputs.view(-1, outputs.size(-1))  # Reshape to [batch_size * seq_length, vocab_size]
        targets_flat = targets.view(-1)  # Flatten targets to [batch_size * seq_length]
        differential_weights_flat = differential_weights.view(
            -1)  # Flatten weights to match sequence length [batch_size * seq_length]

        # Compute the standard loss function without reduction to get a loss value per token.
        loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # Apply the differential weights to each token's loss.
        weighted_loss_per_token = loss_per_token * differential_weights_flat

        # Calculate the mean of the weighted losses to get a single scalar representing the batch's loss.
        mean_weighted_loss = weighted_loss_per_token.mean()

        return mean_weighted_loss
