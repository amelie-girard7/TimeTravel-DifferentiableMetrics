# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model_pg.py
import logging
import torch
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

class PGTrainer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.metrics_evaluator = model.metrics_evaluator

    def training_step_pg(self, batch, input_ids, attention_mask):
        """
        Training-specific logic for Policy Gradient (PG) mode.
        """
        # Forward pass
        generated_tokens, logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Get ground-truth references
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate rewards
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Default values
        delta_m1 = torch.tensor(0.0)  # Ensures `delta_m1` is always a tensor
        dynamic_baseline = 0.0

        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()
            rewards = rewards - dynamic_baseline

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

            # Calculate PG loss
        pg_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=dynamic_baseline)

        # Logging
        self.model.log('training_pg_loss', pg_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.model.log('training_pg_reward_mean', rewards.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.model.log('training_pg_baseline', dynamic_baseline, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.model.log('training_pg_delta_m1_mean', delta_m1.mean().item(), on_step=True, on_epoch=True, prog_bar=True,logger=True)

        logger.info(
            f'[TRAIN] PG Loss: {pg_loss}, Baseline: {dynamic_baseline}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_loss

    def validation_step_pg(self, batch, input_ids, attention_mask):
        """
        Validation-specific logic for Policy Gradient (PG) mode.
        """
        generated_tokens, logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Calculate scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Default values
        delta_m1 = torch.tensor(0.0)  # Ensures `delta_m1` is always a tensor
        dynamic_baseline = 0.0

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG validation loss (baseline = 0.0, since no updates occur)
        pg_val_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log validation metrics
        self.model.log('validation_pg_loss', pg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.model.log('validation_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.model.log('validation_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.model.log('validation_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

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

        logger.info(
            f'[VALIDATION] Epoch {self.model.current_epoch} | PG Loss: {pg_val_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_val_loss

    def test_step_pg(self, batch, input_ids, attention_mask):
        """
        Test-specific logic for Policy Gradient (PG) mode.
        """
        generated_tokens, logits = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        edited_endings = [str(ee) for ee in batch['edited_ending']]
        original_endings = [str(oe) for oe in batch['original_ending']]

        # Compute scores
        score_pred_edited = self.metrics_evaluator.calculate_score(generated_texts, edited_endings).detach()
        score_pred_original = self.metrics_evaluator.calculate_score(generated_texts, original_endings).detach()

        # Default values
        delta_m1 = torch.tensor(0.0)  # Ensures `delta_m1` is always a tensor
        dynamic_baseline = 0.0

        # Handle the different experiments
        if CONFIG["pg_experiment"] == "fixed":
            rewards = score_pred_edited - CONFIG["baseline_score"]
            dynamic_baseline = 0.0  # No dynamic baseline needed

        elif CONFIG["pg_experiment"] == "dynamic":
            dynamic_baseline = score_pred_edited.mean().detach()
            rewards = score_pred_edited - dynamic_baseline

        elif CONFIG["pg_experiment"] == "delta_m1":
            delta_m1 = score_pred_edited - score_pred_original
            rewards = score_pred_edited + delta_m1
            dynamic_baseline = rewards.mean().detach()

        else:
            raise ValueError(f"Invalid PG experiment: {CONFIG['pg_experiment']}")

        # Compute PG test loss (baseline = 0.0, since no updates occur)
        pg_test_loss = self.calculate_policy_gradient_loss(generated_tokens, logits, rewards, baseline=0.0)

        # Log test metrics
        self.model.log('test_pg_loss', pg_test_loss, on_epoch=True, prog_bar=True, logger=True)
        self.model.log('test_pg_reward_mean', rewards.mean(), on_epoch=True, prog_bar=True, logger=True)
        self.model.log('test_pg_baseline', dynamic_baseline, on_epoch=True, prog_bar=True, logger=True)

        if CONFIG["pg_experiment"] == "delta_m1":
            self.model.log('test_pg_delta_m1_mean', delta_m1.mean().item(), on_epoch=True, prog_bar=True, logger=True)

        # Save test details
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

        logger.info(
            f'[TEST] Epoch {self.model.current_epoch} | PG Loss: {pg_test_loss}, ΔM1 Mean: {delta_m1.mean().item() if CONFIG["pg_experiment"] == "delta_m1" else "N/A"}')

        return pg_test_loss

    def calculate_policy_gradient_loss(self, generated_tokens, logits, rewards, baseline):
        """
        Calculates policy gradient loss based on generated tokens and rewards.
        Handles the case where BART scores are negative by flipping the sign of rewards.
        """
        # Stack logits along the sequence dimension and apply log softmax
        logits = torch.log_softmax(torch.stack(logits, dim=1), dim=-1)
        logits = self.model.apply_vocab_masking(logits)  # Apply masking to stacked logits

        # Gather the log probabilities for the generated tokens
        labels_for_indexing = generated_tokens[:, 1:].contiguous()
        token_log_probs = logits.gather(dim=-1, index=labels_for_indexing.unsqueeze(-1)).squeeze(-1)

        # Create a mask to ignore padding tokens
        padding_mask = labels_for_indexing != self.tokenizer.pad_token_id
        token_log_probs = token_log_probs * padding_mask.float()

        # Sum log probabilities across the sequence dimension
        sequence_log_prob_sum = token_log_probs.sum(dim=1)

        # Handle special case for BART (negative rewards)
        if CONFIG.get("reward_metric") == "bart":
           rewards = rewards + 4  # add a Baseline to move to the positif size but you keep the magnitude
            # Calculate policy gradient loss
        return -(rewards * sequence_log_prob_sum).mean()
