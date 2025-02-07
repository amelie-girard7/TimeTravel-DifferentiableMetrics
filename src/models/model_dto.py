# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/models/model_dto.py
import logging
import torch
import torch.nn.functional as F
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

class DTOTrainer:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.metrics_evaluator = model.metrics_evaluator

    def training_step_dto(self, batch, input_ids, attention_mask):
        """
        Handles training logic for Differentiable Training Objectives (DTO) mode.
        Uses soft token embeddings and optimizes differentiable metric-based loss.
        """

        # Compute soft token embeddings instead of discrete tokens.
        outputs = self.model.forward(input_ids, attention_mask)
        token_probs = torch.softmax(outputs.logits, dim=-1)  # Convert logits to probability distributions
        expected_embeds = self.expected_embeddings(token_probs)

        # Retrieve reference edited endings.
        references = batch["edited_ending"]

        # Select the differentiable metric for optimization.
        metric_type = CONFIG.get("reward_metric", "bart")

        # Compute differentiable loss using soft embeddings.
        dif_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        # Log the computed loss.
        self.model.log('training_differentiable_loss', dif_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        logger.info(f'[TRAIN] DTO Loss: {dif_loss}')

        return dif_loss

    def validation_step_dto(self, batch, input_ids, attention_mask):
        """
        Validation-specific logic for Differentiable Training (DTO) mode.
        Uses soft token embeddings and evaluates loss based on a differentiable metric.
        """

        outputs = self.model.forward(input_ids, attention_mask)
        token_probs = torch.softmax(outputs.logits, dim=-1)
        expected_embeds = self.expected_embeddings(token_probs)

        references = batch["edited_ending"]
        metric_type = CONFIG.get("reward_metric", "bart")
        dif_val_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        self.model.log('validation_differentiable_loss', dif_val_loss.detach(), on_epoch=True, prog_bar=True, logger=True)

        # Save validation details for debugging.
        for i in range(len(batch["premise"])):
            self.model.epoch_validation_details.append({
                'Epoch': self.model.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': batch['edited_ending'][i],
            })

        logger.info(f'[VALIDATION] Epoch {self.model.current_epoch} | DTO Loss: {dif_val_loss}')

        return dif_val_loss

    def test_step_dto(self, batch, input_ids, attention_mask):
        """
        Test-specific logic for Differentiable Training Objectives (DTO) mode.
        """

        outputs = self.model.forward(input_ids, attention_mask)
        token_probs = torch.softmax(outputs.logits, dim=-1)
        expected_embeds = self.expected_embeddings(token_probs)

        references = batch["edited_ending"]
        metric_type = CONFIG.get("reward_metric", "bart")
        dif_test_loss = self.differentiable_metric_loss(expected_embeds, references, metric=metric_type)

        self.model.log('test_differentiable_loss', dif_test_loss, on_epoch=True, prog_bar=True, logger=True)

        for i in range(len(batch["premise"])):
            self.model.epoch_test_details.append({
                'Epoch': self.model.current_epoch,
                'Premise': batch['premise'][i],
                'Initial': batch['initial'][i],
                'Counterfactual': batch['counterfactual'][i],
                'Original Ending': batch['original_ending'][i],
                'Edited Ending': batch['edited_ending'][i],
            })

        logger.info(f'[TEST] Epoch {self.model.current_epoch} | DTO Loss: {dif_test_loss}')

        return dif_test_loss

    def expected_embeddings(self, token_probs):
        """
        Compute soft token embeddings using probability-weighted sum over the model's embedding matrix.
        """
        embedding_matrix = self.model.get_input_embeddings().weight  # (vocab_size, hidden_dim)
        return torch.einsum('bsv,vh->bsh', token_probs, embedding_matrix)

    def differentiable_metric_loss(self, generated_embeds, references, metric="bert"):
        """
        Computes differentiable loss based on soft embeddings and a selected evaluation metric.
        """

        if metric == "bert":
            _, _, f1_scores = self.metrics_evaluator.bert_scorer.score(
                generated_embeds, references, use_soft_embeddings=True
            )
            return 1 - f1_scores.mean() if f1_scores is not None else torch.tensor(1.0, device=self.model.device)

        elif metric == "bart":
            scores = self.metrics_evaluator.bart_scorer.score(generated_embeds, references)
            return -torch.tensor(scores, device=self.model.device).mean()

        elif metric == "rouge":
            rouge_scores = self.metrics_evaluator.rouge.get_scores(generated_embeds, references, avg=True)
            return 1 - torch.tensor(rouge_scores["rouge-l"]["f"], device=self.model.device)

        elif metric == "bleu":
            references = [[ref] for ref in references]
            bleu_score = self.metrics_evaluator.sacre_bleu.corpus_score(generated_embeds, references).score
            return 1 - torch.tensor(bleu_score, device=self.model.device) / 100.0

        else:
            raise ValueError(f"Unsupported metric: {metric}")
