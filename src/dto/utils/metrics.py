# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/dto/utils/metrics.py
import logging
import torch
from src.dto.utils.config import CONFIG
from src.BARTScore_metric.bart_score import BARTScorer

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    A class for evaluating text generation models using BARTScore as the primary metric.
    """

    def __init__(self):
        """
        Initializes the metric evaluator with BARTScore.
        """
        # Initialize BARTScorer (used for DTO loss and evaluation)
        self.bart_scorer = BARTScorer(
            device=CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint=CONFIG.get("bart_scorer_checkpoint", "facebook/bart-large-cnn")
        )
        # Set the underlying model to eval mode and freeze its parameters
        self.bart_scorer.model.eval()
        # Freeze BART scorer so it does not update during training
        for param in self.bart_scorer.model.parameters():
            param.requires_grad = False
        print("MetricsEvaluator initialized with BARTScore")

    def calculate_score(self, generated_texts, references):
        """
        Computes the BARTScore similarity between generated texts and reference texts.

        Args:
            generated_texts (list of str): List of generated outputs from the model.
            references (list of str): Corresponding reference texts (ground truth).

        Returns:
            scores_tensor (torch.Tensor): A tensor of BARTScore values.
        """
        if self.bart_scorer is None:
            raise ValueError("BARTScore is not initialized. Set 'use_bart' to True in CONFIG.")

        print("Calculating BARTScore...")
        # Ensure inputs are lists of strings
        generated_texts = [str(gt) for gt in generated_texts]
        references = [str(ref) for ref in references]

        # Compute BARTScore for each generated-reference pair
        scores = self.bart_scorer.score(generated_texts, references)
        # Convert scores to a tensor for logging
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=CONFIG.get("scorer_device", "cpu"))

        print(f"BARTScore Tensor: {scores_tensor}")
        return scores_tensor

    def calculate_bart_similarity(self, generated_texts, edited_endings, 
                              counterfactuals, initials, original_endings):
        """Compute BARTScore similarity across various comparison pairs."""
        comparisons = [
            ('bart/pred_edited', generated_texts, edited_endings),
            ('bart/pred_cf', generated_texts, counterfactuals),
            ('bart/pred_initial', generated_texts, initials),
            ('bart/pred_original', generated_texts, original_endings),
            ('bart/edited_cf', edited_endings, counterfactuals),
            ('bart/edited_initial', edited_endings, initials),
            ('bart/edited_original', edited_endings, original_endings),
        ]
        
        results = {}
        for label, src, tgt in comparisons:
            try:
                scores = self.bart_scorer.score(src, tgt, batch_size=4)
                results[f"{label}_score"] = sum(scores) / len(scores)
            except Exception as e:
                print(f"Error computing BARTScore for {label}: {e}")
                results[f"{label}_score"] = float('nan')
        
        return results
    

    def calculate_score_embeds(self, expected_embeds, endings, validation=False, batch_size=None):
        """
        Helper function to calculate BARTScore with proper gradient handling
        
        Args:
            expected_embeds: Tensor of expected embeddings
            endings: List of ending texts to score
            validation: Boolean indicating if in validation mode
            batch_size: Batch size for scoring (from CONFIG)
            
        Returns:
            Tensor of scores
        """
        if validation:
            with torch.no_grad():
                return self.bart_scorer.score_embeds(
                    expected_embeds,
                    endings,
                    batch_size=batch_size,
                    validation=True
                )
        else:
            if not expected_embeds.requires_grad:
                expected_embeds = expected_embeds.requires_grad_(True)
            return self.bart_scorer.score_embeds(
                expected_embeds,
                endings,
                batch_size=batch_size,
                validation=False
            )
  