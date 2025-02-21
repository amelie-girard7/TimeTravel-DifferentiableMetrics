# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/utils/metrics.py
import logging
import torch
from src.mle.utils.config import CONFIG
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
        #print(f"Initializing MetricsEvaluator with config: {CONFIG}")

        # Initialize BARTScorer (used for DTO loss and evaluation)
        self.bart_scorer = BARTScorer(
            device=CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint=CONFIG.get("bart_scorer_checkpoint", "facebook/bart-large-cnn")
        )

        #print("MetricsEvaluator initialized.")

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

        #print("Calculating BARTScore...")
        # Ensure inputs are lists of strings
        generated_texts = [str(gt) for gt in generated_texts]
        references = [str(ref) for ref in references]

        # Compute BARTScore for each generated-reference pair
        scores = self.bart_scorer.score(generated_texts, references)
        # Convert scores to a tensor for logging
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=CONFIG.get("scorer_device", "cpu"))

        #print(f"BARTScore Tensor: {scores_tensor}")
        return scores_tensor

    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings,
                                          all_counterfactuals, all_initials, all_premises,
                                          all_original_endings, logger):
        """
        Calculates and logs BARTScore similarity for different text comparisons.

        Args:
            all_generated_texts (list of str): Model-generated outputs.
            all_edited_endings (list of str): Ground-truth edited endings.
            all_counterfactuals (list of str): Counterfactual endings.
            all_initials (list of str): Initial story elements.
            all_premises (list of str): Story premises.
            all_original_endings (list of str): Original endings.
            logger (logging.Logger): Logger to record similarity scores.

        Returns:
            bart_scores (dict): Dictionary of computed BART similarity scores.
        """
        #print("Calculating BART similarity scores...")

        # Define different text comparisons for evaluation
        all_comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        bart_scores = {}

        for label, src_texts, tgt_texts in all_comparisons:
            if tgt_texts:
                try:
                    # Compute BARTScore for the given text pair with a batch size of 4
                    scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)
                    # Compute average BARTScore for this comparison
                    avg_score = sum(scores) / len(scores) if scores else float('nan')
                    # Log the score
                    logger.info(f"{label}_avg_score: {avg_score}")
                    bart_scores[f"{label}_avg_score"] = avg_score
                    #print(f"{label}_avg_score: {avg_score}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bart_scores[f"{label}_avg_score"] = 'N/A'
                    #print(f"Error calculating {label}: {e}")

        return bart_scores

    def calculate_score_embeds(self, inputs_embeds, references):
        """
        Computes BARTScore similarity by passing expected embeddings directly as inputs.

        Args:
            inputs_embeds (torch.Tensor): Expected embeddings from the rewriting model.
            references (list of str): Reference texts (edited endings) tokenized as usual.

        Returns:
            scores_tensor (torch.Tensor): A tensor of BARTScore values.
        """
        # Use the CONFIG file to determine the batch size.
        batch_size = CONFIG.get("batch_size", 4)

        # Call the new score_embeds method from the BART scorer.
        scores = self.bart_scorer.score_embeds(inputs_embeds, references, batch_size=batch_size)

        # Convert scores to a tensor on the device specified in the config.
        scorer_device = CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu")
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=scorer_device)

        return scores_tensor
