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

    def calculate_score_embeds(self, inputs_embeds, references, batch_size=CONFIG["batch_size"]):
        """
        Computes BARTScore similarity by passing expected embeddings directly as inputs.

        Args:
            inputs_embeds (torch.Tensor): Expected embeddings from the rewriting model.
            references (list of str): Reference texts (edited endings) tokenized as usual.
            batch_size (int): Batch size for scoring (defaults to CONFIG["batch_size"]).

        Returns:
            scores_tensor (torch.Tensor): A tensor of BARTScore values.
        """

        print("\n=== SCORE EMBEDS DEBUGGING ===")
        print(f"Input embeddings device: {inputs_embeds.device}")
        print(f"BART scorer device: {self.bart_scorer.device}")
        print(f"Input embeddings shape: {inputs_embeds.shape}")
        print(f"Input embeddings stats - mean: {inputs_embeds.mean().item():.4f}, std: {inputs_embeds.std().item():.4f}")
        print(f"References count: {len(references)}")
        print(f"Sample reference: {references[0][:50]}...")

        # Verify BART scorer model is in eval mode
        print(f"BART scorer model training mode: {self.bart_scorer.model.training}")

        # Call the new score_embeds method from the BART scorer.
        scores = self.bart_scorer.score_embeds(inputs_embeds, references, batch_size=batch_size)

        print(f"Raw scores: {scores[:5]}")
        print(f"Scores dtype: {type(scores[0])}")
        print(f"Scores stats - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {sum(scores)/len(scores):.4f}")

        # Convert scores to a tensor on the device specified in the config.
        scorer_device = CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu")
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=scorer_device)

        print(f"Scores tensor stats - mean: {scores_tensor.mean().item():.4f}, std: {scores_tensor.std().item():.4f}")
        print(f"Scores tensor device: {scores_tensor.device}")

        return scores_tensor
