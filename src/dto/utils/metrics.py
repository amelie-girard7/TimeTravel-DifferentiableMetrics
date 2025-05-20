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

    # def calculate_score_embeds(self, inputs_embeds, references, batch_size=CONFIG["batch_size"], validation=False):
    #     """
    #     Computes BARTScore similarity with explicit gradient control and debugging.
        
    #     Args:
    #         inputs_embeds: Expected embeddings from main model [batch, seq_len, hidden_dim]
    #         references: List of reference text strings
    #         batch_size: Batch size for scoring
    #         validation: Boolean flag indicating validation mode
            
    #     Returns:
    #         Tensor of BARTScore values with proper gradient handling
    #     """
    #     # ===== 1. Mode Identification =====
    #     mode = "VALIDATION" if validation else "TRAINING"
    #     print(f"\n=== BARTScore Calculation ({mode}) ===")
        
    #     # ===== 2. Input Verification =====
    #     print("[Input Verification]")
    #     print(f"Embeddings device: {inputs_embeds.device}")
    #     print(f"Embeddings shape: {inputs_embeds.shape}")
    #     print(f"Embeddings requires_grad: {inputs_embeds.requires_grad}")
    #     print(f"Embeddings stats - μ: {inputs_embeds.mean().item():.4f} σ: {inputs_embeds.std().item():.4f}")
    #     print(f"Reference count: {len(references)}")
    #     print(f"Sample reference: {references[0][:100]}{'...' if len(references[0]) > 100 else ''}")
        
    #     # ===== 3. Scorer Status =====
    #     print("\n[Scorer Status]")
    #     print(f"Scorer device: {self.bart_scorer.device}")
    #     print(f"Scorer training mode: {self.bart_scorer.model.training}")
    #     print(f"Scorer parameters frozen: {all(not p.requires_grad for p in self.bart_scorer.parameters())}")
        
    #     # ===== 4. Gradient-Controlled Scoring =====
    #     if validation:
    #         with torch.no_grad():
    #             print("\n[Validation Mode] Gradient context: torch.no_grad()")
    #             scores = self.bart_scorer.score_embeds(
    #                 inputs_embeds, 
    #                 references, 
    #                 batch_size,
    #                 validation=True
    #             )
    #             print(f"Output scores requires_grad: {scores.requires_grad}")
    #     else:
    #         print("\n[Training Mode] Gradient context: enabled")
    #         scores = self.bart_scorer.score_embeds(
    #             inputs_embeds,
    #             references,
    #             batch_size,
    #             validation=False
    #         )
    #         print(f"Output scores requires_grad: {scores.requires_grad}")
        
    #     # ===== 5. Output Verification =====
    #     print("\n[Output Verification]")
    #     print(f"Scores shape: {scores.shape}")
    #     print(f"Scores dtype: {scores.dtype}")
    #     print(f"Scores stats - μ: {scores.mean().item():.4f} σ: {scores.std().item():.4f}")
    #     print(f"Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
    #     # ===== 6. Gradient Flow Check =====
    #     if not validation and not scores.requires_grad:
    #         print("\n!! WARNING: Training mode but scores don't require gradient !!")
    #         print("Possible gradient flow interruption at:")
    #         print("- BART scorer forward pass")
    #         print("- Score aggregation")
        
    #     return scores
