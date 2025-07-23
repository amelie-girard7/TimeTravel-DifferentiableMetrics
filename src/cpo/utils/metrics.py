import logging
import torch
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
from src.cpo.utils.config import CONFIG
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
         # Initialize ROUGE for ROUGE score calculation
        self.rouge = Rouge()

        # Initialize BERTScorer if configured to use BERTScore
        self.bert_scorer = (
            BERTScorer(
                model_type=CONFIG.get("bert_scorer_model_type", "bert-base-uncased"),
                device=CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu"),
                batch_size=CONFIG.get("bert_scorer_batch_size", 16)
            ) if CONFIG.get("use_bert", False) else None
        )

        # Initialize BLEU scorer if configured to use BLEU
        self.sacre_bleu = BLEU() if CONFIG.get("use_bleu", False) else None

        # Initialize BARTScorer (used for CPO loss and evaluation)
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

    def calculate_bart_similarity(self, generated_texts, edited_endings, counterfactuals, original_endings):
        """Compute BARTScore similarity across various comparison pairs."""
        comparisons = [
            ('bart/pred_edited', generated_texts, edited_endings),
            ('bart/pred_cf', generated_texts, counterfactuals),
            #('bart/pred_initial', generated_texts, initials),
            ('bart/pred_original', generated_texts, original_endings),
            ('bart/edited_cf', edited_endings, counterfactuals),
            #('bart/edited_initial', edited_endings, initials),
            ('bart/edited_original', edited_endings, original_endings),
        ]
        
        results = {}
        for label, src, tgt in comparisons:
            try:
                scores = self.bart_scorer.score(src, tgt, batch_size=4)
                results[f"{label}_score"] = sum(scores) / len(scores)
            except Exception as e:
                logger.error(f"BARTScore failed for {label}: {str(e)}")
                results[f"{label}_score"] = float('nan')
        
        return results
    
    def calculate_rouge_similarity(self, generated_texts, edited_endings, counterfactuals, original_endings):
        """
        Calculates and logs ROUGE scores for various comparisons between generated texts and references.
        """
        # print("Calculating ROUGE scores...")

        all_comparisons = [
            ('rouge_prediction_edited', generated_texts, edited_endings),
            ('rouge_prediction_cf', generated_texts, counterfactuals),
            ('rouge_prediction_original', generated_texts, original_endings),
            ('rouge_edited_ending_cf', edited_endings, counterfactuals),
            ('rouge_edited_ending_original', edited_endings, original_endings),
        ]

        rouge_scores = {}
        for label, hypotheses, references in all_comparisons:
            if references:
                try:
                    rouge_scores_set = self.rouge.get_scores(hypotheses, references, avg=True)
                    score_type = 'rouge-l'
                    rouge_scores[f"{label}_{score_type}_f"] = rouge_scores_set[score_type]['f']
                    logger.info(f"{label}_{score_type}_f: {rouge_scores_set[score_type]['f']}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    rouge_scores[f"{label}_f"] = 'N/A'

        return rouge_scores

    def calculate_bert_similarity(self, generated_texts, edited_endings, counterfactuals, original_endings):
        """
        Calculates and logs BERT similarity F1 scores for various comparisons of generated texts and references.
        """
        # print("Calculating BERT similarity F1 scores...")

        all_comparisons = [
            ('bert_prediction_edited', generated_texts, edited_endings),
            ('bert_prediction_cf', generated_texts, counterfactuals),
            ('bert_prediction_original', generated_texts, original_endings),
            ('bert_edited_ending_cf', edited_endings, counterfactuals),
            ('bert_edited_ending_original', edited_endings, original_endings),
        ]

        bert_scores = {}
        for label, texts_a, texts_b in all_comparisons:
            if texts_b:
                try:
                    _, _, f1 = self.bert_scorer.score(texts_a, texts_b)
                    avg_f1 = f1.mean().item()
                    logger.info(f"{label}_f1: {avg_f1}")
                    bert_scores[f"{label}_f1"] = avg_f1
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bert_scores[f"{label}_f1"] = 'N/A'

        return bert_scores

    def calculate_bleu_similarity(self, generated_texts, edited_endings, counterfactuals, original_endings):

        # print("Calculating BLEU scores...")

        # Prepare references for BLEU score calculation
        # edited_endings_refs = [[ending] for ending in all_edited_endings] if all_edited_endings else None
        # counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        # initials_refs = [[init] for init in all_initials]
        # original_endings_refs = [[orig] for orig in all_original_endings]

        # Prepare references for BLEU score calculation
        edited_endings_refs = [edited_endings] if edited_endings else None
        counterfactuals_refs = [counterfactuals]
        original_endings_refs = [original_endings]

        # List of all comparisons we want to calculate BLEU scores for
        all_comparisons = [
            ('bleu_prediction_edited', generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', generated_texts, counterfactuals_refs),
            ('bleu_prediction_original', generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', edited_endings, counterfactuals_refs),
            ('bleu_edited_ending_original', edited_endings, original_endings_refs),
        ]

        # Dictionary to store BLEU scores for each comparison
        bleu_scores = {}
        for label, texts, references in all_comparisons:
            if references is not None:
                try:
                    # Calculate BLEU score
                    bleu_result = self.sacre_bleu.corpus_score(texts, references)
                    bleu_score = bleu_result.score
                    logger.info(f"{label}: {bleu_score}")
                    bleu_scores[label] = bleu_score
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bleu_scores[label] = 'N/A'

        return bleu_scores


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
  