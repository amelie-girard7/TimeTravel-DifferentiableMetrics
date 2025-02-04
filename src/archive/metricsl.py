# /data/agirard/Projects/Timetravel/src/utils/metrics.py
import logging
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer
from src.BARTScore_metric.bart_score import BARTScorer
from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    A class for evaluating text generation models using various metrics such as BLEU, ROUGE, BERTScore, and BARTScore.
    """

    def __init__(self):
        """
        Initializes the metric evaluators with configurations from the CONFIG file.
        """
        print(f"Initializing MetricsEvaluator with config: {CONFIG}")
        self.sacre_bleu = BLEU()  # SacreBLEU score evaluator
        self.rouge = Rouge()  # ROUGE score evaluator
        self.bert_scorer = BERTScorer(
            model_type=CONFIG["bert_scorer_model_type"],
            device=CONFIG["scorer_device"],
            num_layers=None,
            batch_size=CONFIG["bert_scorer_batch_size"]
        )  # BERTScore evaluator
        self.bart_scorer = BARTScorer(
            device=CONFIG["scorer_device"],
            checkpoint=CONFIG["bart_scorer_checkpoint"]
        )  # BARTScore evaluator
        print("MetricsEvaluator initialized.")

    def calculate_and_log_bleu_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs SacreBLEU scores for various comparisons of generated texts and references.
        """
        print("Calculating BLEU scores...")
        # Prepare references for BLEU score calculation
        edited_endings_refs = [[ending] for ending in all_edited_endings] if all_edited_endings else None
        counterfactuals_refs = [[cf] for cf in all_counterfactuals]
        initials_refs = [[init] for init in all_initials]
        original_endings_refs = [[orig] for orig in all_original_endings]

        print(f"References prepared:\nEdited Endings: {edited_endings_refs}\nCounterfactuals: {counterfactuals_refs}\nInitials: {initials_refs}\nOriginal Endings: {original_endings_refs}")

        # Define all comparisons to calculate BLEU scores
        all_comparisons = [
            ('bleu_prediction_edited', all_generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', all_generated_texts, counterfactuals_refs),
            ('bleu_prediction_initial', all_generated_texts, initials_refs),
            ('bleu_prediction_original', all_generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', all_edited_endings, counterfactuals_refs),
            ('bleu_edited_ending_initial', all_edited_endings, initials_refs),
            ('bleu_edited_ending_original', all_edited_endings, original_endings_refs),
        ]

        print(f"BLEU Comparisons: {all_comparisons}")

        # Calculate and log BLEU scores for each comparison
        bleu_scores = {}
        for label, texts, references in all_comparisons:
            if references is not None:
                try:
                    bleu_result = self.sacre_bleu.corpus_score(texts, references)
                    bleu_score = bleu_result.score
                    logger.info(f"{label}: {bleu_score}")
                    bleu_scores[label] = bleu_score
                    print(f"{label}: {bleu_score}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bleu_scores[label] = 'N/A'
                    print(f"Error calculating {label}: {e}")
        return bleu_scores

    def calculate_and_log_rouge_scores(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs ROUGE scores for various comparisons of generated texts and references.
        """
        print("Calculating ROUGE scores...")
        # Define all comparisons to calculate ROUGE scores
        all_comparisons = [
            ('rouge_prediction_edited', all_generated_texts, all_edited_endings),
            ('rouge_prediction_cf', all_generated_texts, all_counterfactuals),
            ('rouge_prediction_initial', all_generated_texts, all_initials),
            ('rouge_prediction_original', all_generated_texts, all_original_endings),
            ('rouge_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('rouge_edited_ending_initial', all_edited_endings, all_initials),
            ('rouge_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        print(f"ROUGE Comparisons: {all_comparisons}")

        # Initialize dictionary to store Rouge-L F-scores
        rouge_scores = {}

        # Calculate and log ROUGE-L F-scores for each comparison
        for label, hypotheses, references in all_comparisons:
            if references:
                try:
                    # Calculate Rouge-L score
                    rouge_scores_set = self.rouge.get_scores(hypotheses, references, avg=True)
                    score_type = 'rouge-l'

                    # Store Rouge-L F-score
                    rouge_scores[f"{label}_{score_type}_f"] = rouge_scores_set[score_type]['f']

                    # Log and print the Rouge-L F-score
                    logger.info(f"{label}_{score_type}_f: {rouge_scores_set[score_type]['f']}")
                    print(f"{label}_{score_type}_f: {rouge_scores_set[score_type]['f']}")
                    
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    rouge_scores[f"{label}_f"] = 'N/A'
                    print(f"Error calculating {label}: {e}")

        return rouge_scores

    def calculate_and_log_bert_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BERT similarity F1 scores for various comparisons of generated texts and references.
        """
        print("Calculating BERT similarity F1 scores...")

        # Define all comparisons to calculate BERT similarity scores
        all_comparisons = [
            ('bert_prediction_edited', all_generated_texts, all_edited_endings),
            ('bert_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bert_prediction_initial', all_generated_texts, all_initials),
            ('bert_prediction_original', all_generated_texts, all_original_endings),
            ('bert_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bert_edited_ending_initial', all_edited_endings, all_initials),
            ('bert_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        print(f"BERT Comparisons: {all_comparisons}")

        # Calculate and log BERT F1 scores for each comparison
        bert_scores = {}
        
        for label, texts_a, texts_b in all_comparisons:
            if texts_b:
                try:
                    # Get the F1 score (third element of the tuple)
                    _, _, f1 = self.bert_scorer.score(texts_a, texts_b)
                    
                    # Calculate the mean F1 score across all text pairs
                    avg_f1 = f1.mean().item()
                    
                    # Log and store the F1 score for the current comparison
                    logger.info(f"{label}_f1: {avg_f1}")
                    bert_scores[f"{label}_f1"] = avg_f1
                    print(f"{label}_f1: {avg_f1}")
                    
                except Exception as e:
                    # Log the error if something goes wrong in the BERT score calculation
                    logger.error(f"Error calculating {label}: {e}")
                    bert_scores[f"{label}_f1"] = 'N/A'
                    print(f"Error calculating {label}: {e}")

        return bert_scores

    def calculate_and_log_bart_similarity(self, all_generated_texts, all_edited_endings, all_counterfactuals, all_initials, all_premises, all_original_endings, logger):
        """
        Calculates and logs BART-based similarity scores for a variety of text comparisons,
        using the BARTScorer to evaluate the similarity between different segments of texts.
        This version only supports single-reference scoring.
        """
        print("Calculating BART similarity scores...")
        
        # Define all pairs of text segments for which to calculate similarity scores
        all_comparisons = [
            ('bart_prediction_edited', all_generated_texts, all_edited_endings),
            ('bart_prediction_cf', all_generated_texts, all_counterfactuals),
            ('bart_prediction_initial', all_generated_texts, all_initials),
            ('bart_prediction_original', all_generated_texts, all_original_endings),
            ('bart_edited_ending_cf', all_edited_endings, all_counterfactuals),
            ('bart_edited_ending_initial', all_edited_endings, all_initials),
            ('bart_edited_ending_original', all_edited_endings, all_original_endings),
        ]

        print(f"BART Comparisons: {all_comparisons}")

        # Calculate and log BARTScores for each comparison
        bart_scores = {}
        for label, src_texts, tgt_texts in all_comparisons:
            if tgt_texts:
                try:
                    # Single-reference scoring
                    scores = self.bart_scorer.score(src_texts, tgt_texts, batch_size=4)

                    # Validate that the number of scores matches the number of source texts
                    if len(scores) != len(src_texts):
                        raise ValueError(f"Mismatch in the number of scores returned. Expected {len(src_texts)} but got {len(scores)}.")

                    avg_score = sum(scores) / len(scores) if scores else float('nan')
                    logger.info(f"{label}_avg_score: {avg_score}")
                    bart_scores[f"{label}_avg_score"] = avg_score
                    print(f"{label}_avg_score: {avg_score}")
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bart_scores[f"{label}_avg_score"] = 'N/A'
                    print(f"Error calculating {label}: {e}")
        return bart_scores
