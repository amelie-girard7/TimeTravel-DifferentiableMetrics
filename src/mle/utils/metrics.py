import logging
import torch
from src.mle.utils.config import CONFIG
from src.BARTScore_metric.bart_score import BARTScorer
from sacrebleu.metrics import BLEU
from rouge import Rouge
from bert_score import BERTScorer

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    A class for evaluating text generation models using BARTScore as the primary metric.
    """

    def __init__(self):
        """
        Initializes the metric evaluator with all required scorers.
        """
        # Initialize BARTScorer
        self.bart_scorer = BARTScorer(
            device=CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint=CONFIG.get("bart_scorer_checkpoint", "facebook/bart-large-cnn")
        )
        self.bart_scorer.model.eval()
        for param in self.bart_scorer.model.parameters():
            param.requires_grad = False

        # Initialize BERTScore
        try:
            from bert_score import BERTScorer
            self.bert_scorer = BERTScorer(
                model_type=CONFIG.get("bert_scorer_model", "bert-base-uncased"),
                device=CONFIG.get("scorer_device", "cuda" if torch.cuda.is_available() else "cpu")
            )
        except ImportError:
            logger.warning("BERTScore not available. Install with: pip install bert-score")
            self.bert_scorer = None

        # Initialize ROUGE
        try:
            from rouge import Rouge
            self.rouge = Rouge()
        except ImportError:
            logger.warning("ROUGE not available. Install with: pip install rouge")
            self.rouge = None

        # Initialize BLEU
        try:
            import sacrebleu
            self.sacre_bleu = sacrebleu
        except ImportError:
            logger.warning("sacrebleu not available. Install with: pip install sacrebleu")
            self.sacre_bleu = None

 
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

        generated_texts = [str(gt) for gt in generated_texts]
        references = [str(ref) for ref in references]

        # Compute BARTScore for each generated-reference pair
        scores = self.bart_scorer.score(generated_texts, references)
        # Convert scores to a tensor for logging
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=CONFIG.get("scorer_device", "cpu"))
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
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bart_scores[f"{label}_avg_score"] = 'N/A'

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

    def calculate_bleu_similarity(self, generated_texts, edited_endings, counterfactuals, original_endings):
        """
        Calculates BLEU scores using sacrebleu.
        """

        # Prepare references in correct format for sacrebleu
        # Each should be a list of reference lists (one per hypothesis)
        edited_endings_refs = [edited_endings] if edited_endings else None
        counterfactuals_refs = [counterfactuals]
        original_endings_refs = [original_endings]

        all_comparisons = [
            ('bleu_prediction_edited', generated_texts, edited_endings_refs),
            ('bleu_prediction_cf', generated_texts, counterfactuals_refs),
            ('bleu_prediction_original', generated_texts, original_endings_refs),
            ('bleu_edited_ending_cf', edited_endings, counterfactuals_refs),
            ('bleu_edited_ending_original', edited_endings, original_endings_refs),
        ]

        bleu_scores = {}
        for label, hypotheses, references in all_comparisons:
            if references is not None:
                try:
                    # Calculate BLEU score
                    # hypotheses: list of strings (predictions)
                    # references: list of lists (each inner list contains references for one prediction)
                    bleu_result = self.sacre_bleu.corpus_bleu(
                        hypotheses, 
                        references
                    )
                    bleu_score = bleu_result.score
                    logger.info(f"{label}: {bleu_score}")
                    bleu_scores[label] = bleu_score
                except Exception as e:
                    logger.error(f"Error calculating {label}: {e}")
                    bleu_scores[label] = 'N/A'

        return bleu_scores