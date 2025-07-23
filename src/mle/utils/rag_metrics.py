import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sacrebleu.metrics import BLEU
from rouge import Rouge
from src.BARTScore_metric.bart_score import BARTScorer
from src.mle.utils.config import CONFIG

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """
    Unified metric evaluator that always calculates all available metrics.
    """
    
    def __init__(self):
        """Initialize all metric scorers"""
        self.sacre_bleu = BLEU()
        self.rouge = Rouge()
        self.bart_scorer = BARTScorer(
            device=CONFIG["scorer_device"],
            checkpoint=CONFIG["bart_scorer_checkpoint"]
        )
        logger.info("MetricsEvaluator initialized with all scorers")

    def calculate_all_metrics(self, results: List[Dict], results_path: Path) -> Dict[str, float]:
        """
        Calculate ALL metrics for given results.
        Args:
            results: List of result dictionaries containing:
                - generated_text
                - edited_ending
                - counterfactual
                - initial
                - premise
                - original_ending
            results_path: Path for saving metrics CSV
        Returns:
            Dictionary of all calculated metrics
        """
        # Extract texts
        texts = [r['generated_text'] for r in results]
        edited = [r['edited_ending'] for r in results]
        counters = [r['counterfactual'] for r in results]
        initials = [r['initial'] for r in results]
        premises = [r['premise'] for r in results]
        originals = [r['original_ending'] for r in results]

        all_metrics = {}
        
        # 1. Calculate BART Scores
        logger.info("Calculating BART scores...")
        bart_scores = self._calculate_bart_scores(
            texts, edited, counters, initials, premises, originals
        )
        all_metrics.update(bart_scores)
        
        # 2. Calculate ROUGE Scores
        logger.info("Calculating ROUGE scores...")
        rouge_scores = self._calculate_rouge_scores(
            texts, edited, counters, initials, premises, originals
        )
        all_metrics.update(rouge_scores)
        
        # 3. Calculate BLEU Scores
        logger.info("Calculating BLEU scores...")
        bleu_scores = self._calculate_bleu_scores(
            texts, edited, counters, initials, premises, originals
        )
        all_metrics.update(bleu_scores)

        # Save metrics
        self._save_metrics(all_metrics, results_path)
        
        return all_metrics

    def _calculate_bart_scores(self, texts, edited, counters, initials, premises, originals):
        """Calculate all BARTScore comparisons"""
        comparisons = [
            ('bart/pred_edited', texts, edited),
            ('bart/pred_cf', texts, counters),
            ('bart/pred_initial', texts, initials),
            ('bart/pred_original', texts, originals),
            ('bart/edited_cf', edited, counters),
            ('bart/edited_initial', edited, initials),
            ('bart/edited_original', edited, originals),
        ]
        
        scores = {}
        for label, src, tgt in comparisons:
            try:
                if src and tgt:
                    batch_scores = self.bart_scorer.score(src, tgt, batch_size=4)
                    avg_score = sum(batch_scores)/len(batch_scores)
                    scores[f"{label}_avg"] = avg_score
                    logger.info(f"{label}_avg: {avg_score}")
            except Exception as e:
                logger.error(f"BARTScore failed for {label}: {str(e)}")
                scores[f"{label}_avg"] = float('nan')
        return scores

    def _calculate_rouge_scores(self, texts, edited, counters, initials, premises, originals):
        """Calculate all ROUGE-L F1 scores"""
        comparisons = [
            ('rouge/pred_edited', texts, edited),
            ('rouge/pred_cf', texts, counters),
            ('rouge/pred_initial', texts, initials),
            ('rouge/pred_original', texts, originals),
            ('rouge/edited_cf', edited, counters),
            ('rouge/edited_initial', edited, initials),
            ('rouge/edited_original', edited, originals),
        ]
        
        scores = {}
        for label, hyp, ref in comparisons:
            try:
                if hyp and ref:
                    rouge_scores = self.rouge.get_scores(hyp, ref, avg=True)
                    f1 = rouge_scores['rouge-l']['f']
                    scores[f"{label}_f1"] = f1
                    logger.info(f"{label}_f1: {f1}")
            except Exception as e:
                logger.error(f"ROUGE failed for {label}: {str(e)}")
                scores[f"{label}_f1"] = float('nan')
        return scores

    def _calculate_bleu_scores(self, texts, edited, counters, initials, premises, originals):
        """Calculate all BLEU scores"""
        comparisons = [
            ('bleu/pred_edited', texts, [edited]),
            ('bleu/pred_cf', texts, [counters]),
            ('bleu/pred_initial', texts, [initials]),
            ('bleu/pred_original', texts, [originals]),
            ('bleu/edited_cf', edited, [counters]),
            ('bleu/edited_initial', edited, [initials]),
            ('bleu/edited_original', edited, [originals]),
        ]
        
        scores = {}
        for label, hyp, ref in comparisons:
            try:
                if hyp and ref[0]:
                    bleu_score = self.sacre_bleu.corpus_score(hyp, ref).score
                    scores[label] = bleu_score
                    logger.info(f"{label}: {bleu_score}")
            except Exception as e:
                logger.error(f"BLEU failed for {label}: {str(e)}")
                scores[label] = float('nan')
        return scores

    def _save_metrics(self, metrics: Dict, results_path: Path):
        """Save metrics to CSV"""
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df.index.name = 'Metric'
        metrics_df.reset_index(inplace=True)
        
        metrics_path = results_path.with_name(results_path.stem + "_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")