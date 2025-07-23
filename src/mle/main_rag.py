import os
import sys
import logging
import pandas as pd
from pathlib import Path

from src.mle.utils.config import CONFIG
from src.mle.utils.rag_utils import (
    load_docs,
    build_vector_store,
    make_rag_chain,
    run_rag_inference
)
from src.mle.utils.rag_metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """RAG pipeline with dual mode support"""
    # In generation mode, we need the API key
    if not CONFIG["run_similarities_only"] and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set (required for generation mode)")
        sys.exit(1)

    # Prepare results directory & path
    results_dir = CONFIG["results_dir"] / "rag"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "rag_results.csv"

    # Determine whether to re-run generation or just metrics
    if CONFIG["run_similarities_only"]:
        logger.info("Running in metrics-only mode")
        if not results_path.exists():
            logger.error(f"No existing results found at {results_path} for metrics calculation")
            sys.exit(1)
        results = pd.read_csv(results_path).to_dict("records")
    else:
        logger.info("Running in full generation+metrics mode")
        results = run_full_rag_pipeline()

    # compute and save metrics
    calculate_and_save_metrics(results, results_path)


def run_full_rag_pipeline() -> list:
    """Load data, build store from test, generate on train"""
    data_dir = CONFIG["data_dir"]
    train_path = data_dir / CONFIG["train_file"]
    test_path  = data_dir / CONFIG["test_file"]

    if not train_path.exists() or not test_path.exists():
        logger.error(f"Missing data file(s): {train_path} or {test_path}")
        return []

    logger.info(f"Loading generation (train) data from {train_path}")
    train_data = pd.read_json(train_path, lines=True, orient='records')

    logger.info(f"Loading retrieval (test) examples from {test_path}")
    test_data = pd.read_json(test_path, lines=True, orient='records')

    # Build vector store from test (example) set
    docs = load_docs(train_data)
    persist_path = CONFIG["rag"]["persist_path"]
    build_vector_store(docs, persist_path)

    # Build chain and run on train set
    chain = make_rag_chain(persist_path, k=CONFIG["rag"]["k"])
    results = run_rag_inference(chain, test_data)

    return results


def calculate_and_save_metrics(results: list, results_path: Path):
    """Compute BLEU/ROUGE/BARTScore and save both results & metrics"""
    if not results:
        logger.warning("No results available for metrics calculation")
        return

    logger.info("Calculating all metrics...")
    try:
        evaluator = MetricsEvaluator()
        metrics = evaluator.calculate_all_metrics(results, results_path)
        # Save generated outputs (unless metrics-only mode)
        if not CONFIG["run_similarities_only"]:
            pd.DataFrame(results).to_csv(results_path, index=False)
            logger.info(f"Results saved to {results_path}")
        logger.info("Metric calculation completed")
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")


if __name__ == "__main__":
    main()
