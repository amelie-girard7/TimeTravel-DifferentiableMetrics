import os
import sys
import pandas as pd
import uuid  # Ensure this import statement is present
from src.mle.utils.config import CONFIG
from src.mle.utils.utils import chatgpt_zero_shot_inference, chatgpt_one_shot_inference
from src.mle.utils.metrics import MetricsEvaluator
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the specified inference mode or just run similarity metrics.
    """

    # Ensure API key is set (used internally by utils)
    if not os.getenv("OPENAI_API_KEY"):
        logger.info("Error: Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Path to the test data file for fine-tuning and inference
    test_data_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/data/transformed/test_data.json"

    # Check if the test data file exists
    if not os.path.exists(test_data_path):
        logger.info(f"Gold data file does not exist: {test_data_path}")
        return

    # Load the test data
    logger.info(f"Loading test data from: {test_data_path}")
    test_data = pd.read_json(test_data_path, lines=True)
    logger.info("test data loaded successfully. Sample data:")
    logger.info(test_data.head())  

    results_path = None
    if not CONFIG["run_similarities_only"]:
        # Run the specified mode
        if CONFIG["inference_mode"] == "zero_shot":
            results = chatgpt_zero_shot_inference(test_data)
            # Save the results to a CSV file
            results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/zero_shot_results.csv"
        elif CONFIG["inference_mode"] == "one_shot":
            # Run one-shot inference using ChatGPT
            results = chatgpt_one_shot_inference(test_data, CONFIG["example_selection"])
            # Save the results to a CSV file
            if CONFIG["example_selection"] == "random":
                results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/one_shot_results_random.csv"
            else:
                results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/one_shot_results_fixed.csv"
        else:
            logger.info(f"Unknown mode: {CONFIG['inference_mode']}")
            return

        # Save results to CSV with correct headers
        pd.DataFrame(results).to_csv(results_path, index=False)
    else:
        # If running similarities only, load the existing results
        if CONFIG["inference_mode"] == "zero_shot":
            results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/zero_shot_results.csv"
        elif CONFIG["inference_mode"] == "one_shot":
            if CONFIG["example_selection"] == "random":
                results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/one_shot_results_random.csv"
            else:
                results_path = "/data/user/Projects/TimeTravel-DifferentiableMetrics/results/gpt-4o/one_shot_results_fixed.csv"
        else:
            logger.info(f"Invalid mode for running similarities: {CONFIG['inference_mode']}")
            return
        
        if not os.path.exists(results_path):
            logger.info(f"Results file does not exist: {results_path}")
            return
        
        results = pd.read_csv(results_path).to_dict('records')

        # Only run similarity metrics if specifically required
        if CONFIG["run_similarities_only"]:
            run_similarity_metrics(results, results_path)

def run_similarity_metrics(results, results_path):
    """
    Function to run similarity metrics on the generated results and save them to a transposed CSV file.
    """
    metrics_evaluator = MetricsEvaluator()

    # Extract the relevant text fields from the results
    generated_texts = [result['generated_text'] for result in results]
    counterfactuals = [result['counterfactual'] for result in results]
    initials = [result['initial'] for result in results]
    premises = [result['premise'] for result in results]
    original_endings = [result['original_ending'] for result in results]
    edited_endings = [result.get('edited_ending', '') for result in results]  # Ensure edited_endings are extracted

    # Initialize all_metrics dictionary
    all_metrics = {}

    # Calculate BART scores
    logger.info("Calculating BART similarity scores...")
    bart_scores = metrics_evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(bart_scores)


    # Calculate ROUGE scores
    logger.info("Calculating ROUGE scores...")
    rouge_scores = metrics_evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )
    all_metrics.update(rouge_scores)

    # Convert the metrics to a DataFrame and transpose it
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Value'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Value']

    # Save the transposed metrics DataFrame to a CSV file
    metrics_results_path = results_path.replace(".csv", "_metrics.csv")
    metrics_df.to_csv(metrics_results_path, index=False)
    logger.info(f"Similarity metrics saved to {metrics_results_path}")

if __name__ == "__main__":
    main()
