import os
import logging
import pandas as pd
from src.utils.metrics import MetricsEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_from_details_file(details_file, epoch, output_metrics_file):
    """
    Evaluate metrics for a specific epoch from a details file.

    Args:
        details_file (str): Path to the details CSV file.
        epoch (int): The epoch to filter data.
        output_metrics_file (str): Path to save the computed metrics.

    Raises:
        FileNotFoundError: If the details file does not exist.
    """
    # Check if details file exists
    if not os.path.exists(details_file):
        logger.error(f"Details file not found at {details_file}")
        raise FileNotFoundError(f"Details file not found at {details_file}")

    # Load details file
    logger.info(f"Loading details file: {details_file}")
    details_df = pd.read_csv(details_file)

    # Filter rows for the specified epoch
    filtered_details = details_df[details_df['Epoch'] == epoch]
    if filtered_details.empty:
        logger.warning(f"No rows found for epoch {epoch}. Using all rows instead.")
        filtered_details = details_df

    # Extract required columns
    try:
        generated_texts = filtered_details['Generated Text'].tolist()
        edited_endings = filtered_details['Edited Ending'].tolist()
        counterfactuals = filtered_details['Counterfactual'].tolist()
        initials = filtered_details['Initial'].tolist()
        premises = filtered_details['Premise'].tolist()
        original_endings = filtered_details['Original Ending'].tolist()
    except KeyError as e:
        logger.error(f"Missing column in details file: {e}")
        raise KeyError(f"Details file is missing required column: {e}")

    # Validate data for metric calculation
    if not (generated_texts and edited_endings):
        logger.error("Generated Text or Edited Ending is empty. Skipping metric calculations.")
        return
    if len(generated_texts) != len(edited_endings):
        logger.error("Mismatch in lengths of Generated Text and Edited Ending. Skipping metric calculations.")
        return

    # Initialize evaluator and calculate metrics
    evaluator = MetricsEvaluator()
    metrics = {}
    try:
        metrics.update(evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
        metrics.update(evaluator.calculate_and_log_bert_similarity(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
        metrics.update(evaluator.calculate_and_log_bleu_scores(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
        metrics.update(evaluator.calculate_and_log_rouge_scores(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

    # Save metrics to file
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    metrics_df.to_csv(output_metrics_file, index=False)
    logger.info(f"Metrics saved to {output_metrics_file}")

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Inputs: Modify these variables as needed
    details_file = "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2024-11-29-11/validation_details_mle.csv"  # Replace with your CSV file path
    epoch = 1  # Replace with your target epoch
    output_metrics_file = "/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2024-11-29-11/validation_metrics_epoch_2_mle_2024-11-29-11.csv"  # Replace with your desired output path

    # Run evaluation
    logger.info(f"Starting evaluation for epoch {epoch}")
    evaluate_from_details_file(details_file, epoch, output_metrics_file)
    logger.info("Evaluation completed.")

if __name__ == "__main__":
    main()
