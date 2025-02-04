import os
import pandas as pd
import logging
from src.utils.metricsl import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory where metrics files will be saved
metrics_output_dir = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/results'
os.makedirs(metrics_output_dir, exist_ok=True)  # Create the directory if it doesn't exist

def process_epoch_data(df, epoch):
    """
    Function to calculate and return similarity metrics for a specific epoch.
    """
    # Filter data for the specific epoch
    epoch_data = df[df['Epoch'] == epoch]

    # Extract necessary columns
    generated_texts = epoch_data['Generated Text'].tolist()
    edited_endings = epoch_data['Edited Ending'].tolist()
    counterfactuals = epoch_data['Counterfactual'].tolist()
    initials = epoch_data['Initial'].tolist()
    premises = epoch_data['Premise'].tolist()
    original_endings = epoch_data['Original Ending'].tolist()
    
    # Initialize the MetricsEvaluator
    evaluator = MetricsEvaluator()

    # Calculate the metrics for the epoch
    all_metrics = {}
    
    # Calculate BART similarity
    all_metrics.update(evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate BERT similarity
    all_metrics.update(evaluator.calculate_and_log_bert_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate BLEU scores
    all_metrics.update(evaluator.calculate_and_log_bleu_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))
    
    # Calculate ROUGE scores
    all_metrics.update(evaluator.calculate_and_log_rouge_scores(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

    # Convert metrics to DataFrame and transpose
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=[f'Epoch {epoch}'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', f'Epoch {epoch}']

    return metrics_df

def process_file(file_path, output_dir, epoch):
    """
    Process a specific file (e.g., validation or test) for the given path and epoch.
    """
    if os.path.exists(file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Process the specified epoch
        metrics_df = process_epoch_data(df, epoch)
        
        # Generate a new file path for the metrics output
        base_name = os.path.basename(file_path).replace('.csv', '')
        metrics_file_name = f'{base_name}_metrics_epoch_{epoch}.csv'
        metrics_file_path = os.path.join(output_dir, metrics_file_name)
        
        # Save the metrics to the output file
        metrics_df.to_csv(metrics_file_path, index=False)
        print(f"Metrics for Epoch {epoch} saved to {metrics_file_path}")
    else:
        print(f"File not found: {file_path}")

def main():
    """
    Main function to process validation and test files for specified epochs.
    """
    # Example input paths
    validation_file_path = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2024-03-22-10/validation_details.csv'
    test_file_path = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/model_2024-03-22-10/test_details.csv'
    output_directory = '/data/agirard/Projects/TimeTravel-PolicyGradientRL/results'
    
    # Specify epochs for validation and test
    validation_epoch = 5  # Process epoch 1 for validation
    test_epoch = 6        # Process epoch 0 for test
    
    # Process validation file
    print("Processing validation metrics...")
    process_file(validation_file_path, output_directory, validation_epoch)
    
    # Process test file
    print("Processing test metrics...")
    process_file(test_file_path, output_directory, test_epoch)

if __name__ == "__main__":
    main()
