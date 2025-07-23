import os
import pandas as pd
import logging
from src.cpo.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(df):
    """
    Extracts necessary columns, computes similarity metrics using MetricsEvaluator,
    and returns a DataFrame of metrics.
    """
    # Verify the expected columns exist
    required_columns = ['generated', 'edited', 'counterfactual', 'original']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"CSV file is missing required columns: {missing_columns}. Found columns: {df.columns.tolist()}")

    # Extract columns
    generated_texts = df['generated'].astype(str).tolist()
    edited_endings = df['edited'].astype(str).tolist()
    counterfactuals = df['counterfactual'].astype(str).tolist()
    original_endings = df['original'].astype(str).tolist()

    evaluator = MetricsEvaluator()
    all_metrics = {}

    # Calculate BART similarity metrics
    comparisons = [
        ('bart/pred_edited', generated_texts, edited_endings),
        ('bart/pred_cf', generated_texts, counterfactuals),
        ('bart/pred_original', generated_texts, original_endings),
        ('bart/edited_cf', edited_endings, counterfactuals),
        ('bart/edited_original', edited_endings, original_endings),
    ]
    
    for label, src, tgt in comparisons:
        try:
            scores = evaluator.bart_scorer.score(src, tgt, batch_size=4)
            all_metrics[f"{label}_score"] = sum(scores) / len(scores)
        except Exception as e:
            print(f"Error computing BARTScore for {label}: {e}")
            all_metrics[f"{label}_score"] = float('nan')

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    return metrics_df

def process_file(file_path):
    """
    Process a single CSV file:
      - Reads the file.
      - Calculates similarity metrics.
      - Saves the output file in the same directory as the input file with suffix '_metrics.csv'.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        metrics_df = process_data(df)

        base_dir = os.path.dirname(file_path)
        base_name, ext = os.path.splitext(os.path.basename(file_path))
        output_file_path = os.path.join(base_dir, f'{base_name}_metrics{ext}')
        metrics_df.to_csv(output_file_path, index=False)
    else:
        print(f"File not found: {file_path}")

def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    The output metric files will be saved in the same repository.
    
    """
    if os.path.isdir(repo_path):
        # List all CSV files in the repository that start with the given prefix.
        csv_files = [f for f in os.listdir(repo_path) 
                     if f.endswith('.csv') and f.startswith(prefix)]
        if not csv_files:
            print(f"No CSV files with prefix '{prefix}' found in {repo_path}")
            return
        for csv_file in csv_files:
            file_path = os.path.join(repo_path, csv_file)
            process_file(file_path)
    else:
        print(f"Repository not found: {repo_path}")

def main():
    """
    Main function to process multiple repositories.
    For each repository, you specify a prefix to select the files you want.
    For example:
      - For test files, use prefix 'test_details_cpo'
    The output file is saved in the same directory as the input file.
    """
    # List of repository directories to process.
    repo_paths = [
        '/data/user/Projects/TimeTravel-DifferentiableMetrics/models/dto_2025-05-29_15-58-59',
    ]

    # # Process validation files
    # for repo in repo_paths:
    #     process_repository(repo, prefix='validation_details_pg_')
    
    # Process test files
    for repo in repo_paths:
        process_repository(repo, prefix='test_details_cpo')

if __name__ == "__main__":
    main()
