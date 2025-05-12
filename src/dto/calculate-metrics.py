import os
import pandas as pd
import logging
from src.dto.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(df):
    """
    Extracts necessary columns, computes similarity metrics using MetricsEvaluator,
    and returns a DataFrame of metrics.
    """
    generated_texts = df['Generated Text'].tolist()
    edited_endings = df['Edited Ending'].tolist()
    counterfactuals = df['Counterfactual'].tolist()
    initials = df['Initial'].tolist()
    premises = df['Premise'].tolist()
    original_endings = df['Original Ending'].tolist()

    evaluator = MetricsEvaluator()
    all_metrics = {}

    # Calculate all similarity metrics (BART, BERT, BLEU, ROUGE)
    all_metrics.update(evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    ))

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
        print(f"Metrics saved to {output_file_path}")
    else:
        print(f"File not found: {file_path}")

def process_repository(repo_path, prefix):
    """
    Process all CSV files in the given repository that start with the specified prefix.
    The output metric files will be saved in the same repository.
    
    For example, if prefix is 'validation_details_pg_', then only files starting with that
    prefix will be processed.
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
            print(f"Processing file: {file_path}")
            process_file(file_path)
    else:
        print(f"Repository not found: {repo_path}")

def main():
    """
    Main function to process multiple repositories.
    For each repository, you specify a prefix to select the files you want.
    For example:
      - For validation files, use prefix 'validation_details_pg_'
      - For test files, use prefix 'test_details_pg_'
    The output file is saved in the same directory as the input file.
    """
    # List of repository directories to process.
    repo_paths = [
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-04-07',  # T5-base, BART, score+Delta_M1, temp 0.7
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-05-17',  # T5-base, BERT, score+Delta_M1, temp 0.7
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-03-11',  # T5-base, BART, Delta_M1, temp 0.7
        # '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-27-09',  # Dynamic, BART, temp 0.7
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-17-09',  # Dynamic, BART+4, temp 1.5
        # '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-12-09',  # Dynamic, BART+4, temp 1
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-06-17',  # Dynamic, BART+4, temp 0.7
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-01-15',  # Fixed0.5, Bart+4, temp 1
        #'/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-02-02-16',  # Fixed0.5, Bart+4, temp 0.7
        # '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-11-11',  # Fixed0.5, Bart, temp 0.7
        '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/pg_2025-03-13-09',  # Fixed0.3, Bart, temp 0.7
    ]

    # Process validation files
    for repo in repo_paths:
        process_repository(repo, prefix='validation_details_pg_')
    
    # Process test files
    for repo in repo_paths:
        process_repository(repo, prefix='test_details_pg_')

if __name__ == "__main__":
    main()
