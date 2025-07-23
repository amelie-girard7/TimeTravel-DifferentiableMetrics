import os
import pandas as pd
import logging
from src.dto.utils.metrics import MetricsEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_scores(df):
    """Calculate scores between Generated Text and Edited Ending"""
    try:
        evaluator = MetricsEvaluator()
        
        # Initialize score columns
        df['bart_score'] = 0.0
        df['rouge_l_score'] = 0.0
        df['bleu_score'] = 0.0

        for idx, row in df.iterrows():
            try:
                # Get all required text fields
                texts = {
                    'generated': str(row['Generated Text']),
                    'edited': str(row['Edited Ending']),
                    'counterfactual': str(row['Counterfactual']),
                    'original': str(row['Original Ending'])
                }

                # Skip empty texts
                if not texts['generated'].strip() or not texts['edited'].strip():
                    continue

                # Calculate all metrics using the proper methods
                bart_metrics = evaluator.calculate_bart_similarity(
                    generated_texts=[texts['generated']],
                    edited_endings=[texts['edited']],
                    counterfactuals=[texts['counterfactual']],
                    original_endings=[texts['original']]
                )
                
                rouge_metrics = evaluator.calculate_rouge_similarity(
                    generated_texts=[texts['generated']],
                    edited_endings=[texts['edited']],
                    counterfactuals=[texts['counterfactual']],
                    original_endings=[texts['original']]
                )
                
                if evaluator.sacre_bleu is not None:
                    bleu_metrics = evaluator.calculate_bleu_similarity(
                        generated_texts=[texts['generated']],
                        edited_endings=[texts['edited']],
                        counterfactuals=[texts['counterfactual']],
                        original_endings=[texts['original']]
                    )
                else:
                    bleu_metrics = {}

                # Extract and store the specific scores we want
                df.at[idx, 'bart_score'] = bart_metrics.get('bart/pred_edited_score', 0.0)
                df.at[idx, 'rouge_l_score'] = rouge_metrics.get('rouge_prediction_edited_rouge-l_f', 0.0)
                df.at[idx, 'bleu_score'] = bleu_metrics.get('bleu_prediction_edited', 0.0)

            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue

        return df

    except Exception as e:
        logger.error(f"Error in score calculation: {str(e)}")
        raise

def main():
    """Main function to process the reward CSV file"""
    input_path = '/data/user/Projects/TimeTravel-DifferentiableMetrics/src/dto/reward.csv'
    output_path = '/data/user/Projects/TimeTravel-DifferentiableMetrics/src/dto/reward_scored.csv'
    
    try:
        logger.info(f"Reading input file: {input_path}")
        df = pd.read_csv(input_path)
        
        logger.info("Calculating scores...")
        scored_df = calculate_scores(df)
        
        logger.info(f"Saving results to: {output_path}")
        scored_df.to_csv(output_path, index=False)
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()