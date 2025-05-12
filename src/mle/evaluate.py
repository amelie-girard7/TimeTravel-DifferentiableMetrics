# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/mle/evaluate.py
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from transformers import BartTokenizer
from pytorch_lightning import Trainer
from src.mle.models.model import BartFineTuner
from src.mle.data_loader import create_dataloaders
from src.mle.utils.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(checkpoint_path):
    """
    Main evaluation function that:
    1. Loads model from checkpoint
    2. Runs validation and test evaluation
    3. Saves generated texts
    (Metric calculation is handled separately by calculate_metrics.py)
    """
    model_dir = Path(checkpoint_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize tokenizer and dataloaders
    tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    dev_key = CONFIG["dev_file"].split('.')[0]
    test_key = CONFIG["test_file"].split('.')[0]

    # Load model from checkpoint
    model = BartFineTuner.load_from_checkpoint(
        checkpoint_path,
        model_name=CONFIG["model_name"],
        model_dir=model_dir,
        file_label="_mle"
    )

    trainer = Trainer(accelerator='gpu', devices=1, logger=False)

    # Run validation
    logger.info("Running validation evaluation...")
    trainer.validate(model, dataloaders[dev_key], verbose=False)
    val_details_df = pd.DataFrame(model.epoch_validation_details)

    # Save validation details
    expected_columns = ['Epoch', 'Premise', 'Initial', 'Counterfactual',
                        'Original Ending', 'Edited Ending', 'Generated Text']
    val_details_df = val_details_df[expected_columns]
    val_details_file = model_dir / f"validation_details_mle_{timestamp}.csv"
    val_details_df.to_csv(val_details_file, index=False)
    logger.info(f"Validation details saved to: {val_details_file}")

    # Run test
    logger.info("Running test evaluation...")
    trainer.test(model, dataloaders[test_key], verbose=False)
    test_details_df = pd.DataFrame(model.epoch_test_details)

    # Save test details
    test_details_df = test_details_df[expected_columns]
    test_details_file = model_dir / f"test_details_mle_{timestamp}.csv"
    test_details_df.to_csv(test_details_file, index=False)
    logger.info(f"Test details saved to: {test_details_file}")

    # Print instructions for running metric calculation
    logger.info("\nEvaluation complete. To calculate metrics, run:")
    logger.info(f"python calculate_metrics.py --files {val_details_file} {test_details_file}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_checkpoint>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    main(checkpoint_path)