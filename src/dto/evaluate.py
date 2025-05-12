import sys
import pandas as pd
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from src.dto.models.model import BartFineTuner
from src.dto.data_loader import create_dataloaders
from src.dto.utils.config import CONFIG
from pathlib import Path
from datetime import datetime


def main(checkpoint_path):
    model_dir = Path(checkpoint_path).parent
    tokenizer = BartFineTuner.from_pretrained(CONFIG["model_name"], legacy=False)

    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    dev_key = CONFIG["dev_file"].split('.')[0]
    test_key = CONFIG["test_file"].split('.')[0]

    model = BartFineTuner.load_from_checkpoint(
        checkpoint_path,
        model_name=CONFIG["model_name"],
        model_dir=model_dir,
        file_label="_pg"
    )

    trainer = Trainer(accelerator='gpu', devices=1, logger=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run validation
    trainer.validate(model, dataloaders[dev_key], verbose=False)
    val_details_df = pd.DataFrame(model.epoch_validation_details)

    expected_columns = ['Premise', 'Initial', 'Counterfactual', 'Original Ending', 'Edited Ending', 'Generated Text']
    val_details_df = val_details_df[expected_columns]

    val_details_file = model.model_dir / f"validation_details_pg_{timestamp}.csv"
    val_details_df.to_csv(val_details_file, index=False)
    print(f"Validation details saved to: {val_details_file}")

    # Run test
    trainer.test(model, dataloaders[test_key], verbose=False)
    test_details_df = pd.DataFrame(model.epoch_test_details)

    test_details_df = test_details_df[expected_columns]

    test_details_file = model.model_dir / f"test_details_pg_{timestamp}.csv"
    test_details_df.to_csv(test_details_file, index=False)
    print(f"Test details saved to: {test_details_file}")


if __name__ == '__main__':
    checkpoint_path = sys.argv[1]
    main(checkpoint_path)
