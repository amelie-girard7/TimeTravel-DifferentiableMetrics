import sys
import os
import datetime
import logging
from transformers import BartForConditionalGeneration, BartTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # Added EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.mle.models.model import BartFineTuner
from src.mle.data_loader import create_dataloaders
from src.mle.utils.metrics import MetricsEvaluator
from src.mle.utils.config import CONFIG
import pandas as pd
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize or load the model from a checkpoint
def setup_model(model_dir, file_label="", checkpoint_path=None):
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = BartFineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )
    else:
        logger.info(f"Initializing fresh model: {CONFIG['model_name']} with label {file_label}")
        model = BartFineTuner(CONFIG["model_name"], model_dir, file_label=file_label)

    return model


# Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing
def setup_trainer(max_epochs, checkpoint_callback, early_stop_callback, wandb_logger):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        # callbacks=[checkpoint_callback],
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=0.1,
        default_root_dir="./"
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer


# Extract the epoch number from the checkpoint filename
def extract_epoch_from_checkpoint(checkpoint_path):
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    return "Unknown"


# Main execution logic
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"mle_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False
    )
    wandb_logger.experiment.config.update(CONFIG)

    tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )

    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], \
    CONFIG["test_file"].split('.')[0]

    # PG Phase
    model = setup_model(
        model_dir,
        file_label="_mle",
        checkpoint_path=CONFIG["mle_checkpoint_path"]
    )

    mle_checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        monitor='validation_mle_loss',
        mode='min',
        save_top_k=1,
        filename="mle_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_mle_loss:.2f}"
    )

    # Early stopping callback to stop training when the validation loss stops improving
    early_stop_callback = EarlyStopping(
        monitor='validation_mle_loss',  
        min_delta=0.00, 
        patience=2,  
        verbose=True, 
        mode='min' 
    )

    # trainer = setup_trainer(CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
    trainer = setup_trainer(CONFIG["mle_epochs"], mle_checkpoint_callback, early_stop_callback, wandb_logger)

    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])

    best_checkpoint = mle_checkpoint_callback.best_model_path
    best_epoch = extract_epoch_from_checkpoint(best_checkpoint)

    # Load explicitly the best checkpoint
    model = setup_model(model_dir, file_label="_mle", checkpoint_path=best_checkpoint)

    # Explicitly set up Trainer without logging for final evaluation
    trainer = Trainer(accelerator='gpu', devices=1, logger=False)

    # Run explicit validation pass to collect and log details
    trainer.validate(model, dataloaders[dev_key], verbose=False)


if __name__ == '__main__':
    logger.info("Starting the MLE process...")
    main()
    logger.info("Process completed.")