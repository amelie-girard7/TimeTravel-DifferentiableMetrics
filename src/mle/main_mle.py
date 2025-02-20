
# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/mle/main_mle.py
import sys
import os
import datetime
import logging
from transformers import BartTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.mle.models.model import FlanT5FineTuner
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

def setup_model(model_dir, file_label="", checkpoint_path=None):
    """
    Initializes the model for MLE training.
    If a checkpoint path is provided, the function loads the model from the checkpoint.
    """
    if checkpoint_path:
        #logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )
    else:
        #logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )

    return model

def setup_trainer(max_epochs, checkpoint_callback, wandb_logger):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        default_root_dir="./"
    )
    #logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer

def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    """
    Evaluates the model using BARTScore metrics and saves the computed metrics.
    """
    #logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")
    #print(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load model from checkpoint
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)

    # Run evaluation (validation or test)
    trainer = Trainer(accelerator='gpu', devices=1)
    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    evaluator = MetricsEvaluator()

    # Load generated texts and references from CSV
    details_file = os.path.join(model_dir, f"{phase}_details{file_label}.csv")
    if not os.path.exists(details_file):
        logger.error(f"{phase.capitalize()} details file not found at {details_file}")
        raise FileNotFoundError(f"{phase.capitalize()} details file not found at {details_file}")

    details_df = pd.read_csv(details_file)
    filtered_details = details_df[details_df['Epoch'] == best_epoch]
    if filtered_details.empty:
        logger.warning(f"No rows found for epoch {best_epoch} in {phase} details. Using all rows instead.")
        filtered_details = details_df

    try:
        generated_texts = filtered_details['Generated Text'].tolist()
        edited_endings = filtered_details['Edited Ending'].tolist()
        counterfactuals = filtered_details.get('Counterfactual', []).tolist()
        initials = filtered_details.get('Initial', []).tolist()
        original_endings = filtered_details.get('Original Ending', []).tolist()
        premises = filtered_details.get('Premise', []).tolist()
    except KeyError as e:
        logger.error(f"Missing column in {phase} details: {e}")
        raise

    if not (generated_texts and edited_endings):
        logger.error(f"Generated texts or edited endings are empty. Skipping metric calculations for {phase}.")
        return

    # Compute all BARTScore comparisons.
    bart_scores_dict = evaluator.calculate_and_log_bart_similarity(
        generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
    )

    # Save metrics to CSV
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    metrics_df = pd.DataFrame.from_dict(bart_scores_dict, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    metrics_df.to_csv(metrics_file, index=False)

    #logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")
    #print(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")

def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts the epoch number from the checkpoint file name.
    """
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    logger.warning(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    return "Unknown"

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Ensure only MLE training is enabled
    if not CONFIG["mle_enabled"]:
        raise ValueError("MLE training is not enabled in CONFIG. Please enable it.")

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
        CONFIG["num_workers"]
    )

    train_key, dev_key, test_key = (
        CONFIG["train_file"].split('.')[0],
        CONFIG["dev_file"].split('.')[0],
        CONFIG["test_file"].split('.')[0]
    )

    #print("Starting MLE phase training...")
    mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None

    model = setup_model(model_dir, file_label="_mle", checkpoint_path=mle_checkpoint)

    mle_checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        monitor='validation_mle_loss',
        mode='min',
        save_top_k=1,
        filename="mle_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss={validation_mle_loss:.2f}"
    )

    trainer = setup_trainer(CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
    #print("MLE training completed.")

    best_checkpoint = mle_checkpoint_callback.best_model_path
    if best_checkpoint:
        best_epoch = extract_epoch_from_checkpoint(best_checkpoint)

        evaluate_and_save(model_dir, dataloaders[test_key], best_checkpoint, "_mle", best_epoch, "test")
        evaluate_and_save(model_dir, dataloaders[dev_key], best_checkpoint, "_mle", best_epoch, "validation")

if __name__ == '__main__':
    #logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
    #print("Process completed.")
