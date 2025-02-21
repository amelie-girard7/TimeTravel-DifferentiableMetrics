import sys
import os
import datetime
import logging
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.dto.models.model import FlanT5FineTuner
from src.dto.data_loader import create_dataloaders
from src.dto.utils.metrics import MetricsEvaluator
from src.dto.utils.config import CONFIG
import pandas as pd
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure safe loading of checkpoints in PyTorch 2.6+ - I don't think we need this line
torch.serialization.add_safe_globals([os.path, re, datetime])

def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    """
    Evaluates the DTO model on a dataset using BARTScore metrics.

    - Uses pre-padded tensors from `collate_fn()` instead of re-padding manually.
    - Extracts already processed input_ids, attention_mask, and labels from the batch.
    - Ensures that all required keys are present in the batch.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load DTO model from checkpoint.
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)
    logger.info("Model loaded.")

    # Run evaluation (validation or test) using the Trainer.
    trainer = Trainer(accelerator='gpu', devices=1)
    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")
    logger.info("Trainer evaluation complete.")

    evaluator = MetricsEvaluator()
    metrics = {}

    # Required keys for evaluation
    required_keys = ['input_ids', 'attention_mask', 'edited_ending', 'counterfactual', 'initial', 'premise', 'original_ending']

    # Extract batch from the loader (these are already padded via collate_fn)
    for batch in loader:
        # Check if all required keys are present
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise KeyError(f"Missing required keys in batch: {missing_keys}")

        # Extract required tensors and lists
        gen_input_ids = batch['input_ids']
        gen_attention_mask = batch['attention_mask']
        edited_endings_list = batch['edited_ending']
        counterfactual_list = batch['counterfactual']
        initial_list = batch['initial']
        premise_list = batch['premise']
        original_endings_list = batch['original_ending']

        logger.info(f"Batch tensor shapes - input_ids: {gen_input_ids.shape}, attention_mask: {gen_attention_mask.shape}")

        # Forward pass on already padded batch
        generated_embeddings = model.forward(gen_input_ids, gen_attention_mask)
        logger.info(f"Generated embeddings shape: {generated_embeddings.shape}")

        # Decode the embeddings to text (argmax over the vocabulary)
        generated_texts = model.tokenizer.batch_decode(
            generated_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Compute BARTScore similarity
        bart_scores_dict = evaluator.calculate_and_log_bart_similarity(
            generated_texts,
            edited_endings_list,  # Ground truth references
            counterfactual_list,
            initial_list,
            premise_list,
            original_endings_list,
            logger
        )
        metrics.update(bart_scores_dict)

    logger.info(f"Average DTO BARTScore for {phase}: {metrics.get('bart_prediction_edited_avg_score', 'N/A')}")

    # Save the computed metrics to a CSV file.
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")

def setup_model(model_dir, file_label="", checkpoint_path=None, load_from_mle=False):
    """
    Initializes and loads the model for training or evaluation.
    """
    if load_from_mle and checkpoint_path:
        logger.info(f"Loading model from MLE checkpoint: {checkpoint_path}")

        # Load checkpoint (allow metadata loading)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False
        )

        # Remove `bart_scorer` weights if present
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("bart_scorer")}
        checkpoint["state_dict"] = state_dict

        # Load model without expecting `bart_scorer`
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label="_dto",
            strict=False  # Allow missing keys
        )

    else:
        logger.info(f"Initializing a fresh DTO model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label="_dto"
        )

    return model


def setup_trainer(max_epochs, model_dir):
    """
    Sets up the PyTorch Lightning Trainer with W&B logger and checkpointing.
    """
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        monitor='validation_dto_loss',
        mode='min',
        save_top_k=1,
        filename="dto_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_dto_loss:.2f}"
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        default_root_dir=model_dir
    )
    
    return trainer, checkpoint_callback


def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts the epoch number from the checkpoint file name.
    """
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    return int(match.group(1)) if match else "Unknown"


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Set up model directory with timestamp
    model_dir = CONFIG["models_dir"] / f"dto_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and data
    tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(CONFIG["data_dir"], tokenizer, CONFIG["batch_size"], CONFIG["num_workers"])

    # Extract dataset keys
    train_key, dev_key, test_key = [CONFIG[key].split('.')[0] for key in ["train_file", "dev_file", "test_file"]]

    # Experiment mode setup
    experiment_mode = CONFIG["experiment_mode"]
    file_label, checkpoint_path, load_from_mle = ("_dto_scratch", None, False) if experiment_mode == "scratch" else (
        "_dto_mle", CONFIG["dto_checkpoint_path"], True)

    logger.info(f"Running Experiment: {experiment_mode}")

    # Ensure dataset keys exist
    if train_key not in dataloaders or dev_key not in dataloaders:
        logger.error(f"Missing required dataloaders: Available keys -> {dataloaders.keys()}")
        return

    # Initialize the model and trainer
    model = setup_model(model_dir, file_label, checkpoint_path, load_from_mle)
    trainer, checkpoint_callback = setup_trainer(CONFIG["dto_epochs"], model_dir)

    # Train the model
    trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
    logger.info(f"Experiment {experiment_mode} completed.")

    # Retrieve the best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        logger.error("No valid checkpoint found. Exiting evaluation.")
        return

    best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
    logger.info(f"Best DTO checkpoint at epoch: {best_epoch}")

    # Evaluate the model on test and validation sets
    for phase, key in [("test", test_key), ("validation", dev_key)]:
        if key in dataloaders:
            evaluate_and_save(model_dir, dataloaders[key], best_checkpoint, file_label, best_epoch, phase)
        else:
            logger.warning(f"Skipping {phase} evaluation: Missing dataloader key {key}")


if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
