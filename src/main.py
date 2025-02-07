import sys
import os
import datetime
import logging
import re
import pandas as pd
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.models.model import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.metrics import MetricsEvaluator
from src.utils.config import CONFIG

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir, file_label="", checkpoint_path=None, use_policy_gradient=False, use_dto=False):
    """
    Initializes or loads a model based on the specified training mode.
    """
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )
    else:
        logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(CONFIG["model_name"], model_dir, file_label=file_label)

    # Set training mode
    model.use_policy_gradient = use_policy_gradient
    model.use_differentiable_metrics = use_dto

    return model

def setup_trainer(max_epochs, checkpoint_callback, wandb_logger):
    """
    Configures the PyTorch Lightning Trainer.
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
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer

def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    """
    Evaluates the model using the best checkpoint and saves metrics.
    """
    logger.info(f"Evaluating {phase} data for epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load model
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)
    trainer = Trainer(accelerator='gpu', devices=1)

    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Load and process results
    details_file = os.path.join(model_dir, f"{phase}_details{file_label}.csv")
    if not os.path.exists(details_file):
        logger.error(f"{phase.capitalize()} details file not found: {details_file}")
        raise FileNotFoundError(f"{phase.capitalize()} details file not found: {details_file}")

    details_df = pd.read_csv(details_file)
    filtered_details = details_df[details_df['Epoch'] == best_epoch]
    if filtered_details.empty:
        filtered_details = details_df

    # Extract relevant columns
    try:
        generated_texts = filtered_details['Generated Text'].tolist()
        edited_endings = filtered_details['Edited Ending'].tolist()
    except KeyError as e:
        logger.error(f"Missing column in {phase}_details: {e}")
        return

    if not (generated_texts and edited_endings):
        logger.error(f"Empty data for {phase}, skipping metric calculations.")
        return

    # Compute evaluation metrics
    evaluator = MetricsEvaluator()
    metrics = {}

    try:
        metrics.update(evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BART scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_bert_similarity(
            generated_texts, edited_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BERT scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_bleu_scores(
            generated_texts, edited_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BLEU scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_rouge_scores(
            generated_texts, edited_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores for {phase}: {e}")

    # Save metrics
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    pd.DataFrame.from_dict(metrics, orient='index', columns=['Score']).to_csv(metrics_file, index=False)
    logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")

def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts epoch number from checkpoint file name.
    """
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    return int(match.group(1)) if match else "Unknown"

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Determine active training mode
    if CONFIG["use_differentiable_metrics"]:
        phase = "dto"
    elif CONFIG["mle_enabled"]:
        phase = "mle"
    elif CONFIG["pg_enabled"]:
        phase = "pg"
    else:
        raise ValueError("No training mode is enabled in CONFIG.")

    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"{phase}_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB logger
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False
    )
    wandb_logger.experiment.config.update(CONFIG)

    # Setup tokenizer and dataloaders
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"], tokenizer, CONFIG["batch_size"], CONFIG["num_workers"]
    )
    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    # --- DTO Training ---
    if CONFIG["use_differentiable_metrics"]:
        print("\n🚀 Starting DTO Training...\n")

        dto_checkpoint = CONFIG["dto_checkpoint_path"] if CONFIG["dto_checkpoint_path"] else None
        model = setup_model(model_dir, file_label="_dto", checkpoint_path=dto_checkpoint, use_dto=True)

        dto_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_differentiable_loss',
            mode='min',
            save_top_k=1,
            filename="dto_checkpoint_epoch-{epoch:02d}-val_loss={validation_differentiable_loss:.2f}"
        )

        trainer = setup_trainer(CONFIG["dto_epochs"], dto_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("\n✅ DTO Training Completed!\n")

        best_checkpoint = dto_checkpoint_callback.best_model_path
        best_epoch = extract_epoch_from_checkpoint(best_checkpoint)

        evaluate_and_save(model_dir, dataloaders[test_key], best_checkpoint, "_dto", best_epoch, "test")
        evaluate_and_save(model_dir, dataloaders[dev_key], best_checkpoint, "_dto", best_epoch, "validation")

if __name__ == '__main__':
    logger.info("Starting the main process...")
    main()
    logger.info("Process completed.")
