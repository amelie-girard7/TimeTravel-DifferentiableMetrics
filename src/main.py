# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/main.py
import os
import datetime
import logging
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.models.model import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.metrics import MetricsEvaluator
from src.utils.config import CONFIG
import pandas as pd
import re

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(model_dir, file_label="", checkpoint_path=None, use_policy_gradient=False):
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

    # Set model training mode
    model.use_policy_gradient = use_policy_gradient
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
        #log_every_n_steps=1,
        default_root_dir="./"
    )
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer

def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    """
    Evaluates data for the specified phase ('test' or 'validation') using the given model checkpoint.
    """
    logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")
    print(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load model from checkpoint
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)

    # Generate predictions using the appropriate evaluation method
    trainer = Trainer(accelerator='gpu', devices=1)
    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Load the details file for the specified phase
    details_file = os.path.join(model_dir, f"{phase}_details{file_label}.csv")
    if not os.path.exists(details_file):
        logger.error(f"{phase.capitalize()} details file not found at {details_file}")
        raise FileNotFoundError(f"{phase.capitalize()} details file not found at {details_file}")
    details_df = pd.read_csv(details_file)

    # Filter rows for the best epoch
    filtered_details = details_df[details_df['Epoch'] == best_epoch]
    if filtered_details.empty:
        logger.warning(f"No rows found for epoch {best_epoch} in {phase}_details_df. Evaluating all rows instead.")
        filtered_details = details_df

    # Extract relevant columns
    try:
        generated_texts = filtered_details['Generated Text'].tolist()
        edited_endings = filtered_details['Edited Ending'].tolist()
        counterfactuals = filtered_details['Counterfactual'].tolist()
        initials = filtered_details['Initial'].tolist()
        premises = filtered_details['Premise'].tolist()
        original_endings = filtered_details['Original Ending'].tolist()
    except KeyError as e:
        logger.error(f"Missing column in filtered_{phase}_details: {e}")
        raise

    # Validate non-empty and matching lengths for metric calculations
    if not (generated_texts and edited_endings):
        logger.error(f"Generated texts or edited endings are empty. Skipping metric calculations for {phase}.")
        return

    if len(generated_texts) != len(edited_endings):
        logger.error(f"Mismatch in lengths of generated texts and edited endings. Skipping metric calculations for {phase}.")
        return

    # Calculate metrics
    evaluator = MetricsEvaluator()
    metrics = {}

    try:
        metrics.update(evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BART similarity scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_bert_similarity(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BERT similarity scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_bleu_scores(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating BLEU scores for {phase}: {e}")

    try:
        metrics.update(evaluator.calculate_and_log_rouge_scores(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        ))
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores for {phase}: {e}")

    # Save metrics
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Metric', 'Score']
    metrics_df.to_csv(metrics_file, index=False)

    logger.info(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")
    print(f"{phase.capitalize()} evaluation metrics saved to {metrics_file}")

def extract_epoch_from_checkpoint(checkpoint_path):
    """
    Extracts the epoch number from the checkpoint file name.
    """
    match = re.search(r"epoch=(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))

    logger.warning(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    print(f"Could not extract epoch from checkpoint path: {checkpoint_path}")
    return "Unknown"

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Define unique directory based on timestamp and phase
    phase = "mle" if CONFIG["mle_enabled"] else "pg"
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    model_dir = CONFIG["models_dir"] / f"{phase}_{model_timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_type = CONFIG.get("dataset_type", "TimeTravel")  # Default fallback is "TimeTravel"
    print(f"Selected dataset type: {dataset_type}")  # Debug dataset type

    # Setup WandB logger
    wandb_logger = WandbLogger(
    project="counterfactualStory",
    entity="counterfactualStory",
    log_model=False  # Avoid logging model checkpoints
    )
    wandb_logger.experiment.config.update(CONFIG)
    wandb_logger.experiment.config.update({
        "log_system_stats": False,  # Turn off system stats
        "log_code": False           # Avoid logging source code
    })


    # Setup tokenizer and dataloaders
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"], 
        tokenizer, 
        CONFIG["batch_size"], 
        CONFIG["num_workers"],
    )
    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    model = None
    best_checkpoint = None

    # --- MLE Phase ---
    if CONFIG["mle_enabled"]:
        print("Starting MLE phase training...")
        mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None
        model = setup_model(
            model_dir,
            file_label="_mle",
            checkpoint_path=mle_checkpoint,
            use_policy_gradient=False
        )

        mle_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_mle_loss', # Metric to monitor during training
            mode='min', # Save the checkpoint when the validation loss decreases (minimum is better)
            save_top_k=1, # Keeping only the top 1 checkpoint, the one with the minimum loss
            filename="mle_checkpoint_epoch={epoch}-val_loss={validation_mle_loss:.2f}" # File name format including epoch and loss
        )

        trainer = setup_trainer(CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("MLE training completed.")

        best_checkpoint = mle_checkpoint_callback.best_model_path
        best_loss = mle_checkpoint_callback.best_model_score
        logger.info(f"Best MLE checkpoint: {best_checkpoint}")
        logger.info(f"Best Validation MLE Loss: {best_loss:.4f}")
        print(f"Best MLE checkpoint: {best_checkpoint}")
        print(f"Best Validation MLE Loss: {best_loss:.4f}")

        if best_checkpoint:
            best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
            logger.info(f"Best MLE checkpoint corresponds to epoch: {best_epoch}")
            print(f"Best MLE checkpoint corresponds to epoch: {best_epoch}")

            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[test_key],
                best_checkpoint=best_checkpoint,
                file_label="_mle",
                best_epoch=best_epoch,
                phase="test"
            )
            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[dev_key],
                best_checkpoint=best_checkpoint,
                file_label="_mle",
                best_epoch=best_epoch,
                phase="validation"
            )

    # --- PG Phase ---
    if CONFIG["pg_enabled"]:
        print("Starting PG phase training...")
        model = setup_model(
            model_dir,
            file_label="_pg",
            checkpoint_path=CONFIG["pg_checkpoint_path"],
            use_policy_gradient=True
        )

        pg_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_pg_loss',
            mode='max',
            save_top_k=1,
            filename="pg_checkpoint_epoch={epoch}-val_loss={validation_pg_loss:.2f}"
        )

        trainer = setup_trainer(CONFIG["pg_epochs"], pg_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("PG training completed.")

        best_checkpoint = pg_checkpoint_callback.best_model_path
        logger.info(f"Best PG checkpoint: {best_checkpoint}")
        print(f"Best PG checkpoint: {best_checkpoint}")

        if best_checkpoint:
            best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
            logger.info(f"Best PG checkpoint corresponds to epoch: {best_epoch}")
            print(f"Best PG checkpoint corresponds to epoch: {best_epoch}")

            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[test_key],
                best_checkpoint=best_checkpoint,
                file_label="_pg",
                best_epoch=best_epoch,
                phase="test"
            )
            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[dev_key],
                best_checkpoint=best_checkpoint,
                file_label="_pg",
                best_epoch=best_epoch,
                phase="validation"
            )

if __name__ == '__main__':
    logger.info("Starting the main process...")
    print("Starting the main process...")
    main()
    logger.info("Process completed.")
    print("Process completed.")
