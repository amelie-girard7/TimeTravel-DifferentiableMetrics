# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/main.py
import sys
import os
import datetime
import logging
from transformers import BartTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.models.model import FlanT5FineTuner
from src.data_loader import create_dataloaders
from src.utils.metrics import MetricsEvaluator
from src.utils.config import CONFIG
import pandas as pd
import re

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_model(model_dir, file_label="", checkpoint_path=None, use_differentiable_training=False):
    """
    Initializes the model for training. Loads from checkpoint if provided.
    """

    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Load model from checkpoint (without passing use_differentiable_training)
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path,
            strict=False,  # Allows partial loading if necessary
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label
        )

        # Manually set `use_differentiable_training` in case it's missing from the checkpoint
        model.use_differentiable_training = use_differentiable_training

    else:
        logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label,
            use_differentiable_training=use_differentiable_training
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
    logger.info(f"Trainer setup complete for {max_epochs} epochs.")
    return trainer


def evaluate_and_save(model_dir, loader, best_checkpoint, file_label, best_epoch, phase):
    """
    Evaluates data for the specified phase ('test' or 'validation') using BARTScore.
    Differentiates between DTO (embedding-based) and MLE (text-based) modes.
    In MLE mode, this version computes multiple comparisons (e.g. prediction_edited,
    prediction_cf, etc.) and saves them individually.
    """
    logger.info(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")
    print(f"Evaluating {phase} data for best epoch {best_epoch} using checkpoint: {best_checkpoint}")

    # Load model from checkpoint
    model = setup_model(model_dir, file_label=file_label, checkpoint_path=best_checkpoint)

    # Check whether we are in DTO mode or MLE mode.
    use_dto = model.use_differentiable_training  # True for DTO, False for MLE

    # Run evaluation (validation or test) with the Trainer.
    trainer = Trainer(accelerator='gpu', devices=1)
    if phase == "test":
        trainer.test(model, loader, verbose=False)
    elif phase == "validation":
        trainer.validate(model, loader, verbose=False)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    evaluator = MetricsEvaluator()
    metrics = {}

    if use_dto:
        # --- DTO Mode ---
        # Assumes loader.dataset.tensors = [gen_input_ids, gen_attention_mask, ref_input_ids, ref_attention_mask]
        gen_input_ids = loader.dataset.tensors[0]
        gen_attention_mask = loader.dataset.tensors[1]
        ref_input_ids = loader.dataset.tensors[2]
        ref_attention_mask = loader.dataset.tensors[3]

        # Forward pass to get soft embeddings
        generated_embeddings = model.forward(gen_input_ids, gen_attention_mask)
        reference_embeddings = model.forward(ref_input_ids, ref_attention_mask)

        # Decode soft embeddings using argmax (to get token IDs)
        generated_texts = model.tokenizer.batch_decode(
            generated_embeddings.argmax(dim=-1), skip_special_tokens=True
        )
        reference_texts = model.tokenizer.batch_decode(
            reference_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Use simple BARTScore calculation for DTO
        bart_scores = evaluator.calculate_score(generated_texts, reference_texts)
        metrics["dto_bart_score_avg"] = bart_scores.mean().item()
        logger.info(f"DTO BARTScore (average) for {phase}: {metrics['dto_bart_score_avg']}")
    else:
        # --- MLE Mode ---
        # Load CSV details that contain the generated texts and all reference columns
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
            counterfactuals = (filtered_details['Counterfactual'].tolist()
                               if 'Counterfactual' in filtered_details.columns else [])
            initials = (filtered_details['Initial'].tolist()
                        if 'Initial' in filtered_details.columns else [])
            original_endings = (filtered_details['Original Ending'].tolist()
                                if 'Original Ending' in filtered_details.columns else [])
            premises = (filtered_details['Premise'].tolist()
                        if 'Premise' in filtered_details.columns else [])
        except KeyError as e:
            logger.error(f"Missing column in filtered_{phase}_details: {e}")
            raise

        # Basic validations
        if not (generated_texts and edited_endings):
            logger.error(f"Generated texts or edited endings are empty. Skipping metric calculations for {phase}.")
            return
        if len(generated_texts) != len(edited_endings):
            logger.error("Mismatch in lengths of generated texts and edited endings. Skipping metric calculations.")
            return

        # Compute all BARTScore comparisons. This method returns a dictionary with keys like:
        # 'bart_prediction_edited_avg_score', 'bart_prediction_cf_avg_score', etc.
        bart_scores_dict = evaluator.calculate_and_log_bart_similarity(
            generated_texts, edited_endings, counterfactuals, initials, premises, original_endings, logger
        )
        # Update the metrics dictionary with all the returned comparisons.
        metrics.update(bart_scores_dict)

    # Save metrics to CSV.
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

    # Determine training phase based on config
    if CONFIG["mle_enabled"]:
        phase = "mle"
    elif CONFIG["dto_enabled"]:
        phase = "dto"
    else:
        raise ValueError("No valid training phase enabled in CONFIG. Enable either MLE or DTO.")

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
        "log_code": False  # Avoid logging source code
    })

    # Setup tokenizer and dataloaders
    tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )
    train_key, dev_key, test_key = CONFIG["train_file"].split('.')[0], CONFIG["dev_file"].split('.')[0], CONFIG["test_file"].split('.')[0]

    # --- MLE Phase ---
    if CONFIG["mle_enabled"]:
        print("Starting MLE phase training...")
        mle_checkpoint = CONFIG["mle_checkpoint_path"] if CONFIG["mle_from_checkpoint"] else None

        model = setup_model(
            model_dir,
            file_label="_mle",
            checkpoint_path=mle_checkpoint,
            use_differentiable_training=False
        )

        mle_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_mle_loss',  # Metric to monitor during training
            mode='min',  # Save checkpoint when validation loss decreases
            save_top_k=1,  # Keeping only the best checkpoint
            filename="mle_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss={validation_mle_loss:.2f}"
        )

        trainer = setup_trainer(CONFIG["mle_epochs"], mle_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("MLE training completed.")

        best_checkpoint = mle_checkpoint_callback.best_model_path
        best_loss = mle_checkpoint_callback.best_model_score
        logger.info(f"Best MLE checkpoint: {best_checkpoint}")
        print(f"Best Validation MLE Loss: {best_loss:.4f}")

        if best_checkpoint:
            best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
            logger.info(f"Best MLE checkpoint corresponds to epoch: {best_epoch}")

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

    # --- DTO Phase ---
    if CONFIG["use_differentiable_metrics"]:
        print("Starting DTO phase training...")
        model = setup_model(
            model_dir,
            file_label="_dto",
            checkpoint_path=CONFIG.get("dto_checkpoint_path", None),
            use_differentiable_training=True  # Set DTO mode explicitly
        )

        dto_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_dto_loss',  # DTO loss monitoring
            mode='min',
            save_top_k=1,
            filename="dto_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_dto_loss:.2f}"
        )

        trainer = setup_trainer(CONFIG["dto_epochs"], dto_checkpoint_callback, wandb_logger)
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("DTO training completed.")

        best_checkpoint = dto_checkpoint_callback.best_model_path
        logger.info(f"Best DTO checkpoint: {best_checkpoint}")
        print(f"Best DTO checkpoint: {best_checkpoint}")

        if best_checkpoint:
            best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
            logger.info(f"Best DTO checkpoint corresponds to epoch: {best_epoch}")
            print(f"Best DTO checkpoint corresponds to epoch: {best_epoch}")

            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[test_key],
                best_checkpoint=best_checkpoint,
                file_label="_dto",
                best_epoch=best_epoch,
                phase="test"
            )
            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[dev_key],
                best_checkpoint=best_checkpoint,
                file_label="_dto",
                best_epoch=best_epoch,
                phase="validation"
            )


if __name__ == '__main__':
    logger.info("Starting the main process...")
    print("Starting the main process...")
    main()
    logger.info("Process completed.")
    print("Process completed.")
