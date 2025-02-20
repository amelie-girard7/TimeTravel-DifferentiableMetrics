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
    Initializes the model for training.   
    If a checkpoint path is provided, the function loads the model from the checkpoint.
    Otherwise, it initializes a fresh model using the pre-trained model specified in CONFIG.
    """

    # If a checkpoint path is provided, load the model from the checkpoint.
    if checkpoint_path:
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Load the model from the checkpoint.
        # The 'strict=False' parameter allows for partial loading of the model if the checkpoint is missing some parameters.
        model = FlanT5FineTuner.load_from_checkpoint(
            checkpoint_path, # Use the pre-trained model specified in the configuration.
            #strict=False,  # Allows partial loading if necessary
            model_name=CONFIG["model_name"],
            model_dir=model_dir, # The directory where the model is (or will be) stored.
            file_label=file_label  # Append a label to help identify the model type (e.g., "_mle" or "_dto").
        )

        # Set the use_differentiable_training flag manually.
        # This is important because the checkpoint might not store this flag,
        # so we explicitly update it to ensure the model behaves as expected (DTO mode if True).
        model.use_differentiable_training = use_differentiable_training
    
    # If no checkpoint is provided, initialize a new (fresh) model.
    else:
        logger.info(f"Initializing a fresh model: {CONFIG['model_name']} with label {file_label}")
        model = FlanT5FineTuner(
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label,
            use_differentiable_training=use_differentiable_training # Set DTO mode if required.
        )
    # Return the initialized (or loaded) model.
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
    Evaluates the model on a given dataset (either test or validation) using BARTScore metrics,
    and saves the computed metrics into a CSV file. The evaluation procedure differs depending on
    whether the model is in DTO (Differentiable Training Objectives) mode or MLE (Maximum Likelihood
    Estimation) mode.
    
    In DTO mode:
      - The dataset is expected to have a tensor list organized as:
          [generated_input_ids, generated_attention_mask, reference_input_ids, reference_attention_mask]
      - The model performs a forward pass on both the generated inputs and the reference inputs to obtain
        soft embeddings.
      - These soft embeddings are then decoded using an argmax operation to obtain discrete token sequences.
      - A simple BARTScore is computed between the generated texts and the reference texts.
      - The average DTO BARTScore is stored as a metric.
    
    In MLE mode:
      - A CSV file containing detailed generation outputs (e.g., "Generated Text", "Edited Ending",
        "Premise", etc.) is loaded.
      - The CSV is filtered to include only the rows corresponding to the best_epoch.
      - The necessary columns (such as generated texts, edited endings, etc.) are extracted.
      - Various BARTScore comparisons are computed (using the evaluator's method that logs multiple metrics).
      - These metrics are merged into a dictionary.
    
    Finally, the function converts the metrics dictionary to a DataFrame and writes it to a CSV file,
    whose filename encodes the phase and epoch number.
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
        #   index 0: generated input IDs, index 1: generated attention masks,
        #   index 2: reference input IDs, index 3: reference attention masks.
        gen_input_ids = loader.dataset.tensors[0]
        gen_attention_mask = loader.dataset.tensors[1]
        ref_input_ids = loader.dataset.tensors[2]
        ref_attention_mask = loader.dataset.tensors[3]

        # Perform a forward pass on the generated inputs to obtain soft embeddings.
        generated_embeddings = model.forward(gen_input_ids, gen_attention_mask)
        # Similarly, obtain soft embeddings for the reference inputs.
        reference_embeddings = model.forward(ref_input_ids, ref_attention_mask)

        # Decode the soft embeddings using argmax to convert them into discrete token IDs.
        # 'skip_special_tokens=True' removes tokens like <pad> from the decoded output.
        generated_texts = model.tokenizer.batch_decode(
            generated_embeddings.argmax(dim=-1), skip_special_tokens=True
        )
        reference_texts = model.tokenizer.batch_decode(
            reference_embeddings.argmax(dim=-1), skip_special_tokens=True
        )

        # Compute a simple BARTScore between the generated texts and the reference texts.
        # This returns a tensor of scores; we take the mean to get a single metric.
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

    # Save the computed metrics to a CSV file.
    # The CSV filename includes the phase and epoch number for clarity.
    metrics_file = os.path.join(model_dir, f"{phase}_metrics_epoch_{best_epoch}{file_label}.csv")
    # Convert the metrics dictionary into a Pandas DataFrame.
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    metrics_df.reset_index(inplace=True)
    # Rename the columns for clarity.
    metrics_df.columns = ['Metric', 'Score']
    # Save the DataFrame to the CSV file without writing an index.
    metrics_df.to_csv(metrics_file, index=False)

    # Log and print that the evaluation metrics have been successfully saved.
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
    # Set the CUDA device to use (here, device '0' is selected).
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Determine the training phase based on configuration flags.
    # If MLE (Maximum Likelihood Estimation) is enabled in the CONFIG, use that.
    # Otherwise, if DTO (Differentiable Training Objectives) is enabled, use DTO.
    # If neither is enabled, raise an error.

    # Determine training phase based on config
    if CONFIG["mle_enabled"]:
        phase = "mle"
    elif CONFIG["dto_enabled"]:
        phase = "dto"
    else:
        raise ValueError("No valid training phase enabled in CONFIG. Enable either MLE or DTO.")

    # Create a timestamp string to uniquely name the model directory.
    model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    # Construct the model directory path using the phase and timestamp.
    model_dir = CONFIG["models_dir"] / f"{phase}_{model_timestamp}"
    # Create the directory (and parent directories) if it does not already exist.
    model_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve the dataset type from the configuration, defaulting to "TimeTravel" if not specified.
    dataset_type = CONFIG.get("dataset_type", "TimeTravel")  # Default fallback is "TimeTravel"
    print(f"Selected dataset type: {dataset_type}")  # Debug dataset type

    # Setup the WandB (Weights & Biases) logger for experiment tracking.
    wandb_logger = WandbLogger(
        project="counterfactualStory",
        entity="counterfactualStory",
        log_model=False  # Avoid logging model checkpoints
    )
    # Update the WandB experiment configuration with the global CONFIG.
    wandb_logger.experiment.config.update(CONFIG)
    wandb_logger.experiment.config.update({
        "log_system_stats": False,  # Turn off system stats
        "log_code": False  # Avoid logging source code
    })

    # Setup the tokenizer using the specified model from CONFIG.
    tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
    dataloaders = create_dataloaders(
        CONFIG["data_dir"],
        tokenizer,
        CONFIG["batch_size"],
        CONFIG["num_workers"],
    )
    # Extract keys for the training, development, and test datasets by removing file extensions.
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
        # Initialize the model for DTO training (differentiable training objectives mode)
        model = setup_model(
            model_dir,
            file_label="_dto",
            checkpoint_path=CONFIG.get("dto_checkpoint_path", None), # Optionally load a checkpoint for DTO.
            use_differentiable_training=True   # Enable DTO mode.
        )
        # Set up a ModelCheckpoint callback to monitor the DTO loss on validation.
        dto_checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            monitor='validation_dto_loss',  # Monitor the DTO loss metric.
            mode='min',
            save_top_k=1,                   # Keep only the best checkpoint.
            filename="dto_checkpoint_epoch-{epoch:02d}-step-{step:06d}-val_loss-{validation_dto_loss:.2f}"
        )
        # Setup the Trainer for DTO training with the specified number of epochs.
        trainer = setup_trainer(CONFIG["dto_epochs"], dto_checkpoint_callback, wandb_logger)
        # Start training using the training and validation dataloaders.
        trainer.fit(model, dataloaders[train_key], dataloaders[dev_key])
        print("DTO training completed.")

        # Retrieve the best DTO checkpoint path.
        best_checkpoint = dto_checkpoint_callback.best_model_path
        logger.info(f"Best DTO checkpoint: {best_checkpoint}")
        print(f"Best DTO checkpoint: {best_checkpoint}")

        # If a best checkpoint is found, extract the corresponding epoch.
        if best_checkpoint:
            best_epoch = extract_epoch_from_checkpoint(best_checkpoint)
            logger.info(f"Best DTO checkpoint corresponds to epoch: {best_epoch}")
            print(f"Best DTO checkpoint corresponds to epoch: {best_epoch}")

            # Evaluate and save metrics on the test set for DTO mode.
            evaluate_and_save(
                model_dir=model_dir,
                loader=dataloaders[test_key],
                best_checkpoint=best_checkpoint,
                file_label="_dto",
                best_epoch=best_epoch,
                phase="test"
            )
            # Evaluate and save metrics on the validation set for DTO mode.
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
