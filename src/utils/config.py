#/data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/utils/config.py
import os
from pathlib import Path

# Set the root directory based on an environment variable or default to a parent directory
ROOT_DIR = Path(os.getenv('TIMETRAVEL_DTO_ROOT', Path(__file__).resolve().parent.parent.parent))

# Configuration dictionary for model training, paths, and other settings
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",  # Directory containing transformed data
    "models_dir": ROOT_DIR / "models",  # Directory to save models
    "logs_dir": ROOT_DIR / "logs",  # Directory for logs
    "results_dir": ROOT_DIR / "results",  # Directory for results (e.g., validation details)
    "dataset_type": "TimeTravel",  # Options: "ART", "TimeTravel", "AblatedTimeTravel"

    # ******** Data files ***********
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large"),  # Use BART model instead of T5
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Learning rate for the optimizer

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # **Training Setup**
    "mle_enabled": False,  # Enable MLE training
    "dto_enabled": True,  # Enable Differentiable Training Objectives (DTO)


    # **MLE Training Configuration**
    "mle_from_checkpoint": False,  # Resume MLE training from checkpoint
    "mle_checkpoint_path": None,  # None: Train MLE from scratch
    "mle_epochs": 3,  # Number of epochs for MLE training

    # **DTO Training Configuration**
    "use_differentiable_metrics": True,  # Ensure DTO mode is enabled
    "dto_checkpoint_path": "/data/agirard/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-02-11-11/mle_checkpoint_epoch-epoch=02-step-step=000003-val_loss=validation_mle_loss=3.82.ckpt",  # Train DTO from scratch
    "dto_epochs": 3,  # Number of epochs for DTO training

    # **Metric Configuration (Only BART is used)**
    "reward_metric": "bart",  # Only BARTScore is used
    "use_bart": True,  # Enable BART scorer
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # Default BART model for evaluation

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)

}

# Create any directories that don't exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
