# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/utils/config.py
import os
from pathlib import Path

# Set the root directory based on an environment variable or default to a parent directory
ROOT_DIR = Path(os.getenv('TIMETRAVEL_DTO_ROOT', Path(__file__).resolve().parent.parent.parent))

# Configuration dictionary for model training, paths, and other settings
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",
    "models_dir": ROOT_DIR / "models",
    "logs_dir": ROOT_DIR / "logs",
    "results_dir": ROOT_DIR / "results",
    "dataset_type": "TimeTravel",  # Options: "ART", "TimeTravel", "AblatedTimeTravel"


    # ******** Data files***********
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),

    # Preprocessing and generation parameters
    "max_length": 512,
    "shuffle": True,
    "max_gen_length": 250,

    # Additional training options
    "use_custom_loss": False,
    "output_attentions": False,

    # **MLE Training Configuration**
    "mle_enabled": False,  # Enable MLE training
    "mle_from_checkpoint": False,
    "mle_checkpoint_path": None,
    "mle_epochs": 3,  # Number of epochs for MLE

    # **Policy Gradient (PG) Training Configuration**
    "pg_enabled": False,  # Enable Policy Gradient (PG) training
    "pg_from_checkpoint": False,
    "pg_checkpoint_path": None,
    "pg_epochs": 3,  # Number of epochs for PG

    # **Differentiable Training Objectives (DTO) Configuration**
    "use_differentiable_metrics": True,  # Enable Differentiable Training Objectives (DTO)
    "dto_loss_weight": 1.0,  # Weight for the DTO loss term
    "dto_epochs": 3,  # Number of epochs for DTO training
    "dto_checkpoint_path": None,  # Starting checkpoint

    # **Ensure Only One Training Mode is Active**
    "training_mode": "differentiable",  # Choose between "mle", "pg", or "differentiable"
    "reward_metric": "bart",  # Options: "rouge", "bart", "bert", "bleu"

    # **Additional configuration for scoring metrics**
    "use_bert": True,
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",
    "scorer_device": "cuda:0",
    "bert_scorer_batch_size": 4,

    "use_bleu": True,
    "use_bart": True,
    "bart_scorer_checkpoint": "facebook/bart-large-cnn"
}

# Ensure that only one training mode is enabled
if sum([CONFIG["mle_enabled"], CONFIG["pg_enabled"], CONFIG["use_differentiable_metrics"]]) > 1:
    raise ValueError("Only one training mode (MLE, PG, or DTO) can be enabled at a time.")

# Create necessary directories
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
