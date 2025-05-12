# /data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/dto/utils/config.py
import os
from pathlib import Path

# Root directory: Set via environment variable or default to project root
ROOT_DIR = Path(os.getenv('TIMETRAVEL_DTO_ROOT', Path(__file__).resolve().parent.parent.parent))

CONFIG = {
    # **Directory Paths**
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",   # Processed dataset location
    "models_dir": ROOT_DIR / "models",               # Checkpoints and trained models
    "logs_dir": ROOT_DIR / "logs",                   # Training logs
    "results_dir": ROOT_DIR / "results",             # Evaluation results (e.g., validation details)

    # **Dataset Type**
    "dataset_type": "TimeTravel",   # Options: "ART", "TimeTravel", "AblatedTimeTravel"

    # **Dataset Files**
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # **Model & Training Settings**
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large-cnn"),  # Base model
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),   # Training batch size
    "num_workers": int(os.getenv('NUM_WORKERS', 3)), # Dataloader workers
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Optimizer learning rate

    # **Training Parameters**
    "max_length": 512,      # Max token length for input sequences
    "max_gen_length": 250,  # Max token length for generated text
    "shuffle": True,        # Shuffle dataset during training

    # **Experiment Mode (DTO Training)**
    "experiment_mode": "mle_checkpoint",  # Options: "scratch" | "mle_checkpoint"
    "dto_epochs": 1,  # Number of DTO training epochs

    # **MLE Checkpoint for DTO Training**
    "dto_checkpoint_path":"/data/agirard/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-03-31-19/mle_checkpoint_epoch-epoch=00-step-step=004180-val_loss-validation_mle_loss=0.91.ckpt", #MLE6

    # **DTO Training Settings**
    "dto_enabled": True, # Enable DTO mode

    # **Evaluation Metrics (BARTScore)**
    "reward_metric": "bart",                      # Primary evaluation metric
    "use_bart": True,                             # Use BART as the reward model
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # BART model for evaluation

    # **Additional Training Options**
    "output_attentions": True,  # Enable model attention output (optional)
}

# Ensure all required directories exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    CONFIG[path_key].mkdir(parents=True, exist_ok=True)
