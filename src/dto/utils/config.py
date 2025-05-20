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
    "train_file": "train_supervised_small_sample.json",
    "dev_file": "dev_data_sample.json",
    "test_file": "test_data_sample.json",

    # **Model & Training Settings**
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large-cnn"),  # Base model
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),   # Training batch size
    "num_workers": int(os.getenv('NUM_WORKERS', 3)), # Dataloader workers
    "learning_rate": float(os.getenv('LEARNING_RATE', 3e-5)),  # Optimizer learning rate

    # **Training Parameters**
    "max_length": 512,      # Max token length for input sequences
    "max_gen_length": 250,  # Max token length for generated text
    "shuffle": True,        # Shuffle dataset during training
    "dto_epochs": 2,  # Number of DTO training epochs

    # **MLE Checkpoint for DTO Training**
    "dto_checkpoint_path":"/data/agirard/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-03-31-19/mle_checkpoint_epoch-epoch=00-step-step=004180-val_loss-validation_mle_loss=0.91.ckpt", #MLE6


    # **Evaluation Metrics (BARTScore)**
    "reward_metric": "bart",                      # Primary evaluation metric
    "use_bart": True,                             # Use BART as the reward model
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # BART model for evaluation

    # **Additional Training Options**
    "output_attentions": True,  # Enable model attention output (optional)

    # **Gumbel-Softmax Settings**
    "use_gumbel": True,                      # Enable/disable Gumbel-Softmax
    "gumbel_temperature": 0.5,   # Fixed temperature (higher than 1.0 for better gradients) 1, 1.5
    "gumbel_hard": False, 

    # Annealing Settings (Disabled for initial experiments)
    "gumbel_annealing": False,   # Disable automatic temperature annealing
    "gumbel_anneal_rate": 0.999, # Default rate (unused when annealing=False)
    "gumbel_min_temp": 0.1,      # Default minimum (unused when annealing=False)
    "gumbel_log_freq": 100,       # Log frequency if enabled later

    # **New Optimization Parameters**
    "gradient_clip_val": 1.0,
    "accumulate_grad_batches": 1,  # No accumulation for now

}

# Ensure all required directories exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    CONFIG[path_key].mkdir(parents=True, exist_ok=True)