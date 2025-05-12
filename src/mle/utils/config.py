#/data/agirard/Projects/TimeTravel-DifferentiableMetrics/src/mle/utils/config.py
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
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",
    
    # Timetravel,AblatedTimeTravel datasets
    #"train_file": "train_supervised_small.json",
    #"dev_file": "dev_data.json",
    #"test_file": "test_data.json",

    # Sample Art dataset
    #"train_file": "art_train_data_sample.json",
    #"dev_file": "art_dev_data_sample.json",
    #"test_file": "art_test_data_sample.json", 
    # 
    # Art dataset
    #"train_file": "art_train_data.json",
    #"dev_file": "art_dev_data.json",
    #"test_file": "art_test_data.json", 

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large-cnn"),  # Use BART model instead of T5
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Learning rate for the optimizer

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # **Training Setup**
    "mle_enabled": True,  # Enable MLE training (set to True)



    # MLE Training Configuration - These may not be required
    "mle_from_checkpoint": True,  # Resume MLE training from checkpoint
    "mle_checkpoint_path": None, # Train MLE from scratch
    #"mle_checkpoint_path": "/data/agirard/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-03-31-09/mle_checkpoint_epoch-epoch=00-step-step=004180-val_loss-validation_mle_loss=0.92.ckpt", #MLE1
    #"mle_checkpoint_path": "/data/agirard/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-03-31-14/mle_checkpoint_epoch-epoch=00-step-step=000418-val_loss-validation_mle_loss=0.95.ckpt", #MLE2   
    "mle_epochs": 6,  # Number of epochs for MLE training

    # **Metric Configuration (Only BART is used)**
    #"reward_metric": "bart",  # Only BARTScore is used
    "use_bart": True,  # Enable BART scorer

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)
}

# Create any directories that don't exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
       
