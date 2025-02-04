# /data/agirard/Projects/TimeTravel-PolicyGradientRL/src/utils/config.py
import os
from pathlib import Path

# Set the root directory based on an environment variable or default to a parent directory
ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

# Configuration dictionary for model training, paths, and other settings
CONFIG = {
    # Paths relative to the root directory
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data" / "transformed",  # Directory containing transformed data
    "models_dir": ROOT_DIR / "models",  # Directory to save models
    "logs_dir": ROOT_DIR / "logs",  # Directory for logs
    "results_dir": ROOT_DIR / "results",  # Directory for results (e.g., validation details)
    "dataset_type": "TimeTravel",  # Options: "ART", "TimeTravel", "AblatedTimeTravel"

    # ******** Data files***********
    # Sample Timetravel sample datasets
    #"train_file": "train_supervised_small_sample.json",
    #"dev_file": "dev_data_sample.json",
    #"test_file": "test_data_sample.json",

    # Timetravel,AblatedTimeTravel datasets
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

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
    "model_name": os.getenv('MODEL_NAME', "google/flan-t5-base"),  # Hugging Face model to load
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),  # Learning rate for the optimizer

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)

    # MLE Training
    "mle_enabled": False,  # Enable MLE training
    "mle_from_checkpoint": True,   # Set to True to resume training from the specified mle_checkpoint_path; False starts training from scratch.
    "mle_checkpoint_path": '/home/agirard/Data/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-01-28-11/mle_checkpoint_epoch=epoch=2-val_loss=validation_mle_loss=2.04.ckpt',  # MLE3_1-1_Ablated-TT
    "mle_epochs": 3,  # Number of epochs to train with MLE

    # PG Training
    "pg_enabled": True,  # Set to True to enable policy gradient (PG) training; False disables it.
    "pg_from_checkpoint": True,  # If True, PG training starts from the specified pg_checkpoint_path;
                              # If False, PG training starts from the best MLE checkpoint.
    #"pg_checkpoint_path": '/home/agirard/Data/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-01-22-14/mle_checkpoint_epoch=epoch=2-val_loss=validation_mle_loss=0.95.ckpt',# MLE3_10-1_TT
    #"pg_checkpoint_path": '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-01-15-12/mle_checkpoint_epoch=epoch=2-val_loss=validation_mle_loss=0.90.ckpt',   # MLE3_5-1_TT
    "pg_checkpoint_path": '/data/agirard/Projects/TimeTravel-PolicyGradientRL/models/mle_2024-12-03-15/mle_checkpoint_epoch=epoch=2-val_loss=validation_mle_loss=0.88.ckpt',   # MLE3_1-1_TT
    # "pg_checkpoint_path": '/home/agirard/Data/Projects/TimeTravel-PolicyGradientRL/models/mle_2025-01-28-11/mle_checkpoint_epoch=epoch=2-val_loss=validation_mle_loss=2.04.ckpt',  # MLE3_1-1_Ablated-TT
    "pg_epochs": 3,  # Number of epochs to fine-tune with PG

    # Additional configuration for scoring metrics
    "reward_metric": "bart",   # "rouge","bart", "bert","bleu" (default to "rouge")

    # **Experiment Selection**
    "pg_experiment": "delta_m1",  # Options: "fixed", "dynamic", "delta_m1"
    "baseline_score": 0.5,  # Used for PG fixed baseline experiment
    "delta_m1_enabled": True,  # Enable Delta_M1 reward adjustments
  
    # Additional configuration for scoring metrics 
    "use_bert": True,  # Disable BERT scorer
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",  # Default BERT model for scorer 
    "scorer_device": "cuda:0",  # Device for the scorer
    "bert_scorer_batch_size": 4,  # Batch size for BERT scorer 

    "use_bleu": True,  # Disable BLEU scorer,

    "use_bart": True,  # Disable BART scorer
    "bart_scorer_checkpoint": "facebook/bart-large-cnn"  # Default BART model for scorer 
}

# Create any directories that don't exist
for path_key in ['data_dir', 'models_dir', 'logs_dir', 'results_dir']:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
