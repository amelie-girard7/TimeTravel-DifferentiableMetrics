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
    
    "beta": 0.5, # inverse temperature β for contrastive sigmoid
    "lamda": 2.0,

    # **Dataset Files**
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # **Model & Training Settings**
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large-cnn"),  # Base model
    "batch_size": int(os.getenv('BATCH_SIZE', 2)),   # Training batch size
    "num_workers": int(os.getenv('NUM_WORKERS', 3)), # Dataloader workers
    "learning_rate": float(os.getenv('LEARNING_RATE', 1e-5)),  

    # **Training Parameters**
    "max_length": 512,      # Max token length for input sequences
    "max_gen_length": 250,  # Max token length for generated text
    "shuffle": True,        # Shuffle dataset during training
    "cpo_epochs": 3,  # Number of CPO training epochs

    # **MLE Checkpoint for CPO Training**
    "cpo_checkpoint_path":"/data/user/Projects/TimeTravel-DifferentiableMetrics/models/mle_2025-03-31-19/mle_checkpoint_epoch-epoch=00-step-step=004180-val_loss-validation_mle_loss=0.91.ckpt", #MLE6


    # **Evaluation Metrics (BARTScore)**
    "reward_metric": "bart",                      # Primary evaluation metric
    "use_bart": True,                             # Use BART as the reward model
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # BART model for evaluation

    # **Additional Training Options**
    "output_attentions": True,  # Enable model attention output (optional)

    # # **Gumbel-Softmax Settings**
    # "use_gumbel": True,                      # Enable/disable Gumbel-Softmax
    # "gumbel_temperature": 1,   # Fixed temperature (higher than 1.0 for better gradients) 1, 1.5
    # "gumbel_hard": False, 

    # # Annealing Settings (Disabled for initial experiments)
    # "gumbel_annealing": False,   # Disable automatic temperature annealing
    # "gumbel_anneal_rate": 0.999, # Default rate (unused when annealing=False)
    # "gumbel_min_temp": 0.1,      # Default minimum (unused when annealing=False)
    # "gumbel_log_freq": 100,       # Log frequency if enabled later

    # **New Optimization Parameters**
    # "gradient_clip_val": 0.5,
    # "accumulate_grad_batches": 1,  # No accumulation for now

    # Additional configuration for scoring metrics 
    "use_bert": True,  # Disable BERT scorer
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",  # Default BERT model for scorer 
    "scorer_device": "cuda:0",  # Device for the scorer
    "bert_scorer_batch_size": 4,  # Batch size for BERT scorer 

    "use_bleu": True,  # Disable BLEU scorer,

}

# Ensure all required directories exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    CONFIG[path_key].mkdir(parents=True, exist_ok=True)