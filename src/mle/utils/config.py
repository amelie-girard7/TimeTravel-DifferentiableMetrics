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
    #"train_file": "train_supervised_small_sample.json",
    #"dev_file": "dev_data_sample.json",
    #"test_file": "test_data_sample.json",
    
    # Timetravel,AblatedTimeTravel datasets
    "train_file": "train_supervised_small.json",
    "dev_file": "dev_data.json",
    "test_file": "test_data.json",

    # Sample Art dataset
    # "train_file": "art_train_data_sample.json",
    # "dev_file": "art_dev_data_sample.json",
    # "test_file": "art_test_data_sample.json", 
    # 
    # Art dataset
    # "train_file": "art_train_data.json",
    # "dev_file": "art_dev_data.json",
    # "test_file": "art_test_data.json", 

    # Model and training configurations
    "model_name": os.getenv('MODEL_NAME', "facebook/bart-large-cnn"),  # Use BART model instead of T5
    "batch_size": int(os.getenv('BATCH_SIZE', 4)),  # Number of samples per batch
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),  # Number of workers for data loading
    "learning_rate": float(os.getenv('LEARNING_RATE', 5e-5)),  # Learning rate for the optimizer

    # Preprocessing and generation parameters
    "max_length": 512,  # Maximum length for input data
    "shuffle": True,  # Shuffle the data during training
    "max_gen_length": 250,  # Maximum length for generated text

    # **Training Setup**
    "mle_enabled": True,  # Enable MLE training (set to True)


    "mle_from_checkpoint": False,  # Resume MLE training from checkpoint
    "mle_checkpoint_path": None, # Train MLE from scratch
    "mle_epochs": 6,  # Number of epochs for MLE training

    # **Metric Configuration (Only BART is used)**
    #"reward_metric": "bart",  # Only BARTScore is used
    "use_bart": True,  # Enable BART scorer

    # Additional training options
    "use_custom_loss": False,  # Whether to use a custom loss function (set to False for MLE)
    "output_attentions": False,  # Set to True to output attentions from the model (optional)

    # GPT Inference and evaluation settings
    # "inference_mode": "one_shot",  # Options: zero_shot, one_shot
    # "example_selection": "random",  # "fixed" or "random" - Example selection for one_shot mode
    #"run_similarities_only": True,  # If True, only run similarities, # False, Generate new results

    # **Evaluation Metrics (BARTScore)**
    "reward_metric": "bart",                      # Primary evaluation metric
    "use_bart": True,                             # Use BART as the reward model
    "bart_scorer_checkpoint": "facebook/bart-large-cnn",  # BART model for evaluation

        # Additional configuration for scoring metrics 
    "use_bert": True,  # Disable BERT scorer
    "bert_scorer_model_type": "microsoft/deberta-xlarge-mnli",  # Default BERT model for scorer 
    "scorer_device": "cuda:0",  # Device for the scorer
    "bert_scorer_batch_size": 4,  # Batch size for BERT scorer 

    "use_bleu": True,  # Disable BLEU scorer,
    
    # RAG
    "run_similarities_only": False,  # # False = generate+metrics, True = metrics only
    "inference_mode": "rag",  # Options: "standard" (GPT), "rag"
    "rag": {
        "persist_path": str(ROOT_DIR / "vector_store"),
        "k": 1,  # Number of retrieved examples
        "embedding_model": "text-embedding-3-small"
    }
}

# Create any directories that don't exist
for path_key in ["data_dir", "models_dir", "logs_dir", "results_dir"]:
    path = CONFIG[path_key]
    if not path.exists():
        print(f"Creating directory: {path}")


       
