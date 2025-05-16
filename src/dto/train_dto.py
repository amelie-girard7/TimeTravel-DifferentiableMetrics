import sys
import os
import datetime
import logging
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.dto.models.model import BartFineTuner
from src.dto.data_loader import create_dataloaders
from src.dto.utils.metrics import MetricsEvaluator
from src.dto.utils.config import CONFIG
import pandas as pd
import re


# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure safe loading of checkpoints in PyTorch 2.6+
torch.serialization.add_safe_globals([os.path, re, datetime])
# torch.serialization.add_safe_globals([pathlib.Path, os.path, re, datetime])

def validate_config():
    """Validate critical configuration parameters"""
    required_keys = ['model_name', 'data_dir', 'batch_size', 'learning_rate',
                    'dto_epochs', 'dto_checkpoint_path']
    missing = [k for k in required_keys if k not in CONFIG]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    if not Path(CONFIG["dto_checkpoint_path"]).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CONFIG['dto_checkpoint_path']}")

def setup_model(model_dir, file_label="", checkpoint_path=None):
    """Initialize model with enhanced safety checks"""
    try:
        if not checkpoint_path:
            raise ValueError("Checkpoint path must be provided")
            
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Add safe globals for PyTorch 2.6+
        # import pathlib
        # torch.serialization.add_safe_globals([pathlib.Path, os.path, re, datetime])

        # Load checkpoint with validation
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False  # Only safe because we control the source
        )
        
        # Verify checkpoint structure
        if "state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing state_dict")
            
        # Filter state dict safely
        state_dict = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("metrics_evaluator.bart_scorer")
        }

        model = BartFineTuner(
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label=file_label or "_dto"
        )
        
        # Load with strict=False to handle architecture changes
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

        # Verify critical weights match
        if "model.lm_head.weight" in checkpoint["state_dict"]:
            mle_head = checkpoint["state_dict"]["model.lm_head.weight"].cpu()
            dto_head = model.model.lm_head.weight.detach().cpu()
            head_match = torch.allclose(dto_head, mle_head, atol=1e-6)
            logger.info(f"LM head weights match: {head_match}")
            if not head_match:
                logger.warning("LM head weights mismatch detected")

        return model
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        raise
    
def setup_trainer(max_epochs, model_dir):
    """Configure trainer with robust settings"""
    try:
        wandb_logger = WandbLogger(
            project="counterfactualStory",
            entity="counterfactualStory",
            save_dir=model_dir,
            offline=os.getenv("WANDB_MODE") == "offline",
            log_model=False # Log best model to W&B
        )

        callbacks = [
            ModelCheckpoint(
                dirpath=model_dir,
                monitor='val/dto_loss',
                mode='min',
                save_top_k=1,
                filename='dto-best-{epoch}-{val/dto_loss:.2f}', 
                auto_insert_metric_name=False,
                save_last=True
            ),
            EarlyStopping(
                monitor='val/dto_loss',
                patience=10,
                mode='min',
                verbose=True,
                check_finite=True
            )
        ]

        return Trainer(
            max_epochs=max_epochs,
            accelerator='auto',
            devices=1,
            logger=wandb_logger,
            callbacks=callbacks,
            #val_check_interval=0.25,  # Validate 4 times per epoch
            val_check_interval=1, 
            log_every_n_steps=10,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            default_root_dir=model_dir
        ), callbacks[0]
        
    except Exception as e:
        logger.error(f"Trainer setup failed: {str(e)}")
        raise

def main():
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # Initial validation
        validate_config()
        
        # Setup directories and logging
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_dir = Path(CONFIG["models_dir"]) / f"dto_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = model_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Starting training session in {model_dir}")
        logger.info(f"Configuration:\n{CONFIG}")

        # Initialize components
        tokenizer = BartTokenizer.from_pretrained(
            CONFIG["model_name"], 
            legacy=False
        )
        
        dataloaders = create_dataloaders(
            CONFIG["data_dir"], 
            tokenizer, 
            CONFIG["batch_size"], 
            CONFIG["num_workers"]
        )
        
        # Verify required dataloaders
        required_keys = {
            CONFIG["train_file"].split('.')[0],
            CONFIG["dev_file"].split('.')[0]
        }
        missing = required_keys - set(dataloaders.keys())
        if missing:
            raise KeyError(f"Missing required dataloaders: {missing}")

        # Model and trainer setup
        model = setup_model(
            model_dir,
            checkpoint_path=CONFIG["dto_checkpoint_path"]
        )
        
        trainer, checkpoint_callback = setup_trainer(
            CONFIG["dto_epochs"],
            model_dir
        )

        # Training
        train_key = CONFIG["train_file"].split('.')[0]
        dev_key = CONFIG["dev_file"].split('.')[0]
        
        logger.info("Starting training...")
        trainer.fit(
            model,
            train_dataloaders=dataloaders[train_key],
            val_dataloaders=dataloaders[dev_key]
        )

        # Final evaluation with best model
        best_path = checkpoint_callback.best_model_path
        if not best_path:
            raise RuntimeError("No valid checkpoint found")
            
        logger.info(f"Loading best model from: {best_path}")
        model = BartFineTuner.load_from_checkpoint(
            best_path,
            model_name=CONFIG["model_name"],
            model_dir=model_dir,
            file_label="_dto"
        )

        # Final validation
        logger.info("Running final validation...")
        val_results = trainer.validate(model, dataloaders[dev_key])
        logger.info(f"Validation results:\n{val_results}")

        # Final test if available
        test_key = CONFIG["test_file"].split('.')[0]
        if test_key in dataloaders:
            logger.info("Running final test...")
            test_results = trainer.test(model, dataloaders[test_key])
            logger.info(f"Test results:\n{test_results}")
            
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.exception("Training failed:")
        sys.exit(1)
    finally:
        # Cleanup
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        sys.stdout.close()
        sys.stderr.close()

if __name__ == '__main__':
    main()