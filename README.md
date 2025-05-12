# Differentiable Training Objectives (DTO) for Counterfactual Story Rewriting

## Overview

This repository implements **Differentiable Training Objectives (DTO)** for counterfactual story rewriting using BART and BARTScore. We first train a base BART model with Maximum Likelihood Estimation (MLE) to convergence, then fine‑tune it using DTO by loading the best MLE checkpoint and applying a differentiable reward based on BARTScore.

## Features

* 🎯 **MLE Pre‑training**: Train BART on paired (story, edited ending) data.
* 🔄 **DTO Fine‑tuning**: Load the best MLE checkpoint and optimize with a differentiable BARTScore reward.
* ⚡️ **Lightning Integration**: PyTorch Lightning for training loops, checkpointing, and logging.
* 📊 **Metrics**: Evaluate using BARTScore similarity and save CSV reports.

## Repository Structure

```
├── src/dto/
│   ├── data_loader.py        # Data loading and collate_fn with differential weights
│   ├── models/model.py       # BartFineTuner LightningModule implementing DTO
│   ├── utils/
│   │   ├── metrics.py        # MetricsEvaluator for BARTScore
│   │   └── config.py         # CONFIG dict for paths and hyperparams
│   └── utils/utils.py        # Helper functions (e.g. weight calculation)
├── scripts/
│   └── main.py               # Entry point: trains MLE or DTO, evaluates, saves metrics
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. **Clone** the repo:

   ```bash
   git clone https://github.com/yourusername/counterfactual-dto.git
   cd counterfactual-dto
   ```
2. **Create** a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\\Scripts\\activate  # Windows
   ```
3. **Install** dependencies:

   ```bash
   pip install -r requirements.txt  # includes torch, transformers, pytorch-lightning, wandb
   ```

## Configuration

Edit `src/dto/utils/config.py` or set environment variables:

* `MODEL_NAME`: base model (default `facebook/bart-large-cnn`)
* `BATCH_SIZE`, `NUM_WORKERS`, `LEARNING_RATE`
* `dto_epochs`: number of DTO fine‑tuning epochs
* `experiment_mode`: `scratch` or `mle_checkpoint`
* `dto_checkpoint_path`: path to the best MLE `.ckpt` for DTO

## Training

1. **MLE Pre‑training** (if using scratch):

   ```bash
   python scripts/main.py --mode scratch
   ```
2. **DTO Fine‑tuning** (loads MLE checkpoint):

   ```bash
   python scripts/main.py --mode mle_checkpoint
   ```

* A ModelCheckpoint callback saves the best DTO model by monitoring `validation_dto_loss`.
* Checkpoints are stored in `models/dto_<timestamp>/`.

## Evaluation

After training, `main.py` automatically:

1. Extracts the best checkpoint epoch.
2. Runs validation and test loops.
3. Computes BARTScore metrics via `MetricsEvaluator`.
4. Saves CSV files in `models/dto_<timestamp>/`:

   * `test_metrics_epoch_<N>_dto_mle.csv`
   * `validation_metrics_epoch_<N>_dto_mle.csv`

## Usage Example

```bash
# Fine-tune DTO from MLE checkpoint, then evaluate
python scripts/main.py --mode mle_checkpoint
```

Metrics will appear in the model directory as CSVs.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on GitHub or email `amelie.girard@student.uts.edu.au`.


