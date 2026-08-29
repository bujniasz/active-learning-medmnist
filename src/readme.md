# Source Code

This directory contains the Python source code for training, running and analyzing the experiments.

The project is organized around two training workflows:

- fully supervised training, used as a baseline;
- Active Learning training, where the model gradually queries new labels from an unlabeled pool.

Both workflows share the same data loading utilities, model wrapper and CSV metric logging format.

## Structure

```text
src/
  run_experiments.py
  training/
    train_active.py
    train_supervised.py
  analysis/
    analyze_active_screening.py
    analyze_active_strategies.py
    analyze_supervised.py
    plotting.py
  utils/
    labels_mapping.py
    load_data.py
    shared.py
```

## Main Entry Points

### `run_experiments.py`

Experiment launcher for larger sweeps.

In Active Learning mode, it runs `training/train_active.py` for combinations of datasets, strategies, seeds and AL hyperparameters, appending results to a shared CSV file.

In supervised mode, it runs `training/train_supervised.py` for selected datasets and seeds, saving all results to one CSV file.

Typical usage:

```bash
python3 -m src.run_experiments -c configs/run_experiments/active/default.yaml
python3 -m src.run_experiments -c configs/run_experiments/supervised/default.yaml
```

## Training

### `training/train_supervised.py`

Trains a ResNet18-based classifier on the full training split of a selected MedMNIST dataset.

The script loads train/validation/test splits, applies dataset-specific binary label mapping where needed, trains the model, saves the best validation checkpoint and appends validation/test metrics to a CSV file.

It can also run in `--eval-only` mode to load an existing checkpoint and evaluate it.

### `training/train_active.py`

Runs one Active Learning experiment.

The script prepares an initial labeled subset, keeps the remaining samples unlabeled, selects new samples in query cycles and retrains the model after each cycle. It supports multiple query strategies, including random sampling, entropy, margin-based uncertainty, MC Dropout variants, BALD-style scoring, diversity-aware variants and EGL for the final fully connected layer.

It saves:

- the best checkpoint according to the selected validation metric;
- validation and final test metrics to CSV;
- an `.asklog.json` file with queried sample IDs for each cycle.

## Analysis

### `analysis/analyze_active_screening.py`

Post-processing script for Active Learning screening experiments.

It reads an AL results CSV, summarizes runs, computes AULC-based comparisons, prepares main-effect and pairwise comparison tables, runs Wilcoxon signed-rank tests and creates supporting plots.

Typical usage:

```bash
python3 -m src.analysis.analyze_active_screening -c configs/analysis/analyze_active_screening/default.yaml
```

### `analysis/analyze_supervised.py`

Post-processing script for supervised learning experiments.

It reads final supervised results, computes mean and standard deviation of test
metrics for each dataset and creates aggregated confusion matrices.

Typical usage:

```bash
python3 -m src.analysis.analyze_supervised -c configs/analysis/analyze_supervised/default.yaml
```

### `analysis/analyze_active_strategies.py`

Post-processing script for the final Active Learning strategy comparison.

It reads final Active Learning results, computes validation AULC summaries, supervised validation baselines, time-to-baseline tables, final test summaries, strategy rankings, overall rankings and aggregated confusion matrices. It also generates validation metric plots, train-loss plots and per-strategy confusion matrix figures.

Typical usage:

```bash
python3 -m src.analysis.analyze_active_strategies -c configs/analysis/analyze_active_strategies/default.yaml
```

### `analysis/plotting.py`

Shared plotting utilities used by the analysis scripts.

It contains functions for screening curves, main-effect plots, pairwise delta plots, supervised and Active Learning confusion matrices, validation curves, train-loss curves and helper utilities for labels, colors and axis formatting.

## Utilities

### `utils/load_data.py`

Data loading and dataset preparation utilities.

It defines the `MedMNISTDataset` wrapper, loads `.npy` image/label splits from `data/`, normalizes images, converts them to PyTorch channel-first format and prepares data either as `DataLoader` objects for supervised training or NumPy arrays for Active Learning.

### `utils/labels_mapping.py`

Dataset-specific label mapping logic.

It converts selected MedMNIST datasets to binary classification tasks:

- `bloodmnist`: selected classes mapped to pathology vs physiology;
- `octmnist`: abnormal classes mapped against the normal class;
- `pathmnist`: selected tissue classes mapped to pathology vs physiology, with background and lymphocytes discarded.

`pneumoniamnist` is already binary and does not require this mapping.

### `utils/shared.py`

Shared model, config and metric helpers.

It defines `ResNet18EmbedDropout`, a ResNet18-based classifier with an added dropout layer before the final classifier. It also contains YAML config loading, metric formatting, prediction helpers, confusion matrix reporting and CSV row appending.

## Typical Flow

1. YAML configs from `configs/` define datasets, seeds, strategies, budgets and output paths.
2. `src.run_experiments` launches either supervised or Active Learning runs.
3. Training scripts load data via `utils/load_data.py` and use shared helpers from `utils/shared.py`.
4. Metrics are appended to CSV files in `results/`.
5. Active Learning runs additionally save checkpoints in `models/` and ask logs in `logs/`.
6. Analysis scripts read result CSV files and generate summary tables and plots.
