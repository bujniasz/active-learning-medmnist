# Active Learning Medmnist

This repository contains code, configuration files and selected experiment artifacts for research on applying **Active Learning** to medical image classification using **MedMNIST2D** datasets.

The goal of the project is to compare standard fully supervised learning with an Active Learning setup, where the model does not use the full labeled training set from the start. Instead, it iteratively selects new samples for annotation. The experiments evaluate whether informed sample selection can achieve results close to fully supervised training while using a smaller labeling budget.

## Project Scope

The project includes:

- preparation of MedMNIST2D data in `.npy` format;
- binary label mapping for selected datasets;
- ResNet18 training in a fully supervised setting;
- Active Learning training with multiple query strategies;
- automated experiment sweeps;
- CSV metric logging;
- model checkpoint saving;
- asklog saving, i.e. storing the sample IDs selected during Active Learning cycles;
- screening analysis and plot generation.

## Data

The project uses selected MedMNIST2D datasets:

- `pneumoniamnist` - binary pneumonia classification;
- `bloodmnist` - used mainly for Active Learning hyperparameter screening;
- `octmnist` - classes mapped to a binary classification task;
- `pathmnist` - classes mapped to a binary classification task, with selected unused classes discarded.

Data is expected in the following structure:

```text
data/<dataset>/
  train_images.npy
  train_labels.npy
  val_images.npy
  val_labels.npy
  test_images.npy
  test_labels.npy
```

Data download and preparation details are described in `data/readme.md`.

## Model

The experiments use a model based on `ResNet18` from `torchvision`.

The model uses ImageNet-pretrained weights. For RGB images, the first convolutional layer remains unchanged from the pretrained model. For single-channel images, the first convolutional layer is adapted to the input channel count and initialized from scratch, while the later ResNet18 blocks still use pretrained weights.

The final classification layer is always replaced to match the number of classes in the current task. A dropout layer is added before the classifier and is used, among other things, by MC Dropout-based strategies.

## Active Learning Strategies

The project supports the following sample query strategies:

- `random` - random sample selection, used as a passive baseline;
- `least_confident` - selects samples for which the highest predicted class probability is lowest;
- `margin` - selects samples with the smallest difference between the two most probable classes;
- `entropy` - selects samples with the highest prediction entropy;
- `mc_entropy` - entropy-based strategy using MC Dropout and multiple stochastic model predictions;
- `mc_bald` - BALD-style strategy using MC Dropout to estimate epistemic uncertainty;
- `entropy_diverse` - first selects uncertain samples, then adds diversity-aware selection in embedding space;
- `mc_entropy_diverse` - combines MC Entropy with diverse batch selection;
- `mc_bald_diverse` - combines BALD with diverse batch selection;
- `egl_fc` - Expected Gradient Length computed for the final fully connected layer.

Not every strategy has to be used in every final experiment. The exact strategy sets are defined in YAML files under `configs/`.

## Metrics

Results are mainly logged using:

- `acc` - accuracy;
- `f1_macro` - macro F1;
- `auc` - ROC AUC for binary classification;
- `ap` - average precision;
- `val_mean` - average of the available validation metrics.

For Active Learning experiments, the current number of labeled samples (`labeled_count`) and labeling budget parameters are also stored.

## Installation

Running the project in a virtual environment is recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

After activating the environment, all scripts can be run as Python modules from the repository root.

## Example Commands

Single Active Learning experiment:

```bash
python3 -m src.training.train_active -c configs/train_active/default.yaml
```

Single fully supervised training run:

```bash
python3 -m src.training.train_supervised -c configs/train_supervised/default.yaml
```

Active Learning experiment sweep:

```bash
python3 -m src.run_experiments -c configs/run_experiments/active/default.yaml
```

Supervised experiment sweep:

```bash
python3 -m src.run_experiments -c configs/run_experiments/supervised/default.yaml
```

Active Learning screening analysis:

```bash
python3 -m src.analysis.analyze_active_screening -c configs/analyze_active_screening/default.yaml
```

## Typical Workflow

1. Activate the `venv` environment.
2. Install dependencies from `requirements.txt`.
3. Make sure the data is available under `data/<dataset>/`.
4. Choose or copy an appropriate YAML file from `configs/`.
5. Run a single training job or an experiment sweep through `src.run_experiments`.
6. Check experiment results in `results/`.
7. For Active Learning, also inspect checkpoints in `models/` and asklogs in `logs/`.
8. If needed, run screening analysis or plot generation.

## Repository Structure

```text
  configs/
  data/
  logs/
  models/
  results/
  src/
  requirements.txt
  README.md
  README-en.md
```

### `configs/`

YAML configuration files for the main scripts. Each main workflow has its own `default.yaml`, which serves both as a runnable example and as parameter documentation.

See: `configs/readme.md`.

### `data/`

Input data in `.npy` format, split into `train`, `val` and `test`. The repository may contain only part of the data, while the remaining datasets should be downloaded according to the instructions.

See: `data/readme.md`.

### `logs/`

Asklogs generated during Active Learning. `.asklog.json` files store which samples were selected for annotation in each cycle.

The repository keeps a representative asklog subset for `final_active_best/octmnist`; the remaining logs are treated as generated artifacts.

See: `logs/readme.md`.

### `models/`

`.pth` model checkpoints saved during training. Models are large, so they are ignored by default. The repository includes one representative example checkpoint.

See: `models/readme.md`.

### `results/`

CSV files with experiment results and directories with plots and tables generated from those CSV files. Final files matching `results/final_*.csv` are versioned, while other generated artifacts are ignored by default.

See: `results/readme.md`.

### `src/`

Project source code. It contains training scripts, the experiment launcher, result analysis, plotting utilities and helper functions for data loading, label mapping and model handling.

See: `src/readme.md`.

## Main Source Files

- `src/training/train_supervised.py` - supervised model training and evaluation.
- `src/training/train_active.py` - single Active Learning experiment.
- `src/run_experiments.py` - larger experiment sweep launcher.
- `src/analysis/analyze_active_screening.py` - Active Learning screening analysis.
- `src/analysis/plotting.py` - plot generation utilities.
- `src/utils/load_data.py` - data loading and preparation.
- `src/utils/labels_mapping.py` - label mapping for binary tasks.
- `src/utils/shared.py` - ResNet18 model, predictions, CSV logging, metrics and shared helpers.

## Final Results

The repository includes three main result files:

```text
results/final_supervised.csv
results/final_active_screening.csv
results/final_active_best.csv
```

Their roles:

- `final_supervised.csv` - supervised baseline for the final datasets;
- `final_active_screening.csv` - Active Learning parameter screening on `bloodmnist`;
- `final_active_best.csv` - final Active Learning strategy comparison using the selected parameter setup.

## Reproducibility Notes

The experiments use fixed seeds, and the scripts set random seeds for Python, NumPy and PyTorch. However, full bit-level reproducibility may still depend on library versions, hardware, CUDA drivers and environment configuration.

The most important experiment parameters are stored in YAML config files and in the resulting CSV files.
