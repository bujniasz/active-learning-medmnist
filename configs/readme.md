# Configs

This directory contains YAML configuration files for the main executable scripts in the project.

Each main script has its own `default.yaml` file. These default configs are intended not only as ready-to-run examples, but also as lightweight documentation: most options are described directly in comments inside the YAML files.

## Structure

- `train_active/default.yaml` - config for a single Active Learning training run.
- `train_supervised/default.yaml` - config for a single fully supervised training run.
- `run_experiments/active/*.yaml` - configs for Active Learning experiment sweeps.
- `run_experiments/supervised/*.yaml` - configs for supervised experiment sweeps.
- `analysis/analyze_active_screening/default.yaml` - config for post-processing and analysis of Active Learning screening results.
- `analysis/analyze_supervised/default.yaml` - config for post-processing and analysis of supervised results.
- `analysis/analyze_active_strategies/default.yaml` - config for post-processing and analysis of final Active Learning strategy-comparison results.
- `analysis/analyze_threshold_optimization/default.yaml` - config for post-training decision-threshold optimization.

## Usage

Configs can be edited directly or copied to create new experiment variants. The scripts accept a config path via the `-c` / `--config` argument, for example:

```bash
python3 -m src.training.train_active -c configs/train_active/default.yaml
python3 -m src.training.train_supervised -c configs/train_supervised/default.yaml
python3 -m src.run_experiments -c configs/run_experiments/active/default.yaml
python3 -m src.analysis.analyze_active_screening -c configs/analysis/analyze_active_screening/default.yaml
python3 -m src.analysis.analyze_supervised -c configs/analysis/analyze_supervised/default.yaml
python3 -m src.analysis.analyze_active_strategies -c configs/analysis/analyze_active_strategies/default.yaml
python3 -m src.analysis.analyze_threshold_optimization -c configs/analysis/analyze_threshold_optimization/default.yaml
```
