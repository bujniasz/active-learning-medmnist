# Logs

This directory is used for automatically generated Active Learning ask logs.

Ask logs are saved by `src/training/train_active.py` during Active Learning runs. Each `.asklog.json` file records which sample IDs were queried in each cycle, together with the planned and actual query batch size.

These files are useful for inspecting or reproducing the exact annotation sequence used in a run. They are generated automatically from the model checkpoint path and normally do not need to be edited by hand.

## Typical Structure

Generated logs usually follow the same experiment layout as model checkpoints:

```text
logs/
  final_active_best/
    <dataset>/
      <run-name>.asklog.json
  final_active_screening/
    <dataset>/
      <run-name>.asklog.json
```

The log files are experiment artifacts. By default, they are ignored by Git to avoid storing every generated run output in the repository.

## Versioned Example

The repository includes one representative subset of generated ask logs:

```text
logs/final_active_best/octmnist/
```

This subset is kept in Git to show the structure and contents of real Active Learning ask logs from the final experiment. The remaining generated log directories are intentionally left untracked to avoid versioning all experiment artifacts.
