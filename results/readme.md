# Results

This directory is used for experiment outputs generated from training and analysis scripts.

Training scripts append metric rows to CSV files in this directory. These CSV files store validation and test results, including dataset name, strategy, seed, active learning parameters, labeled sample counts and evaluation metrics.

Plotting and analysis scripts can also create subdirectories here. These folders usually contain figures and summary tables generated from the result CSV files.

## Typical Outputs

Common result files include:

```text
results/final_supervised.csv
results/final_active_best.csv
results/final_active_screening.csv
```

Common plot directories follow the CSV name:

```text
results/final_active_best/
results/final_supervised/
```

## Versioned Results

The final CSV files matching `results/final_*.csv` are kept in Git because they contain the main experiment results used by the project.

Generated plot directories and older experimental outputs are ignored by default to avoid versioning all derived artifacts.
