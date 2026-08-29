#!/usr/bin/env python3
from __future__ import annotations

# General
import argparse
from pathlib import Path

# Pandas
import pandas as pd

# Custom
from src.analysis.plotting import plot_supervised_confusion_matrices
from src.utils.shared import load_config, fmt


METRIC_COLS = ["acc", "f1_macro", "auc", "ap"]
CONFUSION_COLS = ["tp", "fp", "tn", "fn"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--results-csv", default=None, help="Path to supervised results CSV")
    p.add_argument("--out-dir", default=None, help="Directory for analysis outputs")
    p.add_argument("--plot-confusion", action="store_true", help="Generate confusion matrix plot")
    return p.parse_args()


def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.results_csv = cfg.get("results_csv", args.results_csv)
    args.out_dir = cfg.get("out_dir", args.out_dir)
    args.plot_confusion = cfg.get("plot_confusion", args.plot_confusion)

    return args


def load_supervised_test_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["dataset", "phase", "seed", "split", "step_type", *METRIC_COLS, *CONFUSION_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        (df["phase"] == "supervised")
        & (df["split"] == "test")
        & (df["step_type"] == "final")
    ].copy()

    if len(df) == 0:
        raise ValueError("No supervised final test rows found.")

    for col in ["seed", *METRIC_COLS, *CONFUSION_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["test_mean"] = df[METRIC_COLS].mean(axis=1, skipna=True)

    return df


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, g in df.groupby("dataset", sort=True):
        row = {
            "dataset": dataset,
            "n_runs": int(g["seed"].nunique()),
        }

        for col in [*METRIC_COLS, "test_mean"]:
            row[f"mean_{col}"] = g[col].mean()
            row[f"std_{col}"] = g[col].std()

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_confusion_matrices(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, g in df.groupby("dataset", sort=True):
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        tn = int(g["tn"].sum())
        fn = int(g["fn"].sum())

        rows.append(
            {
                "dataset": dataset,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "support": tp + fp + tn + fn,
            }
        )

    return pd.DataFrame(rows)


def format_summary_for_csv(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()

    numeric_cols = [
        c
        for c in summary.columns
        if c.startswith("mean_") or c.startswith("std_")
    ]

    for col in numeric_cols:
        summary[col] = summary[col].apply(lambda x: fmt(x, ndigits=4))

    return summary


def main():
    args = parse_args()
    args = apply_config(args)

    if args.results_csv is None:
        raise SystemExit("results_csv must be provided via CLI or config")

    if args.out_dir is None:
        raise SystemExit("out_dir must be provided via CLI or config")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_supervised_test_results(Path(args.results_csv))
    print("Loaded supervised final-test rows:", len(df))

    summary = summarize_metrics(df)
    confusion = summarize_confusion_matrices(df)

    format_summary_for_csv(summary).to_csv(out_dir / "summary.csv", index=False)
    confusion.to_csv(out_dir / "confusion_matrices.csv", index=False)

    if args.plot_confusion:
        plot_supervised_confusion_matrices(confusion, out_dir)

    print("Saved:", out_dir / "summary.csv")
    print("Saved:", out_dir / "confusion_matrices.csv")
    if args.plot_confusion:
        print("Saved confusion matrix plots in:", out_dir / "plots")


if __name__ == "__main__":
    main()
