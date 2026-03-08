#!/usr/bin/env python3
"""Run a small Active Learning sweep and generate 5 validation plots.

This script:
  1) Runs train_active.py for a list of (strategy, seed, model_path)
  2) Reads the shared results CSV
  3) Produces 5 plots vs labeled_count for: acc, f1_macro, auc, ap, train_loss
     with one curve per strategy (aggregated over seeds).
  4) Adds a single 'X' marker per strategy at the labeled_count where
     the *mean* val_mean (across seeds) is maximized.

Example (matches your commands):
  python3 run_active_sweep.py \
    --data-dir data/bloodmnist/ \
    --results-csv results/test.csv \
    --train-script src/train_active.py \
    --out-dir results/plots \
    --strategies random uncertainty \
    --seeds 12 24 \
    --models models/random-1.pth models/random-2.pth models/unc-1.pth models/unc-2.pth
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys
from matplotlib.ticker import MultipleLocator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STRATEGY_LABELS = {
    "random": "Random",
    "uncertainty": "Entropy",
    "mc_bald": "BALD",
    "entropy_diverse": "Entropy + Diversity",
    "mc_bald_diverse": "BALD + Diversity",
    "egl_fc": "EGL",
}

def make_color_map(strategies: list[str]) -> dict[str, str]:
    """
    Deterministic strategy->color mapping.
    Uses Matplotlib default color cycle, assigned in sorted strategy order.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not cycle:
        # fallback (shouldn't happen)
        cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    cmap: dict[str, str] = {}
    for i, s in enumerate(sorted(map(str, strategies))):
        cmap[s] = cycle[i % len(cycle)]
    return cmap

def run_cmd(cmd: list[str]) -> None:
    print("\n▶ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_best_x_by_strategy(df_val: pd.DataFrame) -> dict[str, int]:
    """Return {strategy: labeled_count_at_best_mean_val_mean}."""
    best_x = {}
    for strat, d in df_val.groupby("strategy"):
        strat = str(strat)
        # mean val_mean across seeds at each labeled_count
        g = (d.groupby("labeled_count")["val_mean"]
               .mean(numeric_only=True)
               .dropna())
        if len(g) == 0:
            continue
        # if ties, take the smallest labeled_count for stability
        maxv = g.max()
        best_lc = int(g[g == maxv].index.min())
        best_x[strat] = best_lc
    return best_x

def build_model_path(data_dir: str, results_csv: str, strategy: str, seed: int) -> str:
    """
    Build deterministic model path:
    models/{dataset}-{results_name}-{strategy}-{seed}.pth
    """
    dataset = Path(data_dir).resolve().name
    results_name = Path(results_csv).stem
    return f"models/{dataset}-{results_name}-{strategy}-{seed}.pth"

def plot_metric(
    df_val,
    metric: str,
    best_x: dict[str, int],
    title: str,
    out_path,
    color_map: dict[str, str],
):
    fig, ax = plt.subplots(figsize=(15, 6))

    x_ticks_all = sorted({int(v) for v in df_val["labeled_count"].dropna().to_numpy()})

    # show only some x ticks if there are too many
    if len(x_ticks_all) > 12:
        step = int(np.ceil(len(x_ticks_all) / 12))
        x_ticks = x_ticks_all[::step]
        if x_ticks[-1] != x_ticks_all[-1]:
            x_ticks.append(x_ticks_all[-1])
    else:
        x_ticks = x_ticks_all

    for strat, d in df_val.groupby("strategy"):
        strat = str(strat)
        line_color = color_map.get(strat, None)

        seed_df = (
            d.groupby(["seed", "labeled_count"], as_index=False)[metric]
             .mean()
             .sort_values(["seed", "labeled_count"])
        )

        g = (
            seed_df.groupby("labeled_count", as_index=False)[metric]
                   .mean()
                   .sort_values("labeled_count")
        )

        x = g["labeled_count"].to_numpy()
        y = g[metric].to_numpy()

        (line,) = ax.plot(
            x,
            y,
            linewidth=2.1,
            label=STRATEGY_LABELS.get(strat, strat),
            color=line_color,
        )

        line_color = line.get_color()

        # X marker for best point
        if strat in best_x:
            x0 = best_x[strat]
            if x0 in set(x.tolist()):
                y0 = float(g.loc[g["labeled_count"] == x0, metric].iloc[0])
                ax.plot(
                    [x0],
                    [y0],
                    marker="x",
                    markersize=9,
                    mew=2.2,
                    linestyle="None",
                    color=line_color,
                )

    ax.set_title(title)
    ax.set_xlabel("labeled_count")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.2)

    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(v) for v in x_ticks])

    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    legend_loc = "upper right" if metric == "train_loss" else "lower right"
    ax.legend(loc=legend_loc, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="Path to dataset folder (e.g. data/bloodmnist/) (required unless --no-run)")
    p.add_argument("--results-csv", required=True, help="Path to shared CSV (will be appended by train_active.py)")
    p.add_argument("--train-script", default="src/train_active.py", help="Path to train_active.py")
    p.add_argument("--out-dir", default="results/plots", help="Where to write PNG plots")
    p.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to run/plot. If omitted with --no-run, will be auto-detected from CSV.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to run/plot. If omitted with --no-run, will be auto-detected from CSV.",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--no-run", action="store_true", help="Skip running training; just plot from CSV")
    p.add_argument("--overwrite-results", action="store_true", help="Delete results CSV before running")
    p.add_argument("--dataset", default=None, help="Optional dataset filter when plotting (e.g. bloodmnist)")
    args = p.parse_args()

    results_csv = Path(args.results_csv)
    train_script = Path(args.train_script)
    out_dir = Path(args.out_dir)

    # Ensure output directory exists (fixes FileNotFoundError on savefig)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If we are going to run training, validate required args
    if not args.no_run:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required unless --no-run")
        if args.strategies is None or args.seeds is None:
            raise SystemExit("--strategies and --seeds are required unless --no-run")

    # Overwrite CSV if requested
    if args.overwrite_results and results_csv.exists():
        print(f"🧹 Removing existing results CSV: {results_csv}")
        results_csv.unlink()

    # -----------------------------
    # RUN TRAINING (optional)
    # -----------------------------
    if not args.no_run:
        strategies: list[str] = list(map(str, args.strategies))
        seeds: list[int] = list(args.seeds)

        # Run in fixed order: for each strategy, for each seed

        for strat in strategies:
            for seed in seeds:
                model_path = build_model_path(
                    args.data_dir,
                    args.results_csv,
                    strat,
                    seed,
                )

                cmd = [
                    args.python,
                    str(train_script),
                    "-d", str(args.data_dir),
                    "-m", str(model_path),
                    "-r", str(results_csv),
                    "--strategy", str(strat),
                    "--seed", str(seed),
                ]
                run_cmd(cmd)

    # -----------------------------
    # PLOT FROM CSV
    # -----------------------------
    if not results_csv.exists():
        raise SystemExit(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    # Base filter: active learning cycle validation rows
    df_val = df[
        (df["phase"] == "active") &
        (df["split"] == "val") &
        (df["step_type"] == "cycle")
    ].copy()

    # Optional dataset filter
    if args.dataset is not None and "dataset" in df_val.columns:
        df_val = df_val[df_val["dataset"] == args.dataset].copy()

    if len(df_val) == 0:
        msg = "No rows found for plotting (phase=active, split=val, step_type=cycle"
        if args.dataset is not None:
            msg += f", dataset={args.dataset}"
        msg += ")."
        raise SystemExit(msg)

    # Auto-detect strategies and seeds when plotting-only and not provided
    if args.strategies is None:
        strategies = sorted(map(str, df_val["strategy"].dropna().unique().tolist()))
        print("ℹ️ Auto-detected strategies from CSV:", strategies)
    else:
        strategies = list(map(str, args.strategies))

    if args.seeds is None:
        # keep stable order
        seeds = sorted([int(s) for s in pd.to_numeric(df_val["seed"], errors="coerce").dropna().unique().tolist()])
        print("ℹ️ Auto-detected seeds from CSV:", seeds)
    else:
        seeds = list(args.seeds)

    # Apply user-specified filters
    df_val = df_val[df_val["strategy"].isin(strategies)].copy()
    df_val = df_val[df_val["seed"].isin(seeds)].copy()

    if len(df_val) == 0:
        raise SystemExit("After filtering by strategies/seeds, no rows remain to plot.")

    # Ensure numeric columns
    for col in ["train_loss", "acc", "f1_macro", "auc", "ap", "val_mean", "labeled_count"]:
        if col in df_val.columns:
            df_val[col] = pd.to_numeric(df_val[col], errors="coerce")

    # Compute best-x per strategy from mean val_mean curve
    best_x = compute_best_x_by_strategy(df_val)
    color_map = make_color_map(strategies)

    # Pick dataset name for filenames/titles
    dataset_name = "dataset"
    if "dataset" in df_val.columns and df_val["dataset"].notna().any():
        dataset_name = str(df_val["dataset"].dropna().iloc[0])

    plots = [
        ("acc",        f"{dataset_name} — Validation Accuracy vs labeled_count (Active Learning)", out_dir / f"{dataset_name}_val_acc.png"),
        ("f1_macro",   f"{dataset_name} — Validation F1_macro vs labeled_count (Active Learning)", out_dir / f"{dataset_name}_val_f1_macro.png"),
        ("auc",        f"{dataset_name} — Validation AUC vs labeled_count (Active Learning)", out_dir / f"{dataset_name}_val_auc.png"),
        ("ap",         f"{dataset_name} — Validation AP (AUPRC) vs labeled_count (Active Learning)", out_dir / f"{dataset_name}_val_ap.png"),
        ("train_loss", f"{dataset_name} — Training loss vs labeled_count (Active Learning)", out_dir / f"{dataset_name}_train_loss.png"),
    ]

    for metric, title, out_path in plots:
        if metric not in df_val.columns:
            print(f"⚠️ Missing column {metric} in CSV; skipping plot")
            continue
        # Ensure parent exists for each out_path (extra safety)
        ensure_parent(Path(out_path))
        plot_metric(df_val, metric, best_x, title, out_path, color_map)

    print("\n✅ Plots written to:", out_dir.resolve())
    for _, _, out_path in plots:
        out_path = Path(out_path)
        if out_path.exists():
            print(" -", out_path)


if __name__ == "__main__":
    main()