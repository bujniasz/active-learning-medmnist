#!/usr/bin/env python3
"""Run a comprehensive Active Learning sweep and generate validation plots.

This script:
  1) Runs train_active.py for a grid of (data_dir, init_size, batch, epc, strategy, seed)
  2) Reads the shared results CSV
  3) Parses the model names to extract configuration parameters
  4) Produces validation plots grouped by dataset and training hyperparameters
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys
import itertools
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
        g = (d.groupby("labeled_count")["val_mean"]
               .mean(numeric_only=True)
               .dropna())
        if len(g) == 0:
            continue
        maxv = g.max()
        best_lc = int(g[g == maxv].index.min())
        best_x[strat] = best_lc
    return best_x

def build_model_path(
    data_dir: str,
    results_csv: str,
    strategy: str,
    init_size: int,
    batch: int,
    epc: int,
    budget: int,
    seed: int,
) -> str:
    """
    Build deterministic model path:
    models/{dataset}-{results_name}-{strategy}-{init_size}-{batch}-{epc}-{budget}-{seed}.pth
    """
    dataset = Path(data_dir).resolve().name
    results_name = Path(results_csv).stem
    return f"models/{dataset}-{results_name}-{strategy}-{init_size}-{batch}-{epc}-{budget}-{seed}.pth"

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
    # Zmienione na przyjmowanie wielu wartości (grid search)
    p.add_argument("--data-dirs", nargs="+", default=None, help="Paths to dataset folders (e.g. data/bloodmnist/) (required unless --no-run)")
    p.add_argument("--init-sizes", nargs="+", type=int, default=[100], help="List of initial subset sizes")
    p.add_argument("--batches", nargs="+", type=int, default=[20], help="List of query batch sizes")
    p.add_argument("--epochs-per-cycles", nargs="+", type=int, default=[1], help="List of epochs per cycle")
    p.add_argument("--budgets", nargs="+", type=int, default=[200], help="List of AL budgets")
    p.add_argument("--strategies", nargs="+", default=None, help="Strategies to run/plot. Auto-detected if --no-run")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run/plot. Auto-detected if --no-run")
    p.add_argument("--results-csv", required=True, help="Path to shared CSV")
    p.add_argument("--train-script", default="src/train_active.py", help="Path to train_active.py")
    p.add_argument("--out-dir", default="results/plots", help="Where to write PNG plots")
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--no-run", action="store_true", help="Skip running training; just plot from CSV")
    p.add_argument("--overwrite-results", action="store_true", help="Delete results CSV before running")
    p.add_argument("--datasets", nargs="+", default=None, help="Optional dataset filter when plotting (e.g. bloodmnist)")
    args = p.parse_args()

    results_csv = Path(args.results_csv)
    train_script = Path(args.train_script)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_run:
        if args.data_dirs is None:
            raise SystemExit("--data-dirs is required unless --no-run")
        if args.strategies is None or args.seeds is None:
            raise SystemExit("--strategies and --seeds are required unless --no-run")

    if args.overwrite_results and results_csv.exists():
        print(f"🧹 Removing existing results CSV: {results_csv}")
        results_csv.unlink()

    # -----------------------------
    # 1) RUN TRAINING (GRID SEARCH)
    # -----------------------------
    if not args.no_run:
        strategies = list(map(str, args.strategies))
        seeds = list(args.seeds)
        
        # Tworzenie kombinacji wszystkich parametrów
        grid = itertools.product(
            args.data_dirs,
            args.init_sizes,
            args.batches,
            args.epochs_per_cycles,
            args.budgets,
            strategies,
            seeds,
        )

        for data_dir, init_size, batch, epc, budget, strat, seed in grid:
            model_path = build_model_path(
                data_dir, args.results_csv, strat, init_size, batch, epc, budget, seed
            )

            cmd = [
                args.python,
                str(train_script),
                "-d", str(data_dir),
                "-m", str(model_path),
                "-r", str(results_csv),
                "--strategy", str(strat),
                "--seed", str(seed),
                "--init-size", str(init_size),
                "--batch", str(batch),
                "--budget", str(budget),
                "--epochs-per-cycle", str(epc)
            ]
            run_cmd(cmd)

    # -----------------------------
    # 2) PLOT FROM CSV
    # -----------------------------
    if not results_csv.exists():
        raise SystemExit(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    df_val = df[
        (df["phase"] == "active") &
        (df["split"] == "val") &
        (df["step_type"] == "cycle")
    ].copy()

    if len(df_val) == 0:
        raise SystemExit("No rows found for plotting (phase=active, split=val, step_type=cycle).")

    # Wyciąganie parametrów z nazwy modelu przy użyciu Regex 
    # (Format: dataset-resultscsv-strategy-initsize-batch-epc-budget-seed.pth)
    regex = r'-(?P<ext_strategy>[A-Za-z0-9_]+)-(?P<ext_init_size>\d+)-(?P<ext_batch>\d+)-(?P<ext_epc>\d+)-(?P<ext_budget>\d+)-(?P<ext_seed>\d+)\.pth$'
    extracted = df_val["model"].str.extract(regex)
    
    for col in ["ext_init_size", "ext_batch", "ext_epc", "ext_budget", "ext_seed"]:
        extracted[col] = pd.to_numeric(extracted[col], errors="coerce")

    # Uzupełnienie DataFrame o wyekstrahowane dane
    df_val["init_size"] = extracted["ext_init_size"]
    df_val["batch"] = extracted["ext_batch"]
    df_val["epc"] = extracted["ext_epc"]
    df_val["budget"] = extracted["ext_budget"]

    # Filtrowanie przy plotowaniu
    if args.datasets is not None:
        df_val = df_val[df_val["dataset"].isin(args.datasets)]

    if args.strategies is None:
        strategies = sorted(map(str, df_val["strategy"].dropna().unique().tolist()))
        print("ℹ️ Auto-detected strategies from CSV:", strategies)
    else:
        strategies = list(map(str, args.strategies))
        df_val = df_val[df_val["strategy"].isin(strategies)]

    if args.seeds is None:
        seeds = sorted([int(s) for s in pd.to_numeric(df_val["seed"], errors="coerce").dropna().unique().tolist()])
        print("ℹ️ Auto-detected seeds from CSV:", seeds)
    else:
        seeds = list(args.seeds)
        df_val = df_val[df_val["seed"].isin(seeds)]
        
    # Opcjonalne filtry, jeśli chcemy plotować tylko określone wielkości (gdy --no-run)
    if args.init_sizes:
        df_val = df_val[df_val["init_size"].isin(args.init_sizes)]
    if args.batches:
        df_val = df_val[df_val["batch"].isin(args.batches)]
    if args.epochs_per_cycles:
        df_val = df_val[df_val["epc"].isin(args.epochs_per_cycles)]
    if args.budgets:
        df_val = df_val[df_val["budget"].isin(args.budgets)]

    if len(df_val) == 0:
        raise SystemExit("After filtering, no rows remain to plot.")

    for col in ["train_loss", "acc", "f1_macro", "auc", "ap", "val_mean", "labeled_count"]:
        if col in df_val.columns:
            df_val[col] = pd.to_numeric(df_val[col], errors="coerce")

    color_map = make_color_map(strategies)

    # Generowanie wykresów w grupach, aby krzywe uczyły się na tych samych parametrach bazy
    group_cols = ["dataset", "init_size", "batch", "epc", "budget"]
    valid_groups = df_val.dropna(subset=group_cols)

    for name, group_df in valid_groups.groupby(group_cols):
        dataset_name, init_size, batch, epc, budget = name
        init_size, batch, epc, budget = int(init_size), int(batch), int(epc), int(budget)
        
        best_x = compute_best_x_by_strategy(group_df)
        
        prefix = f"{dataset_name}_init{init_size}_b{batch}_epc{epc}_budget{budget}"
        title_suffix = f"(Init: {init_size}, Batch: {batch}, Epochs: {epc}, Budget: {budget})"
        
        plots = [
            ("acc",        f"{dataset_name} — Val Accuracy {title_suffix}", out_dir / f"{prefix}_val_acc.png"),
            ("f1_macro",   f"{dataset_name} — Val F1_macro {title_suffix}", out_dir / f"{prefix}_val_f1_macro.png"),
            ("auc",        f"{dataset_name} — Val AUC {title_suffix}", out_dir / f"{prefix}_val_auc.png"),
            ("ap",         f"{dataset_name} — Val AP {title_suffix}", out_dir / f"{prefix}_val_ap.png"),
            ("train_loss", f"{dataset_name} — Train loss {title_suffix}", out_dir / f"{prefix}_train_loss.png"),
        ]

        for metric, title, out_path in plots:
            if metric not in group_df.columns:
                print(f"⚠️ Missing column {metric} in CSV; skipping plot")
                continue
            ensure_parent(Path(out_path))
            plot_metric(group_df, metric, best_x, title, out_path, color_map)

    print("\n✅ Plots written to:", out_dir.resolve())


if __name__ == "__main__":
    main()