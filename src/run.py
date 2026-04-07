#!/usr/bin/env python3
"""Run a comprehensive Active Learning sweep and generate validation plots.

This script:
  1) Runs train_active.py for a grid of AL configurations
  2) Reads the shared results CSV
  3) Uses parameter columns already stored in CSV
  4) Produces validation plots grouped by dataset and training hyperparameters
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

STRATEGY_LABELS = {
    "random": "Random",
    "least_confident": "Least confident",
    "margin": "Margin Sampling",
    "entropy": "Entropy",
    "mc_entropy": "MC Entropy",
    "mc_bald": "BALD",
    "entropy_diverse": "Entropy + Diversity",
    "mc_entropy_diverse": "MC Entropy + Diversity",
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
    best_x: dict[str, int] = {}
    for strat, d in df_val.groupby("strategy"):
        strat = str(strat)
        g = (
            d.groupby("labeled_count")["val_mean"]
            .mean(numeric_only=True)
            .dropna()
        )
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
    config_tag: str,
    seed: int,
) -> str:
    """
    Build deterministic model path:
    models/{dataset}-{results_name}-{strategy}-{config_tag}-{seed}.pth
    """
    dataset = Path(data_dir).resolve().name
    results_name = Path(results_csv).stem
    return f"models/{dataset}-{results_name}-{strategy}-{config_tag}-{seed}.pth"


def pct_str(x: float) -> str:
    """Pretty % string for filenames/tags."""
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def plot_metric(
    df_val: pd.DataFrame,
    metric: str,
    best_x: dict[str, int],
    title: str,
    out_path: Path,
    color_map: dict[str, str],
) -> None:
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

        # mean over seeds for each labeled_count
        seed_df = (
            d.groupby(["seed", "labeled_count"])[metric]
            .mean()
            .reset_index(name=metric)
            .sort_values(by=["seed", "labeled_count"])
        )

        g = (
            seed_df.groupby("labeled_count")[metric]
            .mean()
            .reset_index(name=metric)
            .sort_values(by="labeled_count")
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

    # Data / execution
    p.add_argument("--data-dirs", nargs="+", default=None,
                   help="Paths to dataset folders (required unless --no-run)")
    p.add_argument("--results-csv", required=True, help="Path to shared CSV")
    p.add_argument("--train-script", default="src/train_active.py", help="Path to train_active.py")
    p.add_argument("--out-dir", default="results/plots", help="Where to write PNG plots")
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--no-run", action="store_true", help="Skip running training; just plot from CSV")
    p.add_argument("--overwrite-results", action="store_true", help="Delete results CSV before running")

    # Filters
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Optional dataset filter when plotting (e.g. bloodmnist)")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Strategies to run/plot. Auto-detected if --no-run")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to run/plot. Auto-detected if --no-run")

    # Absolute AL params
    p.add_argument("--init-sizes", nargs="+", type=int, default=None,
                   help="List of initial subset sizes (absolute)")
    p.add_argument("--batches", nargs="+", type=int, default=None,
                   help="List of query batch sizes (absolute)")
    p.add_argument("--budgets", nargs="+", type=int, default=None,
                   help="List of AL budgets (absolute)")

    # Percentage AL params
    p.add_argument("--init-size-pcts", nargs="+", type=float, default=None,
                   help="List of initial labeled set sizes as percent of train set")
    p.add_argument("--budget-pcts", nargs="+", type=float, default=None,
                   help="List of AL budgets as percent of train set")
    p.add_argument("--batch-pcts-of-budget", nargs="+", type=float, default=None,
                   help="List of batch sizes as percent of resolved budget")

    # Other params
    p.add_argument("--epochs-per-cycles", nargs="+", type=int, default=[1],
                   help="List of epochs per cycle")

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

    # ---- mode validation ----
    use_pct_init = args.init_size_pcts is not None
    use_pct_batch = args.batch_pcts_of_budget is not None
    use_pct_budget = args.budget_pcts is not None

    use_abs_init = args.init_sizes is not None
    use_abs_batch = args.batches is not None
    use_abs_budget = args.budgets is not None

    if use_abs_init and use_pct_init:
        raise SystemExit("Use either --init-sizes or --init-size-pcts, not both.")
    if use_abs_batch and use_pct_batch:
        raise SystemExit("Use either --batches or --batch-pcts-of-budget, not both.")
    if use_abs_budget and use_pct_budget:
        raise SystemExit("Use either --budgets or --budget-pcts, not both.")

    pct_flags = [use_pct_init, use_pct_batch, use_pct_budget]
    abs_flags = [use_abs_init, use_abs_batch, use_abs_budget]

    use_pct_mode = any(pct_flags)
    use_abs_mode = any(abs_flags)

    if use_pct_mode and not all(pct_flags):
        raise SystemExit(
            "For percentage mode, provide all of: "
            "--init-size-pcts, --batch-pcts-of-budget, --budget-pcts"
        )

    if use_abs_mode and not all(abs_flags):
        raise SystemExit(
            "For absolute mode, provide all of: "
            "--init-sizes, --batches, --budgets"
        )

    if not args.no_run and not use_pct_mode and not use_abs_mode:
        raise SystemExit(
            "Provide either absolute AL params "
            "(--init-sizes, --batches, --budgets) "
            "or percentage AL params "
            "(--init-size-pcts, --batch-pcts-of-budget, --budget-pcts)."
        )

    if use_pct_mode and use_abs_mode:
        raise SystemExit("Cannot mix percentage mode and absolute mode in one run.")

    if args.overwrite_results and results_csv.exists():
        print(f"🧹 Removing existing results CSV: {results_csv}")
        results_csv.unlink()

    # -----------------------------
    # 1) RUN TRAINING (GRID SEARCH)
    # -----------------------------
    if not args.no_run:
        strategies = list(map(str, args.strategies))
        seeds = list(args.seeds)

        if use_pct_mode:
            grid = itertools.product(
                args.data_dirs,
                args.init_size_pcts,
                args.batch_pcts_of_budget,
                args.epochs_per_cycles,
                args.budget_pcts,
                strategies,
                seeds,
            )

            for data_dir, init_pct, batch_pct, epc, budget_pct, strat, seed in grid:
                config_tag = (
                    f"init{pct_str(init_pct)}p-"
                    f"batch{pct_str(batch_pct)}pb-"
                    f"epc{epc}-"
                    f"budget{pct_str(budget_pct)}p"
                )

                model_path = build_model_path(
                    data_dir=data_dir,
                    results_csv=args.results_csv,
                    strategy=strat,
                    config_tag=config_tag,
                    seed=seed,
                )

                cmd = [
                    args.python,
                    str(train_script),
                    "-d", str(data_dir),
                    "-m", str(model_path),
                    "-r", str(results_csv),
                    "--strategy", str(strat),
                    "--seed", str(seed),
                    "--init-size-pct", str(init_pct),
                    "--batch-pct-of-budget", str(batch_pct),
                    "--budget-pct", str(budget_pct),
                    "--epochs-per-cycle", str(epc),
                ]
                run_cmd(cmd)

        else:
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
                config_tag = (
                    f"init{init_size}-"
                    f"batch{batch}-"
                    f"epc{epc}-"
                    f"budget{budget}"
                )

                model_path = build_model_path(
                    data_dir=data_dir,
                    results_csv=args.results_csv,
                    strategy=strat,
                    config_tag=config_tag,
                    seed=seed,
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
                    "--epochs-per-cycle", str(epc),
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

    # Use CSV columns directly instead of parsing model names
    numeric_cols = [
        "seed",
        "init_size",
        "batch",
        "budget",
        "epc",
        "init_size_pct",
        "budget_pct",
        "batch_pct_of_budget",
        "train_loss",
        "acc",
        "f1_macro",
        "auc",
        "ap",
        "val_mean",
        "labeled_count",
    ]
    for col in numeric_cols:
        if col in df_val.columns:
            df_val[col] = pd.to_numeric(df_val[col], errors="coerce")

    # ---- plotting filters ----
    if args.datasets is not None:
        df_val = df_val[df_val["dataset"].isin(args.datasets)]

    if args.strategies is None:
        strategies = sorted(map(str, df_val["strategy"].dropna().unique().tolist()))
        print("ℹ️ Auto-detected strategies from CSV:", strategies)
    else:
        strategies = list(map(str, args.strategies))
        df_val = df_val[df_val["strategy"].isin(strategies)]

    if args.seeds is None:
        seeds = sorted(
            int(s) for s in pd.to_numeric(df_val["seed"], errors="coerce").dropna().unique().tolist()
        )
        print("ℹ️ Auto-detected seeds from CSV:", seeds)
    else:
        seeds = list(args.seeds)
        df_val = df_val[df_val["seed"].isin(seeds)]

    # Absolute filters
    if args.init_sizes is not None and "init_size" in df_val.columns:
        df_val = df_val[df_val["init_size"].isin(args.init_sizes)]
    if args.batches is not None and "batch" in df_val.columns:
        df_val = df_val[df_val["batch"].isin(args.batches)]
    if args.budgets is not None and "budget" in df_val.columns:
        df_val = df_val[df_val["budget"].isin(args.budgets)]

    # Percentage filters
    if args.init_size_pcts is not None and "init_size_pct" in df_val.columns:
        df_val = df_val[df_val["init_size_pct"].isin(args.init_size_pcts)]
    if args.batch_pcts_of_budget is not None and "batch_pct_of_budget" in df_val.columns:
        df_val = df_val[df_val["batch_pct_of_budget"].isin(args.batch_pcts_of_budget)]
    if args.budget_pcts is not None and "budget_pct" in df_val.columns:
        df_val = df_val[df_val["budget_pct"].isin(args.budget_pcts)]

    if args.epochs_per_cycles is not None and "epc" in df_val.columns:
        df_val = df_val[df_val["epc"].isin(args.epochs_per_cycles)]

    if len(df_val) == 0:
        raise SystemExit("After filtering, no rows remain to plot.")

    color_map = make_color_map(strategies)

    # Decide plotting/grouping mode:
    # 1) If CLI explicitly selected percentage mode -> use pct grouping
    # 2) Else if CSV has pct columns and absolute columns are all missing -> use pct grouping
    # 3) Otherwise default to absolute grouping
    has_pct_cols = all(c in df_val.columns for c in ["init_size_pct", "batch_pct_of_budget", "budget_pct"])
    has_abs_cols = all(c in df_val.columns for c in ["init_size", "batch", "budget"])

    if use_pct_mode:
        plot_pct_mode = True
    elif use_abs_mode:
        plot_pct_mode = False
    else:
        plot_pct_mode = has_pct_cols and not has_abs_cols

    if plot_pct_mode:
        group_cols = ["dataset", "init_size_pct", "batch_pct_of_budget", "epc", "budget_pct"]
    else:
        group_cols = ["dataset", "init_size", "batch", "epc", "budget"]

    valid_groups = df_val.dropna(subset=group_cols)

    for name, group_df in valid_groups.groupby(group_cols):
        best_x = compute_best_x_by_strategy(group_df)

        if plot_pct_mode:
            dataset_name, init_pct, batch_pct, epc, budget_pct = name
            epc = int(epc)

            init_pct = float(init_pct)
            batch_pct = float(batch_pct)
            budget_pct = float(budget_pct)

            prefix = (
                f"{dataset_name}_"
                f"init{pct_str(init_pct)}p_"
                f"b{pct_str(batch_pct)}pb_"
                f"epc{epc}_"
                f"budget{pct_str(budget_pct)}p"
            )
            title_suffix = (
                f"(Init: {init_pct:g}%, "
                f"Batch: {batch_pct:g}% of budget, "
                f"Epochs: {epc}, "
                f"Budget: {budget_pct:g}%)"
            )
        else:
            dataset_name, init_size, batch, epc, budget = name
            init_size = int(init_size)
            batch = int(batch)
            epc = int(epc)
            budget = int(budget)

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