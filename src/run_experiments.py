#!/usr/bin/env python3
"""Run experiment sweeps and save raw result CSV files.

This script supports two modes:

1) Active Learning sweep:
   - runs train_active.py for a grid of AL configurations
   - appends validation/test results to a shared CSV

2) Supervised sweep:
   - runs train_supervised.py for selected datasets and seeds
   - writes results to the shared CSV
"""

from __future__ import annotations

# General
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

# Custom
from src.utils.shared import load_config

def script_path_to_module(script_path: str | Path) -> str:
    """
    Convert script path like:
        src/training/train_active.py
    to module path:
        src.training.train_active
    """
    path = Path(script_path)

    if path.suffix == ".py":
        path = path.with_suffix("")

    return ".".join(path.parts)

def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.mode = cfg.get("mode", args.mode)

    args.data_dirs = cfg.get("data_dirs", args.data_dirs)
    args.results_csv = cfg.get("results_csv", args.results_csv)
    args.train_script = cfg.get("train_script", args.train_script)
    args.supervised_train_script = cfg.get("supervised_train_script", args.supervised_train_script)
    args.python = cfg.get("python", args.python)
    args.overwrite_results = cfg.get("overwrite_results", args.overwrite_results)

    args.strategies = cfg.get("strategies", args.strategies)
    args.seeds = cfg.get("seeds", args.seeds)

    al_mode = cfg.get("al_mode", None)

    if al_mode == "percent":
        args.init_size_pcts = cfg.get("init_size_pcts", args.init_size_pcts)
        args.batch_pcts_of_budget = cfg.get("batch_pcts_of_budget", args.batch_pcts_of_budget)
        args.budget_pcts = cfg.get("budget_pcts", args.budget_pcts)

        args.init_sizes = None
        args.batches = None
        args.budgets = None

    elif al_mode == "absolute":
        args.init_sizes = cfg.get("init_sizes", args.init_sizes)
        args.batches = cfg.get("batches", args.batches)
        args.budgets = cfg.get("budgets", args.budgets)

        args.init_size_pcts = None
        args.batch_pcts_of_budget = None
        args.budget_pcts = None

    elif al_mode is not None:
        raise ValueError(f"Unknown al_mode: {al_mode}")

    args.epochs_per_cycles = cfg.get("epochs_per_cycles", args.epochs_per_cycles)

    return args

def run_cmd(cmd: list[str]) -> None:
    print("\n▶ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def model_root_from_results_csv(results_csv: str | Path) -> Path:
    """
    Build model output root from results CSV path.

    Examples:
        results/my-experiment/new.csv -> models/my-experiment/new
        results/screening_official.csv -> models/screening_official
        some-dir/exp.csv -> models/some-dir/exp
    """
    p = Path(results_csv).with_suffix("")

    # Common project convention:
    # results/foo/bar.csv -> models/foo/bar
    if not p.is_absolute() and len(p.parts) > 0 and p.parts[0] == "results":
        rel = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(p.name)
    else:
        # Fallback for non-standard paths.
        # For absolute paths, avoid recreating the whole absolute tree under models/.
        rel = Path(p.name) if p.is_absolute() else p

    return Path("models") / rel

def build_model_path(
    data_dir: str,
    results_csv: str,
    strategy: str,
    config_tag: str,
    seed: int,
) -> str:
    """
    Build deterministic active model path.

    Example:
        results_csv = results/my-experiment/new.csv
        data_dir    = data/bloodmnist

    Output:
        models/my-experiment/new/bloodmnist/
            bloodmnist-new-{strategy}-{config_tag}-{seed}.pth
    """
    dataset = Path(data_dir).resolve().name
    results_name = Path(results_csv).stem

    model_dir = model_root_from_results_csv(results_csv) / dataset
    model_name = f"{dataset}-{results_name}-{strategy}-{config_tag}-{seed}.pth"

    return str(model_dir / model_name)

def build_supervised_model_path(
    data_dir: str,
    results_csv: str,
    seed: int,
) -> str:
    """
    Build deterministic supervised model path.

    Example:
        results_csv = results/my-experiment/new.csv
        data_dir    = data/bloodmnist

    Output:
        models/my-experiment/new/bloodmnist/
            bloodmnist-new-supervised-{seed}.pth
    """
    dataset = Path(data_dir).resolve().name
    results_name = Path(results_csv).stem

    model_dir = model_root_from_results_csv(results_csv) / dataset
    model_name = f"{dataset}-{results_name}-supervised-{seed}.pth"

    return str(model_dir / model_name)

def pct_str(x: float) -> str:
    """Pretty % string for filenames/tags."""
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")

def main() -> None:
    p = argparse.ArgumentParser()

    # Data / execution
    p.add_argument("-c", "--config", type=str, default=None,
               help="Path to YAML config file")
    p.add_argument("--mode", choices=["active", "supervised"], default="active",
                   help="Run mode: active learning sweep or supervised sweep")
    p.add_argument("--data-dirs", nargs="+", default=None,
                   help="Paths to dataset folders")
    p.add_argument("--results-csv", default=None, help="Path to shared CSV")
    p.add_argument("--train-script", default="src/training/train_active.py", help="Path to train_active.py")
    p.add_argument("--supervised-train-script", default="src/training/train_supervised.py", help="Path to train_supervised.py")
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--overwrite-results", action="store_true", help="Delete results CSV before running")

    # Sweep parameters
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Strategies to run")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to run")

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

    # Active params
    p.add_argument("--epochs-per-cycles", nargs="+", type=int, default=[1],
                   help="List of epochs per active learning cycle")

    args = p.parse_args()

    args = apply_config(args)

    if args.results_csv is None:
        raise SystemExit("results_csv must be provided either via CLI or config")

    results_csv = Path(args.results_csv)
    train_script = Path(args.train_script)
    supervised_train_script = Path(args.supervised_train_script)

    if args.data_dirs is None:
        raise SystemExit("--data-dirs is required")
    if args.seeds is None:
        raise SystemExit("--seeds is required")
    if args.mode == "active" and args.strategies is None:
        raise SystemExit("--strategies is required for active mode")

    # ---- active mode validation ----
    if args.mode == "active":
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

        if not use_pct_mode and not use_abs_mode:
            raise SystemExit(
                "Provide either absolute AL params "
                "(--init-sizes, --batches, --budgets) "
                "or percentage AL params "
                "(--init-size-pcts, --batch-pcts-of-budget, --budget-pcts)."
            )

        if use_pct_mode and use_abs_mode:
            raise SystemExit("Cannot mix percentage mode and absolute mode in one run.")

    else:
        use_pct_mode = False
        use_abs_mode = False

    if args.overwrite_results and results_csv.exists():
        print(f"🧹 Removing existing results CSV: {results_csv}")
        results_csv.unlink()

    # -----------------------------
    # RUN TRAINING (GRID SEARCH)
    # -----------------------------
    seeds = list(args.seeds)

    if args.mode == "supervised":
        grid = itertools.product(
            args.data_dirs,
            seeds,
        )

        for data_dir, seed in grid:
            model_path = build_supervised_model_path(
                data_dir=data_dir,
                results_csv=args.results_csv,
                seed=seed,
            )

            cmd = [
                args.python,
                "-m",
                script_path_to_module(supervised_train_script),
                "-d", str(data_dir),
                "-mp", str(model_path),
                "-r", str(results_csv),
                "--seed", str(seed),
            ]

            run_cmd(cmd)

    else:
        strategies = list(map(str, args.strategies))

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
                    "-m",
                    script_path_to_module(train_script),
                    "-d", str(data_dir),
                    "-mp", str(model_path),
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
                    "-m",
                    script_path_to_module(train_script),
                    "-d", str(data_dir),
                    "-mp", str(model_path),
                    "-r", str(results_csv),
                    "--strategy", str(strat),
                    "--seed", str(seed),
                    "--init-size", str(init_size),
                    "--batch", str(batch),
                    "--budget", str(budget),
                    "--epochs-per-cycle", str(epc),
                ]
                run_cmd(cmd)

    print(f"\n✅ {args.mode} sweep finished.")
    print("Results CSV:", results_csv.resolve())

if __name__ == "__main__":
    main()
