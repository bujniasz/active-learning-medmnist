#!/usr/bin/env python3
from __future__ import annotations

# General
import argparse
from itertools import combinations
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

# Scipy
from scipy.stats import wilcoxon

# Custom
from src.analysis.plotting import (
    plot_absolute_colored_effect,
    plot_main_effects,
    plot_pairwise_deltas,
    plot_screening_curves,
    sorted_unique_numeric_values,
)
from src.utils.shared import load_config, fmt, fmt_p_value

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--results-csv", default=None, help="Path to results CSV")
    p.add_argument("--out-dir", default=None, help="Directory for analysis outputs")
    p.add_argument("--metric", default="val_mean", help="Metric used for AULC / validation comparison")
    return p.parse_args()

def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.results_csv = cfg.get("results_csv", args.results_csv)
    args.out_dir = cfg.get("out_dir", args.out_dir)
    args.metric = cfg.get("metric", args.metric)

    return args

def format_numeric_columns(df: pd.DataFrame, cols: list[str], ndigits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt(x, ndigits=ndigits))
    return df

def fmt_level(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return str(v).replace(".", "p")

def pretty_param_label(col: str) -> str:
    mapping = {
        "init_size": "init size",
        "init_size_pct": "init size (%)",
        "batch": "batch",
        "batch_pct_of_budget": "batch (% of budget)",
        "budget": "budget",
        "budget_pct": "budget (%)",
        "epc": "epochs per cycle",
        "labeled_count": "labeled count",
        "val_mean": "val_mean",
        "last_iteration_model_metric": "last-iteration model",
        "best_val_metric": "best validation metric",
        "final_test_model_mean": "final test model mean",
        "aulc_norm": "AULC (normalized)",
    }
    return mapping.get(col, col.replace("_", " "))

def add_composite_mean(
    df: pd.DataFrame,
    *,
    out_col: str,
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Adds a composite mean score computed from available metric columns.
    Useful for test rows where val_mean is empty.
    """
    if metric_cols is None:
        metric_cols = ["acc", "f1_macro", "auc", "ap"]

    df = df.copy()
    existing = [c for c in metric_cols if c in df.columns]

    if not existing:
        df[out_col] = np.nan
        return df

    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[out_col] = df[existing].mean(axis=1, skipna=True)
    return df

def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [
        "dataset",
        "strategy",
        "seed",
        "init_size",
        "batch",
        "budget",
        "epc",
        "labeled_count",
        "phase",
        "split",
        "step_type",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    supervised_rows = df[df["phase"] == "supervised"]
    if len(supervised_rows) > 0:
        print(
            f"ℹ️ Found {len(supervised_rows)} supervised rows; "
            "ignoring them in AL screening analysis."
        )

    df = df[df["phase"] == "active"].copy()

    if len(df) == 0:
        raise ValueError("No Active Learning rows found. Expected rows with phase='active'.")

    numeric_cols = [
        "seed",
        "init_size",
        "batch",
        "budget",
        "epc",
        "init_size_pct",
        "budget_pct",
        "batch_pct_of_budget",
        "final_labeled_target",
        "labeled_count",
        "val_mean",
        "acc",
        "f1_macro",
        "auc",
        "ap",
        "train_loss",
        "is_best",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def detect_param_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Decide whether to analyze pct params or absolute params.
    For screening with pct-based runs, we want:
      init_size_pct, batch_pct_of_budget, budget_pct, epc
    """
    pct_candidates = {
        "init_size": "init_size_pct",
        "batch": "batch_pct_of_budget",
        "budget": "budget_pct",
        "epc": "epc",
    }

    abs_candidates = {
        "init_size": "init_size",
        "batch": "batch",
        "budget": "budget",
        "epc": "epc",
    }

    pct_ok = all(c in df.columns for c in pct_candidates.values())

    if pct_ok:
        batch_pct_n = df["batch_pct_of_budget"].dropna().nunique() if "batch_pct_of_budget" in df.columns else 0
        budget_pct_n = df["budget_pct"].dropna().nunique() if "budget_pct" in df.columns else 0
        init_pct_n = df["init_size_pct"].dropna().nunique() if "init_size_pct" in df.columns else 0

        if batch_pct_n > 0 or budget_pct_n > 0 or init_pct_n > 0:
            return pct_candidates

    return abs_candidates

def compute_aulc(g: pd.DataFrame, x: str = "labeled_count", y: str = "val_mean"):
    g = g.sort_values(x)
    x_vals = g[x].to_numpy()
    y_vals = g[y].to_numpy()

    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]

    if len(x_vals) < 2:
        return np.nan, np.nan

    x_range = x_vals.max() - x_vals.min()
    if x_range == 0:
        return np.nan, np.nan

    area = np.trapezoid(y_vals, x_vals)
    norm = area / x_range

    return area, norm

def build_run_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "strategy",
        "seed",
        "init_size",
        "batch",
        "budget",
        "epc",
    ]

    if "init_size_pct" in df.columns:
        group_cols.append("init_size_pct")
    if "budget_pct" in df.columns:
        group_cols.append("budget_pct")
    if "batch_pct_of_budget" in df.columns:
        group_cols.append("batch_pct_of_budget")

    val_df = df[
        (df["split"] == "val")
        & (df["step_type"] == "cycle")
    ].copy()

    if len(val_df) == 0:
        raise ValueError(
            "No Active Learning validation-cycle rows found. "
            "Expected rows with phase='active', split='val', step_type='cycle'."
        )

    if metric not in val_df.columns:
        raise ValueError(f"Metric column not found in validation rows: {metric}")

    test_df = df[
        (df["split"] == "test")
        & (df["step_type"] == "final")
    ].copy()
    test_df = add_composite_mean(test_df, out_col="final_test_model_mean")

    rows = []

    for key, g in val_df.groupby(group_cols):
        g = g.sort_values("labeled_count").copy()

        valid_metric = pd.to_numeric(g[metric], errors="coerce").dropna()
        if len(valid_metric) == 0:
            continue

        aulc, aulc_norm = compute_aulc(g, "labeled_count", metric)

        last_row = g.iloc[-1]
        best_idx = g[metric].idxmax()
        best_row = g.loc[best_idx]

        row = {
            **dict(zip(group_cols, key)),
            "aulc": aulc,
            "aulc_norm": aulc_norm,
            "last_iteration_model_metric": last_row[metric],
            "last_iteration_labeled_count": last_row["labeled_count"],
            "best_val_metric": best_row[metric],
            "best_labeled_count": best_row["labeled_count"],
            "n_cycles": len(g),
        }

        test_match = test_df.copy()
        for col, value in zip(group_cols, key):
            test_match = test_match[test_match[col] == value]

        if len(test_match) == 0:
            row["final_test_model_mean"] = np.nan
            row["final_test_acc"] = np.nan
            row["final_test_f1_macro"] = np.nan
            row["final_test_auc"] = np.nan
            row["final_test_ap"] = np.nan
            row["final_test_labeled_count"] = np.nan
        else:
            if len(test_match) > 1:
                print(
                    "⚠️ Multiple final test rows found for run; using the first one:",
                    dict(zip(group_cols, key)),
                )

            t = test_match.iloc[0]
            row["final_test_model_mean"] = t.get("final_test_model_mean", np.nan)
            row["final_test_acc"] = t.get("acc", np.nan)
            row["final_test_f1_macro"] = t.get("f1_macro", np.nan)
            row["final_test_auc"] = t.get("auc", np.nan)
            row["final_test_ap"] = t.get("ap", np.nan)
            row["final_test_labeled_count"] = t.get("labeled_count", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)

def main_effects(summary: pd.DataFrame, param_col: str) -> pd.DataFrame:
    agg_kwargs = {
        "mean_aulc_norm": ("aulc_norm", "mean"),
        "std_aulc_norm": ("aulc_norm", "std"),
        "mean_last_iteration_model": ("last_iteration_model_metric", "mean"),
        "std_last_iteration_model": ("last_iteration_model_metric", "std"),
        "mean_best_val": ("best_val_metric", "mean"),
        "std_best_val": ("best_val_metric", "std"),
        "count": ("aulc_norm", "count"),
    }

    if "final_test_model_mean" in summary.columns:
        agg_kwargs["mean_final_test_model"] = ("final_test_model_mean", "mean")
        agg_kwargs["std_final_test_model"] = ("final_test_model_mean", "std")

    return (
        summary.groupby(param_col)
        .agg(**agg_kwargs)
        .reset_index()
        .sort_values("mean_aulc_norm", ascending=False)
    )

def pairwise_compare_all(
    summary: pd.DataFrame,
    param_col: str,
    use_pct_mode: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build all pairwise comparisons for a parameter with potentially >2 levels.
    Returns:
      - concatenated detailed pairwise rows
      - concatenated summary rows
    """
    values = sorted_unique_numeric_values(summary, param_col)
    if len(values) < 2:
        print(f"⚠️ Pairwise only works for >=2 values of {param_col}, got: {values}")
        return pd.DataFrame(), pd.DataFrame()

    eps = 0.002

    def win_tie_loss_counts(vals: pd.Series) -> tuple[int, int, int]:
        wins_v1 = int((vals > eps).sum())
        wins_v2 = int((vals < -eps).sum())
        ties = int((vals.abs() <= eps).sum())
        return wins_v1, wins_v2, ties

    if use_pct_mode:
        base_cols = ["dataset", "strategy", "seed", "epc"]
        for c in ["init_size_pct", "budget_pct", "batch_pct_of_budget"]:
            if c in summary.columns:
                base_cols.append(c)
    else:
        base_cols = ["dataset", "strategy", "seed", "epc"]
        for c in ["init_size", "budget", "batch"]:
            if c in summary.columns:
                base_cols.append(c)

    seen = set()
    base_cols = [c for c in base_cols if not (c in seen or seen.add(c))]
    base_cols = [c for c in base_cols if c != param_col]

    pairwise_metrics = [
        "aulc_norm",
        "last_iteration_model_metric",
        "final_test_model_mean",
    ]

    all_pw = []
    all_summary = []

    for v1, v2 in combinations(values, 2):
        sub = summary[summary[param_col].isin([v1, v2])].copy()

        metrics_available = [m for m in pairwise_metrics if m in sub.columns]
        if not metrics_available:
            continue

        pivot = sub.pivot_table(
            index=base_cols,
            columns=param_col,
            values=metrics_available,
        )

        pivot = pivot.dropna(how="all")
        if len(pivot) == 0:
            continue

        pivot.columns = [f"{metric}_{fmt_level(float(value))}" for metric, value in pivot.columns]

        v1_key = fmt_level(v1)
        v2_key = fmt_level(v2)

        if f"aulc_norm_{v1_key}" in pivot.columns and f"aulc_norm_{v2_key}" in pivot.columns:
            pivot["delta_aulc_norm"] = pivot[f"aulc_norm_{v1_key}"] - pivot[f"aulc_norm_{v2_key}"]

        if f"last_iteration_model_metric_{v1_key}" in pivot.columns and f"last_iteration_model_metric_{v2_key}" in pivot.columns:
            pivot["delta_last_iteration_model"] = (
                pivot[f"last_iteration_model_metric_{v1_key}"]
                - pivot[f"last_iteration_model_metric_{v2_key}"]
            )

        if f"final_test_model_mean_{v1_key}" in pivot.columns and f"final_test_model_mean_{v2_key}" in pivot.columns:
            pivot["delta_final_test_model"] = (
                pivot[f"final_test_model_mean_{v1_key}"]
                - pivot[f"final_test_model_mean_{v2_key}"]
            )

        delta_cols = [
            c
            for c in ["delta_aulc_norm", "delta_last_iteration_model", "delta_final_test_model"]
            if c in pivot.columns
        ]

        if not delta_cols:
            continue

        pivot = pivot.dropna(subset=delta_cols, how="all")
        if len(pivot) == 0:
            continue

        pw = pivot.reset_index()
        pw["param"] = param_col
        pw["comparison"] = f"{fmt_level(v1)}_vs_{fmt_level(v2)}"
        pw["v1"] = v1
        pw["v2"] = v2

        n = len(pw)
        summary_data = {
            "param": param_col,
            "comparison": f"{fmt_level(v1)}_vs_{fmt_level(v2)}",
            "v1": v1,
            "v2": v2,
            "n_pairs": n,
        }

        if "delta_aulc_norm" in pw.columns:
            vals = pd.to_numeric(pw["delta_aulc_norm"], errors="coerce").dropna()
            n_metric = len(vals)
            wins_v1, wins_v2, ties = win_tie_loss_counts(vals)
            summary_data.update({
                "n_pairs_aulc": n_metric,
                "median_delta_aulc_norm": vals.median() if n_metric else np.nan,
                "mean_delta_aulc_norm": vals.mean() if n_metric else np.nan,
                "win_rate_v1_aulc_pct": 100.0 * wins_v1 / n_metric if n_metric else np.nan,
                "win_rate_v2_aulc_pct": 100.0 * wins_v2 / n_metric if n_metric else np.nan,
                "ties_aulc": ties,
            })

        if "delta_last_iteration_model" in pw.columns:
            vals = pd.to_numeric(pw["delta_last_iteration_model"], errors="coerce").dropna()
            n_metric = len(vals)
            wins_v1, wins_v2, ties = win_tie_loss_counts(vals)
            summary_data.update({
                "n_pairs_last_iteration_model": n_metric,
                "median_delta_last_iteration_model": vals.median() if n_metric else np.nan,
                "mean_delta_last_iteration_model": vals.mean() if n_metric else np.nan,
                "win_rate_v1_last_iteration_model_pct": 100.0 * wins_v1 / n_metric if n_metric else np.nan,
                "win_rate_v2_last_iteration_model_pct": 100.0 * wins_v2 / n_metric if n_metric else np.nan,
                "ties_last_iteration_model": ties,
            })

        if "delta_final_test_model" in pw.columns:
            vals = pd.to_numeric(pw["delta_final_test_model"], errors="coerce").dropna()
            n_metric = len(vals)
            wins_v1, wins_v2, ties = win_tie_loss_counts(vals)
            summary_data.update({
                "n_pairs_final_test_model": n_metric,
                "median_delta_final_test_model": vals.median() if n_metric else np.nan,
                "mean_delta_final_test_model": vals.mean() if n_metric else np.nan,
                "win_rate_v1_final_test_model_pct": 100.0 * wins_v1 / n_metric if n_metric else np.nan,
                "win_rate_v2_final_test_model_pct": 100.0 * wins_v2 / n_metric if n_metric else np.nan,
                "ties_final_test_model": ties,
            })

        all_pw.append(pw)
        all_summary.append(pd.DataFrame([summary_data]))

    if not all_pw:
        print(f"⚠️ No valid pairs found for {param_col}")
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(all_pw, ignore_index=True), pd.concat(all_summary, ignore_index=True)

def wilcoxon_from_pairwise(pw: pd.DataFrame, param_name: str) -> pd.DataFrame:
    """
    Computes paired Wilcoxon signed-rank tests from pairwise deltas.
    Tests whether median(delta) differs from 0.
    """
    if len(pw) == 0:
        return pd.DataFrame()

    rows = []

    metric_map = {
        "delta_aulc_norm": "aulc_norm",
        "delta_last_iteration_model": "last-iteration-model",
        "delta_final_test_model": "final-test-model",
    }

    for comparison, g in pw.groupby("comparison"):
        v1 = g["v1"].iloc[0]
        v2 = g["v2"].iloc[0]

        for delta_col, metric_name in metric_map.items():
            if delta_col not in g.columns:
                continue

            vals = pd.to_numeric(g[delta_col], errors="coerce").dropna().to_numpy()

            n_pairs = int(len(vals))
            if n_pairs == 0:
                continue

            nonzero_vals = vals[vals != 0]
            n_nonzero = int(len(nonzero_vals))

            median_delta = float(np.median(vals))
            mean_delta = float(np.mean(vals))

            if n_nonzero == 0:
                stat = np.nan
                p_value = np.nan
                significant_005 = False
                significant_001 = False
            else:
                try:
                    result = cast(tuple[float, float], wilcoxon(
                        vals,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    ))

                    stat = float(result[0])
                    p_value = float(result[1])
                except ValueError:
                    stat = np.nan
                    p_value = np.nan

                significant_005 = bool(p_value < 0.05) if np.isfinite(p_value) else False
                significant_001 = bool(p_value < 0.01) if np.isfinite(p_value) else False

            direction = "tie"
            if median_delta > 0:
                direction = "v1_better"
            elif median_delta < 0:
                direction = "v2_better"

            rows.append(
                {
                    "param": param_name,
                    "comparison": comparison,
                    "v1": v1,
                    "v2": v2,
                    "metric": metric_name,
                    "n_pairs": n_pairs,
                    "n_nonzero": n_nonzero,
                    "median_delta": median_delta,
                    "mean_delta": mean_delta,
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "significant_0.05": significant_005,
                    "significant_0.01": significant_001,
                    "direction": direction,
                }
            )

    return pd.DataFrame(rows)

def main():
    args = parse_args()
    args = apply_config(args)

    if args.results_csv is None:
        raise SystemExit("results_csv must be provided via CLI or config")

    if args.out_dir is None:
        raise SystemExit("out_dir must be provided via CLI or config")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_csv))
    print("Loaded active rows:", len(df))

    val_cycle_rows = df[(df["split"] == "val") & (df["step_type"] == "cycle")]
    print("Validation-cycle rows:", len(val_cycle_rows))

    test_final_rows = df[(df["split"] == "test") & (df["step_type"] == "final")]
    print("Final-test rows:", len(test_final_rows))

    if len(val_cycle_rows) == 0:
        raise SystemExit("No rows found for validation curves: split=val, step_type=cycle.")

    param_cols = detect_param_columns(df)
    print("Using analysis columns:", param_cols)

    use_pct_mode = (
        param_cols["init_size"] == "init_size_pct"
        or param_cols["batch"] == "batch_pct_of_budget"
        or param_cols["budget"] == "budget_pct"
    )

    summary = build_run_summary(df, args.metric)
    if len(summary) == 0:
        raise SystemExit("No valid runs found to summarize.")

    summary_to_save = format_numeric_columns(
        summary,
        cols=[
            "aulc",
            "aulc_norm",
            "last_iteration_model_metric",
            "best_val_metric",
            "final_test_model_mean",
            "final_test_acc",
            "final_test_f1_macro",
            "final_test_auc",
            "final_test_ap",
        ],
    )
    summary_path = out_dir / "run_summary.csv"
    summary_to_save.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    analysis_plan = {
        "batch": param_cols["batch"],
        "epc": param_cols["epc"],
        "init_size": param_cols["init_size"],
        "budget": param_cols["budget"],
    }

    all_wilcoxon = []

    # --- main effects ---
    for pretty_name, param_col in analysis_plan.items():
        table = main_effects(summary, param_col)
        path = out_dir / f"main_effects_{pretty_name}.csv"

        table_to_save = format_numeric_columns(
            table,
            cols=[
                "mean_aulc_norm",
                "std_aulc_norm",
                "mean_last_iteration_model",
                "std_last_iteration_model",
                "mean_best_val",
                "std_best_val",
                "mean_final_test_model",
                "std_final_test_model",
            ],
        )
        table_to_save.to_csv(path, index=False)

        print(f"\n=== MAIN EFFECT: {pretty_name} ({param_col}) ===")
        print(table_to_save.to_string(index=False))

        plot_main_effects(table, out_dir, param_col, pretty_name)

    # --- pairwise ---
    for pretty_name, param_col in analysis_plan.items():
        pw, pw_summary = pairwise_compare_all(summary, param_col, use_pct_mode=use_pct_mode)
        if len(pw) == 0:
            continue

        pw_path = out_dir / f"pairwise_{pretty_name}.csv"
        pw_to_save = format_numeric_columns(
            pw,
            cols=[
                c
                for c in pw.columns
                if c.startswith("aulc_norm_")
                or c.startswith("last_iteration_model_metric_")
                or c.startswith("final_test_model_mean_")
                or c in [
                    "delta_aulc_norm",
                    "delta_last_iteration_model",
                    "delta_final_test_model",
                    "v1",
                    "v2",
                ]
            ],
        )
        pw_to_save.to_csv(pw_path, index=False)

        pw_summary_path = out_dir / f"pairwise_summary_{pretty_name}.csv"
        pw_summary_to_save = format_numeric_columns(
            pw_summary,
            cols=[
                "v1",
                "v2",
                "median_delta_aulc_norm",
                "mean_delta_aulc_norm",
                "win_rate_v1_aulc_pct",
                "win_rate_v2_aulc_pct",
                "median_delta_last_iteration_model",
                "mean_delta_last_iteration_model",
                "win_rate_v1_last_iteration_model_pct",
                "win_rate_v2_last_iteration_model_pct",
                "median_delta_final_test_model",
                "mean_delta_final_test_model",
                "win_rate_v1_final_test_model_pct",
                "win_rate_v2_final_test_model_pct",
            ],
        )
        pw_summary_to_save.to_csv(pw_summary_path, index=False)

        print(f"\n=== PAIRWISE: {pretty_name} ({param_col}) ===")
        delta_cols = [
            c for c in [
                "delta_aulc_norm",
                "delta_last_iteration_model",
                "delta_final_test_model",
            ]
            if c in pw.columns
        ]
        desc = pw[delta_cols].describe()
        desc_to_print = format_numeric_columns(desc.reset_index(), cols=delta_cols)
        print(desc_to_print.to_string(index=False))
        print("\nPairwise summary:")
        print(pw_summary_to_save.to_string(index=False))

        plot_pairwise_deltas(pw, out_dir, pretty_name)

        # --- wilcoxon ---
        wilcox_df = wilcoxon_from_pairwise(pw, pretty_name)
        if len(wilcox_df) > 0:
            wilcox_path = out_dir / f"wilcoxon_{pretty_name}.csv"

            wilcox_to_save = format_numeric_columns(
                wilcox_df,
                cols=["v1", "v2", "median_delta", "mean_delta", "wilcoxon_stat"],
            )
            wilcox_to_save["p_value"] = wilcox_df["p_value"].apply(fmt_p_value)

            wilcox_to_save.to_csv(wilcox_path, index=False)
            all_wilcoxon.append(wilcox_df)

            print(f"\nWilcoxon ({pretty_name}):")
            print(wilcox_to_save.to_string(index=False))

    if all_wilcoxon:
        wilcoxon_all = pd.concat(all_wilcoxon, ignore_index=True)

        wilcoxon_all_to_save = format_numeric_columns(
            wilcoxon_all,
            cols=["v1", "v2", "median_delta", "mean_delta", "wilcoxon_stat"],
        )
        wilcoxon_all_to_save["p_value"] = wilcoxon_all["p_value"].apply(fmt_p_value)

        wilcoxon_all_path = out_dir / "wilcoxon_all.csv"
        wilcoxon_all_to_save.to_csv(wilcoxon_all_path, index=False)
        print("Saved:", wilcoxon_all_path)

    # --- screening curves ---
    for pretty_name, param_col in analysis_plan.items():
        plot_screening_curves(
            df,
            out_dir,
            param_col,
            metric=args.metric,
            use_pct_mode=use_pct_mode,
        )

    # --- auxiliary plots: absolute values colored by pct level ---
    if use_pct_mode:
        plot_absolute_colored_effect(
            summary,
            out_dir,
            abs_col="batch",
            pct_col="batch_pct_of_budget",
            pretty_name="batch",
        )
        plot_absolute_colored_effect(
            summary,
            out_dir,
            abs_col="budget",
            pct_col="budget_pct",
            pretty_name="budget",
        )

    print("\n✅ Analysis done:", out_dir.resolve())

if __name__ == "__main__":
    main()
