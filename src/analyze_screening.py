#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from metrics import fmt


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True, help="Path to results CSV")
    p.add_argument("--out-dir", required=True, help="Directory for analysis outputs")
    p.add_argument("--metric", default="val_mean", help="Metric used for AULC/final comparison")
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def format_numeric_columns(df: pd.DataFrame, cols: list[str], ndigits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt(x, ndigits=ndigits))
    return df


def sanitize_filename_part(x) -> str:
    s = str(x)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def sorted_unique_numeric_values(df: pd.DataFrame, col: str) -> list[float]:
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    vals = s.drop_duplicates().to_list()
    vals.sort()
    return vals


def fmt_level(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return str(v).replace(".", "p")


# -----------------------------
# Load + mode detection
# -----------------------------
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
        "val_mean",
        "phase",
        "split",
        "step_type",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        (df["phase"] == "active")
        & (df["split"] == "val")
        & (df["step_type"] == "cycle")
    ].copy()

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


# -----------------------------
# AULC
# -----------------------------
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


# -----------------------------
# Build run summary
# -----------------------------
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

    rows = []

    for key, g in df.groupby(group_cols):
        g = g.sort_values("labeled_count").copy()

        valid_metric = g[metric].dropna()
        if len(valid_metric) == 0:
            continue

        aulc, aulc_norm = compute_aulc(g, "labeled_count", metric)

        final_row = g.iloc[-1]
        best_idx = g[metric].idxmax()
        best_row = g.loc[best_idx]

        rows.append(
            {
                **dict(zip(group_cols, key)),
                "aulc": aulc,
                "aulc_norm": aulc_norm,
                "final_metric": final_row[metric],
                "best_metric": best_row[metric],
                "final_labeled_count": final_row["labeled_count"],
                "best_labeled_count": best_row["labeled_count"],
                "n_cycles": len(g),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main effects
# -----------------------------
def main_effects(summary: pd.DataFrame, param_col: str) -> pd.DataFrame:
    return (
        summary.groupby(param_col)
        .agg(
            mean_aulc_norm=("aulc_norm", "mean"),
            std_aulc_norm=("aulc_norm", "std"),
            mean_final=("final_metric", "mean"),
            std_final=("final_metric", "std"),
            count=("aulc_norm", "count"),
        )
        .reset_index()
        .sort_values("mean_aulc_norm", ascending=False)
    )

def plot_absolute_colored_effect(
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    abs_col: str,
    pct_col: str,
    pretty_name: str,
) -> None:
    """
    Auxiliary plot for pct-based screening:
    x-axis  -> absolute value (e.g. batch or budget)
    color   -> corresponding percentage level
    y-axis  -> mean_aulc_norm

    Example:
      abs_col="batch", pct_col="batch_pct_of_budget", pretty_name="batch"
    """
    if abs_col not in summary.columns or pct_col not in summary.columns:
        return

    tmp = (
        summary.groupby([abs_col, pct_col], as_index=False)
        .agg(
            mean_aulc_norm=("aulc_norm", "mean"),
            count=("aulc_norm", "count"),
        )
        .sort_values([pct_col, abs_col])
    )

    if len(tmp) == 0:
        return

    aux_dir = out_dir / "plots" / "absolute_colored_effects"
    aux_dir.mkdir(parents=True, exist_ok=True)

    pct_levels = sorted_unique_numeric_values(tmp, pct_col)
    cmap = plt.get_cmap("tab10")
    color_map = {lvl: cmap(i % 10) for i, lvl in enumerate(pct_levels)}

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    x_vals = tmp[abs_col].tolist()
    y_vals = tmp["mean_aulc_norm"].tolist()
    colors = [color_map[float(v)] for v in tmp[pct_col].tolist()]

    bars = ax.bar([str(int(x)) if float(x).is_integer() else str(x) for x in x_vals], y_vals, color=colors)

    # legenda po procentach
    handles = []
    labels = []
    for lvl in pct_levels:
        handles.append(Rectangle((0, 0), 1, 1, color=color_map[lvl]))
        labels.append(f"{lvl:g}%")

    ax.legend(handles, labels, title=f"{pretty_name} %", loc="best", framealpha=0.9)

    ax.set_title(f"{pretty_name.capitalize()} effect by absolute value (colored by % level)")
    ax.set_xlabel(abs_col)
    ax.set_ylabel("mean_aulc_norm")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(aux_dir / f"{pretty_name}_absolute_colored.png", dpi=180)
    plt.close(fig)

# -----------------------------
# Pairwise comparisons
# -----------------------------
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

    # Use coherent identifier space
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

    # remove duplicates while preserving order
    seen = set()
    base_cols = [c for c in base_cols if not (c in seen or seen.add(c))]

    # varied param cannot be part of pivot index
    base_cols = [c for c in base_cols if c != param_col]

    all_pw = []
    all_summary = []

    for v1, v2 in combinations(values, 2):
        sub = summary[summary[param_col].isin([v1, v2])].copy()

        pivot = sub.pivot_table(
            index=base_cols,
            columns=param_col,
            values=["aulc_norm", "final_metric"],
        )

        pivot = pivot.dropna()
        if len(pivot) == 0:
            continue

        pivot.columns = [f"{metric}_{fmt_level(float(value))}" for metric, value in pivot.columns]

        v1_key = fmt_level(v1)
        v2_key = fmt_level(v2)

        pivot["delta_aulc_norm"] = pivot[f"aulc_norm_{v1_key}"] - pivot[f"aulc_norm_{v2_key}"]
        pivot["delta_final"] = pivot[f"final_metric_{v1_key}"] - pivot[f"final_metric_{v2_key}"]

        pw = pivot.reset_index()
        pw["param"] = param_col
        pw["comparison"] = f"{fmt_level(v1)}_vs_{fmt_level(v2)}"
        pw["v1"] = v1
        pw["v2"] = v2

        n = len(pw)
        wins_v1_aulc = int((pw["delta_aulc_norm"] > 0).sum())
        wins_v2_aulc = int((pw["delta_aulc_norm"] < 0).sum())
        ties_aulc = int((pw["delta_aulc_norm"] == 0).sum())

        wins_v1_final = int((pw["delta_final"] > 0).sum())
        wins_v2_final = int((pw["delta_final"] < 0).sum())
        ties_final = int((pw["delta_final"] == 0).sum())

        summary_row = pd.DataFrame(
            [
                {
                    "param": param_col,
                    "comparison": f"{fmt_level(v1)}_vs_{fmt_level(v2)}",
                    "v1": v1,
                    "v2": v2,
                    "n_pairs": n,
                    "median_delta_aulc_norm": pw["delta_aulc_norm"].median(),
                    "mean_delta_aulc_norm": pw["delta_aulc_norm"].mean(),
                    "win_rate_v1_aulc_pct": 100.0 * wins_v1_aulc / n if n else np.nan,
                    "win_rate_v2_aulc_pct": 100.0 * wins_v2_aulc / n if n else np.nan,
                    "ties_aulc": ties_aulc,
                    "median_delta_final": pw["delta_final"].median(),
                    "mean_delta_final": pw["delta_final"].mean(),
                    "win_rate_v1_final_pct": 100.0 * wins_v1_final / n if n else np.nan,
                    "win_rate_v2_final_pct": 100.0 * wins_v2_final / n if n else np.nan,
                    "ties_final": ties_final,
                }
            ]
        )

        all_pw.append(pw)
        all_summary.append(summary_row)

    if not all_pw:
        print(f"⚠️ No valid pairs found for {param_col}")
        return pd.DataFrame(), pd.DataFrame()

    return pd.concat(all_pw, ignore_index=True), pd.concat(all_summary, ignore_index=True)


# -----------------------------
# Plots
# -----------------------------
def plot_screening_curves(df: pd.DataFrame, out_dir: Path, param_col: str, metric: str = "val_mean") -> None:
    all_group_cols = ["dataset", "strategy", "epc"]
    for c in ["init_size_pct", "budget_pct", "batch_pct_of_budget", "init_size", "budget", "batch"]:
        if c in df.columns:
            all_group_cols.append(c)

    group_cols = [c for c in all_group_cols if c != param_col]

    curves_dir = out_dir / "plots" / "screening_curves" / param_col
    curves_dir.mkdir(parents=True, exist_ok=True)

    for key, g in df.groupby(group_cols):
        values = sorted_unique_numeric_values(g, param_col)
        if len(values) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for val in values:
            sub = g[g[param_col] == val].copy()
            avg = (
                sub.groupby("labeled_count", as_index=False)[metric]
                .mean()
                .sort_values("labeled_count")
            )

            if len(avg) == 0:
                continue

            ax.plot(
                avg["labeled_count"],
                avg[metric],
                linewidth=2.0,
                label=f"{param_col}={val:g}",
            )

        key_dict = dict(zip(group_cols, key))
        title = " | ".join(f"{k}={key_dict[k]}" for k in group_cols if k in key_dict)

        ax.set_title(f"{param_col} comparison | {title}")
        ax.set_xlabel("labeled_count")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()

        fname_parts = [sanitize_filename_part(v) for v in key]
        fname = "_".join(fname_parts) + f"_compare_{param_col}.png"

        fig.tight_layout()
        fig.savefig(curves_dir / fname, dpi=180)
        plt.close(fig)


def plot_pairwise_deltas(pw: pd.DataFrame, out_dir: Path, pretty_name: str) -> None:
    if len(pw) == 0:
        return

    pairwise_dir = out_dir / "plots" / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    for comparison, pw_cmp in pw.groupby("comparison"):
        for col, suffix, xlabel in [
            ("delta_aulc_norm", "aulc", "delta_aulc_norm"),
            ("delta_final", "final", "delta_final"),
        ]:
            vals = pd.to_numeric(pw_cmp[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.hist(vals, bins=min(15, max(5, len(vals))), edgecolor="black")
            ax.axvline(0.0, linestyle="--", linewidth=1.5)
            ax.set_title(f"Pairwise delta ({pretty_name}): {comparison} | {col}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("count")
            ax.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(pairwise_dir / f"pairwise_{pretty_name}_{comparison}_{suffix}.png", dpi=180)
            plt.close(fig)


def plot_main_effects(
    table: pd.DataFrame,
    out_dir: Path,
    param_col: str,
    pretty_name: str,
) -> None:
    if len(table) == 0:
        return

    main_effects_dir = out_dir / "plots" / "main_effects"
    main_effects_dir.mkdir(parents=True, exist_ok=True)

    vals = table[param_col].tolist()
    labels = [f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v) for v in vals]
    y = pd.to_numeric(table["mean_aulc_norm"], errors="coerce")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, y, color=colors)

    ax.legend(bars, [f"{pretty_name}={lab}" for lab in labels], loc="best", framealpha=0.9)

    ax.set_title(f"Main effect: {pretty_name}")
    ax.set_xlabel(pretty_name)
    ax.set_ylabel("mean_aulc_norm")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(main_effects_dir / f"main_effects_{pretty_name}.png", dpi=180)
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_csv))
    print("Loaded rows:", len(df))

    if len(df) == 0:
        raise SystemExit("No rows found after filtering to phase=active, split=val, step_type=cycle.")

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
        cols=["aulc", "aulc_norm", "final_metric", "best_metric"],
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

    # --- main effects ---
    for pretty_name, param_col in analysis_plan.items():
        table = main_effects(summary, param_col)
        path = out_dir / f"main_effects_{pretty_name}.csv"

        table_to_save = format_numeric_columns(
            table,
            cols=["mean_aulc_norm", "std_aulc_norm", "mean_final", "std_final"],
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
                or c.startswith("final_metric_")
                or c in ["delta_aulc_norm", "delta_final", "v1", "v2"]
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
                "median_delta_final",
                "mean_delta_final",
                "win_rate_v1_final_pct",
                "win_rate_v2_final_pct",
            ],
        )
        pw_summary_to_save.to_csv(pw_summary_path, index=False)

        print(f"\n=== PAIRWISE: {pretty_name} ({param_col}) ===")
        desc = pw[["delta_aulc_norm", "delta_final"]].describe()
        desc_to_print = format_numeric_columns(
            desc.reset_index(),
            cols=["delta_aulc_norm", "delta_final"],
        )
        print(desc_to_print.to_string(index=False))
        print("\nPairwise summary:")
        print(pw_summary_to_save.to_string(index=False))

        plot_pairwise_deltas(pw, out_dir, pretty_name)

    # --- screening curves ---
    for pretty_name, param_col in analysis_plan.items():
        plot_screening_curves(df, out_dir, param_col, metric=args.metric)

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