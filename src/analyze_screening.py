#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from typing import cast

from metrics import fmt, fmt_p_value


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
        "final_metric": "final metric",
        "aulc_norm": "AULC (normalized)",
    }
    return mapping.get(col, col.replace("_", " "))


def maybe_set_zoomed_yaxis(ax, y: pd.Series | np.ndarray) -> None:
    vals = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(vals) == 0:
        return

    y_min = float(vals.min())
    y_max = float(vals.max())

    if y_min >= 0.8 and y_max <= 0.88:
        ax.set_ylim(0.8, 0.88)
    else:
        pad = max(0.005, 0.08 * (y_max - y_min if y_max > y_min else 0.01))
        ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax.grid(axis="y", which="major", alpha=0.35)
    ax.grid(axis="y", which="minor", alpha=0.15)


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
    if abs_col not in summary.columns or pct_col not in summary.columns:
        return

    tmp = (
        summary.groupby([abs_col, pct_col], as_index=False)
        .agg(
            mean_aulc_norm=("aulc_norm", "mean"),
            std_aulc_norm=("aulc_norm", "std"),
            count=("aulc_norm", "count"),
        )
    )

    if len(tmp) == 0:
        return

    tmp[abs_col] = pd.to_numeric(tmp[abs_col], errors="coerce")
    tmp[pct_col] = pd.to_numeric(tmp[pct_col], errors="coerce")
    tmp["mean_aulc_norm"] = pd.to_numeric(tmp["mean_aulc_norm"], errors="coerce")

    tmp = tmp.dropna(subset=[abs_col, pct_col, "mean_aulc_norm"]).copy()

    tmp = tmp.sort_values(
        by=["mean_aulc_norm", abs_col, pct_col],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    aux_dir = out_dir / "plots" / "absolute_colored_effects"
    aux_dir.mkdir(parents=True, exist_ok=True)

    pct_levels = sorted_unique_numeric_values(tmp, pct_col)
    cmap = plt.get_cmap("tab10")
    color_map = {lvl: cmap(i % 10) for i, lvl in enumerate(pct_levels)}

    x_pos = np.arange(len(tmp))
    y_vals = tmp["mean_aulc_norm"].to_numpy()
    colors = [color_map[float(v)] for v in tmp[pct_col].to_list()]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x_pos,
        y_vals,
        color=colors,
        width=0.8,
        edgecolor="black",
        linewidth=1.0,
    )

    x_labels = [
        str(int(v)) if float(v).is_integer() else f"{v:g}"
        for v in tmp[abs_col]
    ]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)

    handles = []
    labels = []
    for lvl in pct_levels:
        handles.append(Rectangle((0, 0), 1, 1, color=color_map[lvl]))
        labels.append(f"{lvl:g}%")

    ax.legend(
        handles,
        labels,
        title=f"{pretty_name} (%)",
        loc="upper left",
        framealpha=0.95,
    )

    ax.set_title(f"Wpływ parametru {pretty_name} (wartości absolutne)")
    ax.set_xlabel(f"{pretty_name} – wartość absolutna")
    ax.set_ylabel("Średni AULC (znormalizowany)")

    ax.grid(axis="y", alpha=0.3)
    maybe_set_zoomed_yaxis(ax, y_vals)

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
# Wilcoxon
# -----------------------------
def wilcoxon_from_pairwise(pw: pd.DataFrame, param_name: str) -> pd.DataFrame:
    """
    Computes paired Wilcoxon signed-rank tests from pairwise deltas.
    Tests whether median(delta) differs from 0.

    Output metrics:
      - delta_aulc_norm
      - delta_final
    """
    if len(pw) == 0:
        return pd.DataFrame()

    rows = []

    metric_map = {
        "delta_aulc_norm": "aulc_norm",
        "delta_final": "final_metric",
    }

    for comparison, g in pw.groupby("comparison"):
        v1 = g["v1"].iloc[0]
        v2 = g["v2"].iloc[0]

        for delta_col, metric_name in metric_map.items():
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


# -----------------------------
# Plots
# -----------------------------
def plot_screening_curves(
    df: pd.DataFrame,
    out_dir: Path,
    param_col: str,
    metric: str = "val_mean",
    *,
    use_pct_mode: bool,
) -> None:
    """
    Rysuje krzywe uczenia dla porównania poziomów jednego parametru
    przy ustalonych pozostałych parametrach.
    """
    curves_dir = out_dir / "plots" / "screening_curves" / param_col
    curves_dir.mkdir(parents=True, exist_ok=True)

    if use_pct_mode:
        candidate_group_cols = [
            "dataset",
            "strategy",
            "epc",
            "init_size_pct",
            "batch_pct_of_budget",
            "budget_pct",
        ]
    else:
        candidate_group_cols = [
            "dataset",
            "strategy",
            "epc",
            "init_size",
            "batch",
            "budget",
        ]

    group_cols = [c for c in candidate_group_cols if c in df.columns and c != param_col]

    ylabel_map = {
        "val_mean": "Średnia metryka walidacyjna",
        "acc": "Accuracy",
        "f1_macro": "F1 macro",
        "auc": "AUC",
        "ap": "Average Precision",
        "train_loss": "Strata treningowa",
    }

    param_label_map = {
        "init_size": "init_size",
        "init_size_pct": "init_size (%)",
        "batch": "batch",
        "batch_pct_of_budget": "batch (% budżetu)",
        "budget": "budget",
        "budget_pct": "budget (%)",
        "epc": "epc",
    }

    short_label_map = {
        "strategy": "strategia",
        "epc": "epc",
        "init_size": "init",
        "init_size_pct": "init",
        "batch": "batch",
        "batch_pct_of_budget": "batch",
        "budget": "budżet",
        "budget_pct": "budżet",
    }

    for key, g in df.groupby(group_cols):
        values = sorted_unique_numeric_values(g, param_col)
        if len(values) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8.4, 5.2))

        x_all = sorted_unique_numeric_values(g, "labeled_count")
        if not x_all:
            plt.close(fig)
            continue

        x_min = min(x_all)
        x_max = max(x_all)

        for val in values:
            sub = g[g[param_col] == val].copy()

            avg = (
                sub.groupby("labeled_count", as_index=False)[metric]
                .mean()
                .sort_values("labeled_count")
            )

            if len(avg) == 0:
                continue

            val_txt = f"{val:g}" if isinstance(val, (int, float, np.floating)) else str(val)

            ax.plot(
                avg["labeled_count"],
                avg[metric],
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=f"{param_label_map.get(param_col, param_col)} = {val_txt}",
            )

        key_dict = dict(zip(group_cols, key))

        fixed_parts = []
        for c in group_cols:
            if c == "dataset":
                continue

            v = key_dict.get(c)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue

            if isinstance(v, (int, float, np.floating)):
                v_txt = f"{v:g}"
            else:
                v_txt = str(v)

            fixed_parts.append(f"{short_label_map.get(c, c)}={v_txt}")

        title_suffix = " | ".join(fixed_parts)

        ax.set_title(
            f"Krzywe screeningu: {param_label_map.get(param_col, param_col)}"
            + (f"\n{title_suffix}" if title_suffix else "")
        )

        ax.set_xlabel("Liczba oznaczonych próbek")
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.grid(alpha=0.3)

        ax.axvline(
            x_min,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.8,
        )

        y_top = ax.get_ylim()[1]
        ax.text(
            x_min,
            y_top,
            f" start = {int(x_min)}",
            ha="left",
            va="bottom",
            fontsize=9,
        )

        if len(x_all) <= 10:
            xticks = x_all
        else:
            step = max(1, len(x_all) // 8)
            xticks = x_all[::step]
            if x_min not in xticks:
                xticks = [x_min] + xticks
            if x_max not in xticks:
                xticks = xticks + [x_max]
            xticks = sorted(set(int(x) for x in xticks))

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks])

        ax.legend(framealpha=0.95, loc="best")

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

    eps = 0.002

    metric_labels = {
        "delta_aulc_norm": "Różnica AULC (znormalizowanego)",
        "delta_final": "Różnica wyniku końcowego",
    }

    metric_titles = {
        "delta_aulc_norm": "Rozkład różnic AULC",
        "delta_final": "Rozkład różnic wyniku końcowego",
    }

    param_labels = {
        "batch": "batch",
        "budget": "budget",
        "epc": "epc",
        "init_size": "init_size",
    }

    color_win_v1 = "tab:green"
    color_tie = "lightgray"
    color_win_v2 = "tab:red"
    alpha_fill = 0.9

    for comparison, pw_cmp in pw.groupby("comparison"):
        v1 = pw_cmp["v1"].iloc[0]
        v2 = pw_cmp["v2"].iloc[0]

        v1_txt = f"{v1:g}" if isinstance(v1, (int, float, np.floating)) else str(v1)
        v2_txt = f"{v2:g}" if isinstance(v2, (int, float, np.floating)) else str(v2)

        for col, suffix in [
            ("delta_aulc_norm", "aulc"),
            ("delta_final", "final"),
        ]:
            vals = pd.to_numeric(pw_cmp[col], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                continue

            neg = vals[vals < -eps]
            tie = vals[np.abs(vals) <= eps]
            pos = vals[vals > eps]

            n_neg = len(neg)
            n_tie = len(tie)
            n_pos = len(pos)
            n_all = len(vals)

            xmin = float(vals.min())
            xmax = float(vals.max())

            total_bins = min(14, max(6, int(np.sqrt(len(vals)) * 1.5)))
            edges = np.linspace(xmin, xmax, total_bins + 1)

            # =========================
            # 1) HISTOGRAM DELT
            # =========================
            fig_hist, ax_hist = plt.subplots(figsize=(8.2, 4.8))

            neg_edges: list[float] = [float(x) for x in edges if x <= -eps]
            if n_neg > 0 and len(neg_edges) >= 2:
                ax_hist.hist(
                    neg,
                    bins=neg_edges,
                    color=color_win_v2,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=alpha_fill,
                    label=f"< -{eps:g}: wygrywa {v2_txt}",
                )

            if n_tie > 0:
                ax_hist.hist(
                    tie,
                    bins=[-eps, eps],
                    color=color_tie,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=alpha_fill,
                    label=f"|Δ| ≤ {eps:g}: remis",
                )

            pos_edges: list[float] = [float(x) for x in edges if x >= eps]
            if n_pos > 0 and len(pos_edges) >= 2:
                ax_hist.hist(
                    pos,
                    bins=pos_edges,
                    color=color_win_v1,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=alpha_fill,
                    label=f"> {eps:g}: wygrywa {v1_txt}",
                )

            ax_hist.axvline(0.0, linestyle="--", linewidth=1.8, color="black")

            ax_hist.set_title(
                f"{metric_titles[col]}: {param_labels.get(pretty_name, pretty_name)} ({v1_txt} vs {v2_txt})"
            )
            ax_hist.set_xlabel(metric_labels[col])
            ax_hist.set_ylabel("Liczba par")
            ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_hist.grid(axis="y", alpha=0.3)
            ax_hist.legend(loc="best", framealpha=0.95)

            fig_hist.tight_layout()
            fig_hist.savefig(
                pairwise_dir / f"pairwise_hist_{pretty_name}_{comparison}_{suffix}.png",
                dpi=180,
            )
            plt.close(fig_hist)

            # =========================
            # 2) BILANS PORÓWNAŃ
            # =========================
            fig_bar, ax_bar = plt.subplots(figsize=(5.6, 4.8))

            categories = [f"Wygrywa {v1_txt}", "Remis", f"Wygrywa {v2_txt}"]
            counts = [n_pos, n_tie, n_neg]
            colors = [color_win_v1, color_tie, color_win_v2]

            x = np.arange(3)
            ax_bar.bar(
                x,
                counts,
                color=colors,
                edgecolor="black",
                linewidth=1.0,
                width=0.72,
                alpha=alpha_fill,
            )

            ax_bar.set_title(
                f"Bilans porównań: {param_labels.get(pretty_name, pretty_name)} ({v1_txt} vs {v2_txt})"
            )
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(categories, rotation=20, ha="right")
            ax_bar.set_ylabel("Liczba par")
            ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_bar.grid(axis="y", alpha=0.3)

            max_idx = int(np.argmax(counts))
            if max_idx == 0:
                text_x = 0.98
                text_ha = "right"
            elif max_idx == 2:
                text_x = 0.02
                text_ha = "left"
            else:
                text_x = 0.98
                text_ha = "right"

            summary_text = (
                f"Wygrywa {v1_txt}: {n_pos}/{n_all} ({100*n_pos/n_all:.1f}%)\n"
                f"Remis: {n_tie}/{n_all} ({100*n_tie/n_all:.1f}%)\n"
                f"Wygrywa {v2_txt}: {n_neg}/{n_all} ({100*n_neg/n_all:.1f}%)"
            )

            ax_bar.text(
                text_x,
                0.98,
                summary_text,
                transform=ax_bar.transAxes,
                ha=text_ha,
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="black",
                ),
            )

            fig_bar.tight_layout()
            fig_bar.savefig(
                pairwise_dir / f"pairwise_winrate_{pretty_name}_{comparison}_{suffix}.png",
                dpi=180,
            )
            plt.close(fig_bar)


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

    plot_df = table.copy()

    try:
        plot_df[param_col] = pd.to_numeric(plot_df[param_col], errors="raise")
        plot_df = plot_df.sort_values(param_col, ascending=True)
    except Exception:
        plot_df = plot_df.sort_values(param_col, ascending=True, key=lambda s: s.astype(str))

    vals = plot_df[param_col].tolist()
    labels = []
    for v in vals:
        if isinstance(v, (int, float, np.floating)):
            labels.append(f"{v:g}")
        else:
            labels.append(str(v))

    y = pd.to_numeric(plot_df["mean_aulc_norm"], errors="coerce").to_numpy()

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    x_pos = np.arange(len(labels))
    ax.bar(
        x_pos,
        y,
        color=colors,
        width=0.8,
        edgecolor="black",
        linewidth=1.0,
    )

    xlabel_map = {
        "batch": "batch (% budżetu)",
        "budget": "budget (%)",
        "epc": "liczba epok na cykl",
        "init_size": "rozmiar zbioru początkowego (%)",
    }

    title_map = {
        "batch": "Wpływ parametru batch",
        "budget": "Wpływ parametru budget",
        "epc": "Wpływ parametru epc",
        "init_size": "Wpływ parametru init_size",
    }

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    ax.set_title(title_map.get(pretty_name, f"Wpływ parametru {pretty_name}"))
    ax.set_xlabel(xlabel_map.get(pretty_name, pretty_name))
    ax.set_ylabel("Średni AULC (znormalizowany)")

    ax.set_ylim(0.8, 0.9)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))

    ax.grid(axis="y", which="major", alpha=0.35)
    ax.grid(axis="y", which="minor", alpha=0.15)

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

    all_wilcoxon = []

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