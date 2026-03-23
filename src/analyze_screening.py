#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import fmt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True, help="Path to results CSV")
    p.add_argument("--out-dir", required=True, help="Directory for analysis outputs")
    p.add_argument("--metric", default="val_mean", help="Metric used for AULC/final comparison")
    return p.parse_args()


def format_numeric_columns(df: pd.DataFrame, cols: list[str], ndigits: int = 4) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt(x, ndigits=ndigits))
    return df


def sanitize_filename_part(x) -> str:
    s = str(x)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def sorted_unique_int_values(df: pd.DataFrame, col: str) -> list[int]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return sorted(s.astype(int).unique().tolist())


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


def main_effects(summary: pd.DataFrame, param: str) -> pd.DataFrame:
    return (
        summary.groupby(param)
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


def pairwise_compare(summary: pd.DataFrame, param: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [
        "dataset",
        "strategy",
        "seed",
        "init_size",
        "batch",
        "budget",
        "epc",
    ]
    base_cols.remove(param)

    vals = sorted_unique_int_values(summary, param)
    if len(vals) != 2:
        print(f"⚠️ Pairwise only works for exactly 2 values of {param}, got: {vals}")
        return pd.DataFrame(), pd.DataFrame()

    pivot = summary.pivot_table(
        index=base_cols,
        columns=param,
        values=["aulc_norm", "final_metric"],
    )

    pivot = pivot.dropna()
    if len(pivot) == 0:
        print(f"⚠️ No valid pairs found for {param}")
        return pd.DataFrame(), pd.DataFrame()

    pivot.columns = [f"{metric}_{value}" for metric, value in pivot.columns]

    v1, v2 = vals
    pivot["delta_aulc_norm"] = pivot[f"aulc_norm_{v1}"] - pivot[f"aulc_norm_{v2}"]
    pivot["delta_final"] = pivot[f"final_metric_{v1}"] - pivot[f"final_metric_{v2}"]

    pw = pivot.reset_index()

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
                "param": param,
                "comparison": f"{v1}_vs_{v2}",
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

    return pw, summary_row


def plot_screening_curves(df: pd.DataFrame, out_dir: Path, param: str, metric: str = "val_mean") -> None:
    all_group_cols = ["dataset", "strategy", "init_size", "batch", "budget", "epc"]
    group_cols = [c for c in all_group_cols if c != param]

    curves_dir = out_dir / "plots" / "screening_curves" / param
    curves_dir.mkdir(parents=True, exist_ok=True)

    for key, g in df.groupby(group_cols):
        values = sorted_unique_int_values(g, param)
        if len(values) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for val in values:
            sub = g[g[param] == val].copy()
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
                label=f"{param}={val}",
            )

        key_dict = dict(zip(group_cols, key))
        title = (
            f"{param} comparison | "
            f"dataset={key_dict.get('dataset')} | "
            f"strategy={key_dict.get('strategy')} | "
            f"init={key_dict.get('init_size')} | "
            f"batch={key_dict.get('batch', '-')} | "
            f"epc={key_dict.get('epc', '-')} | "
            f"budget={key_dict.get('budget', '-')}"
        )

        ax.set_title(title)
        ax.set_xlabel("labeled_count")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()

        fname_parts = [sanitize_filename_part(v) for v in key]
        fname = "_".join(fname_parts) + f"_compare_{param}.png"

        fig.tight_layout()
        fig.savefig(curves_dir / fname, dpi=180)
        plt.close(fig)


def plot_pairwise_deltas(pw: pd.DataFrame, out_dir: Path, param: str) -> None:
    if len(pw) == 0:
        return

    pairwise_dir = out_dir / "plots" / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    v1 = None
    v2 = None
    aulc_cols = [c for c in pw.columns if c.startswith("aulc_norm_")]
    if len(aulc_cols) == 2:
        try:
            vals = sorted(int(c.split("_")[-1]) for c in aulc_cols)
            v1, v2 = vals
        except Exception:
            pass

    for col, suffix, xlabel in [
        ("delta_aulc_norm", "aulc", "delta_aulc_norm"),
        ("delta_final", "final", "delta_final"),
    ]:
        if col not in pw.columns:
            continue

        vals = pd.to_numeric(pw[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(vals, bins=min(15, max(5, len(vals))), edgecolor="black")
        ax.axvline(0.0, linestyle="--", linewidth=1.5)

        if v1 is not None and v2 is not None:
            title = f"Pairwise delta ({param}): {v1} - {v2} | {col}"
        else:
            title = f"Pairwise delta ({param}) - {col}"

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(pairwise_dir / f"pairwise_{param}_{suffix}.png", dpi=180)
        plt.close(fig)


def plot_main_effects(table: pd.DataFrame, out_dir: Path, param: str) -> None:
    if len(table) == 0:
        return

    main_effects_dir = out_dir / "plots" / "main_effects"
    main_effects_dir.mkdir(parents=True, exist_ok=True)

    vals = table[param].astype(str).tolist()
    y = pd.to_numeric(table["mean_aulc_norm"], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(vals, y)
    ax.set_title(f"Main effect: {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("mean_aulc_norm")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(main_effects_dir / f"main_effects_{param}.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(Path(args.results_csv))
    print("Loaded rows:", len(df))

    if len(df) == 0:
        raise SystemExit("No rows found after filtering to phase=active, split=val, step_type=cycle.")

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

    params = ["batch", "epc", "init_size", "budget"]

    for param in params:
        table = main_effects(summary, param)
        path = out_dir / f"main_effects_{param}.csv"

        table_to_save = format_numeric_columns(
            table,
            cols=["mean_aulc_norm", "std_aulc_norm", "mean_final", "std_final"],
        )
        table_to_save.to_csv(path, index=False)

        print(f"\n=== MAIN EFFECT: {param} ===")
        print(table_to_save.to_string(index=False))

        plot_main_effects(table, out_dir, param)

    for param in params:
        pw, pw_summary = pairwise_compare(summary, param)
        if len(pw) == 0:
            continue

        pw_path = out_dir / f"pairwise_{param}.csv"
        pw_to_save = format_numeric_columns(
            pw,
            cols=[
                c
                for c in pw.columns
                if c.startswith("aulc_norm_")
                or c.startswith("final_metric_")
                or c in ["delta_aulc_norm", "delta_final"]
            ],
        )
        pw_to_save.to_csv(pw_path, index=False)

        pw_summary_path = out_dir / f"pairwise_summary_{param}.csv"
        pw_summary_to_save = format_numeric_columns(
            pw_summary,
            cols=[
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

        print(f"\n=== PAIRWISE: {param} ===")
        desc = pw[["delta_aulc_norm", "delta_final"]].describe()
        desc_to_print = format_numeric_columns(
            desc.reset_index(),
            cols=["delta_aulc_norm", "delta_final"],
        )
        print(desc_to_print.to_string(index=False))
        print("\nPairwise summary:")
        print(pw_summary_to_save.to_string(index=False))

        plot_pairwise_deltas(pw, out_dir, param)

    for param in params:
        plot_screening_curves(df, out_dir, param, metric=args.metric)

    print("\n✅ Analysis done:", out_dir.resolve())


if __name__ == "__main__":
    main()