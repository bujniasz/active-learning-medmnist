#!/usr/bin/env python3
from __future__ import annotations

# General
import argparse
from pathlib import Path

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Custom
from src.analysis.plotting import (
    make_color_map,
    plot_active_strategy_confusion_matrices,
    plot_active_strategy_train_loss,
    plot_active_strategy_validation_metric,
)
from src.utils.shared import load_config


METRIC_COLS = ["acc", "f1_macro", "auc", "ap"]
CONFUSION_COLS = ["tp", "fp", "tn", "fn"]


def as_int(value) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Cannot convert value to int: {value!r}")
    return int(numeric)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--active-results-csv", default=None, help="Path to active learning results CSV")
    p.add_argument("--supervised-results-csv", default=None, help="Optional path to supervised results CSV")
    p.add_argument("--out-dir", default=None, help="Directory for analysis outputs")
    p.add_argument("--plot-validation", action="store_true", help="Generate validation metric plots")
    p.add_argument("--plot-train-loss", action="store_true", help="Generate train loss plots")
    p.add_argument("--plot-confusion", action="store_true", help="Generate confusion matrix plots")
    return p.parse_args()


def apply_config(args):
    if args.config is None:
        args.datasets = None
        args.strategies = None
        args.metrics = METRIC_COLS
        return args

    cfg = load_config(args.config)

    args.active_results_csv = cfg.get("active_results_csv", args.active_results_csv)
    args.supervised_results_csv = cfg.get("supervised_results_csv", args.supervised_results_csv)
    args.out_dir = cfg.get("out_dir", args.out_dir)
    args.datasets = cfg.get("datasets", None)
    args.strategies = cfg.get("strategies", None)
    args.metrics = cfg.get("metrics", METRIC_COLS)
    args.plot_validation = cfg.get("plot_validation", args.plot_validation)
    args.plot_train_loss = cfg.get("plot_train_loss", args.plot_train_loss)
    args.plot_confusion = cfg.get("plot_confusion", args.plot_confusion)

    return args


def require_columns(df: pd.DataFrame, required: list[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def convert_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_active_results(path: Path, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = [
        "dataset",
        "phase",
        "strategy",
        "seed",
        "split",
        "step_type",
        "step",
        "labeled_count",
        *metrics,
        "val_mean",
        "train_loss",
        *CONFUSION_COLS,
    ]
    require_columns(df, required, name=str(path))

    df = df[df["phase"] == "active"].copy()
    if len(df) == 0:
        raise ValueError("No active learning rows found.")

    numeric_cols = [
        "seed",
        "step",
        "labeled_count",
        "train_loss",
        "val_mean",
        *metrics,
        *CONFUSION_COLS,
    ]
    df = convert_numeric(df, numeric_cols)

    val_df = df[(df["split"] == "val") & (df["step_type"] == "cycle")].copy()
    test_df = df[(df["split"] == "test") & (df["step_type"] == "final")].copy()

    if len(val_df) == 0:
        raise ValueError("No active validation-cycle rows found.")
    if len(test_df) == 0:
        raise ValueError("No active final-test rows found.")

    return val_df, test_df


def load_supervised_results(path: Path, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = [
        "dataset",
        "phase",
        "seed",
        "split",
        "step_type",
        *metrics,
        *CONFUSION_COLS,
    ]
    require_columns(df, required, name=str(path))

    df = df[df["phase"] == "supervised"].copy()
    if len(df) == 0:
        raise ValueError("No supervised rows found.")

    numeric_cols = ["seed", *metrics, *CONFUSION_COLS]
    df = convert_numeric(df, numeric_cols)

    val_df = df[(df["split"] == "val") & (df["step_type"] == "epoch")].copy()
    test_df = df[(df["split"] == "test") & (df["step_type"] == "final")].copy()

    if len(val_df) == 0:
        raise ValueError("No supervised validation-epoch rows found.")
    if len(test_df) == 0:
        raise ValueError("No supervised final-test rows found.")

    return val_df, test_df


def apply_filters(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    datasets: list[str] | None,
    strategies: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if datasets:
        val_df = val_df[val_df["dataset"].isin(datasets)].copy()
        test_df = test_df[test_df["dataset"].isin(datasets)].copy()

    if strategies:
        val_df = val_df[val_df["strategy"].isin(strategies)].copy()
        test_df = test_df[test_df["strategy"].isin(strategies)].copy()

    if len(val_df) == 0:
        raise ValueError("No active validation rows remain after filtering.")
    if len(test_df) == 0:
        raise ValueError("No active test rows remain after filtering.")

    return val_df, test_df


def compute_aulc(g: pd.DataFrame, metric: str) -> tuple[float, float]:
    g = g.sort_values("labeled_count")
    x_vals = g["labeled_count"].to_numpy()
    y_vals = g[metric].to_numpy()

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

    return float(area), float(norm)


def compute_val_aulc(val_df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for (dataset, strategy, seed), g in val_df.groupby(["dataset", "strategy", "seed"], sort=True):
        for metric in metrics:
            area, norm = compute_aulc(g, metric)
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "seed": as_int(seed),
                    "metric": metric,
                    "aulc": area,
                    "aulc_norm": norm,
                    "n_cycles": int(g["step"].nunique()),
                }
            )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["dataset", "strategy", "metric"], as_index=False)
        .agg(
            mean_aulc_norm=("aulc_norm", "mean"),
            std_aulc_norm=("aulc_norm", "std"),
            mean_aulc=("aulc", "mean"),
            std_aulc=("aulc", "std"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["dataset", "metric", "mean_aulc_norm"], ascending=[True, True, False])
    )

    return runs, summary


def compute_supervised_val_baselines(supervised_val: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for (dataset, seed), g in supervised_val.groupby(["dataset", "seed"], sort=True):
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                best_val = np.nan
            else:
                best_val = float(vals.max())

            rows.append(
                {
                    "dataset": dataset,
                    "seed": as_int(seed),
                    "metric": metric,
                    "best_supervised_val_metric": best_val,
                }
            )

    run_baselines = pd.DataFrame(rows)
    return (
        run_baselines.groupby(["dataset", "metric"], as_index=False)
        .agg(
            supervised_val_baseline=("best_supervised_val_metric", "mean"),
            std_supervised_val_baseline=("best_supervised_val_metric", "std"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["dataset", "metric"])
    )


def compute_time_to_baseline(
    val_df: pd.DataFrame,
    baselines: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    baseline_map = {
        (str(r["dataset"]), str(r["metric"])): float(r["supervised_val_baseline"])
        for r in baselines.to_dict(orient="records")
    }

    for (dataset, strategy), g_strategy in val_df.groupby(["dataset", "strategy"], sort=True):
        for metric in metrics:
            baseline = baseline_map.get((str(dataset), metric), np.nan)
            curve = (
                g_strategy.groupby("labeled_count", as_index=False)
                .agg(
                    mean_metric=(metric, "mean"),
                    std_metric=(metric, "std"),
                    mean_step=("step", "mean"),
                )
                .sort_values("labeled_count")
            )
            curve["mean_metric"] = pd.to_numeric(curve["mean_metric"], errors="coerce")
            reached_rows = curve[curve["mean_metric"] >= baseline] if np.isfinite(baseline) else curve.iloc[0:0]

            reached = len(reached_rows) > 0
            if reached:
                first = reached_rows.iloc[0]
                first_labeled_count = int(first["labeled_count"])
                first_step = int(round(float(first["mean_step"])))
                first_mean_metric = float(first["mean_metric"])
            else:
                first_labeled_count = np.nan
                first_step = np.nan
                first_mean_metric = np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "metric": metric,
                    "supervised_val_baseline": baseline,
                    "reached": bool(reached),
                    "first_labeled_count": first_labeled_count,
                    "first_step": first_step,
                    "first_mean_metric": first_mean_metric,
                }
            )

    return pd.DataFrame(rows)


def compute_best_val_mean_cycles(val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, strategy), g_strategy in val_df.groupby(["dataset", "strategy"], sort=True):
        curve = (
            g_strategy.groupby("labeled_count", as_index=False)
            .agg(
                mean_val_mean=("val_mean", "mean"),
                std_val_mean=("val_mean", "std"),
                mean_step=("step", "mean"),
            )
            .sort_values("labeled_count")
        )
        curve["mean_val_mean"] = pd.to_numeric(curve["mean_val_mean"], errors="coerce")
        curve = curve.dropna(subset=["mean_val_mean"])

        if len(curve) == 0:
            best_step = np.nan
            best_labeled_count = np.nan
            best_val_mean = np.nan
            best_val_mean_std = np.nan
        else:
            best_idx = curve["mean_val_mean"].idxmax()
            best = curve.loc[best_idx]
            best_step = int(round(float(best["mean_step"])))
            best_labeled_count = int(best["labeled_count"])
            best_val_mean = float(best["mean_val_mean"])
            best_val_mean_std = (
                float(best["std_val_mean"])
                if pd.notna(best["std_val_mean"])
                else np.nan
            )

        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "best_step": best_step,
                "best_labeled_count": best_labeled_count,
                "best_val_mean": best_val_mean,
                "std_val_mean": best_val_mean_std,
            }
        )

    return pd.DataFrame(rows)


def compute_validation_cycle_summary(
    time_to_baseline: pd.DataFrame,
    best_val_mean_cycles: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    if len(time_to_baseline) == 0:
        return pd.DataFrame()

    rows = []
    best_map = {
        (str(r["dataset"]), str(r["strategy"])): r
        for r in best_val_mean_cycles.to_dict(orient="records")
    }

    for (dataset, strategy), g_strategy in time_to_baseline.groupby(["dataset", "strategy"], sort=True):
        row = {
            "dataset": dataset,
            "strategy": strategy,
        }

        for metric in metrics:
            metric_rows = g_strategy[g_strategy["metric"] == metric]
            if len(metric_rows) == 0:
                row[f"first_step_{metric}"] = np.nan
                row[f"reached_{metric}"] = False
                continue

            metric_row = metric_rows.iloc[0]
            row[f"first_step_{metric}"] = metric_row["first_step"]
            row[f"reached_{metric}"] = bool(metric_row["reached"])

        best = best_map.get((str(dataset), str(strategy)), {})
        row["best_val_mean_step"] = best.get("best_step", np.nan)
        row["best_val_mean_labeled_count"] = best.get("best_labeled_count", np.nan)
        row["best_val_mean"] = best.get("best_val_mean", np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_test_results(test_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg_kwargs = {"n_runs": ("seed", "nunique")}
    for metric in metrics:
        agg_kwargs[f"mean_{metric}"] = (metric, "mean")
        agg_kwargs[f"std_{metric}"] = (metric, "std")

    return (
        test_df.groupby(["dataset", "strategy"], as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["dataset", "strategy"])
    )


def summarize_supervised_test_results(supervised_test: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    summary = summarize_test_results(
        supervised_test.assign(strategy="supervised"),
        metrics,
    )
    return summary


def compute_test_ranking(test_summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for dataset, g in test_summary.groupby("dataset", sort=True):
        tmp = g.copy()
        for metric in metrics:
            tmp[f"rank_{metric}"] = tmp[f"mean_{metric}"].rank(
                ascending=False,
                method="average",
            )

        rank_cols = [f"rank_{metric}" for metric in metrics]
        tmp["mean_test_rank"] = tmp[rank_cols].mean(axis=1, skipna=True)
        tmp["test_position"] = (
            tmp["mean_test_rank"]
            .rank(ascending=True, method="min")
            .astype(int)
        )
        tmp["top1_metrics"] = (tmp[rank_cols] == 1).sum(axis=1)
        tmp["top3_metrics"] = (tmp[rank_cols] <= 3).sum(axis=1)

        columns = ["dataset", "strategy"]
        if "method_type" in tmp.columns:
            columns.append("method_type")

        rows.append(
            tmp[
                [
                    *columns,
                    "mean_test_rank",
                    "test_position",
                    "top1_metrics",
                    "top3_metrics",
                    *rank_cols,
                ]
            ]
        )

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["dataset", "mean_test_rank", "strategy"])
    )


def compute_overall_test_ranking(test_ranking: pd.DataFrame) -> pd.DataFrame:
    if len(test_ranking) == 0:
        return pd.DataFrame()

    require_columns(
        test_ranking,
        ["dataset", "strategy", "mean_test_rank"],
        name="test_ranking",
    )

    test_ranking = test_ranking.copy()
    test_ranking["test_position"] = (
        test_ranking.groupby("dataset")["mean_test_rank"]
        .rank(ascending=True, method="min")
        .astype(int)
    )

    dataset_names = sorted(map(str, test_ranking["dataset"].dropna().unique().tolist()))
    index_cols = ["strategy"]
    if "method_type" in test_ranking.columns:
        index_cols.append("method_type")

    rank_pivot = test_ranking.pivot_table(
        index=index_cols,
        columns="dataset",
        values="mean_test_rank",
        aggfunc="mean",
    )
    rank_pivot = rank_pivot.rename(columns={dataset: f"rank_{dataset}" for dataset in dataset_names})

    position_pivot = test_ranking.pivot_table(
        index=index_cols,
        columns="dataset",
        values="test_position",
        aggfunc="first",
    )
    position_pivot = position_pivot.rename(
        columns={dataset: f"position_{dataset}" for dataset in dataset_names}
    )

    ranking = rank_pivot.join(position_pivot).reset_index()
    rank_cols = [f"rank_{dataset}" for dataset in dataset_names if f"rank_{dataset}" in ranking.columns]
    position_cols = [
        f"position_{dataset}"
        for dataset in dataset_names
        if f"position_{dataset}" in ranking.columns
    ]
    ranking["mean_rank"] = ranking[rank_cols].mean(axis=1, skipna=True)
    ranking["n_datasets"] = ranking[rank_cols].count(axis=1)
    ranking["overall_position"] = ranking["mean_rank"].rank(
        ascending=True,
        method="min",
    ).astype(int)

    ordered_cols = ["strategy"]
    if "method_type" in ranking.columns:
        ordered_cols.append("method_type")
    ordered_cols.extend([*position_cols, *rank_cols, "mean_rank", "n_datasets", "overall_position"])

    return ranking[ordered_cols].sort_values(["overall_position", "mean_rank", "strategy"])


def compute_strategy_spread(active_test_summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []

    for dataset, g_dataset in active_test_summary.groupby("dataset", sort=True):
        for metric in metrics:
            mean_col = f"mean_{metric}"
            std_col = f"std_{metric}"

            if mean_col not in g_dataset.columns:
                continue

            if std_col not in g_dataset.columns:
                g_dataset = g_dataset.assign(**{std_col: np.nan})

            ranked = (
                g_dataset[["strategy", mean_col, std_col]]
                .dropna(subset=[mean_col])
                .sort_values([mean_col, "strategy"], ascending=[False, True])
                .reset_index(drop=True)
            )

            if len(ranked) == 0:
                continue

            best = ranked.iloc[0]
            second = ranked.iloc[1] if len(ranked) > 1 else ranked.iloc[0]
            worst = ranked.iloc[-1]

            best_value = float(best[mean_col])
            second_value = float(second[mean_col])
            worst_value = float(worst[mean_col])

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "best_strategy": best["strategy"],
                    "best_value": best_value,
                    "best_std": float(best[std_col]) if pd.notna(best[std_col]) else np.nan,
                    "second_strategy": second["strategy"],
                    "second_value": second_value,
                    "second_std": float(second[std_col]) if pd.notna(second[std_col]) else np.nan,
                    "worst_strategy": worst["strategy"],
                    "worst_value": worst_value,
                    "worst_std": float(worst[std_col]) if pd.notna(worst[std_col]) else np.nan,
                    "best_minus_second": best_value - second_value,
                    "best_minus_worst": best_value - worst_value,
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "metric"])


def compute_active_vs_supervised_spread(
    active_test_summary: pd.DataFrame,
    supervised_test_summary: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    supervised_by_dataset = {
        str(row["dataset"]): row
        for _, row in supervised_test_summary.iterrows()
    }

    for dataset, g_dataset in active_test_summary.groupby("dataset", sort=True):
        supervised_row = supervised_by_dataset.get(str(dataset))
        if supervised_row is None:
            continue

        for metric in metrics:
            mean_col = f"mean_{metric}"
            std_col = f"std_{metric}"

            if mean_col not in g_dataset.columns or mean_col not in supervised_row.index:
                continue

            if std_col not in g_dataset.columns:
                g_dataset = g_dataset.assign(**{std_col: np.nan})

            ranked = (
                g_dataset[["strategy", mean_col, std_col]]
                .dropna(subset=[mean_col])
                .sort_values([mean_col, "strategy"], ascending=[False, True])
                .reset_index(drop=True)
            )

            if len(ranked) == 0 or pd.isna(supervised_row[mean_col]):
                continue

            best_active = ranked.iloc[0]
            worst_active = ranked.iloc[-1]

            best_active_value = float(best_active[mean_col])
            worst_active_value = float(worst_active[mean_col])
            supervised_value = float(supervised_row[mean_col])

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "supervised_value": supervised_value,
                    "supervised_std": float(supervised_row[std_col])
                    if std_col in supervised_row.index and pd.notna(supervised_row[std_col])
                    else np.nan,
                    "best_active_strategy": best_active["strategy"],
                    "best_active_value": best_active_value,
                    "best_active_std": float(best_active[std_col]) if pd.notna(best_active[std_col]) else np.nan,
                    "worst_active_strategy": worst_active["strategy"],
                    "worst_active_value": worst_active_value,
                    "worst_active_std": float(worst_active[std_col]) if pd.notna(worst_active[std_col]) else np.nan,
                    "best_active_minus_supervised": best_active_value - supervised_value,
                    "supervised_minus_best_active": supervised_value - best_active_value,
                    "best_active_minus_worst_active": best_active_value - worst_active_value,
                    "best_active_above_supervised": best_active_value > supervised_value,
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "metric"])


def summarize_confusion_matrices(test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, strategy), g in test_df.groupby(["dataset", "strategy"], sort=True):
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        tn = int(g["tn"].sum())
        fn = int(g["fn"].sum())

        rows.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "support": tp + fp + tn + fn,
            }
        )

    return pd.DataFrame(rows)


def round_float_columns(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    df = df.copy()
    float_cols = df.select_dtypes(include=["float"]).columns
    for col in float_cols:
        df[col] = df[col].round(ndigits)
    return df


def baseline_dict_for_dataset(baselines: pd.DataFrame, dataset: str) -> dict[str, float]:
    if len(baselines) == 0:
        return {}

    tmp = baselines[baselines["dataset"] == dataset]
    return {
        str(row["metric"]): float(row["supervised_val_baseline"])
        for row in tmp.to_dict(orient="records")
    }


def best_labeled_count_dict_for_dataset(best_cycles: pd.DataFrame, dataset: str) -> dict[str, int]:
    if len(best_cycles) == 0:
        return {}

    tmp = best_cycles[best_cycles["dataset"] == dataset]
    result = {}
    for row in tmp.to_dict(orient="records"):
        labeled_count = row.get("best_labeled_count", np.nan)
        if pd.notna(labeled_count):
            result[str(row["strategy"])] = int(labeled_count)
    return result


def main():
    args = parse_args()
    args = apply_config(args)

    if args.active_results_csv is None:
        raise SystemExit("active_results_csv must be provided via CLI or config")
    if args.out_dir is None:
        raise SystemExit("out_dir must be provided via CLI or config")

    metrics = [str(m) for m in args.metrics]
    unknown_metrics = [m for m in metrics if m not in METRIC_COLS]
    if unknown_metrics:
        raise SystemExit(f"Unsupported metrics: {unknown_metrics}. Supported metrics: {METRIC_COLS}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df, test_df = load_active_results(Path(args.active_results_csv), metrics)
    val_df, test_df = apply_filters(
        val_df,
        test_df,
        datasets=args.datasets,
        strategies=args.strategies,
    )

    print("Loaded active validation-cycle rows:", len(val_df))
    print("Loaded active final-test rows:", len(test_df))

    aulc_runs, aulc_summary = compute_val_aulc(val_df, metrics)
    best_val_mean_cycles = compute_best_val_mean_cycles(val_df)
    active_test_summary = summarize_test_results(test_df, metrics)
    test_ranking = compute_test_ranking(active_test_summary, metrics)
    overall_test_ranking = compute_overall_test_ranking(test_ranking)
    strategy_spread = compute_strategy_spread(active_test_summary, metrics)
    confusion = summarize_confusion_matrices(test_df)

    round_float_columns(aulc_runs).to_csv(out_dir / "val_aulc_runs.csv", index=False)
    round_float_columns(aulc_summary).to_csv(out_dir / "val_aulc_summary.csv", index=False)
    round_float_columns(best_val_mean_cycles).to_csv(out_dir / "best_val_mean_cycles.csv", index=False)
    round_float_columns(active_test_summary).to_csv(out_dir / "active_test_summary.csv", index=False)
    round_float_columns(test_ranking).to_csv(out_dir / "test_ranking.csv", index=False)
    round_float_columns(overall_test_ranking).to_csv(out_dir / "overall_test_ranking.csv", index=False)
    round_float_columns(strategy_spread).to_csv(out_dir / "strategy_spread.csv", index=False)
    confusion.to_csv(out_dir / "confusion_matrices.csv", index=False)

    baselines = pd.DataFrame()
    if args.supervised_results_csv:
        supervised_val, supervised_test = load_supervised_results(Path(args.supervised_results_csv), metrics)
        if args.datasets:
            supervised_val = supervised_val[supervised_val["dataset"].isin(args.datasets)].copy()
            supervised_test = supervised_test[supervised_test["dataset"].isin(args.datasets)].copy()

        baselines = compute_supervised_val_baselines(supervised_val, metrics)
        time_to_baseline = compute_time_to_baseline(val_df, baselines, metrics)
        validation_cycle_summary = compute_validation_cycle_summary(
            time_to_baseline,
            best_val_mean_cycles,
            metrics,
        )
        supervised_test_summary = summarize_supervised_test_results(supervised_test, metrics)
        combined_test_summary = pd.concat(
            [
                supervised_test_summary.assign(method_type="supervised"),
                active_test_summary.assign(method_type="active"),
            ],
            ignore_index=True,
        )
        combined_test_ranking = compute_test_ranking(combined_test_summary, metrics)
        overall_test_ranking_with_supervised = compute_overall_test_ranking(combined_test_ranking)
        active_vs_supervised_spread = compute_active_vs_supervised_spread(
            active_test_summary,
            supervised_test_summary,
            metrics,
        )

        round_float_columns(baselines).to_csv(out_dir / "supervised_val_baselines.csv", index=False)
        round_float_columns(time_to_baseline).to_csv(out_dir / "time_to_supervised_baseline.csv", index=False)
        round_float_columns(validation_cycle_summary).to_csv(out_dir / "validation_cycle_summary.csv", index=False)
        round_float_columns(combined_test_summary).to_csv(out_dir / "test_summary_with_supervised.csv", index=False)
        round_float_columns(combined_test_ranking).to_csv(
            out_dir / "test_ranking_with_supervised.csv",
            index=False,
        )
        round_float_columns(overall_test_ranking_with_supervised).to_csv(
            out_dir / "overall_test_ranking_with_supervised.csv",
            index=False,
        )
        round_float_columns(active_vs_supervised_spread).to_csv(
            out_dir / "active_vs_supervised_spread.csv",
            index=False,
        )

    if args.plot_validation or args.plot_train_loss or args.plot_confusion:
        strategies = sorted(map(str, val_df["strategy"].dropna().unique().tolist()))
        color_map = make_color_map(strategies)

    if args.plot_validation:
        plots_dir = out_dir / "plots" / "validation"
        for dataset, g in val_df.groupby("dataset", sort=True):
            dataset_baselines = baseline_dict_for_dataset(baselines, str(dataset))
            best_labeled_count_by_strategy = best_labeled_count_dict_for_dataset(
                best_val_mean_cycles,
                str(dataset),
            )
            for metric in metrics:
                plot_active_strategy_validation_metric(
                    g,
                    str(dataset),
                    metric,
                    plots_dir / f"{dataset}_val_{metric}.png",
                    color_map,
                    dataset_baselines.get(metric),
                    best_labeled_count_by_strategy,
                )

    if args.plot_train_loss:
        plots_dir = out_dir / "plots" / "train_loss"
        for dataset, g in val_df.groupby("dataset", sort=True):
            plot_active_strategy_train_loss(
                g,
                str(dataset),
                plots_dir / f"{dataset}_train_loss.png",
                color_map,
            )

    if args.plot_confusion:
        plot_active_strategy_confusion_matrices(confusion, out_dir)

    print("Saved:", out_dir / "val_aulc_runs.csv")
    print("Saved:", out_dir / "val_aulc_summary.csv")
    print("Saved:", out_dir / "best_val_mean_cycles.csv")
    print("Saved:", out_dir / "active_test_summary.csv")
    print("Saved:", out_dir / "test_ranking.csv")
    print("Saved:", out_dir / "overall_test_ranking.csv")
    print("Saved:", out_dir / "strategy_spread.csv")
    print("Saved:", out_dir / "confusion_matrices.csv")
    if args.supervised_results_csv:
        print("Saved:", out_dir / "supervised_val_baselines.csv")
        print("Saved:", out_dir / "time_to_supervised_baseline.csv")
        print("Saved:", out_dir / "validation_cycle_summary.csv")
        print("Saved:", out_dir / "test_summary_with_supervised.csv")
        print("Saved:", out_dir / "test_ranking_with_supervised.csv")
        print("Saved:", out_dir / "overall_test_ranking_with_supervised.csv")
        print("Saved:", out_dir / "active_vs_supervised_spread.csv")
    if args.plot_validation or args.plot_train_loss or args.plot_confusion:
        print("Saved plots in:", out_dir / "plots")


if __name__ == "__main__":
    main()
