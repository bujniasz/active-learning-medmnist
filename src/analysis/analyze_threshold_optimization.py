#!/usr/bin/env python3
from __future__ import annotations

# General
import argparse
import re
from pathlib import Path
from typing import Any, Callable

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Torch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# Custom
from src.analysis.plotting import STRATEGY_LABELS, plot_threshold_score_curves
from src.utils.load_data import prepare_split_active
from src.utils.shared import ResNet18EmbedDropout, load_config


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--active-models-dir", default=None, help="Directory with active learning checkpoints")
    p.add_argument("--supervised-models-dir", default=None, help="Directory with supervised checkpoints")
    p.add_argument("--active-results-csv", default=None, help="Optional active learning results CSV")
    p.add_argument("--supervised-results-csv", default=None, help="Optional supervised results CSV")
    p.add_argument("--out-dir", default=None, help="Directory for analysis outputs")
    p.add_argument("--batch-size", type=int, default=256, help="Evaluation batch size")
    p.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter")
    p.add_argument("--strategies", nargs="*", default=None, help="Optional active strategy filter")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed filter")
    p.add_argument("--threshold-min", type=float, default=0.01, help="Minimum threshold")
    p.add_argument("--threshold-max", type=float, default=0.99, help="Maximum threshold")
    p.add_argument("--threshold-step", type=float, default=0.01, help="Threshold step")
    p.add_argument("--make-plots", action="store_true", help="Generate threshold score plots")
    return p.parse_args()


def apply_config(args):
    if args.config is None:
        return args

    cfg = load_config(args.config)

    args.active_models_dir = cfg.get("active_models_dir", args.active_models_dir)
    args.supervised_models_dir = cfg.get("supervised_models_dir", args.supervised_models_dir)
    args.active_results_csv = cfg.get("active_results_csv", args.active_results_csv)
    args.supervised_results_csv = cfg.get("supervised_results_csv", args.supervised_results_csv)
    args.out_dir = cfg.get("out_dir", cfg.get("output_dir", args.out_dir))
    args.batch_size = int(cfg.get("batch_size", args.batch_size))
    args.make_plots = bool(cfg.get("make_plots", args.make_plots))

    thresholds = cfg.get("thresholds", {})
    args.threshold_min = float(thresholds.get("min", 0.01))
    args.threshold_max = float(thresholds.get("max", 0.99))
    args.threshold_step = float(thresholds.get("step", 0.01))

    args.datasets = cfg.get("datasets", None)
    args.strategies = cfg.get("strategies", None)
    args.seeds = cfg.get("seeds", None)

    return args


def as_optional_set(
    values: list[Any] | None,
    *,
    cast: Callable[[Any], Any] = str,
) -> set[Any] | None:
    if values is None:
        return None
    return {cast(v) for v in values}


def require_columns(df: pd.DataFrame, required: list[str], *, name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def dataset_name_from_data_dir(data_dir: str | Path) -> str:
    return Path(str(data_dir)).name


def seed_from_filename(path: Path) -> int | None:
    match = re.search(r"-(\d+)\.pth$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_expected_models(path: str | Path | None, *, phase: str) -> set[str] | None:
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results CSV does not exist: {path}")

    df = pd.read_csv(path)
    if "model" not in df.columns:
        return None

    if "phase" in df.columns:
        df = df[df["phase"] == phase].copy()
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if "step_type" in df.columns:
        df = df[df["step_type"] == "final"].copy()

    models = {
        str(model)
        for model in df["model"].dropna().astype(str).tolist()
        if str(model).endswith(".pth")
    }
    return models if models else None


def discover_model_records(
    models_dir: str | Path | None,
    *,
    method_type: str,
    expected_models: set[str] | None,
    dataset_filter: set[str] | None,
    strategy_filter: set[str] | None,
    seed_filter: set[int] | None,
) -> list[dict]:
    if models_dir is None:
        return []

    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    records = []
    for model_path in sorted(models_dir.rglob("*.pth")):
        if expected_models is not None and model_path.name not in expected_models:
            continue

        checkpoint = torch.load(model_path, map_location="cpu")
        data_dir = checkpoint.get("data_dir")
        if data_dir is None:
            raise ValueError(f"Missing data_dir in checkpoint: {model_path}")

        dataset = dataset_name_from_data_dir(data_dir)
        seed = checkpoint.get("seed", seed_from_filename(model_path))
        if seed is None:
            raise ValueError(f"Cannot determine seed for checkpoint: {model_path}")
        seed = int(seed)

        if method_type == "active":
            strategy = str(checkpoint.get("strategy", model_path.stem))
            method_label = STRATEGY_LABELS.get(strategy, strategy)
        elif method_type == "supervised":
            strategy = "supervised"
            method_label = "Uczenie nadzorowane"
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if seed_filter is not None and seed not in seed_filter:
            continue
        if method_type == "active" and strategy_filter is not None and strategy not in strategy_filter:
            continue

        records.append(
            {
                "dataset": dataset,
                "method_type": method_type,
                "strategy": strategy,
                "method_label": method_label,
                "seed": seed,
                "model_path": str(model_path),
                "data_dir": str(data_dir),
                "in_channels": int(checkpoint["in_channels"]),
                "num_classes": int(checkpoint["num_classes"]),
            }
        )

    return records


def tensor_loader_from_arrays(X, y, *, batch_size: int) -> DataLoader:
    X_t = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    y_t = torch.from_numpy(y) if isinstance(y, np.ndarray) else y

    if X_t.dtype == torch.uint8:
        X_t = X_t.float() / 255.0
    else:
        X_t = X_t.float()
        if X_t.max() > 1.5:
            X_t = X_t / 255.0

    if X_t.ndim == 3:
        X_t = X_t.unsqueeze(1)

    y_t = y_t.long().view(-1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)


def load_eval_loaders(data_dir: str, *, batch_size: int) -> tuple[DataLoader, DataLoader]:
    X_val, y_val, _, _ = prepare_split_active(data_dir, split="val", to_nchw=True)
    X_test, y_test, _, _ = prepare_split_active(data_dir, split="test", to_nchw=True)
    return (
        tensor_loader_from_arrays(X_val, y_val, batch_size=batch_size),
        tensor_loader_from_arrays(X_test, y_test, batch_size=batch_size),
    )


def predict_positive_probs(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_score = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs, backbone_mode="eval", enable_dropout=False)
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(targets.cpu().numpy().tolist())
            y_score.extend(probs.detach().cpu().numpy().tolist())

    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def build_threshold_grid(min_value: float, max_value: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if min_value <= 0 or max_value >= 1:
        raise ValueError("threshold range must stay inside (0, 1)")
    if min_value > max_value:
        raise ValueError("threshold min cannot be larger than threshold max")

    n_steps = int(np.floor((max_value - min_value) / step))
    values = min_value + step * np.arange(n_steps + 1)
    if values[-1] < max_value - 1e-10:
        values = np.append(values, max_value)
    return np.round(values, 10)


def metrics_for_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "acc": float(acc),
        "f1_macro": float(f1_macro),
        "threshold_score": float((acc + f1_macro) / 2.0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def select_threshold(curve: pd.DataFrame, *, default_threshold: float = 0.5) -> pd.Series:
    max_score = curve["val_threshold_score"].max()
    candidates: pd.DataFrame = curve.loc[curve["val_threshold_score"] == max_score].copy()
    candidates["distance_to_default"] = (candidates["threshold"] - default_threshold).abs()
    candidates = candidates.sort_values(
        by=["distance_to_default", "threshold"],
        ascending=[True, True],
    )
    return candidates.iloc[0]


def evaluate_record(record: dict, thresholds: np.ndarray, loaders_cache: dict) -> tuple[dict, list[dict]]:
    if int(record["num_classes"]) != 2:
        raise ValueError(f"Only binary classifiers are supported: {record['model_path']}")

    data_dir = str(record["data_dir"])
    if data_dir not in loaders_cache:
        loaders_cache[data_dir] = load_eval_loaders(data_dir, batch_size=int(record["batch_size"]))
    val_loader, test_loader = loaders_cache[data_dir]

    checkpoint = torch.load(record["model_path"], map_location=DEVICE)
    model = ResNet18EmbedDropout(
        in_channels=int(record["in_channels"]),
        num_classes=int(record["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    val_y_true, val_y_score = predict_positive_probs(model, val_loader)
    test_y_true, test_y_score = predict_positive_probs(model, test_loader)

    curve_rows = []
    for threshold in thresholds:
        threshold = float(threshold)
        m = metrics_for_threshold(val_y_true, val_y_score, threshold)
        curve_rows.append(
            {
                "dataset": record["dataset"],
                "method_type": record["method_type"],
                "strategy": record["strategy"],
                "method_label": record["method_label"],
                "seed": record["seed"],
                "model": Path(record["model_path"]).name,
                "threshold": threshold,
                "val_acc": m["acc"],
                "val_f1_macro": m["f1_macro"],
                "val_threshold_score": m["threshold_score"],
            }
        )

    curve = pd.DataFrame(curve_rows)
    selected = select_threshold(curve)
    selected_threshold = float(selected["threshold"])

    val_default = metrics_for_threshold(val_y_true, val_y_score, 0.5)
    val_optimized = metrics_for_threshold(val_y_true, val_y_score, selected_threshold)
    test_default = metrics_for_threshold(test_y_true, test_y_score, 0.5)
    test_optimized = metrics_for_threshold(test_y_true, test_y_score, selected_threshold)

    test_auc = roc_auc_score(test_y_true, test_y_score)
    test_ap = average_precision_score(test_y_true, test_y_score)

    run_row = {
        "dataset": record["dataset"],
        "method_type": record["method_type"],
        "strategy": record["strategy"],
        "method_label": record["method_label"],
        "seed": record["seed"],
        "model": Path(record["model_path"]).name,
        "model_path": record["model_path"],
        "data_dir": record["data_dir"],
        "selected_threshold": selected_threshold,
        "val_acc_default": val_default["acc"],
        "val_f1_macro_default": val_default["f1_macro"],
        "val_threshold_score_default": val_default["threshold_score"],
        "val_acc_optimized": val_optimized["acc"],
        "val_f1_macro_optimized": val_optimized["f1_macro"],
        "val_threshold_score_optimized": val_optimized["threshold_score"],
        "test_acc_default": test_default["acc"],
        "test_f1_macro_default": test_default["f1_macro"],
        "test_acc_optimized": test_optimized["acc"],
        "test_f1_macro_optimized": test_optimized["f1_macro"],
        "test_delta_acc": test_optimized["acc"] - test_default["acc"],
        "test_delta_f1_macro": test_optimized["f1_macro"] - test_default["f1_macro"],
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
    }

    for prefix, metrics in [
        ("test_default", test_default),
        ("test_optimized", test_optimized),
    ]:
        for col in ["tp", "fp", "tn", "fn"]:
            run_row[f"{prefix}_{col}"] = metrics[col]

    return run_row, curve_rows


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "method_type", "strategy", "method_label"]
    metric_cols = [
        "selected_threshold",
        "val_acc_default",
        "val_f1_macro_default",
        "val_threshold_score_default",
        "val_acc_optimized",
        "val_f1_macro_optimized",
        "val_threshold_score_optimized",
        "test_acc_default",
        "test_f1_macro_default",
        "test_acc_optimized",
        "test_f1_macro_optimized",
        "test_delta_acc",
        "test_delta_f1_macro",
        "test_auc",
        "test_ap",
    ]

    agg_kwargs = {"n_runs": ("seed", "nunique")}
    for col in metric_cols:
        agg_kwargs[f"{col}_mean"] = (col, "mean")
        agg_kwargs[f"{col}_std"] = (col, "std")

    summary = runs.groupby(group_cols, as_index=False).agg(**agg_kwargs)

    threshold_summary = (
        runs.groupby(group_cols, as_index=False)
        .agg(
            selected_threshold_median=("selected_threshold", "median"),
            selected_threshold_min=("selected_threshold", "min"),
            selected_threshold_max=("selected_threshold", "max"),
        )
    )
    summary = summary.merge(threshold_summary, on=group_cols, how="left")

    return summary.sort_values(["dataset", "method_type", "strategy"])


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    return (
        curves.groupby(
            ["dataset", "method_type", "strategy", "method_label", "threshold"],
            as_index=False,
        )
        .agg(
            mean_val_acc=("val_acc", "mean"),
            std_val_acc=("val_acc", "std"),
            mean_val_f1_macro=("val_f1_macro", "mean"),
            std_val_f1_macro=("val_f1_macro", "std"),
            mean_val_threshold_score=("val_threshold_score", "mean"),
            std_val_threshold_score=("val_threshold_score", "std"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["dataset", "method_type", "strategy", "threshold"])
    )


def round_float_columns(df: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    df = df.copy()
    float_cols = df.select_dtypes(include=["float"]).columns
    for col in float_cols:
        df[col] = df[col].round(ndigits)
    return df


def main():
    args = apply_config(parse_args())

    if args.out_dir is None:
        raise SystemExit("out_dir must be provided via CLI or config")
    if args.active_models_dir is None and args.supervised_models_dir is None:
        raise SystemExit("At least one models directory must be provided")

    dataset_filter = as_optional_set(args.datasets, cast=str)
    strategy_filter = as_optional_set(args.strategies, cast=str)
    seed_filter = as_optional_set(args.seeds, cast=int)

    active_expected = load_expected_models(args.active_results_csv, phase="active")
    supervised_expected = load_expected_models(args.supervised_results_csv, phase="supervised")

    records = []
    records.extend(
        discover_model_records(
            args.active_models_dir,
            method_type="active",
            expected_models=active_expected,
            dataset_filter=dataset_filter,
            strategy_filter=strategy_filter,
            seed_filter=seed_filter,
        )
    )
    records.extend(
        discover_model_records(
            args.supervised_models_dir,
            method_type="supervised",
            expected_models=supervised_expected,
            dataset_filter=dataset_filter,
            strategy_filter=strategy_filter,
            seed_filter=seed_filter,
        )
    )

    if len(records) == 0:
        raise SystemExit("No model checkpoints selected for threshold optimization")

    for record in records:
        record["batch_size"] = int(args.batch_size)

    thresholds = build_threshold_grid(
        float(args.threshold_min),
        float(args.threshold_max),
        float(args.threshold_step),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Selected models:", len(records))
    print("Thresholds:", len(thresholds), f"({thresholds[0]:.2f}-{thresholds[-1]:.2f})")
    print("Device:", DEVICE)

    loaders_cache = {}
    run_rows = []
    curve_rows = []

    for i, record in enumerate(records, start=1):
        print(
            f"[{i}/{len(records)}] "
            f"{record['dataset']} | {record['method_label']} | seed={record['seed']}"
        )
        run_row, record_curve_rows = evaluate_record(record, thresholds, loaders_cache)
        run_rows.append(run_row)
        curve_rows.extend(record_curve_rows)

    runs = pd.DataFrame(run_rows)
    curves = pd.DataFrame(curve_rows)
    summary = summarize_runs(runs)
    curve_summary = summarize_curves(curves)

    round_float_columns(runs).to_csv(out_dir / "threshold_runs.csv", index=False)
    round_float_columns(summary).to_csv(out_dir / "threshold_summary.csv", index=False)
    round_float_columns(curves).to_csv(out_dir / "threshold_curves.csv", index=False)
    round_float_columns(curve_summary).to_csv(out_dir / "threshold_curve_summary.csv", index=False)

    if args.make_plots:
        plot_threshold_score_curves(curve_summary, out_dir, summary)

    print("Saved:", out_dir / "threshold_runs.csv")
    print("Saved:", out_dir / "threshold_summary.csv")
    print("Saved:", out_dir / "threshold_curves.csv")
    print("Saved:", out_dir / "threshold_curve_summary.csv")
    if args.make_plots:
        print("Saved plots in:", out_dir / "plots" / "threshold_curves")


if __name__ == "__main__":
    main()
