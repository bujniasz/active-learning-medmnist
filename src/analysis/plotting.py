from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

# analyze_active_screening.py
def sanitize_filename_part(x) -> str:
    s = str(x)
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)

def sorted_unique_numeric_values(df: pd.DataFrame, col: str) -> list[float]:
    s = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    vals = s.drop_duplicates().to_list()
    vals.sort()
    return vals

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

    if abs_col == "batch":
        ax.set_title("Wpływ nominalnego batcha (wartości absolutne)")
        ax.set_xlabel("Nominalny batch – zwykle rozmiar pierwszych cykli")
    else:
        ax.set_title(f"Wpływ parametru {pretty_name} (wartości absolutne)")
        ax.set_xlabel(f"{pretty_name} – wartość absolutna")

    ax.set_ylabel("Średni AULC (znormalizowany)")

    ax.grid(axis="y", alpha=0.3)
    maybe_set_zoomed_yaxis(ax, y_vals)

    fig.tight_layout()
    fig.savefig(aux_dir / f"{pretty_name}_absolute_colored.png", dpi=180)
    plt.close(fig)

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

    df = df[(df["split"] == "val") & (df["step_type"] == "cycle")].copy()
    if len(df) == 0:
        return

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

        if not isinstance(key, tuple):
            key = (key,)
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
        "delta_last_iteration_model": "Różnica wyniku modelu z ostatniej iteracji",
        "delta_final_test_model": "Różnica wyniku finalnego modelu testowego",
    }

    metric_titles = {
        "delta_aulc_norm": "Rozkład różnic AULC",
        "delta_last_iteration_model": "Rozkład różnic modelu z ostatniej iteracji",
        "delta_final_test_model": "Rozkład różnic finalnego modelu testowego",
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
            ("delta_last_iteration_model", "last-iteration-model"),
            ("delta_final_test_model", "final-test-model"),
        ]:
            if col not in pw_cmp.columns:
                continue

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

            if xmin == xmax:
                xmin -= eps
                xmax += eps

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