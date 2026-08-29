from __future__ import annotations

# General
from pathlib import Path
import numpy as np
import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
        "val_mean": "Średnia metryka walidacyjna (val_mean) [-]",
        "acc": "Accuracy [-]",
        "f1_macro": "F1 macro [-]",
        "auc": "AUC [-]",
        "ap": "Average Precision [-]",
        "train_loss": "Strata treningowa [-]",
    }

    param_label_map = {
        "init_size": "rozmiar zbioru początkowego",
        "init_size_pct": "rozmiar zbioru początkowego",
        "batch": "rozmiar wsadu anotacyjnego",
        "batch_pct_of_budget": "rozmiar wsadu anotacyjnego",
        "budget": "budżet anotacji",
        "budget_pct": "budżet anotacji",
        "epc": "liczba epok na cykl",
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

        x_max = max(x_all)
        start_markers = []
        end_markers = []
        batch_step_markers = []

        def curve_value_label(value) -> str:
            if isinstance(value, (int, float, np.floating)):
                text = f"{value:g}"
            else:
                text = str(value)

            if param_col in {"init_size_pct", "batch_pct_of_budget", "budget_pct"}:
                return f"{text}%"
            return text

        for val in values:
            sub = g[g[param_col] == val].copy()

            avg = (
                sub.groupby("labeled_count", as_index=False)[metric]
                .mean()
                .sort_values("labeled_count")
            )

            if len(avg) == 0:
                continue

            val_txt = curve_value_label(val)

            (line,) = ax.plot(
                avg["labeled_count"],
                avg[metric],
                linewidth=2.0,
                marker="o",
                markersize=4.0,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=val_txt,
            )

            x_start = int(avg["labeled_count"].min())
            start_markers.append((x_start, f"{val_txt} = {x_start}", line.get_color()))

            x_end = int(avg["labeled_count"].max())
            if param_col in {"budget", "budget_pct"}:
                end_markers.append((x_end, f"{val_txt} (+{x_end - x_start})", line.get_color()))

            if param_col in {"batch", "batch_pct_of_budget"}:
                xs = [int(x) for x in avg["labeled_count"].tolist()]
                first_after_start = next((x for x in xs if x > x_start), None)
                if first_after_start is not None:
                    batch_abs = first_after_start - x_start
                    batch_step_markers.append(
                        (x_start, first_after_start, f"+{val_txt} (+{batch_abs})", line.get_color())
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
            f"{param_label_map.get(param_col, param_col).capitalize()}: przykładowa krzywa uczenia dla ustalonej konfiguracji"
        )

        ax.set_xlabel("Liczba oznaczonych próbek [-]")
        ax.set_ylabel(ylabel_map.get(metric, metric))
        ax.grid(alpha=0.3)

        show_start_markers = param_col in {"init_size", "init_size_pct"}
        if show_start_markers:
            y_min, y_top = ax.get_ylim()
            y_span = y_top - y_min
            for i, (x_start, label, color) in enumerate(start_markers):
                y_text = y_top - (0.025 + 0.055 * i) * y_span
                ax.axvline(
                    x_start,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                )
                ax.text(
                    x_start,
                    y_text,
                    f" {label}",
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                )

        show_end_markers = param_col in {"budget", "budget_pct"}
        if show_end_markers:
            y_min, y_top = ax.get_ylim()
            y_span = y_top - y_min
            for i, (x_end, label, color) in enumerate(end_markers):
                y_text = y_top - 0.35 * y_span
                is_last_marker = i == len(end_markers) - 1
                ax.axvline(
                    x_end,
                    linestyle=":",
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                )
                ax.text(
                    x_end,
                    y_text,
                    f"{label} " if is_last_marker else f" {label}",
                    ha="right" if is_last_marker else "left",
                    va="top",
                    fontsize=9,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                )

        if len(x_all) <= 10:
            xticks = x_all
        else:
            step = max(1, len(x_all) // 8)
            xticks = x_all[::step]
            if show_start_markers:
                for x_start, _, _ in start_markers:
                    if x_start not in xticks:
                        xticks = [x_start] + xticks
            if show_end_markers:
                for x_end, _, _ in end_markers:
                    if x_end not in xticks:
                        xticks = xticks + [x_end]
            if x_max not in xticks:
                xticks = xticks + [x_max]
            xticks = sorted(set(int(x) for x in xticks))

        show_batch_markers = param_col in {"batch", "batch_pct_of_budget"}
        if show_batch_markers and batch_step_markers:
            y_min, y_top = ax.get_ylim()
            y_span = y_top - y_min
            common_start = min(x_start for x_start, _, _, _ in batch_step_markers)
            marker_levels = [
                y_top - 0.16 * y_span,
                y_min + 0.18 * y_span,
                y_min + 0.27 * y_span,
            ]
            text_offsets = [
                0.015 * y_span,
                -0.035 * y_span,
                -0.035 * y_span,
            ]

            ax.axvline(
                common_start,
                linestyle="--",
                linewidth=1.5,
                color="black",
                alpha=0.85,
            )
            ax.text(
                common_start,
                y_top - 0.035 * y_span,
                f"start = {common_start}",
                ha="left",
                va="top",
                fontsize=9,
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
            )

            for i, (x_start, x_step, label, color) in enumerate(batch_step_markers):
                ax.axvline(
                    x_step,
                    linestyle=":",
                    linewidth=1.5,
                    color=color,
                    alpha=0.9,
                )

                local_y = marker_levels[min(i, len(marker_levels) - 1)]
                ax.annotate(
                    "",
                    xy=(x_step, local_y),
                    xytext=(x_start, local_y),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color=color,
                        linewidth=1.2,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )
                ax.text(
                    x_step,
                    local_y + text_offsets[min(i, len(text_offsets) - 1)],
                    f" {label}",
                    ha="left",
                    va="bottom" if i == 0 else "top",
                    fontsize=9,
                    color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5),
                )

        if show_end_markers or show_batch_markers:
            protected = {int(x_end) for x_end, _, _ in end_markers}
            if show_batch_markers:
                protected.update(int(x_start) for x_start, _, _, _ in batch_step_markers)
            min_gap = max(1, int(0.04 * (max(x_all) - min(x_all))))
            filtered_desc: list[int] = []

            for tick in sorted((int(x) for x in xticks), reverse=True):
                if tick in protected:
                    filtered_desc.append(tick)
                    continue

                if all(abs(tick - kept) >= min_gap for kept in filtered_desc):
                    filtered_desc.append(tick)

            xticks = sorted(filtered_desc)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks])
        for tick in ax.get_xticklabels():
            tick.set_fontweight("normal")

        legend_loc = "upper left" if show_end_markers else "best"
        ax.legend(framealpha=0.95, loc=legend_loc)

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
        "delta_aulc_norm": "Różnica AULC (znormalizowanego) [-]",
        "delta_last_iteration_model": "Różnica wyniku walidacyjnego [-]",
        "delta_final_test_model": "Różnica średniej z metryk testowych [-]",
    }

    metric_titles = {
        "delta_aulc_norm": "rozkład różnic AULC",
        "delta_last_iteration_model": "rozkład różnic wyniku walidacyjnego",
        "delta_final_test_model": "rozkład różnic średniej z metryk testowych",
    }

    winrate_titles = {
        "delta_aulc_norm": "bilans porównań AULC",
        "delta_last_iteration_model": "bilans porównań wyniku walidacyjnego",
        "delta_final_test_model": "bilans porównań średniej z metryk testowych",
    }

    param_labels = {
        "batch": "rozmiar wsadu anotacyjnego",
        "budget": "budżet anotacji",
        "epc": "liczba epok na cykl",
        "init_size": "rozmiar zbioru początkowego",
    }

    def value_label(value) -> str:
        if isinstance(value, (int, float, np.floating)):
            text = f"{value:g}"
        else:
            text = str(value)

        if pretty_name in {"batch", "budget", "init_size"}:
            return f"{text}%"
        return text

    color_win_v1 = "tab:green"
    color_tie = "lightgray"
    color_win_v2 = "tab:red"
    alpha_fill = 0.9

    for comparison, pw_cmp in pw.groupby("comparison"):
        v1 = pw_cmp["v1"].iloc[0]
        v2 = pw_cmp["v2"].iloc[0]

        v1_txt = value_label(v1)
        v2_txt = value_label(v2)

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
            if n_neg > 0 and len(neg_edges) < 2:
                neg_edges = [float(neg.min()), -eps]
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
            if n_pos > 0 and len(pos_edges) < 2:
                pos_edges = [eps, float(pos.max())]
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
                f"{param_labels.get(pretty_name, pretty_name).capitalize()}:\n{metric_titles[col]}"
            )
            ax_hist.set_xlabel(metric_labels[col])
            ax_hist.set_ylabel("Liczba par [-]")
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
                f"{param_labels.get(pretty_name, pretty_name).capitalize()}:\n{winrate_titles[col]}"
            )
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(categories, rotation=20, ha="right")
            ax_bar.set_ylabel("Liczba par [-]")
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

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]
    x_pos = np.arange(len(labels))

    xlabel_map = {
        "batch": "Rozmiar wsadu anotacyjnego [% budżetu]",
        "budget": "Budżet anotacji [% zbioru treningowego]",
        "epc": "Liczba epok na cykl [-]",
        "init_size": "Rozmiar zbioru początkowego [% zbioru treningowego]",
    }

    title_aulc_map = {
        "batch": "Wpływ rozmiaru wsadu anotacyjnego na przebieg aktywnego uczenia",
        "budget": "Wpływ budżetu anotacji na przebieg aktywnego uczenia",
        "epc": "Wpływ liczby epok na cykl na przebieg aktywnego uczenia",
        "init_size": "Wpływ rozmiaru zbioru początkowego na przebieg aktywnego uczenia",
    }

    title_test_map = {
        "batch": "Wpływ rozmiaru wsadu anotacyjnego na wynik testowy",
        "budget": "Wpływ budżetu anotacji na wynik testowy",
        "epc": "Wpływ liczby epok na cykl na wynik testowy",
        "init_size": "Wpływ rozmiaru zbioru początkowego na wynik testowy",
    }

    ylim_map = {
        "batch": (0.84, 0.885),
        "budget": (0.835, 0.885),
        "epc": (0.83, 0.895),
        "init_size": (0.835, 0.885),
    }

    test_ylim_map = {
        "batch": (0.92, 0.945),
        "budget": (0.908, 0.952),
        "epc": (0.92, 0.945),
        "init_size": (0.925, 0.942),
    }

    def draw_bar_chart(
        *,
        metric_col: str,
        title: str,
        ylabel: str,
        out_name: str,
        ylim: tuple[float, float] | None,
    ) -> None:
        if metric_col not in plot_df.columns:
            return

        y = pd.to_numeric(plot_df[metric_col], errors="coerce").to_numpy()
        if len(y) == 0 or np.all(~np.isfinite(y)):
            return

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        ax.bar(
            x_pos,
            y,
            color=colors,
            width=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)

        ax.set_title(title)
        ax.set_xlabel(xlabel_map.get(pretty_name, pretty_name))
        ax.set_ylabel(ylabel)

        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            maybe_set_zoomed_yaxis(ax, y)

        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))

        ax.grid(axis="y", which="major", alpha=0.35)
        ax.grid(axis="y", which="minor", alpha=0.15)

        fig.tight_layout()
        fig.savefig(main_effects_dir / out_name, dpi=180)
        plt.close(fig)

    draw_bar_chart(
        metric_col="mean_aulc_norm",
        title=title_aulc_map.get(pretty_name, f"Wpływ parametru {pretty_name} na przebieg aktywnego uczenia"),
        ylabel="Średni AULC (znormalizowany) [-]",
        out_name=f"main_effects_{pretty_name}.png",
        ylim=ylim_map.get(pretty_name, (0.8, 0.9)),
    )

    draw_bar_chart(
        metric_col="mean_final_test_model",
        title=title_test_map.get(pretty_name, f"Wpływ parametru {pretty_name} na wynik testowy"),
        ylabel="Średnia metryka testowa [-]",
        out_name=f"main_effects_{pretty_name}_test.png",
        ylim=test_ylim_map.get(pretty_name),
    )

def plot_supervised_confusion_matrices(
    confusion_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if len(confusion_df) == 0:
        return

    required_cols = ["dataset", "tn", "fp", "fn", "tp"]
    missing = [c for c in required_cols if c not in confusion_df.columns]
    if missing:
        raise ValueError(f"Missing required confusion matrix columns: {missing}")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = confusion_df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df = df.sort_values("dataset").reset_index(drop=True)

    dataset_label_map = {
        "octmnist": "OCTMNIST",
        "pathmnist": "PathMNIST",
        "pneumoniamnist": "PneumoniaMNIST",
    }

    for row in df.to_dict(orient="records"):
        matrix = np.array(
            [
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ]
        )

        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        max_count = float(matrix.max())
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)

        dataset = str(row["dataset"])
        ax.set_title(dataset_label_map.get(dataset, dataset))
        ax.set_xlabel("Klasa przewidziana")
        ax.set_ylabel("Klasa rzeczywista")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        threshold = max_count / 2.0
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=11,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots_dir / f"confusion_matrix_{dataset}.png", dpi=180)
        plt.close(fig)

# run_experiments.py
STRATEGY_LABELS = {
    "random": "Random Sampling",
    "least_confident": "Least confident",
    "margin": "Margin Sampling",
    "entropy": "Entropy Sampling",
    "mc_entropy": "MC Entropy",
    "mc_bald": "MC BALD",
    "entropy_diverse": "Entropy Sampling Diverse",
    "mc_entropy_diverse": "MC Entropy Diverse",
    "mc_bald_diverse": "MC BALD Diverse",
    "egl_fc": "Expected Gradient Length",
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

def compute_best_labeled_count_by_strategy(df_val: pd.DataFrame) -> dict[str, int]:
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

def plot_al_metric_by_strategy(
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


# analyze_active_strategies.py
DATASET_LABELS = {
    "octmnist": "OCTMNIST",
    "pathmnist": "PathMNIST",
    "pneumoniamnist": "PneumoniaMNIST",
    "bloodmnist": "BloodMNIST",
}

METRIC_LABELS = {
    "acc": "Accuracy [-]",
    "f1_macro": "$F_1$ macro [-]",
    "auc": "ROC AUC [-]",
    "ap": "Average Precision [-]",
    "train_loss": "Strata treningowa [-]",
}

METRIC_TITLES = {
    "acc": "Accuracy",
    "f1_macro": "$F_1$ macro",
    "auc": "ROC AUC",
    "ap": "Average Precision",
}


def _mean_std_by_labeled_count(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp["labeled_count"] = pd.to_numeric(tmp["labeled_count"], errors="coerce")
    tmp = tmp.dropna(subset=["labeled_count", metric])

    return (
        tmp.groupby("labeled_count", as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"))
        .sort_values("labeled_count")
    )


def _as_float(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return float("nan")
    return float(numeric)


def _set_endpoint_xticks(ax, x_values) -> None:
    vals = pd.to_numeric(pd.Series(x_values), errors="coerce").dropna()
    if len(vals) == 0:
        return

    x_min = int(vals.min())
    x_max = int(vals.max())

    ax.figure.canvas.draw()
    ticks = [
        int(round(float(tick)))
        for tick in ax.get_xticks()
        if np.isfinite(tick) and x_min < float(tick) < x_max
    ]
    ticks = sorted(set([x_min, *ticks, x_max]))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])


def plot_active_strategy_validation_metric(
    df_val: pd.DataFrame,
    dataset: str,
    metric: str,
    out_path: Path,
    color_map: dict[str, str],
    baseline: float | None = None,
    best_labeled_count_by_strategy: dict[str, int] | None = None,
) -> None:
    if len(df_val) == 0:
        return

    fig, ax = plt.subplots(figsize=(11.6, 5.8))

    for strategy, g in df_val.groupby("strategy", sort=True):
        strategy = str(strategy)
        curve = _mean_std_by_labeled_count(g, metric)
        if len(curve) == 0:
            continue

        x = curve["labeled_count"].to_numpy()
        y = curve["mean"].to_numpy()
        color = color_map.get(strategy)

        (line,) = ax.plot(
            x,
            y,
            linewidth=1.9,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=color,
        )

        if best_labeled_count_by_strategy:
            best_x = best_labeled_count_by_strategy.get(strategy)
            if best_x is not None:
                best_rows = curve[curve["labeled_count"] == best_x]
                if len(best_rows) > 0:
                    best_y = _as_float(best_rows.iloc[0]["mean"])
                    ax.plot(
                        [best_x],
                        [best_y],
                        marker="x",
                        markersize=8.5,
                        mew=2.0,
                        linestyle="None",
                        color=line.get_color(),
                    )

    if baseline is not None and np.isfinite(baseline):
        baseline = float(baseline)
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.35,
            color="black",
            alpha=0.8,
        )
        ax.annotate(
            f"Supervised = {baseline:.4f}",
            xy=(0.01, baseline),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, -5),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )

    dataset_label = DATASET_LABELS.get(str(dataset), str(dataset))
    ax.set_title(
        f"{dataset_label}: przebieg metryki walidacyjnej "
        f"{METRIC_TITLES.get(metric, metric)}"
    )
    ax.set_xlabel("Liczba oznaczonych próbek [-]")
    ax.set_ylabel("Wartość metryki [-]")
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    _set_endpoint_xticks(ax, df_val["labeled_count"])
    ax.legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_active_strategy_train_loss(
    df_val: pd.DataFrame,
    dataset: str,
    out_path: Path,
    color_map: dict[str, str],
) -> None:
    if len(df_val) == 0 or "train_loss" not in df_val.columns:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.4))

    for strategy, g in df_val.groupby("strategy", sort=True):
        strategy = str(strategy)
        curve = _mean_std_by_labeled_count(g, "train_loss")
        if len(curve) == 0:
            continue

        x = curve["labeled_count"].to_numpy()
        y = curve["mean"].to_numpy()
        color = color_map.get(strategy)

        ax.plot(
            x,
            y,
            linewidth=1.9,
            label=STRATEGY_LABELS.get(strategy, strategy),
            color=color,
        )

    dataset_label = DATASET_LABELS.get(str(dataset), str(dataset))
    ax.set_title(f"{dataset_label}: przebieg straty treningowej")
    ax.set_xlabel("Liczba oznaczonych próbek [-]")
    ax.set_ylabel(METRIC_LABELS["train_loss"])
    ax.grid(True, alpha=0.25)
    _set_endpoint_xticks(ax, df_val["labeled_count"])
    ax.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_active_strategy_confusion_matrices(
    confusion_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    if len(confusion_df) == 0:
        return

    required_cols = ["dataset", "strategy", "tn", "fp", "fn", "tp"]
    missing = [c for c in required_cols if c not in confusion_df.columns]
    if missing:
        raise ValueError(f"Missing required confusion matrix columns: {missing}")

    plots_dir = out_dir / "plots" / "confusion_matrices"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = confusion_df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["strategy"] = df["strategy"].astype(str)
    df = df.sort_values(["dataset", "strategy"]).reset_index(drop=True)

    for row in df.to_dict(orient="records"):
        matrix = np.array(
            [
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ]
        )

        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        max_count = float(matrix.max())
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_count)

        dataset = str(row["dataset"])
        strategy = str(row["strategy"])
        title = (
            f"{DATASET_LABELS.get(dataset, dataset)}\n"
            f"Strategia: {STRATEGY_LABELS.get(strategy, strategy)}"
        )
        ax.set_title(title)
        ax.set_xlabel("Klasa przewidziana")
        ax.set_ylabel("Klasa rzeczywista")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        threshold = max_count / 2.0
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=11,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        dataset_part = sanitize_filename_part(dataset)
        strategy_part = sanitize_filename_part(strategy)
        fig.savefig(plots_dir / f"confusion_matrix_{dataset_part}_{strategy_part}.png", dpi=180)
        plt.close(fig)
