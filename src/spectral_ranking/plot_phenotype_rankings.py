#!/usr/bin/env python3
"""Python replacement for plot_phenotype_rankings.R.

This script keeps the same CLI shape and output file names, but exact pixel-level
identity with ggplot2 is not possible across rendering engines.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    }
)

PREFERRED_PRS_METHOD_ORDER = [
    "C+T",
    "LDpred",
    "lassosum",
    "PRS-CS",
    "PRS-CS-auto",
    "SBayesR",
    "SCT",
    "DBSLMM",
    "LDpred2",
    "LDpred2-auto",
    "LDpred2-inf",
    "LDpred-funct",
    "lassosum2",
]
_PREFERRED_PRS_METHOD_SET = set(PREFERRED_PRS_METHOD_ORDER)

# CI forest / ranking plot accent colors
_CI_FOREST_RANK_POINT_COLOR = "#d26e66"
_CI_FOREST_RANK_LABEL_COLOR = "#6e80a2"

# Phenotype rank heatmap: warm orange → stronger blue (blue end tuned for cell legibility)
_HEATMAP_RANK_ORANGE = "#F29A3A"
_HEATMAP_RANK_BLUE = "#2563EB"

# Heatmap y-axis label in stratified combined export; keep in sync with compositing in generate_visualizations.
STRATIFIED_COMBINED_HEATMAP_YLABEL_FONTSIZE = 18
COMBINED_PANEL_FIGURE_DPI = 150


def _mm_to_pt(mm: float) -> float:
    """ggplot size/linewidth values are in mm; Matplotlib uses points."""
    return mm * 72.0 / 25.4


def _parse_args(argv: list[str]) -> dict[str, str | None]:
    data_path = str(Path("data") / "examples" / "example_data_multiway_phenotype.csv")
    out_dir = None
    out_path = None
    heatmap_out_path = None
    ci_plot_path = None
    ci_out_path = None
    bigbetter = "0"
    pre_ranked = "0"
    heatmap_ylabel = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--csv" and i + 1 < len(argv):
            data_path = argv[i + 1]
            i += 2
        elif arg == "--out-dir" and i + 1 < len(argv):
            out_dir = argv[i + 1]
            i += 2
        elif arg == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]
            i += 2
        elif arg == "--heatmap-out" and i + 1 < len(argv):
            heatmap_out_path = argv[i + 1]
            i += 2
        elif arg == "--heatmap-ylabel" and i + 1 < len(argv):
            heatmap_ylabel = argv[i + 1]
            i += 2
        elif arg == "--ci-plot" and i + 1 < len(argv):
            ci_plot_path = argv[i + 1]
            i += 2
        elif arg == "--ci-out" and i + 1 < len(argv):
            ci_out_path = argv[i + 1]
            i += 2
        elif arg == "--bigbetter" and i + 1 < len(argv):
            bigbetter = argv[i + 1]
            i += 2
        elif arg == "--pre-ranked" and i + 1 < len(argv):
            pre_ranked = argv[i + 1]
            i += 2
        else:
            i += 1

    return {
        "data_path": data_path,
        "out_dir": out_dir,
        "out_path": out_path,
        "heatmap_out_path": heatmap_out_path,
        "ci_plot_path": ci_plot_path,
        "ci_out_path": ci_out_path,
        "bigbetter": bigbetter,
        "pre_ranked": pre_ranked,
        "heatmap_ylabel": heatmap_ylabel,
    }


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _preferred_method_reorder_indices(
    methods: list[str],
    ranks: list[float] | None = None,
) -> list[int]:
    if not methods:
        return []

    index_by_method = {method: idx for idx, method in enumerate(methods)}
    preferred_indices = [index_by_method[method] for method in PREFERRED_PRS_METHOD_ORDER if method in index_by_method]
    if len(preferred_indices) < 2:
        if ranks is not None and len(ranks) == len(methods):
            return sorted(range(len(methods)), key=lambda i: (float(ranks[i]), i))
        return list(range(len(methods)))
    return preferred_indices + [idx for idx, method in enumerate(methods) if method not in _PREFERRED_PRS_METHOD_SET]


def _apply_preferred_method_order(methods: list[str]) -> list[str]:
    reorder = _preferred_method_reorder_indices(methods)
    return [methods[idx] for idx in reorder]


def _reorder_aligned(values: list[Any], order: list[int]) -> list[Any]:
    return [values[idx] for idx in order]


def _read_phenotype_csv(data_path: str) -> tuple[pd.DataFrame, list[str]]:
    df_wide = pd.read_csv(data_path)
    if df_wide.shape[1] < 2:
        raise ValueError("CSV must contain at least one trait column and one method column.")
    df_wide = df_wide.copy()
    cols = list(df_wide.columns)
    cols[0] = "Traits"
    df_wide.columns = cols
    method_cols = _apply_preferred_method_order([c for c in df_wide.columns if c != "Traits"])
    df_wide = df_wide[["Traits", *method_cols]]
    return df_wide, method_cols


def _parse_bigbetter(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    raise ValueError(f"--bigbetter must be 0 or 1, got: {value!r}")


def _rank_ascending_from_bigbetter(bigbetter: int) -> bool:
    # bigbetter=1 => higher raw values are better => descending rank => ascending=False.
    return int(bigbetter) != 1


def _parse_pre_ranked(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if text in {"0", "1"}:
        return text == "1"
    raise ValueError(f"--pre-ranked must be 0 or 1, got: {value!r}")


def _build_rank_table(df_wide: pd.DataFrame, *, bigbetter: int = 0, pre_ranked: bool = False) -> pd.DataFrame:
    table = df_wide.melt(id_vars=["Traits"], var_name="Method", value_name="Ranking")
    table["Ranking"] = pd.to_numeric(table["Ranking"], errors="coerce")
    table = table.dropna(subset=["Ranking"]).copy()
    if not pre_ranked:
        table["Ranking"] = table.groupby("Traits")["Ranking"].rank(
            method="average",
            ascending=_rank_ascending_from_bigbetter(bigbetter),
        )
    return table


def _row_rank_matrix(
    df_wide: pd.DataFrame,
    method_cols: list[str],
    *,
    bigbetter: int = 0,
    pre_ranked: bool = False,
) -> pd.DataFrame:
    numeric = df_wide[method_cols].apply(pd.to_numeric, errors="coerce")
    if pre_ranked:
        ranked = numeric
    else:
        ranked = numeric.rank(
            axis=1,
            method="average",
            na_option="keep",
            ascending=_rank_ascending_from_bigbetter(bigbetter),
        )
    ranked.index = df_wide["Traits"].astype(str)
    ranked = ranked[method_cols]
    return ranked


def _palette_martin(n: int) -> list[tuple[float, float, float]]:
    """Exact paletteMartin from colorBlindness::paletteMartin (R)."""
    base = [
        "#000000",
        "#004949",
        "#009292",
        "#FF6DB6",
        "#FFB6DB",
        "#490092",
        "#006DDB",
        "#B66DFF",
        "#6DB6FF",
        "#B6DBFF",
        "#920000",
        "#924900",
        "#DB6D00",
        "#24FF24",
    ]
    if n <= len(base):
        colors = base[:n]
    else:
        colors = [base[i % len(base)] for i in range(n)]
    return [matplotlib.colors.to_rgb(c) for c in colors]


def _save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150, *, tight: bool = True) -> None:
    _ensure_parent(path)
    save_kwargs: dict[str, Any] = {"dpi": dpi, "facecolor": "white"}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **save_kwargs)
    plt.close(fig)


def _style_axes_bw(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="black")


def _apply_ci_axis_labels(ax: plt.Axes) -> None:
    """X label matches tick label size/weight; no y-axis title (item names are tick labels)."""
    ax.set_xlabel("Rank", fontsize=12, color="black")
    ax.set_ylabel("")


def _plot_ci_forest(
    items: list[str],
    ranks: list[float],
    theta_hat: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    out_path: str,
    *,
    all_methods: list[str] | None = None,
) -> None:
    """Plot CI forest matching R ggplot2 exactly.
    If all_methods is given, y-axis shows all methods (R factor levels); only plot points for items.
    """
    y_labels = all_methods if all_methods is not None else items
    n_labels = len(y_labels)
    fig_h = max(5.0, 0.4 * max(1, n_labels))
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=150)

    if n_labels == 0:
        ax.axis("off")
        ax.set_title("No ranking data available", fontsize=14)
        _save_fig(fig, out_path, dpi=150)
        return

    # R: factor levels = rev(sorted_names) -> ggplot2 first level at bottom, last at top
    # So top = first of sorted_names (C+T), bottom = last (DBSLMM)
    # Matplotlib: y=1 at bottom by default. Use invert_yaxis so y=1 at top.
    # y_positions: top=1, bottom=n_labels
    y_pos_map = {m: i + 1 for i, m in enumerate(y_labels)}

    if not items:
        ax.set_xlim(0.5, 14.5)
        # ggplot2 discrete scale default expansion ~= add 0.6 on both sides
        ax.set_ylim(n_labels + 0.6, 0.4)
        ax.set_yticks(np.arange(1, n_labels + 1))
        ax.set_yticklabels(y_labels, fontsize=12, color="black")
        _apply_ci_axis_labels(ax)
        _style_axes_bw(ax)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(left=0.24, right=0.99, top=0.984, bottom=0.106)
        _save_fig(fig, out_path, dpi=150, tight=True)
        return

    ranks_arr = np.asarray(ranks, dtype=float)
    lower_arr = np.asarray(ci_lower, dtype=float)
    upper_arr = np.asarray(ci_upper, dtype=float)

    x_min = max(0.5, float(np.nanmin(lower_arr)) - 0.5)
    x_max = min(float(np.nanmax(upper_arr)) + 0.5, float(np.nanmax(ranks_arr)) + 2.0)
    x_breaks = [x for x in range(1, int(math.ceil(x_max)) + 1) if x_min <= x <= x_max]
    # R: scale_x_continuous expand = c(0.02, 0)
    x_span = x_max - x_min
    x_min_exp = x_min - 0.02 * x_span
    x_max_exp = x_max + 0.02 * x_span

    y_positions = np.array([y_pos_map[m] for m in items], dtype=float)

    # R: geom_vline color=grey85, linewidth=0.5
    for xb in x_breaks:
        ax.axvline(x=xb, color="#D9D9D9", linewidth=0.5, zorder=0)

    # R: geom_errorbar(..., width = 0.3, linewidth = 0.8, orientation = "y")
    # Draw manually so endpoint cap height is in data units (0.3), like ggplot2.
    cap_half_height = 0.15
    err_lw = _mm_to_pt(0.8)
    for x, y, lo, hi in zip(ranks_arr, y_positions, lower_arr, upper_arr):
        ax.hlines(y, lo, hi, colors="black", linewidth=err_lw, zorder=2)
        ax.vlines([lo, hi], y - cap_half_height, y + cap_half_height, colors="black", linewidth=err_lw, zorder=2)

    # R: geom_point(size = 3, shape = 21, stroke = 0, fill); brand palette
    ax.plot(
        ranks_arr,
        y_positions,
        linestyle="None",
        marker="o",
        markersize=_mm_to_pt(3.0),
        markerfacecolor=_CI_FOREST_RANK_POINT_COLOR,
        markeredgewidth=0,
        markeredgecolor=_CI_FOREST_RANK_POINT_COLOR,
        color=_CI_FOREST_RANK_POINT_COLOR,
        zorder=3,
    )

    # Rank digits above points: bold face (explicit FontProperties so Agg reliably uses a heavy face)
    rank_label_fp = FontProperties(
        family="sans-serif",
        weight="bold",
        size=_mm_to_pt(4.5),
    )
    # R: geom_text y=as.numeric(Method)+0.35 (above point); with invert, use y-0.35 for above
    for x, y in zip(ranks_arr, y_positions):
        ax.text(
            x,
            y - 0.35,
            f"{int(round(x))}",
            color=_CI_FOREST_RANK_LABEL_COLOR,
            fontproperties=rank_label_fp,
            ha="center",
            va="center",
            zorder=4,
        )

    ax.set_axisbelow(True)
    ax.set_xlim(x_min_exp, x_max_exp)
    # ggplot2 discrete y scale default expansion (add 0.6) to match row/grid placement exactly
    ax.set_ylim(n_labels + 0.6, 0.4)  # y=1 at top to match R (first item at top)
    ax.set_xticks(x_breaks)
    ax.set_yticks(np.arange(1, n_labels + 1))
    ax.set_yticklabels(y_labels, fontsize=12, color="black")
    _apply_ci_axis_labels(ax)
    ax.tick_params(axis="both", labelsize=12, colors="black")
    # R: panel.grid.major color=grey90, linewidth=0.3 (both x and y)
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.3, linestyle="-")
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.3, linestyle="-")
    _style_axes_bw(ax)
    fig.patch.set_facecolor("white")
    # Match ggplot2 panel placement from ggsave(width=10, height=max(5, 0.4*n), dpi=150).
    # Left margin increased to fit y-axis label.
    fig.subplots_adjust(left=0.24, right=0.99, top=0.984, bottom=0.106)
    _save_fig(fig, out_path, dpi=150, tight=True)


def _draw_half_violin(ax: plt.Axes, values: np.ndarray, pos: float, color: Any) -> None:
    if values.size == 0:
        return
    if values.size < 2 or np.allclose(values, values[0]):
        ax.plot([pos, pos + 0.33], [values[0], values[0]], color=color, alpha=0.25, linewidth=3, zorder=1)
        return
    parts = ax.violinplot(
        [values],
        positions=[pos + 0.15],
        widths=0.70,
        points=256,
        bw_method=lambda kde: float(kde.scotts_factor()) * 2.0,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0]
    body.set_facecolor(matplotlib.colors.to_rgba(color, alpha=0.25))
    body.set_edgecolor(matplotlib.colors.to_rgba("black", alpha=0.6))
    body.set_linewidth(1.2)
    body.set_zorder(1)
    # Clip to the right half to mimic gghalves::geom_half_violin(side = "r").
    clip = Rectangle((pos + 0.15, -1e6), 1e6, 2e6, transform=ax.transData)
    body.set_clip_path(clip)


def _shift_annotation_display(ax: plt.Axes, ann: matplotlib.text.Annotation, dx_px: float = 0.0, dy_px: float = 0.0) -> None:
    """Move an annotation text position by display-space pixels."""
    x_data, y_data = ann.xyann
    x_disp, y_disp = ax.transData.transform((x_data, y_data))
    x_new, y_new = x_disp + dx_px, y_disp + dy_px
    x_data_new, y_data_new = ax.transData.inverted().transform((x_new, y_new))
    ann.set_position((x_data_new, y_data_new))


def _repel_violin_mean_labels(
    fig: plt.Figure,
    ax: plt.Axes,
    anns: list[matplotlib.text.Annotation],
    anchors: list[tuple[float, float]],
) -> None:
    """Approximate ggrepel label placement for mean labels (no external dependency)."""
    if not anns:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    x_margin = 6.0
    y_margin = 5.0
    pair_pad_x = 10.0
    pair_pad_y = 7.0
    n = len(anns)
    sep_boost = 2.0 + min(4.0, 0.12 * max(0, n - 10))

    # First pass: choose left/right side if a label would overflow the panel.
    for ann, anchor in zip(anns, anchors):
        x_anchor_disp, y_anchor_disp = ax.transData.transform(anchor)
        bbox = ann.get_window_extent(renderer=renderer)
        if bbox.x1 > ax_bbox.x1 - x_margin:
            ann.set_ha("right")
            # Small gap from the point, similar to ggrepel's point.padding.
            x_data_new, y_data_new = ax.transData.inverted().transform((x_anchor_disp - 16.0, y_anchor_disp))
            ann.set_position((x_data_new, y_data_new))
        elif bbox.x0 < ax_bbox.x0 + x_margin:
            ann.set_ha("left")
            x_data_new, y_data_new = ax.transData.inverted().transform((x_anchor_disp + 16.0, y_anchor_disp))
            ann.set_position((x_data_new, y_data_new))

    # Iteratively resolve collisions by vertical nudging (display-space).
    for _ in range(140):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moved = False
        bboxes = [ann.get_window_extent(renderer=renderer) for ann in anns]

        # Keep labels inside the panel vertically and horizontally.
        for ann, bbox in zip(anns, bboxes):
            if bbox.y1 > ax_bbox.y1 - y_margin:
                _shift_annotation_display(ax, ann, dy_px=(ax_bbox.y1 - y_margin) - bbox.y1)
                moved = True
            if bbox.y0 < ax_bbox.y0 + y_margin:
                _shift_annotation_display(ax, ann, dy_px=(ax_bbox.y0 + y_margin) - bbox.y0)
                moved = True
            if bbox.x1 > ax_bbox.x1 - x_margin:
                _shift_annotation_display(ax, ann, dx_px=(ax_bbox.x1 - x_margin) - bbox.x1)
                moved = True
            if bbox.x0 < ax_bbox.x0 + x_margin:
                _shift_annotation_display(ax, ann, dx_px=(ax_bbox.x0 + x_margin) - bbox.x0)
                moved = True

        if moved:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bboxes = [ann.get_window_extent(renderer=renderer) for ann in anns]

        # Pairwise overlap resolution.
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                b1 = bboxes[i]
                b2 = bboxes[j]
                x_overlap = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                y_overlap = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if x_overlap > -pair_pad_x and y_overlap > -pair_pad_y:
                    ax_i, _ = anchors[i]
                    ax_j, _ = anchors[j]
                    # Adjacent violin columns with overlapping boxes: push apart horizontally first.
                    if abs(ax_i - ax_j) <= 1.25 and x_overlap > -pair_pad_x * 0.5:
                        hx = 10.0 + sep_boost
                        if ax_i <= ax_j:
                            _shift_annotation_display(ax, anns[i], dx_px=-hx)
                            _shift_annotation_display(ax, anns[j], dx_px=+hx)
                        else:
                            _shift_annotation_display(ax, anns[i], dx_px=+hx)
                            _shift_annotation_display(ax, anns[j], dx_px=-hx)
                        moved = True
                    else:
                        sep = (max(0.0, y_overlap) + pair_pad_y) / 2.0 + sep_boost
                        c1 = (b1.y0 + b1.y1) / 2.0
                        c2 = (b2.y0 + b2.y1) / 2.0
                        if c1 <= c2:
                            _shift_annotation_display(ax, anns[i], dy_px=-sep)
                            _shift_annotation_display(ax, anns[j], dy_px=+sep)
                        else:
                            _shift_annotation_display(ax, anns[i], dy_px=+sep)
                            _shift_annotation_display(ax, anns[j], dy_px=-sep)
                        moved = True

        if not moved:
            break


def _plot_violin(
    table: pd.DataFrame,
    method_cols: list[str],
    out_path: str,
    *,
    stratified_combined_panel: bool = False,
) -> None:
    n_methods = len(method_cols)
    n_rank_levels = max(n_methods, 2)
    rank_breaks = list(range(1, n_rank_levels + 1))

    # Taller canvas when there are many rank ticks (y-axis) to reduce crowding
    fig_h = max(7.0, 4.5 + 0.32 * float(n_rank_levels))
    fig_w = max(22.0, 10.0 + 0.48 * float(n_methods))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = _palette_martin(n_methods)
    rng = np.random.default_rng(42)
    mean_label_anns: list[matplotlib.text.Annotation] = []
    mean_label_anchors: list[tuple[float, float]] = []
    label_fs = _mm_to_pt(6.4) if n_methods > 16 else _mm_to_pt(7.0)

    # R: sample.size.label = "K = " -> each method shows (n = K)
    n_per_method = table.groupby("Method").size()
    for idx, method in enumerate(method_cols, start=1):
        vals = table.loc[table["Method"] == method, "Ranking"].dropna().to_numpy(dtype=float)
        color = colors[idx - 1]
        _draw_half_violin(ax, vals, float(idx), color)

        if vals.size:
            # ggbetweenstats default boxplot layer (kept centered; half violin is nudged right).
            bp = ax.boxplot(
                [vals],
                positions=[idx],
                widths=0.28,
                vert=True,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops=dict(facecolor=(1, 1, 1, 0.55), edgecolor="black", linewidth=1.2),
                whiskerprops=dict(color="black", linewidth=1.1),
                capprops=dict(color="black", linewidth=1.1),
                medianprops=dict(color="black", linewidth=1.2),
            )
            for artists in bp.values():
                if isinstance(artists, list):
                    for artist in artists:
                        artist.set_zorder(3)

            # R: point.args jitter.width=0.1, dodge.width=0.6
            jitter_x = idx + rng.uniform(-0.1, 0.1, size=vals.size)
            ax.scatter(jitter_x, vals, s=28, color=[color], alpha=0.8, linewidths=0, zorder=4)
            mean_rank = float(np.nanmean(vals))
            # Mean point + boxed repelled label (ggbetweenstats centrality label).
            ax.scatter(
                [idx],
                [mean_rank],
                s=72,
                facecolors="#B00000",
                edgecolors="#7A0000",
                linewidths=0.6,
                zorder=6,
            )
            # Alternate label side by column to reduce pairwise overlap when means align.
            side = 1 if idx % 2 == 1 else -1
            dx = (0.38 + (0.06 if n_methods > 16 else 0.0)) * float(side)
            ann = ax.annotate(
                rf"$\hat{{R}}_{{\mathrm{{mean}}}} = {mean_rank:.2f}$",
                xy=(idx, mean_rank),
                xytext=(float(idx) + dx, mean_rank),
                textcoords="data",
                fontsize=label_fs,
                color="black",
                ha="left" if side > 0 else "right",
                va="center",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=(1, 1, 1, 0.75), edgecolor="0.45", linewidth=0.8),
                arrowprops=dict(arrowstyle="-", color="0.35", linewidth=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False,
            )
            mean_label_anns.append(ann)
            mean_label_anchors.append((float(idx), mean_rank))

    ax.set_axisbelow(True)
    # Match ggplot2 scale_y_reverse default expansion (~5% of range).
    ax.set_ylim(n_rank_levels + 0.65, 0.35)
    ax.set_xticks(np.arange(1, n_methods + 1))
    # R: sample.size.label = "K = " -> "Item (n = K)" on x-axis
    x_labels = [f"{m}\n(n = {int(n_per_method.get(m, 0))})" for m in method_cols]
    if stratified_combined_panel:
        ax.set_xticklabels(
            x_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=17,
            color="black",
        )
        ax.set_ylabel("Rank", fontsize=23, color="black")
        ax.set_xlabel("")
        # Extra x-range so repelled mean labels at alternating sides stay in-frame.
        ax.set_xlim(0.35, float(n_methods) + 0.75)
    else:
        ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=17, color="black")
        ax.set_ylabel("Rank", fontsize=23, fontweight="bold", color="black")
        ax.set_xlabel("Item", fontsize=23, fontweight="bold", color="black")
        ax.set_xlim(0.5, n_methods + 0.6)

    ax.set_yticks(rank_breaks)
    ax.set_yticklabels(rank_breaks, fontsize=17, color="black")

    ax.grid(axis="both", which="major", color="#D9D9D9", linewidth=0.8)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.grid(axis="y", which="minor", color="#EFEFEF", linewidth=0.6)
    ax.tick_params(axis="y", which="minor", length=0)
    _style_axes_bw(ax)
    _repel_violin_mean_labels(fig, ax, mean_label_anns, mean_label_anchors)
    # Approximate ggplot margin(12,16,12,12); extra bottom room for 45° x labels on combined panel
    bottom_margin = 0.32 if stratified_combined_panel else 0.22
    fig.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=bottom_margin)
    _save_fig(fig, out_path, dpi=150, tight=False)


def _plot_heatmap(
    rank_matrix: pd.DataFrame,
    method_cols: list[str],
    out_path: str,
    *,
    stratified_combined_panel: bool = False,
    heatmap_row_axis_label: str | None = None,
) -> None:
    n_rank_levels = max(len(method_cols), 2)
    rank_breaks = list(range(1, n_rank_levels + 1))

    n_rows = len(rank_matrix.index)
    n_cols = len(method_cols)
    fig_h = max(5.2, min(32.0, 1.9 + 0.50 * float(n_rows)))
    fig_w = max(9.5, min(19.0, 6.2 + 0.23 * float(n_cols)))
    if stratified_combined_panel:
        # Keep panel B from collapsing when there are only a few row groups.
        # We enforce a square-ish minimum canvas so the heatmap body is not overly short.
        fig_h = max(fig_h, fig_w)
    if not stratified_combined_panel:
        fig_w = max(fig_w, 12.0)
        fig_h = max(fig_h, 8.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    data = rank_matrix.to_numpy(dtype=float)
    data_rev = data[::-1, :]
    y_labels = [str(x) for x in rank_matrix.index[::-1]]
    x_labels = method_cols

    cmap = LinearSegmentedColormap.from_list(
        "orange_blue_rank",
        [_HEATMAP_RANK_ORANGE, _HEATMAP_RANK_BLUE],
        N=max(n_rank_levels, 256),
    )
    cmap = cmap.copy()
    cmap.set_bad(color="darkgrey")
    masked = np.ma.masked_invalid(data_rev)

    im = ax.imshow(masked, cmap=cmap, vmin=1, vmax=n_rank_levels, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ytick_fs = 12 if n_rows > 36 else 14
    ax.set_xticklabels(x_labels, fontsize=14, color="black")
    ax.set_yticklabels(y_labels, fontsize=ytick_fs, color="black")
    # R: axis.text.x angle=315, hjust=0, vjust=1
    plt.setp(ax.get_xticklabels(), rotation=315, ha="left", va="top", rotation_mode="anchor")

    row_axis_title = heatmap_row_axis_label if heatmap_row_axis_label else "Phenotype"
    if stratified_combined_panel:
        ax.set_xlabel("")
        ax.set_ylabel(row_axis_title, fontsize=STRATIFIED_COMBINED_HEATMAP_YLABEL_FONTSIZE, color="black")
    else:
        ax.set_xlabel("Item", fontsize=18, fontweight="bold", color="black")
        ax.set_ylabel(row_axis_title, fontsize=STRATIFIED_COMBINED_HEATMAP_YLABEL_FONTSIZE, fontweight="bold", color="black")

    # R: geom_tile color="grey60", linewidth=0.2
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="0.6", linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="major", visible=False)

    _style_axes_bw(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_ticks(rank_breaks)
    cbar.ax.invert_yaxis()  # 1 (orange) at top, N (blue) at bottom
    cbar.ax.tick_params(labelsize=14, colors="black")
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("black")

    max_y_lab = max((len(lab) for lab in y_labels), default=10)
    left_margin = 0.16 + min(0.18, 0.0055 * max(0, max_y_lab - 10) + 0.012 * max(0, n_rows - 10))
    left_margin = min(0.36, left_margin)
    fig.subplots_adjust(left=left_margin, right=0.90, top=0.98, bottom=0.19)
    _save_fig(fig, out_path, dpi=150)


def _safe_json_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def _ci_from_json(ci_plot_path: str, ci_out_path: str) -> None:
    with open(ci_plot_path, "r", encoding="utf-8") as f:
        dat = json.load(f)

    items = [str(x) for x in _safe_json_list(dat, "items")]
    theta_hat = [float(x) for x in _safe_json_list(dat, "theta_hat")]
    ranks = [float(x) for x in _safe_json_list(dat, "ranks")]
    ci_lower = [float(x) for x in _safe_json_list(dat, "ci_lower")]
    ci_upper = [float(x) for x in _safe_json_list(dat, "ci_upper")]

    # Align lengths defensively to avoid malformed JSON crashes.
    n = min(len(items), len(theta_hat), len(ranks), len(ci_lower), len(ci_upper))
    items = items[:n]
    theta_hat = theta_hat[:n]
    ranks = ranks[:n]
    ci_lower = ci_lower[:n]
    ci_upper = ci_upper[:n]
    reorder = _preferred_method_reorder_indices(items, ranks=ranks)
    _plot_ci_forest(
        _reorder_aligned(items, reorder),
        _reorder_aligned(ranks, reorder),
        _reorder_aligned(theta_hat, reorder),
        _reorder_aligned(ci_lower, reorder),
        _reorder_aligned(ci_upper, reorder),
        ci_out_path,
    )
    print(f"Saved CI forest plot to {ci_out_path}")


def _ci_standalone_from_table(table: pd.DataFrame, method_cols: list[str], out_path: str) -> None:
    ordered_methods = _apply_preferred_method_order(method_cols)
    rank_summary = (
        table.groupby("Method", as_index=False)["Ranking"]
        .mean()
        .rename(columns={"Ranking": "mean_rank"})
    )
    rank_summary["rank"] = rank_summary["mean_rank"].rank(method="average", ascending=True)

    # Use original data order like the R script.
    order_map = {name: i for i, name in enumerate(ordered_methods)}
    rank_summary = rank_summary[rank_summary["Method"].isin(order_map)].copy()
    rank_summary["_ord"] = rank_summary["Method"].map(order_map)
    rank_summary = rank_summary.sort_values("_ord").drop(columns="_ord")

    n_m = max(1, len(rank_summary))
    ranks_ci = rank_summary["rank"].to_numpy(dtype=float)
    ci_lower_ci = np.maximum(1.0, ranks_ci - 1.2)
    ci_upper_ci = np.minimum(float(n_m), ranks_ci + 1.2)

    _plot_ci_forest(
        rank_summary["Method"].astype(str).tolist(),
        ranks_ci.tolist(),
        rank_summary["mean_rank"].to_numpy(dtype=float).tolist(),
        ci_lower_ci.tolist(),
        ci_upper_ci.tolist(),
        out_path,
        all_methods=ordered_methods,
    )
    print(f"Saved CI forest plot to {out_path}")


def main(argv: list[str] | None = None) -> int:
    opts = _parse_args(list(sys.argv[1:] if argv is None else argv))

    data_path = str(opts["data_path"])
    out_dir = Path(opts["out_dir"] or "output")
    violin_out_path = opts["out_path"]
    heatmap_out_path = opts["heatmap_out_path"]
    ci_plot_path = opts["ci_plot_path"]
    ci_out_path = opts["ci_out_path"]
    bigbetter = _parse_bigbetter(opts["bigbetter"])
    pre_ranked = _parse_pre_ranked(opts["pre_ranked"])
    heatmap_ylabel_raw = opts.get("heatmap_ylabel")
    heatmap_row_axis_label = str(heatmap_ylabel_raw) if heatmap_ylabel_raw else None

    # CI forest plot (all modes) - run and exit if requested.
    if ci_plot_path is not None and ci_out_path is not None:
        _ci_from_json(ci_plot_path, ci_out_path)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    df_wide, method_cols = _read_phenotype_csv(data_path)
    table = _build_rank_table(df_wide, bigbetter=bigbetter, pre_ranked=pre_ranked)
    rank_matrix = _row_rank_matrix(df_wide, method_cols, bigbetter=bigbetter, pre_ranked=pre_ranked)

    run_heatmaps = violin_out_path is None and heatmap_out_path is None
    if run_heatmaps:
        violin_out_path = str(out_dir / "violin_ranking_over_phenotypes.png")

    stratified_combined = violin_out_path is not None and heatmap_out_path is not None

    if violin_out_path is not None:
        if run_heatmaps:
            _plot_violin(table, method_cols, str(out_dir / "violin_ranking_over_phenotypes.pdf"))
        _plot_violin(
            table,
            method_cols,
            str(violin_out_path),
            stratified_combined_panel=stratified_combined,
        )
        print(f"Saved violin plot to {violin_out_path}")

    run_heatmap = run_heatmaps or heatmap_out_path is not None
    if run_heatmap:
        heatmap_png = str(heatmap_out_path) if heatmap_out_path is not None else str(out_dir / "heatmap_phenotype_rankings.png")
        if run_heatmaps:
            _plot_heatmap(
                rank_matrix,
                method_cols,
                str(out_dir / "heatmap_phenotype_rankings.pdf"),
                stratified_combined_panel=False,
                heatmap_row_axis_label=heatmap_row_axis_label,
            )
        _plot_heatmap(
            rank_matrix,
            method_cols,
            heatmap_png,
            stratified_combined_panel=stratified_combined,
            heatmap_row_axis_label=heatmap_row_axis_label,
        )
        print(f"Saved heatmap to {heatmap_png}")

    if run_heatmaps:
        _ci_standalone_from_table(table, method_cols, str(out_dir / "ci_forest_ranking.png"))

    print("Done. Check output/ for violin_ranking_over_phenotypes.*, heatmap_phenotype_rankings.*, and ci_forest_ranking.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
