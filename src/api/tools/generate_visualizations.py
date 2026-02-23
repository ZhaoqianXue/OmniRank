"""Tool: generate_visualizations."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.r_executor import RScriptExecutor
from core.schemas import EngineConfig, PlotSpec, RankingMode, RankingResults, VisualizationOutput


def _stable_block_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _fmt_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "0"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _fmt_rank_bound(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1e-9:
        return str(rounded)
    return _fmt_number(value, digits=2)


def _ordered_indices(results: RankingResults) -> list[int]:
    return sorted(
        range(len(results.items)),
        key=lambda idx: (results.ranks[idx], -results.theta_hat[idx], results.items[idx]),
    )


def _linear_scale(
    domain_min: float,
    domain_max: float,
    range_min: float,
    range_max: float,
):
    span = domain_max - domain_min
    if abs(span) < 1e-12:
        span = 1.0

    def _map(value: float) -> float:
        ratio = (value - domain_min) / span
        return range_min + ratio * (range_max - range_min)

    return _map


def _rank_color(rank: int, total: int) -> str:
    if total <= 1:
        return "#2b6cb0"
    ratio = (rank - 1) / (total - 1)
    start = (33, 102, 172)
    end = (120, 168, 219)
    red = int(round(start[0] + ratio * (end[0] - start[0])))
    green = int(round(start[1] + ratio * (end[1] - start[1])))
    blue = int(round(start[2] + ratio * (end[2] - start[2])))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _render_svg(width: int, height: int, elements: list[str], background: str = "#070e19") -> str:
    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="OmniRank visualization">\n'
        "  <style>text { font-family: Arial, sans-serif; font-weight: 700; }</style>\n"
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="{background}" />\n'
        f"  {body}\n"
        "</svg>\n"
    )


def _write_svg(svg_path: Path, svg_text: str) -> None:
    svg_path.write_text(svg_text, encoding="utf-8")


def _empty_plot_svg(message: str, width: int = 840, height: int = 220) -> str:
    safe_message = _xml_escape(message)
    elements = [
        (
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            f'font-size="14" fill="#f8fbff">{safe_message}</text>'
        ),
    ]
    return _render_svg(width=width, height=height, elements=elements)


def _indicator_label(indicator_col: str) -> tuple[str, str]:
    """Return (title_case, plural) for indicator column name."""
    if not indicator_col:
        return "Indicator", "Indicators"
    tc = indicator_col[0].upper() + indicator_col[1:].lower()
    plural = tc + ("es" if tc.endswith("s") else "s")
    return tc, plural


def _ci_forest_r(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    """Generate Ranking Confidence Interval Plot using R script."""
    payload = (
        "|".join(results.items)
        + ":"
        + "|".join(f"{r:.6f}" for r in results.ranks)
        + ":"
        + "|".join(f"{lo:.6f}-{hi:.6f}" for lo, hi in zip(results.ci_lower, results.ci_upper, strict=True))
    )
    block_id = _stable_block_id("figure-ci-forest", payload)
    png_path = artifact_dir / f"{block_id}.png"

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    r_script = project_root / "scripts" / "plot_phenotype_rankings.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plot script not found: {r_script}")

    ci_data = {
        "items": results.items,
        "theta_hat": results.theta_hat,
        "ranks": results.ranks,
        "ci_lower": results.ci_lower,
        "ci_upper": results.ci_upper,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ci_data, f, indent=None)
        json_path = Path(f.name)

    try:
        result = subprocess.run(  # noqa: S603
            ["Rscript", str(r_script), "--ci-plot", str(json_path), "--ci-out", str(png_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"R CI plot script failed: {result.stderr or result.stdout or 'unknown error'}"
            )
        if not png_path.exists():
            raise RuntimeError(f"R CI plot script did not produce output: {png_path}")
    finally:
        json_path.unlink(missing_ok=True)

    return PlotSpec(
        type="ci_forest",
        data={
            "names": results.items,
            "rank_point": results.ranks,
            "theta_hat": results.theta_hat,
            "ci_lower": results.ci_lower,
            "ci_upper": results.ci_upper,
        },
        config={"x_label": "rank", "point": "rank", "interval": "rank_ci", "source": "r"},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain="Ranking Confidence Interval Plot",
        caption_academic="Forest plot of 95% rank confidence intervals with rank point estimates.",
        hint_ids=["hint-ci"],
    )


def _normalized_ranking_over_indicator_r(
    csv_path: str,
    artifact_dir: Path,
    indicator_col: str = "phenotype",
) -> PlotSpec:
    """Generate Normalized Ranking Over Individual [Indicators] using R script."""
    source = Path(csv_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    r_script = project_root / "scripts" / "plot_phenotype_rankings.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plot script not found: {r_script}")

    block_id = _stable_block_id("figure-normalized-ranking-over-indicator", str(source))
    png_path = artifact_dir / f"{block_id}.png"

    result = subprocess.run(  # noqa: S603
        ["Rscript", str(r_script), "--csv", str(source), "--out", str(png_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R plot script failed: {result.stderr or result.stdout or 'unknown error'}"
        )
    if not png_path.exists():
        raise RuntimeError(f"R plot script did not produce output: {png_path}")

    _, plural = _indicator_label(indicator_col)
    caption = f"Normalized Ranking Over Individual {plural}"
    return PlotSpec(
        type="normalized_ranking_over_indicator",
        data={"indicator_col": indicator_col},
        config={"x_label": "method", "y_label": "normalized_rank", "source": "r"},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain=caption,
        caption_academic=(
            f"Per-indicator normalized rank distributions across methods; "
            "box-and-whisker summaries and mean markers quantify stability and coverage."
        ),
        hint_ids=["hint-ci"],
    )


def _indicator_rankings_heatmap_r(
    csv_path: str,
    artifact_dir: Path,
    indicator_col: str = "phenotype",
) -> PlotSpec:
    """Generate [Indicator] Rankings heatmap using R script."""
    source = Path(csv_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    r_script = project_root / "scripts" / "plot_phenotype_rankings.R"
    if not r_script.exists():
        raise FileNotFoundError(f"R plot script not found: {r_script}")

    block_id = _stable_block_id("figure-indicator-rankings-heatmap", str(source))
    png_path = artifact_dir / f"{block_id}.png"

    result = subprocess.run(  # noqa: S603
        ["Rscript", str(r_script), "--csv", str(source), "--heatmap-out", str(png_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R heatmap script failed: {result.stderr or result.stdout or 'unknown error'}"
        )
    if not png_path.exists():
        raise RuntimeError(f"R heatmap script did not produce output: {png_path}")

    tc, _ = _indicator_label(indicator_col)
    caption = f"{tc} Rankings"
    return PlotSpec(
        type="indicator_rankings_heatmap",
        data={"indicator_col": indicator_col},
        config={"source": "r"},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain=caption,
        caption_academic=(
            f"Per-{indicator_col.lower()} ranking heatmap across methods; "
            "orange indicates better (lower) rank, blue indicates worse (higher) rank."
        ),
        hint_ids=["hint-ci"],
    )


def _ci_forest(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    order = _ordered_indices(results)
    names = [results.items[i] for i in order]
    scores = [results.theta_hat[i] for i in order]
    ranks = [results.ranks[i] for i in order]
    lower = [results.ci_lower[i] for i in order]
    upper = [results.ci_upper[i] for i in order]

    payload = (
        "|".join(names)
        + ":"
        + "|".join(f"{rank:.6f}" for rank in ranks)
        + ":"
        + "|".join(f"{lo:.6f}-{hi:.6f}" for lo, hi in zip(lower, upper, strict=True))
    )
    block_id = _stable_block_id("figure-ci-forest", payload)
    svg_path = artifact_dir / f"{block_id}.svg"

    if not names:
        _write_svg(
            svg_path,
            _empty_plot_svg(
                message="No ranking data available",
            ),
        )
        return PlotSpec(
            type="ci_forest",
            data={"names": [], "rank_point": [], "ci_lower": [], "ci_upper": [], "theta_hat": []},
            config={"x_label": "rank", "point": "rank", "interval": "rank_ci"},
            svg_path=str(svg_path),
            block_id=block_id,
            caption_plain="Lines show rank uncertainty intervals; points show inferred rank.",
            caption_academic="Forest plot of 95% rank confidence intervals with rank point estimates.",
            hint_ids=["hint-ci"],
        )

    max_name_len = max(len(name) for name in names)
    left_margin = min(360, max(190, 48 + max_name_len * 7))
    right_margin = 96
    plot_width = 700
    width = left_margin + right_margin + plot_width
    row_height = 38
    top_margin = 30
    bottom_margin = 64
    height = top_margin + bottom_margin + row_height * len(names)

    plot_left = float(left_margin)
    plot_right = float(width - right_margin)
    plot_top = float(top_margin)
    plot_bottom = float(height - bottom_margin)

    min_rank = min(min(lower), min(ranks), 1.0)
    max_rank = max(max(upper), max(ranks), float(len(names)))
    domain_min = min_rank - 0.5
    domain_max = max_rank + 0.5
    x_to_px = _linear_scale(domain_min, domain_max, plot_left, plot_right)

    tick_start = int(math.floor(domain_min))
    tick_end = int(math.ceil(domain_max))

    elements: list[str] = [
        (
            f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" '
            f'y2="{plot_bottom:.2f}" stroke="#4a5568" stroke-width="1.2" />'
        ),
    ]

    for tick in range(tick_start, tick_end + 1):
        if tick < domain_min or tick > domain_max:
            continue
        x_pos = x_to_px(float(tick))
        elements.append(
            (
                f'<line x1="{x_pos:.2f}" y1="{plot_top:.2f}" x2="{x_pos:.2f}" y2="{plot_bottom:.2f}" '
                f'stroke="#e2e8f0" stroke-width="1" />'
            )
        )
        elements.append(
            (
                f'<text x="{x_pos:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" '
                f'font-family="Arial, sans-serif" fill="#4a5568">{tick}</text>'
            )
        )

    total = len(names)
    for row_index, (name, rank, score, lo, hi) in enumerate(zip(names, ranks, scores, lower, upper, strict=True)):
        y_center = plot_top + row_height * row_index + row_height / 2
        safe_name = _xml_escape(name)
        x_lo = x_to_px(lo)
        x_hi = x_to_px(hi)
        x_rank = x_to_px(float(rank))
        color = _rank_color(rank, total)

        elements.append(
            (
                f'<line x1="{x_lo:.2f}" y1="{y_center:.2f}" x2="{x_hi:.2f}" y2="{y_center:.2f}" '
                f'stroke="{color}" stroke-width="3.0" />'
            )
        )
        elements.append(
            (
                f'<line x1="{x_lo:.2f}" y1="{y_center - 7:.2f}" x2="{x_lo:.2f}" y2="{y_center + 7:.2f}" '
                f'stroke="{color}" stroke-width="2.0" />'
            )
        )
        elements.append(
            (
                f'<line x1="{x_hi:.2f}" y1="{y_center - 7:.2f}" x2="{x_hi:.2f}" y2="{y_center + 7:.2f}" '
                f'stroke="{color}" stroke-width="2.0" />'
            )
        )
        elements.append(
            (
                f'<circle cx="{x_rank:.2f}" cy="{y_center:.2f}" r="5.5" fill="#1a202c" '
                f'stroke="#ffffff" stroke-width="1.1" />'
            )
        )
        elements.append(
            (
                f'<text x="{plot_left - 12:.2f}" y="{y_center + 4:.2f}" text-anchor="end" font-size="13" '
                f'font-family="Arial, sans-serif" fill="#1a202c">{safe_name}</text>'
            )
        )
        elements.append(
            (
                f'<text x="{plot_right + 10:.2f}" y="{y_center + 4:.2f}" text-anchor="start" font-size="12" '
                f'font-family="Arial, sans-serif" fill="#2d3748">{_xml_escape(_fmt_number(score, digits=4))}</text>'
            )
        )

    elements.append(
        (
            f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{height - 20:.2f}" text-anchor="middle" '
            f'font-size="13" font-family="Arial, sans-serif" fill="#2d3748">Rank (lower is better)</text>'
        )
    )
    elements.append(
        (
            f'<text x="{plot_right + 8:.2f}" y="{plot_top - 14:.2f}" text-anchor="start" font-size="12" '
            f'font-family="Arial, sans-serif" fill="#4a5568">theta_hat</text>'
        )
    )

    _write_svg(svg_path, _render_svg(width=width, height=height, elements=elements))

    return PlotSpec(
        type="ci_forest",
        data={
            "names": names,
            "rank_point": ranks,
            "theta_hat": scores,
            "ci_lower": lower,
            "ci_upper": upper,
        },
        config={"x_label": "rank", "point": "rank", "interval": "rank_ci"},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Horizontal lines show 95% rank confidence intervals; dots show point ranks.",
        caption_academic="Forest plot of rank confidence intervals with rank point estimates for each item.",
        hint_ids=["hint-ci"],
    )


@dataclass
class _DeepRankingData:
    indicator_col: str
    method_order: list[str]
    indicator_order: list[str]
    matrix: dict[str, dict[str, float]]
    rank_min: float
    rank_max: float


def _first_seen(values: list[Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        value = str(raw)
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    frac = index - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _stable_jitter(key: str, amplitude: float) -> float:
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    value = int(digest, 16) / 0xFFFFFFFF
    return (value * 2 - 1) * amplitude


def _interpolate_color(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    bounded = max(0.0, min(1.0, ratio))
    red = int(round(start[0] + bounded * (end[0] - start[0])))
    green = int(round(start[1] + bounded * (end[1] - start[1])))
    blue = int(round(start[2] + bounded * (end[2] - start[2])))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _deep_rank_color(rank_value: float, rank_min: float, rank_max: float) -> str:
    if rank_max <= rank_min:
        return "#bfdbfe"
    ratio = (rank_value - rank_min) / (rank_max - rank_min)
    # Better ranks (smaller) are warm; worse ranks (larger) are blue.
    return _interpolate_color((245, 158, 11), (37, 99, 235), ratio)


def _compute_deep_ranking_data(
    *,
    csv_path: str,
    indicator_col: str,
    selected_indicator_values: list[str] | None,
    selected_items: list[str] | None,
    bigbetter: int,
    bootstrap_iterations: int,
    seed: int,
    r_script_path: str,
) -> _DeepRankingData:
    if not csv_path:
        raise ValueError("Deep ranking requires csv_path.")

    source = Path(csv_path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")

    df = pd.read_csv(source)
    if indicator_col not in df.columns:
        raise ValueError(f"Indicator column '{indicator_col}' not found in data.")

    if selected_items:
        method_order = [item for item in selected_items if item in df.columns and pd.api.types.is_numeric_dtype(df[item])]
    else:
        method_order = [col for col in df.columns if col != indicator_col and pd.api.types.is_numeric_dtype(df[col])]

    if len(method_order) < 2:
        raise ValueError("Deep ranking needs at least 2 numeric method columns.")

    raw_indicator_values = _first_seen(df[indicator_col].tolist())
    if selected_indicator_values:
        selected_set = {str(value) for value in selected_indicator_values}
        indicator_order = [value for value in raw_indicator_values if value in selected_set]
    else:
        indicator_order = raw_indicator_values

    if not indicator_order:
        raise ValueError("No indicator values available after filtering.")

    deep_bootstrap = max(100, min(500, int(bootstrap_iterations // 4) if bootstrap_iterations > 0 else 250))
    global_method_count = len(method_order)
    matrix: dict[str, dict[str, float]] = {}
    executor = RScriptExecutor(timeout_seconds=180)

    for idx, indicator_value in enumerate(indicator_order):
        subset = df[df[indicator_col].astype(str) == indicator_value]
        local_methods = [method for method in method_order if subset[method].notna().any()]
        if len(local_methods) < 2:
            continue

        local_frame = subset[local_methods].copy()
        with tempfile.TemporaryDirectory(prefix="omnirank-deep-rank-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            local_csv = tmp_path / "indicator_input.csv"
            local_frame.to_csv(local_csv, index=False)

            config = EngineConfig(
                csv_path=str(local_csv),
                bigbetter=bigbetter,
                selected_items=local_methods,
                selected_indicator_values=None,
                ranking_mode=RankingMode.FLASH,
                B=deep_bootstrap,
                seed=seed + idx,
                r_script_path=r_script_path,
            )
            execution = executor.run(config=config, session_work_dir=tmp_path)

        if not execution.success or not execution.results:
            continue

        local_n = len(execution.results.items)
        if local_n < 2:
            continue

        row: dict[str, float] = {}
        for method_name, rank in zip(execution.results.items, execution.results.ranks, strict=True):
            normalized_rank = 1.0 + (float(rank) - 1.0) * (global_method_count - 1.0) / (local_n - 1.0)
            row[method_name] = normalized_rank
        matrix[indicator_value] = row

    if not matrix:
        raise ValueError("No valid indicator groups for deep ranking.")

    return _DeepRankingData(
        indicator_col=indicator_col,
        method_order=method_order,
        indicator_order=[value for value in indicator_order if value in matrix],
        matrix=matrix,
        rank_min=1.0,
        rank_max=float(global_method_count),
    )


def _normalized_ranking_over_indicator(
    deep: _DeepRankingData,
    artifact_dir: Path,
) -> PlotSpec:
    method_order = deep.method_order
    indicator_order = deep.indicator_order
    payload = (
        f"{deep.indicator_col}|"
        + "|".join(method_order)
        + ":"
        + "|".join(indicator_order)
        + ":"
        + "|".join(
            f"{indicator}:{method}:{deep.matrix.get(indicator, {}).get(method, float('nan')):.6f}"
            for indicator in indicator_order
            for method in method_order
        )
    )
    block_id = _stable_block_id("figure-normalized-ranking-over-indicator", payload)
    svg_path = artifact_dir / f"{block_id}.svg"

    width = max(920, 220 + len(method_order) * 80)
    height = 520
    left_margin = 72
    right_margin = 28
    top_margin = 50
    bottom_margin = 140
    plot_left = float(left_margin)
    plot_right = float(width - right_margin)
    plot_top = float(top_margin)
    plot_bottom = float(height - bottom_margin)
    y_to_px = _linear_scale(deep.rank_min, deep.rank_max, plot_top, plot_bottom)

    elements: list[str] = []
    for rank_tick in range(int(deep.rank_min), int(math.ceil(deep.rank_max)) + 1):
        y_pos = y_to_px(float(rank_tick))
        elements.append(
            (
                f'<line x1="{plot_left:.2f}" y1="{y_pos:.2f}" x2="{plot_right:.2f}" y2="{y_pos:.2f}" '
                f'stroke="#dbe5f2" stroke-width="1" />'
            )
        )
        elements.append(
            (
                f'<text x="{plot_left - 10:.2f}" y="{y_pos + 4:.2f}" text-anchor="end" font-size="11" '
                f'fill="#334155">{rank_tick}</text>'
            )
        )

    if method_order:
        step = (plot_right - plot_left) / len(method_order)
    else:
        step = 1.0

    summary_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(method_order):
        x_center = plot_left + step * (method_index + 0.5)
        values = [deep.matrix.get(indicator, {}).get(method) for indicator in indicator_order]
        clean_values = sorted([value for value in values if value is not None])
        if not clean_values:
            continue

        q1 = _quantile(clean_values, 0.25)
        median = _quantile(clean_values, 0.5)
        q3 = _quantile(clean_values, 0.75)
        mean_value = sum(clean_values) / len(clean_values)
        min_value = clean_values[0]
        max_value = clean_values[-1]

        method_color = _rank_color(method_index + 1, max(1, len(method_order)))
        box_width = max(18.0, min(28.0, step * 0.38))
        y_q1 = y_to_px(q1)
        y_q3 = y_to_px(q3)
        y_min = y_to_px(min_value)
        y_max = y_to_px(max_value)

        elements.append(
            (
                f'<line x1="{x_center:.2f}" y1="{y_min:.2f}" x2="{x_center:.2f}" y2="{y_max:.2f}" '
                f'stroke="#334155" stroke-width="1.5" />'
            )
        )
        elements.append(
            (
                f'<line x1="{x_center - box_width / 2:.2f}" y1="{y_min:.2f}" '
                f'x2="{x_center + box_width / 2:.2f}" y2="{y_min:.2f}" stroke="#334155" stroke-width="1.5" />'
            )
        )
        elements.append(
            (
                f'<line x1="{x_center - box_width / 2:.2f}" y1="{y_max:.2f}" '
                f'x2="{x_center + box_width / 2:.2f}" y2="{y_max:.2f}" stroke="#334155" stroke-width="1.5" />'
            )
        )
        elements.append(
            (
                f'<rect x="{x_center - box_width / 2:.2f}" y="{y_q1:.2f}" width="{box_width:.2f}" '
                f'height="{max(1.0, y_q3 - y_q1):.2f}" fill="{method_color}" fill-opacity="0.2" '
                f'stroke="#334155" stroke-width="1.4" />'
            )
        )
        elements.append(
            (
                f'<line x1="{x_center - box_width / 2:.2f}" y1="{y_to_px(median):.2f}" '
                f'x2="{x_center + box_width / 2:.2f}" y2="{y_to_px(median):.2f}" '
                f'stroke="#334155" stroke-width="1.6" />'
            )
        )

        for indicator in indicator_order:
            value = deep.matrix.get(indicator, {}).get(method)
            if value is None:
                continue
            jitter_x = x_center + _stable_jitter(f"{method}|{indicator}", max(6.0, step * 0.12))
            elements.append(
                (
                    f'<circle cx="{jitter_x:.2f}" cy="{y_to_px(value):.2f}" r="2.8" '
                    f'fill="{method_color}" fill-opacity="0.7" />'
                )
            )

        elements.append(
            (
                f'<circle cx="{x_center:.2f}" cy="{y_to_px(mean_value):.2f}" r="5.4" '
                f'fill="#b91c1c" stroke="#7f1d1d" stroke-width="1.2" />'
            )
        )
        elements.append(
            (
                f'<text x="{x_center:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="10" '
                f'transform="rotate(35 {x_center:.2f} {plot_bottom + 22:.2f})" fill="#0f172a">{_xml_escape(method)}</text>'
            )
        )
        elements.append(
            (
                f'<text x="{x_center:.2f}" y="{plot_bottom + 62:.2f}" text-anchor="middle" font-size="10" '
                f'fill="#334155">(K={len(clean_values)})</text>'
            )
        )

        summary_rows.append(
            {
                "method": method,
                "k": len(clean_values),
                "mean_rank": round(mean_value, 4),
                "q1": round(q1, 4),
                "median": round(median, 4),
                "q3": round(q3, 4),
            }
        )

    elements.append(
        (
            f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{height - 22:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#334155">Methods</text>'
        )
    )
    elements.append(
        (
            f'<text x="18" y="{(plot_top + plot_bottom) / 2:.2f}" text-anchor="middle" font-size="12" '
            f'fill="#334155" transform="rotate(-90 18 {(plot_top + plot_bottom) / 2:.2f})">Normalized Rank (lower is better)</text>'
        )
    )

    _write_svg(svg_path, _render_svg(width=width, height=height, elements=elements, background="#f8fafc"))
    return PlotSpec(
        type="normalized_ranking_over_indicator",
        data={
            "indicator_col": deep.indicator_col,
            "methods": method_order,
            "indicator_values": indicator_order,
            "matrix": deep.matrix,
            "method_summary": summary_rows,
            "rank_min": deep.rank_min,
            "rank_max": deep.rank_max,
        },
        config={"x_label": "method", "y_label": "normalized_rank"},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Normalized Ranking Over Individual Phenotypes",
        caption_academic=(
            "Per-indicator normalized rank distributions across methods; "
            "box-and-whisker summaries and mean markers quantify stability and coverage."
        ),
        hint_ids=["hint-ci"],
    )


def _indicator_rankings_heatmap(
    deep: _DeepRankingData,
    artifact_dir: Path,
) -> PlotSpec:
    method_order = deep.method_order
    indicator_order = deep.indicator_order
    payload = (
        f"{deep.indicator_col}|heatmap|"
        + "|".join(method_order)
        + ":"
        + "|".join(indicator_order)
        + ":"
        + "|".join(
            f"{indicator}:{method}:{deep.matrix.get(indicator, {}).get(method, float('nan')):.6f}"
            for indicator in indicator_order
            for method in method_order
        )
    )
    block_id = _stable_block_id("figure-indicator-rankings-heatmap", payload)
    svg_path = artifact_dir / f"{block_id}.svg"

    cell_w = 26
    cell_h = 20
    max_label = max((len(value) for value in indicator_order), default=8)
    left_margin = min(420, max(140, 10 + max_label * 7))
    top_margin = 48
    right_margin = 160
    bottom_margin = 138
    width = left_margin + len(method_order) * cell_w + right_margin
    height = top_margin + len(indicator_order) * cell_h + bottom_margin

    elements: list[str] = []
    plot_left = float(left_margin)
    plot_top = float(top_margin)
    plot_width = len(method_order) * cell_w
    plot_height = len(indicator_order) * cell_h

    for row_index, indicator in enumerate(indicator_order):
        y = plot_top + row_index * cell_h
        elements.append(
            (
                f'<text x="{plot_left - 8:.2f}" y="{y + cell_h / 2 + 4:.2f}" text-anchor="end" '
                f'font-size="10.5" fill="#0f172a">{_xml_escape(indicator)}</text>'
            )
        )
        row = deep.matrix.get(indicator, {})
        for col_index, method in enumerate(method_order):
            x = plot_left + col_index * cell_w
            value = row.get(method)
            fill = "#d1d5db" if value is None else _deep_rank_color(value, deep.rank_min, deep.rank_max)
            elements.append(
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" '
                    f'fill="{fill}" stroke="#94a3b8" stroke-width="0.6" />'
                )
            )

    for col_index, method in enumerate(method_order):
        x = plot_left + col_index * cell_w + cell_w * 0.5
        label_y = plot_top + plot_height + 14
        elements.append(
            (
                f'<text x="{x:.2f}" y="{label_y:.2f}" text-anchor="start" font-size="10" fill="#0f172a" '
                f'transform="rotate(45 {x:.2f} {label_y:.2f})">{_xml_escape(method)}</text>'
            )
        )

    legend_x = plot_left + plot_width + 36
    legend_y = plot_top + 20
    legend_h = 140
    steps = 32
    for step in range(steps):
        ratio = step / max(1, steps - 1)
        rank_value = deep.rank_min + ratio * (deep.rank_max - deep.rank_min)
        y = legend_y + (legend_h * ratio)
        elements.append(
            (
                f'<rect x="{legend_x:.2f}" y="{y:.2f}" width="16" height="{legend_h / steps + 1:.2f}" '
                f'fill="{_deep_rank_color(rank_value, deep.rank_min, deep.rank_max)}" stroke="none" />'
            )
        )
    for rank_tick in [deep.rank_min, (deep.rank_min + deep.rank_max) / 2, deep.rank_max]:
        ratio = (rank_tick - deep.rank_min) / max(1e-9, deep.rank_max - deep.rank_min)
        y = legend_y + legend_h * ratio
        elements.append(
            f'<text x="{legend_x + 24:.2f}" y="{y + 4:.2f}" text-anchor="start" font-size="11" fill="#334155">{_fmt_rank_bound(rank_tick)}</text>'
        )
    elements.append(
        (
            f'<text x="{legend_x:.2f}" y="{legend_y - 10:.2f}" text-anchor="start" font-size="11" fill="#334155">'
            "Rank scale</text>"
        )
    )

    elements.append(
        (
            f'<text x="{(plot_left + plot_left + plot_width) / 2:.2f}" y="{height - 22:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#334155">Methods</text>'
        )
    )

    _write_svg(svg_path, _render_svg(width=width, height=height, elements=elements, background="#f8fafc"))

    matrix_rows: list[list[float | None]] = [
        [deep.matrix.get(indicator, {}).get(method) for method in method_order]
        for indicator in indicator_order
    ]
    return PlotSpec(
        type="indicator_rankings_heatmap",
        data={
            "indicator_col": deep.indicator_col,
            "methods": method_order,
            "indicator_values": indicator_order,
            "matrix_rows": matrix_rows,
            "rank_min": deep.rank_min,
            "rank_max": deep.rank_max,
        },
        config={"x_label": "method", "y_label": deep.indicator_col},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Phenotype Rankings",
        caption_academic=(
            "Indicator-by-method heatmap of normalized spectral ranks, enabling direct comparison of method "
            "stability and subgroup-specific performance patterns."
        ),
        hint_ids=["hint-ci"],
    )


def _as_ranking_mode(value: RankingMode | str | None) -> RankingMode:
    if isinstance(value, RankingMode):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == RankingMode.DEEP.value:
            return RankingMode.DEEP
    return RankingMode.FLASH


def generate_visualizations(
    results: RankingResults,
    viz_types: list[str],
    artifact_dir: str,
    csv_path: str | None = None,
    indicator_col: str | None = None,
    selected_indicator_values: list[str] | None = None,
    selected_items: list[str] | None = None,
    bigbetter: int = 1,
    ranking_mode: RankingMode | str | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
    r_script_path: str = "src/spectral_ranking/spectral_ranking.R",
) -> VisualizationOutput:
    """Create deterministic SVG artifacts from ranking results."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[PlotSpec] = []
    errors: list[str] = []
    mode = _as_ranking_mode(ranking_mode)

    for viz_type in viz_types:
        try:
            if viz_type == "ci_forest":
                plots.append(_ci_forest_r(results, output_dir))
                continue

            if viz_type not in {"normalized_ranking_over_indicator", "indicator_rankings_heatmap"}:
                errors.append(f"Unsupported viz_type: {viz_type}")
                continue

            if mode != RankingMode.DEEP:
                errors.append(f"Skipped {viz_type}: ranking mode is '{mode.value}', not 'deep'.")
                continue
            if not indicator_col:
                errors.append(f"Skipped {viz_type}: indicator column is not configured.")
                continue
            if not csv_path:
                errors.append(f"Skipped {viz_type}: csv_path is missing.")
                continue

            if viz_type == "normalized_ranking_over_indicator":
                plots.append(_normalized_ranking_over_indicator_r(csv_path, output_dir, indicator_col))
            elif viz_type == "indicator_rankings_heatmap":
                plots.append(_indicator_rankings_heatmap_r(csv_path, output_dir, indicator_col))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{viz_type} failed: {exc}")

    return VisualizationOutput(plots=plots, errors=errors)
