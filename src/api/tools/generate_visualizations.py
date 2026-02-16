"""Tool: generate_visualizations."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from core.schemas import PlotSpec, RankingResults, VisualizationOutput


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


def _render_svg(width: int, height: int, elements: list[str]) -> str:
    body = "\n  ".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="OmniRank visualization">\n'
        "  <style>text { font-family: Arial, sans-serif; font-weight: 700; }</style>\n"
        f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#070e19" />\n'
        f"  {body}\n"
        "</svg>\n"
    )


def _write_svg(svg_path: Path, svg_text: str) -> None:
    svg_path.write_text(svg_text, encoding="utf-8")


def _empty_plot_svg(message: str, width: int = 840, height: int = 220) -> str:
    safe_message = _xml_escape(message)
    elements = [
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" font-size="14" fill="#f8fbff">{safe_message}</text>',
    ]
    return _render_svg(width=width, height=height, elements=elements)


def _ranking_bar(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    order = _ordered_indices(results)
    names = [results.items[i] for i in order]
    scores = [results.theta_hat[i] for i in order]
    ranks = [results.ranks[i] for i in order]
    lower = [results.ci_lower[i] for i in order]
    upper = [results.ci_upper[i] for i in order]

    payload = (
        "|".join(names)
        + ":"
        + "|".join(f"{value:.6f}" for value in scores)
        + ":"
        + "|".join(f"{value:.6f}" for value in lower)
        + ":"
        + "|".join(f"{value:.6f}" for value in upper)
    )
    block_id = _stable_block_id("figure-ranking-bar", payload)
    svg_path = artifact_dir / f"{block_id}.svg"

    if not names:
        _write_svg(
            svg_path,
            _empty_plot_svg(
                message="No ranking data available",
            ),
        )
        return PlotSpec(
            type="ranking_bar",
            data={"names": [], "scores": [], "ranks": [], "rank_ci_lower": [], "rank_ci_upper": []},
            config={"x_label": "theta_hat", "order": "ascending_rank", "ci_axis": "rank"},
            svg_path=str(svg_path),
            block_id=block_id,
            caption_plain="Bars show estimated preference scores.",
            caption_academic="Bar plot of theta_hat scores ordered by inferred rank.",
            hint_ids=["hint-theta-hat"],
        )

    max_name_len = max(len(name) for name in names)
    left_margin = min(360, max(190, 48 + max_name_len * 7))
    right_margin = 220
    plot_width = 620
    width = left_margin + right_margin + plot_width
    row_height = 38
    top_margin = 30
    bottom_margin = 64
    height = top_margin + bottom_margin + row_height * len(names)

    plot_left = float(left_margin)
    plot_right = float(width - right_margin)
    plot_top = float(top_margin)
    plot_bottom = float(height - bottom_margin)

    raw_min = min(min(scores), 0.0)
    raw_max = max(max(scores), 0.0)
    span = max(raw_max - raw_min, 0.6)
    padding = span * 0.12
    domain_min = raw_min - padding
    domain_max = raw_max + padding
    x_to_px = _linear_scale(domain_min, domain_max, plot_left, plot_right)
    zero_x = x_to_px(0.0)

    tick_count = 5
    ticks = [domain_min + (domain_max - domain_min) * i / (tick_count - 1) for i in range(tick_count)]

    elements: list[str] = [
        f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#4a5568" stroke-width="1.2" />',
        f'<line x1="{zero_x:.2f}" y1="{plot_top:.2f}" x2="{zero_x:.2f}" y2="{plot_bottom:.2f}" stroke="#2d3748" stroke-width="1.0" stroke-dasharray="4 4" />',
    ]

    for tick in ticks:
        x_pos = x_to_px(tick)
        elements.append(
            f'<line x1="{x_pos:.2f}" y1="{plot_top:.2f}" x2="{x_pos:.2f}" y2="{plot_bottom:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_pos:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#4a5568">{_xml_escape(_fmt_number(tick, digits=3))}</text>'
        )

    total = len(names)
    for row_index, (name, score, rank, lo, hi) in enumerate(zip(names, scores, ranks, lower, upper, strict=True)):
        y_center = plot_top + row_height * row_index + row_height / 2
        y_top = y_center - 11
        x_value = x_to_px(score)
        bar_left = min(zero_x, x_value)
        bar_width = max(abs(x_value - zero_x), 1.0)
        color = _rank_color(rank, total)

        value_anchor = "start" if score >= 0 else "end"
        value_x = x_value + 7 if score >= 0 else x_value - 7
        safe_name = _xml_escape(name)
        ci_label = f"rank #{rank} [{_fmt_rank_bound(lo)}, {_fmt_rank_bound(hi)}]"

        elements.append(
            f'<rect x="{bar_left:.2f}" y="{y_top:.2f}" width="{bar_width:.2f}" height="22" rx="3" fill="{color}" fill-opacity="0.88" />'
        )
        elements.append(
            f'<text x="{plot_left - 12:.2f}" y="{y_center + 4:.2f}" text-anchor="end" font-size="13" font-family="Arial, sans-serif" fill="#1a202c">{safe_name}</text>'
        )
        elements.append(
            f'<text x="{value_x:.2f}" y="{y_center + 4:.2f}" text-anchor="{value_anchor}" font-size="12" font-family="Arial, sans-serif" fill="#1a202c">{_xml_escape(_fmt_number(score, digits=4))}</text>'
        )
        elements.append(
            f'<text x="{plot_right + 12:.2f}" y="{y_center + 4:.2f}" text-anchor="start" font-size="12" font-family="Arial, sans-serif" fill="#2d3748">{_xml_escape(ci_label)}</text>'
        )

    elements.append(
        f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{height - 20:.2f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#2d3748">theta_hat (higher is better when bigbetter=1)</text>'
    )

    _write_svg(svg_path, _render_svg(width=width, height=height, elements=elements))

    return PlotSpec(
        type="ranking_bar",
        data={
            "names": names,
            "scores": scores,
            "ranks": ranks,
            "rank_ci_lower": lower,
            "rank_ci_upper": upper,
        },
        config={"x_label": "theta_hat", "order": "ascending_rank", "ci_axis": "rank"},
        svg_path=str(svg_path),
        block_id=block_id,
        caption_plain="Bars show theta_hat scores, with rank confidence intervals listed per item.",
        caption_academic="Bar chart of theta_hat point estimates ordered by rank, with 95% rank confidence intervals provided as labels.",
        hint_ids=["hint-theta-hat", "hint-ci"],
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
            hint_ids=["hint-ci", "hint-rank-interpretation"],
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
        f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#4a5568" stroke-width="1.2" />',
    ]

    for tick in range(tick_start, tick_end + 1):
        if tick < domain_min or tick > domain_max:
            continue
        x_pos = x_to_px(float(tick))
        elements.append(
            f'<line x1="{x_pos:.2f}" y1="{plot_top:.2f}" x2="{x_pos:.2f}" y2="{plot_bottom:.2f}" stroke="#e2e8f0" stroke-width="1" />'
        )
        elements.append(
            f'<text x="{x_pos:.2f}" y="{plot_bottom + 22:.2f}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#4a5568">{tick}</text>'
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
            f'<line x1="{x_lo:.2f}" y1="{y_center:.2f}" x2="{x_hi:.2f}" y2="{y_center:.2f}" stroke="{color}" stroke-width="3.0" />'
        )
        elements.append(
            f'<line x1="{x_lo:.2f}" y1="{y_center - 7:.2f}" x2="{x_lo:.2f}" y2="{y_center + 7:.2f}" stroke="{color}" stroke-width="2.0" />'
        )
        elements.append(
            f'<line x1="{x_hi:.2f}" y1="{y_center - 7:.2f}" x2="{x_hi:.2f}" y2="{y_center + 7:.2f}" stroke="{color}" stroke-width="2.0" />'
        )
        elements.append(
            f'<circle cx="{x_rank:.2f}" cy="{y_center:.2f}" r="5.5" fill="#1a202c" stroke="#ffffff" stroke-width="1.1" />'
        )
        elements.append(
            f'<text x="{plot_left - 12:.2f}" y="{y_center + 4:.2f}" text-anchor="end" font-size="13" font-family="Arial, sans-serif" fill="#1a202c">{safe_name}</text>'
        )
        elements.append(
            f'<text x="{plot_right + 10:.2f}" y="{y_center + 4:.2f}" text-anchor="start" font-size="12" font-family="Arial, sans-serif" fill="#2d3748">{_xml_escape(_fmt_number(score, digits=4))}</text>'
        )

    elements.append(
        f'<text x="{(plot_left + plot_right) / 2:.2f}" y="{height - 20:.2f}" text-anchor="middle" font-size="13" font-family="Arial, sans-serif" fill="#2d3748">Rank (lower is better)</text>'
    )
    elements.append(
        f'<text x="{plot_right + 8:.2f}" y="{plot_top - 14:.2f}" text-anchor="start" font-size="12" font-family="Arial, sans-serif" fill="#4a5568">theta_hat</text>'
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
        hint_ids=["hint-ci", "hint-rank-interpretation"],
    )


def generate_visualizations(
    results: RankingResults,
    viz_types: list[str],
    artifact_dir: str,
) -> VisualizationOutput:
    """Create deterministic SVG artifacts from ranking results."""
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[PlotSpec] = []
    errors: list[str] = []

    for viz_type in viz_types:
        try:
            if viz_type == "ranking_bar":
                plots.append(_ranking_bar(results, output_dir))
            elif viz_type == "ci_forest":
                plots.append(_ci_forest(results, output_dir))
            else:
                errors.append(f"Unsupported viz_type: {viz_type}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{viz_type} failed: {exc}")

    return VisualizationOutput(plots=plots, errors=errors)
