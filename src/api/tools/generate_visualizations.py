"""Tool: generate_visualizations."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from core.r_executor import RScriptExecutor
from core.schemas import EngineConfig, PlotSpec, RankingMode, RankingResults, VisualizationOutput


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


def _phenotype_plot_py_script(project_root: Path) -> Path:
    script = project_root / "src" / "spectral_ranking" / "plot_phenotype_rankings.py"
    if not script.exists():
        raise FileNotFoundError(f"Python plot script not found: {script}")
    return script


def _item_rank_matrix_from_csv(
    csv_path: str,
    indicator_col: str,
    bigbetter: int,
    pre_ranked: bool = False,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Build indicator x item rank matrix from CSV.

    When ``pre_ranked=True``, values are treated as already-ranked values and are
    not re-ranked in Python.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return [], [], []

    indicator_key = indicator_col if indicator_col in df.columns else str(df.columns[0])
    item_order = [col for col in df.columns if col != indicator_key]
    if not item_order:
        return [], [], []
    item_order = _apply_preferred_method_order(item_order)

    numeric = df[item_order].apply(pd.to_numeric, errors="coerce")
    if pre_ranked:
        ranked = numeric
    else:
        ascending = int(bigbetter) != 1
        ranked = numeric.rank(
            axis=1,
            method="average",
            na_option="keep",
            ascending=ascending,
        )
    phenotype_order = df[indicator_key].astype(str).tolist()
    rank_rows: list[list[float | None]] = [
        [None if pd.isna(value) else float(value) for value in row]
        for row in ranked.to_numpy().tolist()
    ]
    return item_order, phenotype_order, rank_rows


def _ci_forest_py(results: RankingResults, artifact_dir: Path) -> PlotSpec:
    """Generate overall ranking plot (CI forest) using Python plot script."""
    order = _preferred_method_reorder_indices(
        [str(item) for item in results.items],
        ranks=list(results.ranks),
    )
    names = [results.items[i] for i in order]
    theta_hat = [results.theta_hat[i] for i in order]
    ranks = [results.ranks[i] for i in order]
    lower = [results.ci_lower[i] for i in order]
    upper = [results.ci_upper[i] for i in order]

    payload = (
        "|".join(names)
        + ":"
        + "|".join(f"{r:.6f}" for r in ranks)
        + ":"
        + "|".join(f"{lo:.6f}-{hi:.6f}" for lo, hi in zip(lower, upper, strict=True))
    )
    block_id = _stable_block_id("figure-ci-forest", payload)
    png_path = artifact_dir / f"{block_id}.png"

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    py_script = _phenotype_plot_py_script(project_root)

    ci_data = {
        "items": names,
        "theta_hat": theta_hat,
        "ranks": ranks,
        "ci_lower": lower,
        "ci_upper": upper,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ci_data, f, indent=None)
        json_path = Path(f.name)

    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(py_script), "--ci-plot", str(json_path), "--ci-out", str(png_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Python CI plot script failed: {result.stderr or result.stdout or 'unknown error'}"
            )
        if not png_path.exists():
            raise RuntimeError(f"Python CI plot script did not produce output: {png_path}")
    finally:
        json_path.unlink(missing_ok=True)

    return PlotSpec(
        type="ci_forest",
        data={
            "names": names,
            "rank_point": ranks,
            "theta_hat": theta_hat,
            "ci_lower": lower,
            "ci_upper": upper,
        },
        config={"x_label": "rank", "point": "rank", "interval": "rank_ci", "source": "python"},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain="Overall Ranking Plot",
        caption_academic="Forest plot of 95% rank confidence intervals with rank point estimates.",
        hint_ids=["hint-ci"],
    )


def _normalized_ranking_over_indicator_py(
    csv_path: str,
    artifact_dir: Path,
    indicator_col: str = "phenotype",
    bigbetter: int = 0,
    pre_ranked: bool = False,
    source: str = "python",
) -> PlotSpec:
    """Generate Normalized Ranking Over Individual [Indicators] using Python plot script."""
    csv_source = Path(csv_path).resolve()
    if not csv_source.exists():
        raise FileNotFoundError(f"Data file not found: {csv_source}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    py_script = _phenotype_plot_py_script(project_root)

    block_id = _stable_block_id(
        "figure-normalized-ranking-over-indicator",
        f"{csv_source}|bigbetter={int(bigbetter)}",
    )
    png_path = artifact_dir / f"{block_id}.png"

    command = [
        sys.executable,
        str(py_script),
        "--csv",
        str(csv_source),
        "--out",
        str(png_path),
        "--bigbetter",
        str(int(bigbetter)),
    ]
    if pre_ranked:
        command.extend(["--pre-ranked", "1"])

    result = subprocess.run(  # noqa: S603
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Python plot script failed: {result.stderr or result.stdout or 'unknown error'}"
        )
    if not png_path.exists():
        raise RuntimeError(f"Python plot script did not produce output: {png_path}")

    _, plural = _indicator_label(indicator_col)
    caption = f"Normalized Ranking Over Individual {plural}"
    return PlotSpec(
        type="normalized_ranking_over_indicator",
        data={"indicator_col": indicator_col},
        config={
            "x_label": "method",
            "y_label": "normalized_rank",
            "source": source,
            "bigbetter": int(bigbetter),
            "pre_ranked": bool(pre_ranked),
        },
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain=caption,
        caption_academic=(
            f"Per-indicator normalized rank distributions across methods; "
            "box-and-whisker summaries and mean markers quantify stability and coverage."
        ),
        hint_ids=["hint-ci"],
    )


def _write_deep_matrix_csv(deep: _DeepRankingData, out_path: Path) -> None:
    """Write phenotype x method rank matrix to CSV for plot_phenotype_rankings.py."""
    rows = []
    for indicator in deep.indicator_order:
        row = {deep.indicator_col: indicator}
        for method in deep.method_order:
            row[method] = deep.matrix.get(indicator, {}).get(method)
        rows.append(row)
    df = pd.DataFrame(rows, columns=[deep.indicator_col, *deep.method_order])
    df.to_csv(out_path, index=False)


def _indicator_rankings_heatmap_py(
    csv_path: str,
    artifact_dir: Path,
    indicator_col: str = "phenotype",
    bigbetter: int = 0,
    pre_ranked: bool = False,
    source: str = "python",
) -> PlotSpec:
    """Generate [Indicator] Rankings heatmap using Python plot script."""
    csv_source = Path(csv_path).resolve()
    if not csv_source.exists():
        raise FileNotFoundError(f"Data file not found: {csv_source}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    py_script = _phenotype_plot_py_script(project_root)

    block_id = _stable_block_id(
        "figure-indicator-rankings-heatmap",
        f"{csv_source}|bigbetter={int(bigbetter)}",
    )
    png_path = artifact_dir / f"{block_id}.png"

    command = [
        sys.executable,
        str(py_script),
        "--csv",
        str(csv_source),
        "--heatmap-out",
        str(png_path),
        "--bigbetter",
        str(int(bigbetter)),
    ]
    if pre_ranked:
        command.extend(["--pre-ranked", "1"])

    result = subprocess.run(  # noqa: S603
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Python heatmap script failed: {result.stderr or result.stdout or 'unknown error'}"
        )
    if not png_path.exists():
        raise RuntimeError(f"Python heatmap script did not produce output: {png_path}")

    tc, _ = _indicator_label(indicator_col)
    caption = f"{tc} Rankings"
    return PlotSpec(
        type="indicator_rankings_heatmap",
        data={"indicator_col": indicator_col},
        config={"source": source, "bigbetter": int(bigbetter), "pre_ranked": bool(pre_ranked)},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain=caption,
        caption_academic=(
            f"Per-{indicator_col.lower()} ranking heatmap across methods; "
            "orange indicates better (lower) rank, blue indicates worse (higher) rank."
        ),
        hint_ids=["hint-ci"],
    )


def _indicator_rankings_combined_py(
    csv_path: str,
    artifact_dir: Path,
    indicator_col: str = "phenotype",
    bigbetter: int = 0,
    pre_ranked: bool = False,
    source: str = "python",
    table_rank_rows: list[list[float | int | None]] | None = None,
) -> PlotSpec:
    """Generate a single 2-panel figure for normalized ranks and heatmap."""
    from PIL import Image, ImageDraw, ImageFont

    style_version = "v6-leftalign-b-and-tablefit"

    csv_source = Path(csv_path).resolve()
    if not csv_source.exists():
        raise FileNotFoundError(f"Data file not found: {csv_source}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    py_script = _phenotype_plot_py_script(project_root)

    block_id = _stable_block_id(
        "figure-indicator-rankings-combined",
        f"{csv_source}|bigbetter={int(bigbetter)}|style={style_version}",
    )
    png_path = artifact_dir / f"{block_id}.png"

    with tempfile.TemporaryDirectory(prefix="omnirank-indicator-combined-") as tmp_dir:
        panel_a_path = Path(tmp_dir) / "panel_a.png"
        panel_b_path = Path(tmp_dir) / "panel_b.png"
        result = subprocess.run(  # noqa: S603
            (
                [
                    sys.executable,
                    str(py_script),
                    "--csv",
                    str(csv_source),
                    "--out",
                    str(panel_a_path),
                    "--heatmap-out",
                    str(panel_b_path),
                    "--bigbetter",
                    str(int(bigbetter)),
                ]
                + (["--pre-ranked", "1"] if pre_ranked else [])
            ),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Python combined plot script failed: {result.stderr or result.stdout or 'unknown error'}"
            )
        if not panel_a_path.exists() or not panel_b_path.exists():
            raise RuntimeError("Python combined plot script did not produce both panel images.")

        with Image.open(panel_a_path) as panel_a:
            panel_a_img = panel_a.convert("RGB")
        with Image.open(panel_b_path) as panel_b:
            panel_b_img = panel_b.convert("RGB")

    panel_a_w, panel_a_h = panel_a_img.size
    panel_b_w, panel_b_h = panel_b_img.size

    tc, plural = _indicator_label(indicator_col)
    panel_a_title = f"(A) Distribution of {tc}-Specific Ranks across {plural}"
    panel_b_title = f"(B) Items' Ranks by {tc}"

    # Keep panel A unchanged. Upscale panel B for better readability in the stacked layout.
    target_b_w = max(panel_b_w, int(panel_a_w * 0.82))
    target_b_w = min(target_b_w, panel_a_w)
    if target_b_w > panel_b_w:
        scale = target_b_w / panel_b_w
        target_b_h = int(round(panel_b_h * scale))
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        panel_b_img = panel_b_img.resize((target_b_w, target_b_h), resample)
        panel_b_w, panel_b_h = panel_b_img.size

    # Keep each panel at original pixel size; only add title rows and vertical stacking.
    panel_w_max = max(panel_a_w, panel_b_w)
    side_padding = 24
    panel_gap = 24

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    title_font = None
    for candidate in font_candidates:
        try:
            title_font = ImageFont.truetype(candidate, size=39)
            break
        except OSError:
            continue
    if title_font is None:
        title_font = ImageFont.load_default()
    title_font_b = title_font
    if hasattr(title_font, "size"):
        for candidate in font_candidates:
            try:
                title_font_b = ImageFont.truetype(candidate, size=int(title_font.size) + 1)
                break
            except OSError:
                continue

    measure_draw = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    title_bbox_a = measure_draw.textbbox((0, 0), panel_a_title, font=title_font)
    title_bbox_b = measure_draw.textbbox((0, 0), panel_b_title, font=title_font_b)
    title_h_a = title_bbox_a[3] - title_bbox_a[1]
    title_h_b = title_bbox_b[3] - title_bbox_b[1]
    title_height = max(52, title_h_a, title_h_b) + 16

    canvas_w = panel_w_max + side_padding * 2
    canvas_h = title_height + panel_a_h + panel_gap + title_height + panel_b_h + 12
    combined = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(combined)

    y = 0
    title_bbox_a = draw.textbbox((0, 0), panel_a_title, font=title_font)
    title_a_w = title_bbox_a[2] - title_bbox_a[0]
    draw.text(
        ((canvas_w - title_a_w) // 2, y + (title_height - title_h_a) // 2),
        panel_a_title,
        fill=(0, 0, 0),
        font=title_font,
    )
    y += title_height

    panel_a_x = side_padding + (panel_w_max - panel_a_w) // 2
    combined.paste(panel_a_img, (panel_a_x, y))
    y += panel_a_h + panel_gap

    title_bbox_b = draw.textbbox((0, 0), panel_b_title, font=title_font_b)
    title_b_w = title_bbox_b[2] - title_bbox_b[0]
    draw.text(
        ((canvas_w - title_b_w) // 2, y + (title_height - title_h_b) // 2),
        panel_b_title,
        fill=(0, 0, 0),
        font=title_font_b,
    )
    y += title_height

    panel_b_x = panel_a_x
    combined.paste(panel_b_img, (panel_b_x, y))
    combined.save(png_path, format="PNG")

    item_order, phenotype_order, rank_rows = _item_rank_matrix_from_csv(
        str(csv_source),
        indicator_col=indicator_col,
        bigbetter=bigbetter,
        pre_ranked=pre_ranked,
    )
    if table_rank_rows is not None and len(table_rank_rows) == len(phenotype_order):
        coerced_rows: list[list[float | int | None]] = []
        for row in table_rank_rows:
            padded = list(row[: len(item_order)])
            if len(padded) < len(item_order):
                padded.extend([None] * (len(item_order) - len(padded)))
            coerced_rows.append(padded)
        rank_rows = coerced_rows

    caption = f"{panel_a_title}; {panel_b_title}"
    return PlotSpec(
        type="indicator_rankings_combined",
        data={
            "indicator_col": indicator_col,
            "item_order": item_order,
            "phenotype_order": phenotype_order,
            "rank_rows": rank_rows,
        },
        config={"source": source, "bigbetter": int(bigbetter), "pre_ranked": bool(pre_ranked)},
        svg_path=str(png_path),
        block_id=block_id,
        caption_plain=caption,
        caption_academic=(
            "Two-panel view combining per-indicator normalized rank distributions and indicator-by-item heatmap "
            "for vertically stacked stability and subgroup comparison."
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
    preferred_order = _preferred_method_reorder_indices([str(name) for name in names], ranks=ranks)
    names = [names[i] for i in preferred_order]
    scores = [scores[i] for i in preferred_order]
    ranks = [ranks[i] for i in preferred_order]
    lower = [lower[i] for i in preferred_order]
    upper = [upper[i] for i in preferred_order]

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
            caption_plain="Overall Ranking Plot",
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
        caption_plain="Overall Ranking Plot",
        caption_academic="Forest plot of rank confidence intervals with rank point estimates for each item.",
        hint_ids=["hint-ci"],
    )


@dataclass
class _DeepRankingData:
    indicator_col: str
    method_order: list[str]
    indicator_order: list[str]
    matrix: dict[str, dict[str, float]]
    raw_rank_matrix: dict[str, dict[str, int]]
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
    # Match softened heatmap endpoints (#F29A3A .. #3A78D4) used in plot_phenotype_rankings.
    return _interpolate_color((242, 154, 58), (58, 120, 212), ratio)


def _is_multiway_phenotype_format(csv_path: str, indicator_col: str) -> bool:
    """Return True if data is phenotype x method with multiple values per row (multiway).
    Multiway: each row has many method values; R script expects raw scores and ranks internally.
    Pairwise: each row has ~2 method values; we must run spectral ranking per indicator."""
    source = Path(csv_path)
    if not source.exists():
        return False
    df = pd.read_csv(source)
    if indicator_col not in df.columns:
        return False
    method_cols = [
        c for c in df.columns
        if c != indicator_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(method_cols) < 2:
        return False
    non_na_per_row = df[method_cols].notna().sum(axis=1)
    median_non_na = non_na_per_row.median()
    return median_non_na > 2


def _is_known_na_rm_failure(error_text: str | None, stderr_text: str | None, stdout_text: str | None) -> bool:
    needle = "missing values and nan's not allowed if 'na.rm' is false"
    merged = "\n".join([error_text or "", stderr_text or "", stdout_text or ""]).lower()
    return needle in merged


def _normalize_rank_to_global_scale(rank: float, global_method_count: int, local_n: int) -> float:
    # normalized_rank = 1 + (rank - 1) * (global_method_count - 1) / (local_n - 1)
    return 1.0 + (float(rank) - 1.0) * (float(global_method_count) - 1.0) / (float(local_n) - 1.0)


def _matrix_rows_from_deep(
    deep: _DeepRankingData,
    *,
    use_raw_rank: bool,
) -> list[list[float | int | None]]:
    source = deep.raw_rank_matrix if use_raw_rank else deep.matrix
    rows: list[list[float | int | None]] = []
    for indicator in deep.indicator_order:
        row_map = source.get(indicator, {})
        rows.append([row_map.get(method) for method in deep.method_order])
    return rows


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
    method_order = _apply_preferred_method_order(method_order)

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
    raw_rank_matrix: dict[str, dict[str, int]] = {}
    executor = RScriptExecutor(timeout_seconds=180)

    for idx, indicator_value in enumerate(indicator_order):
        subset = df[df[indicator_col].astype(str) == indicator_value]
        local_methods = [method for method in method_order if subset[method].notna().any()]
        if len(local_methods) < 2:
            matrix[indicator_value] = {}
            raw_rank_matrix[indicator_value] = {}
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
            if _is_known_na_rm_failure(
                execution.error,
                execution.trace.stderr if execution.trace else "",
                execution.trace.stdout if execution.trace else "",
            ):
                matrix[indicator_value] = {}
                raw_rank_matrix[indicator_value] = {}
            continue

        local_n = len(execution.results.items)
        if local_n < 2:
            matrix[indicator_value] = {}
            raw_rank_matrix[indicator_value] = {}
            continue

        row: dict[str, float] = {}
        raw_row: dict[str, int] = {}
        for method_name, rank in zip(execution.results.items, execution.results.ranks, strict=True):
            normalized_rank = _normalize_rank_to_global_scale(
                rank=float(rank),
                global_method_count=global_method_count,
                local_n=local_n,
            )
            row[method_name] = normalized_rank
            raw_row[method_name] = int(round(float(rank)))
        matrix[indicator_value] = row
        raw_rank_matrix[indicator_value] = raw_row

    if not matrix:
        return _DeepRankingData(
            indicator_col=indicator_col,
            method_order=method_order,
            indicator_order=[],
            matrix={},
            raw_rank_matrix={},
            rank_min=1.0,
            rank_max=float(max(1, global_method_count)),
        )

    return _DeepRankingData(
        indicator_col=indicator_col,
        method_order=method_order,
        indicator_order=[value for value in indicator_order if value in matrix],
        matrix=matrix,
        raw_rank_matrix=raw_rank_matrix,
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
            f'font-size="12" fill="#334155">Items</text>'
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
            f'font-size="12" fill="#334155">Items</text>'
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
    deep_cache: _DeepRankingData | None = None
    combine_indicator_plots = {
        "normalized_ranking_over_indicator",
        "indicator_rankings_heatmap",
    }.issubset(set(viz_types))
    combined_indicator_generated = False

    for viz_type in viz_types:
        try:
            if viz_type == "ci_forest":
                plots.append(_ci_forest_py(results, output_dir))
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

            if deep_cache is None:
                deep_cache = _compute_deep_ranking_data(
                    csv_path=csv_path,
                    indicator_col=indicator_col,
                    selected_indicator_values=selected_indicator_values,
                    selected_items=selected_items,
                    bigbetter=bigbetter,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                    r_script_path=r_script_path,
                )
            matrix_csv = output_dir / "_plot_matrix.csv"
            _write_deep_matrix_csv(deep_cache, matrix_csv)
            plot_csv = str(matrix_csv)
            # Deep mode plots consume pre-computed per-indicator ranks from R output.
            plot_bigbetter = 0
            plot_pre_ranked = True
            plot_source = "r_spectral"

            if combine_indicator_plots:
                if combined_indicator_generated:
                    continue
                plots.append(
                    _indicator_rankings_combined_py(
                        plot_csv,
                        output_dir,
                        indicator_col,
                        bigbetter=plot_bigbetter,
                        pre_ranked=plot_pre_ranked,
                        source=plot_source,
                        table_rank_rows=_matrix_rows_from_deep(deep_cache, use_raw_rank=True),
                    )
                )
                combined_indicator_generated = True
                continue

            if viz_type == "normalized_ranking_over_indicator":
                plots.append(
                    _normalized_ranking_over_indicator_py(
                        plot_csv,
                        output_dir,
                        indicator_col,
                        bigbetter=plot_bigbetter,
                        pre_ranked=plot_pre_ranked,
                        source=plot_source,
                    )
                )
            elif viz_type == "indicator_rankings_heatmap":
                plots.append(
                    _indicator_rankings_heatmap_py(
                        plot_csv,
                        output_dir,
                        indicator_col,
                        bigbetter=plot_bigbetter,
                        pre_ranked=plot_pre_ranked,
                        source=plot_source,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{viz_type} failed: {exc}")

    return VisualizationOutput(plots=plots, errors=errors)
