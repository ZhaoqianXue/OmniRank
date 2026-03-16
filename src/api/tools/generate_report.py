"""Tool: generate_report.

Generates a single-page, publication-ready markdown report with citable blocks,
interleaved figures, and structured analysis following the SOP Deep Research
style contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from core.llm_client import LLMCallError, get_llm_client
from core.schemas import (
    ArtifactRef,
    CitationBlock,
    CitationKind,
    HintKind,
    HintSpec,
    PlotSpec,
    RankingResults,
    ReportOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_block_id(prefix: str, payload: Any) -> str:
    """Build deterministic block id from stable JSON payload."""
    if isinstance(payload, str):
        serialized = payload
    else:
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    digest = hashlib.sha1(
        serialized.encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _escape_table_cell(value: Any) -> str:
    """Escape markdown table cell content (pipes and newlines)."""
    text = str(value)
    # Escape HTML-sensitive chars so untrusted item names cannot render raw tags.
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def _sanitize_inline_text(value: str) -> str:
    """Escape HTML-sensitive characters for inline markdown text."""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _top_item(results: RankingResults) -> tuple[str, float, int]:
    """Return (name, score, rank) for the top-ranked item."""
    best_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
    return (
        results.items[best_idx],
        results.theta_hat[best_idx],
        results.ranks[best_idx],
    )


def _section(block_id: str, kind: str, body: str) -> str:
    """Wrap *body* in a citable ``<section>`` block.

    Blank lines after ``<section>`` and before ``</section>`` are mandatory
    so that CommonMark parsers treat the body as markdown rather than raw
    HTML block content.
    """
    return (
        f'<section data-omni-block-id="{block_id}" data-omni-kind="{kind}">\n'
        f"\n{body}\n\n"
        f"</section>"
    )


_CI_PAIR_PATTERNS = [
    re.compile(r"((?:95%\s*)?CI(?:\s*[:=])?\s*\[)\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(\])", re.IGNORECASE),
    re.compile(r"(confidence intervals?(?:\s*[:=])?\s*\[)\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(\])", re.IGNORECASE),
]


def _ci_int(value: float) -> int:
    return int(round(float(value)))


def _ci_pair(lo: float, hi: float) -> str:
    return f"[{_ci_int(lo)}, {_ci_int(hi)}]"


def _integerize_ci_text(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        prefix, lo, hi, suffix = match.groups()
        return f"{prefix}{_ci_int(float(lo))}, {_ci_int(float(hi))}{suffix}"

    normalized = text
    for pattern in _CI_PAIR_PATTERNS:
        normalized = pattern.sub(_replace, normalized)
    return normalized


def _mentions_item(text: str, item: str) -> bool:
    return item.lower() in text.lower()


def _has_top_claim(text: str, item: str) -> bool:
    escaped = re.escape(item)
    patterns = (
        rf"\*\*{escaped}\*\*\s+ranks?\s+#?1",
        rf"\*\*{escaped}\*\*\s+is\s+ranked\s+#?1",
        rf"\*\*{escaped}\*\*\s+ranks?\s+highest",
        rf"\*\*Top result\*\*:\s+\*\*{escaped}\*\*",
        rf"Top-ranked item[^.\n]*{escaped}",
        rf"{escaped}\s+is\s+top-ranked",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _validate_llm_narrative(
    narrative: dict[str, str],
    results: RankingResults,
    analysis: dict[str, Any],
) -> list[str]:
    top_item, _, _ = _top_item(results)
    summary = narrative.get("summary", "")
    results_narrative = narrative.get("results_narrative", "")
    combined = f"{summary}\n{results_narrative}"
    issues: list[str] = []

    if "key takeaways" not in summary.lower():
        issues.append("Summary must include a '**Key Takeaways:**' header.")
    if not _mentions_item(summary, top_item):
        issues.append(f"Summary must explicitly name **{top_item}** as the top-ranked item.")

    for item in results.items:
        if item == top_item:
            continue
        if _has_top_claim(summary, item):
            issues.append(
                f"Summary incorrectly presents **{item}** as top-ranked; the actual top item is **{top_item}**."
            )
            break

    widest_ci_item = str(analysis.get("widest_ci_item") or "").strip()
    if widest_ci_item and not _mentions_item(combined, widest_ci_item):
        issues.append(
            f"Key takeaways should mention **{widest_ci_item}** as the item with the widest interval."
        )

    largest_gap = analysis.get("largest_gap")
    if isinstance(largest_gap, dict):
        gap_from = str(largest_gap.get("from") or "").strip()
        gap_to = str(largest_gap.get("to") or "").strip()
        if gap_from and gap_to and not (_mentions_item(combined, gap_from) and _mentions_item(combined, gap_to)):
            issues.append(
                f"Narrative should mention the largest estimated-score gap between **{gap_from}** and **{gap_to}**."
            )

    if re.search(r"\btier(s)?\b", combined, re.IGNORECASE):
        issues.append("Avoid 'tier' language; use groups or overlapping-interval clusters instead.")

    return issues


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _analyze_ranking(results: RankingResults) -> dict[str, Any]:
    """Derive clusters, gaps, and near-ties from ranking results."""
    order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])
    n = len(order)

    # Cluster analysis based on pairwise CI overlap with cluster head
    clusters: list[list[int]] = []
    current: list[int] = [order[0]]
    for i in range(1, n):
        idx = order[i]
        overlaps = any(
            results.ci_upper[idx] >= results.ci_lower[c]
            and results.ci_lower[idx] <= results.ci_upper[c]
            for c in current
        )
        if overlaps:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    # Near-ties with the top-ranked item
    top_idx = order[0]
    near_ties_with_top = [
        results.items[idx]
        for idx in order[1:]
        if (
            results.ci_upper[idx] >= results.ci_lower[top_idx]
            and results.ci_lower[idx] <= results.ci_upper[top_idx]
        )
    ]

    # Score gaps between consecutive ranks
    gaps = []
    for i in range(1, n):
        prev, curr = order[i - 1], order[i]
        gaps.append(
            {
                "from": results.items[prev],
                "to": results.items[curr],
                "gap": results.theta_hat[prev] - results.theta_hat[curr],
            }
        )

    largest_gap = max(gaps, key=lambda g: g["gap"]) if gaps else None

    # CI width extremes
    ci_widths = [results.ci_upper[i] - results.ci_lower[i] for i in range(n)]
    widest_idx = max(range(n), key=lambda i: ci_widths[i])
    narrowest_idx = min(range(n), key=lambda i: ci_widths[i])

    return {
        "order": order,
        "clusters": [[results.items[i] for i in c] for c in clusters],
        "n_clusters": len(clusters),
        "near_ties_with_top": near_ties_with_top,
        "gaps": gaps,
        "largest_gap": largest_gap,
        "ci_widths": ci_widths,
        "widest_ci_item": results.items[widest_idx],
        "widest_ci": ci_widths[widest_idx],
        "narrowest_ci_item": results.items[narrowest_idx],
        "narrowest_ci": ci_widths[narrowest_idx],
    }


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------

def _render_ranking_table(results: RankingResults) -> str:
    """Render ranking table with confidence interval and score columns."""
    order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])

    lines = [
        "| Rank | Item | Confidence Interval | Estimated Score |",
        "|---:|---|---|---:|",
    ]
    for idx in order:
        item = _escape_table_cell(results.items[idx])
        score = results.theta_hat[idx]
        ci_lo = results.ci_lower[idx]
        ci_hi = results.ci_upper[idx]

        lines.append(
            f"| {results.ranks[idx]} | {item} | {_ci_pair(ci_lo, ci_hi)} | {score:.4f} |"
        )
    return "\n".join(lines)


def _render_indicator_ranking_table(plot: PlotSpec) -> str | None:
    data = plot.data or {}
    item_order_raw = data.get("item_order")
    phenotype_order_raw = data.get("phenotype_order")
    rank_rows_raw = data.get("rank_rows")
    if not isinstance(item_order_raw, list) or not isinstance(phenotype_order_raw, list) or not isinstance(rank_rows_raw, list):
        return None
    if not item_order_raw or not phenotype_order_raw or not rank_rows_raw:
        return None

    item_order = [str(item) for item in item_order_raw]
    phenotype_order = [str(value) for value in phenotype_order_raw]
    rank_rows: list[list[Any]] = [row if isinstance(row, list) else [] for row in rank_rows_raw]
    if len(rank_rows) != len(phenotype_order):
        return None

    indicator = str((data or {}).get("indicator_col") or "indicator")
    indicator_title = indicator[0].upper() + indicator[1:].lower() if indicator else "Indicator"

    def _rank_cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return "NA"
            rounded = int(round(float(value)))
            if abs(float(value) - rounded) < 1e-9:
                return str(rounded)
        return _escape_table_cell(value)

    header = [indicator_title, *item_order]
    lines = [
        "| " + " | ".join(_escape_table_cell(cell) for cell in header) + " |",
        "|" + "---|" * len(header),
    ]
    for phenotype_value, row in zip(phenotype_order, rank_rows, strict=False):
        padded = list(row[: len(item_order)])
        if len(padded) < len(item_order):
            padded.extend([None] * (len(item_order) - len(padded)))
        cells = [_escape_table_cell(phenotype_value), *[_rank_cell(value) for value in padded]]
        lines.append("| " + " | ".join(cells) + " |")

    table_title = f"{indicator_title} Ranking Table"
    return f"### {table_title}\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Narrative builder (LLM + fallback)
# ---------------------------------------------------------------------------

def _build_llm_narrative(
    results: RankingResults,
    session_meta: dict[str, Any],
) -> dict[str, str]:
    """Generate markdown narrative via LLM with rich deterministic fallback.

    The returned dict values may contain markdown formatting (bold, bullets,
    inline code) but no raw HTML.
    """
    top_item, _, _ = _top_item(results)
    analysis = _analyze_ranking(results)
    order = analysis["order"]

    top_idx = order[0]
    top_ci_lo = results.ci_lower[top_idx]
    top_ci_hi = results.ci_upper[top_idx]
    top_ci_pair = _ci_pair(top_ci_lo, top_ci_hi)
    near_ties = analysis["near_ties_with_top"]
    clusters = analysis["clusters"]
    runner_up_idx = order[1] if len(order) > 1 else None
    runner_up_item = results.items[runner_up_idx] if runner_up_idx is not None else None

    # ── Executive Summary ────────────────────────────────────────────────
    if near_ties:
        tied_items = ", ".join(f"**{item}**" for item in near_ties)
        uncertainty = (
            f"**{top_item}** currently leads, but its rank interval overlaps with "
            f"{tied_items}, so the exact ordering at the top remains uncertain."
        )
    elif runner_up_item is not None:
        uncertainty = (
            f"**{top_item}** is separated from **{runner_up_item}** by non-overlapping "
            "rank intervals, suggesting a clearer lead in this run."
        )
    else:
        uncertainty = f"Only **{top_item}** is present, so no head-to-head uncertainty applies."

    bullets: list[str] = [
        f"**Top result**: **{top_item}** is ranked #1 (95% CI: {top_ci_pair}).",
    ]
    if near_ties:
        bullets.append(
            f"**Close competition**: {', '.join(f'**{item}**' for item in near_ties)} "
            "still overlap with the leader."
        )
    elif runner_up_item is not None:
        bullets.append(
            f"**Top separation**: **{top_item}** is cleanly separated from **{runner_up_item}** "
            "by the reported rank intervals."
        )
    bullets.append(
        f"**Most uncertainty**: **{analysis['widest_ci_item']}** has the widest interval, "
        "so its rank is estimated less precisely."
    )
    if analysis["largest_gap"]:
        lg = analysis["largest_gap"]
        bullets.append(
            f"**Largest score gap**: estimated scores drop most sharply between "
            f"**{lg['from']}** and **{lg['to']}**."
        )

    summary = (
        f"**{top_item}** ranks #1 in this run, meaning it has the strongest estimated score "
        "among the compared items. "
        f"{uncertainty} "
        "The intervals shown are 95% bootstrap confidence intervals, so overlap should be read "
        "as ranking uncertainty rather than as a formal significance claim.\n\n"
        "**Key Takeaways:**\n\n"
        + "\n".join(f"- {b}" for b in bullets)
    )

    # ── Results Narrative ────────────────────────────────────────────────
    parts: list[str] = [
        f"The analysis compares **{len(results.items)} items** and orders them by estimated score."
    ]
    if len(clusters) > 1:
        parts.append(
            f"Grouping items by overlapping rank intervals yields **{len(clusters)} CI-overlap groups**, "
            "which helps separate clearer gaps from crowded parts of the ranking."
        )
    else:
        parts.append(
            "Most items fall into a single CI-overlap group, so much of the table remains tightly packed."
        )
    for cluster_i, cluster in enumerate(clusters, start=1):
        label = f"Group {cluster_i}"

        items_md = ", ".join(f"**{it}**" for it in cluster)
        if len(cluster) == 1:
            ci = results.items.index(cluster[0])
            parts.append(
                f"**{label}**: {items_md} "
                f"(estimated score {results.theta_hat[ci]:.4f}, "
                f"rank CI {_ci_pair(results.ci_lower[ci], results.ci_upper[ci])})."
            )
        else:
            scores = [results.theta_hat[results.items.index(it)] for it in cluster]
            parts.append(
                f"**{label}**: {items_md} "
                f"(scores range {min(scores):.4f} -- {max(scores):.4f}, "
                "with overlapping rank intervals inside this group)."
            )
    if analysis["largest_gap"]:
        lg = analysis["largest_gap"]
        parts.append(
            f"The largest drop in estimated score occurs between **{lg['from']}** and **{lg['to']}**."
        )
    results_narrative = " ".join(parts)

    # ── Methods ──────────────────────────────────────────────────────────
    B = session_meta.get("B", 2000)
    seed = session_meta.get("seed", 42)
    input_path = _sanitize_inline_text(str(session_meta.get("current_file_path") or "N/A"))
    methods = (
        "### Estimation Procedure\n"
        "- **Estimator**: Convert the comparison data into a spectral ranking model and estimate an overall score for each item.\n"
        "- **Ranking rule**: Sort items by estimated score, with higher values ranked first when `bigbetter=1`.\n"
        f"- **Analysis scope**: **{len(results.items)} items** included in this run.\n\n"
        "### Uncertainty Quantification\n"
        "- **Interval type**: 95% bootstrap confidence intervals for rank uncertainty.\n"
        "- **Bootstrap engine**: Gaussian multiplier bootstrap "
        "(Spectral Ranking Inferences based on General Multiway Comparisons, "
        "https://arxiv.org/html/2308.02918).\n\n"
        "### Run Configuration\n"
        f"- **Input file**: `{input_path}`\n"
        f"- **Bootstrap iterations (B)**: {B}\n"
        f"- **Random seed**: {seed}"
    )

    # ── Limitations ──────────────────────────────────────────────────────
    limitations = (
        "- CI overlap is **not** a formal hypothesis test; use overlap as uncertainty context, not proof of equivalence.\n"
        "- Sparse comparison data can widen intervals and reduce rank precision.\n"
        "- Results assume strong connectivity of the comparison graph "
        "(required for a unique stationary distribution).\n"
        "- Ranks are derived from estimated-score ordering; CI overlap affects "
        "rank certainty."
    )

    fallback = {
        "summary": summary,
        "results_narrative": results_narrative,
        "methods": methods,
        "limitations": limitations,
    }
    fallback = {k: _integerize_ci_text(v) for k, v in fallback.items()}

    # ── LLM generation (optional) ────────────────────────────────────────
    client = get_llm_client()
    if not client.is_available():
        return fallback

    payload = {
        "results": {
            "items": results.items,
            "theta_hat": results.theta_hat,
            "ranks": results.ranks,
            "ci_lower": results.ci_lower,
            "ci_upper": results.ci_upper,
            "metadata": (
                results.metadata.model_dump() if results.metadata else None
            ),
        },
        "session_meta": session_meta,
        "analysis": {
            "clusters": analysis["clusters"],
            "n_clusters": analysis["n_clusters"],
            "near_ties_with_top": analysis["near_ties_with_top"],
            "largest_gap": analysis["largest_gap"],
            "widest_ci_item": analysis["widest_ci_item"],
            "narrowest_ci_item": analysis["narrowest_ci_item"],
        },
    }
    try:
        validation_feedback: list[str] = []
        for _ in range(3):
            attempt_payload = dict(payload)
            if validation_feedback:
                attempt_payload["validation_feedback"] = validation_feedback
            llm_output = client.generate_json(
                "generate_report", payload=attempt_payload, max_completion_tokens=4000
            )
            llm_narrative = {k: str(llm_output.get(k) or fallback[k]) for k in fallback}
            # Keep Methodology deterministic and clean for consistent report quality.
            llm_narrative["methods"] = methods
            issues = _validate_llm_narrative(llm_narrative, results, analysis)
            if not issues:
                return {k: _integerize_ci_text(v) for k, v in llm_narrative.items()}
            validation_feedback = issues
        return fallback
    except (LLMCallError, ValueError, TypeError, KeyError):
        return fallback


# ---------------------------------------------------------------------------
# Hints (static, always attached)
# ---------------------------------------------------------------------------

_HINTS: list[HintSpec] = [
    HintSpec(
        hint_id="hint-theta-hat",
        title="Estimated Score",
        body=(
            "Score inferred from the spectral ranking model. Higher values indicate "
            "stronger estimated preference when `bigbetter=1`, but rank and confidence intervals "
            "are usually easier to interpret directly."
        ),
        kind=HintKind.DEFINITION,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-ci",
        title="95% Confidence Interval",
        body=(
            "Computed with Gaussian multiplier bootstrap "
            "(Spectral Ranking Inferences based on General Multiway Comparisons, "
            "https://arxiv.org/html/2308.02918). Narrower intervals mean more precise estimates; "
            "wider intervals mean more uncertainty."
        ),
        kind=HintKind.DEFINITION,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-ci-overlap",
        title="How to Read Overlapping Intervals",
        body=(
            "If two intervals overlap, the ordering between those items is still uncertain. "
            "If they do not overlap, the separation is more clearly measurable, but overlap does "
            "not prove the items are equivalent."
        ),
        kind=HintKind.CAVEAT,
        sources=[],
    ),
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    results: RankingResults,
    session_meta: dict[str, Any],
    plots: list[PlotSpec],
) -> ReportOutput:
    """Generate a single-page markdown report with citable blocks.

    The markdown uses ``<section data-omni-block-id data-omni-kind>``
    wrappers for the Quote UX.  Blank lines separate HTML tags from
    markdown content so that CommonMark parsers render headings, bold,
    bullets, tables, etc. correctly inside sections.
    """
    render_results = results.model_copy(deep=True)
    render_results.items = [_sanitize_inline_text(item) for item in results.items]

    narrative = _build_llm_narrative(render_results, session_meta)
    top_item, top_score, top_rank = _top_item(render_results)
    analysis = _analyze_ranking(render_results)
    ranking_table = _render_ranking_table(render_results)

    # ── Block IDs ────────────────────────────────────────────────────────
    summary_bid = _stable_block_id(
        "summary",
        {
            "top_item": top_item,
            "top_rank": top_rank,
            "top_score": round(float(top_score), 8),
        },
    )
    result_bid = _stable_block_id(
        "result",
        {
            "items": results.items,
            "ranks": results.ranks,
            "theta_hat": [round(x, 8) for x in results.theta_hat],
        },
    )
    table_bid = _stable_block_id(
        "table",
        {
            "rows": len(results.items),
            "items": results.items,
            "ranks": results.ranks,
        },
    )
    # ── Construct named section markdowns ────────────────────────────────
    summary_md = _section(
        summary_bid,
        "summary",
        f"## Executive Summary\n\n{narrative['summary']}",
    )
    result_md = _section(
        result_bid,
        "result",
        "## Ranking Results",
    )
    table_md = _section(
        table_bid,
        "table",
        ranking_table,
    )
    # ── Figures (interleaved in the narrative) ───────────────────────────
    figure_mds: list[str] = []
    figure_blocks: list[CitationBlock] = []
    artifacts: list[ArtifactRef] = []

    for idx, plot in enumerate(plots, start=1):
        fig_bid = plot.block_id or _stable_block_id(
            "figure",
            {"type": plot.type, "index": idx, "data": plot.data, "config": plot.config},
        )
        cap_plain = plot.caption_plain or plot.type
        cap_acad = plot.caption_academic or plot.type
        if plot.type == "ci_forest":
            figure_title = "Ranking Confidence Intervals"
        elif plot.type == "indicator_rankings_combined":
            ind = (plot.data or {}).get("indicator_col") or "phenotype"
            tc = ind[0].upper() + ind[1:].lower() if ind else "Indicator"
            figure_title = f"Rankings by {tc}"
        elif plot.type == "normalized_ranking_over_indicator":
            ind = (plot.data or {}).get("indicator_col") or "phenotype"
            tc = ind[0].upper() + ind[1:].lower() if ind else "Indicator"
            figure_title = f"Normalized Ranks by {tc}"
        elif plot.type == "indicator_rankings_heatmap":
            ind = (plot.data or {}).get("indicator_col") or "phenotype"
            tc = ind[0].upper() + ind[1:].lower() if ind else "Indicator"
            figure_title = f"Rank Heatmap by {tc}"
        else:
            figure_title = cap_plain

        fig_body = (
            f"## {figure_title}\n\n"
            f"![{figure_title}]({plot.svg_path})\n\n"
            f"*{cap_acad}*"
        )
        if plot.type == "indicator_rankings_combined":
            indicator_table_md = _render_indicator_ranking_table(plot)
            if indicator_table_md:
                fig_body += f"\n\n{indicator_table_md}"
        fig_md = _section(fig_bid, "figure", fig_body)
        figure_mds.append(fig_md)

        figure_blocks.append(
            CitationBlock(
                block_id=fig_bid,
                kind=CitationKind.FIGURE,
                markdown=fig_md,
                text=f"{figure_title}: {cap_acad}",
                hint_ids=plot.hint_ids,
                artifact_paths=[plot.svg_path],
            )
        )
        mime = "image/png" if plot.svg_path.lower().endswith(".png") else "image/svg+xml"
        artifacts.append(
            ArtifactRef(
                kind="figure",
                path=plot.svg_path,
                title=plot.type,
                mime_type=mime,
            )
        )

    # ── Assemble full markdown ───────────────────────────────────────────
    parts: list[str] = [
        "# OmniRank Report",
        result_md,
        table_md,
        summary_md,
        *figure_mds,
    ]

    full_markdown = "\n\n".join(parts)

    # ── Citation blocks ──────────────────────────────────────────────────
    citation_blocks: list[CitationBlock] = [
        CitationBlock(
            block_id=summary_bid,
            kind=CitationKind.SUMMARY,
            markdown=summary_md,
            text=narrative["summary"],
            hint_ids=[],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=result_bid,
            kind=CitationKind.RESULT,
            markdown=result_md,
            text="Ranking results section",
            hint_ids=[],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=table_bid,
            kind=CitationKind.TABLE,
            markdown=table_md,
            text="Ranking table",
            hint_ids=["hint-theta-hat", "hint-ci"],
            artifact_paths=[],
        ),
        *figure_blocks,
    ]

    # ── Key findings (machine-readable) ──────────────────────────────────
    key_findings: dict[str, Any] = {
        "top_item": top_item,
        "top_rank": top_rank,
        "top_score": top_score,
        "n_items": len(results.items),
        "n_clusters": analysis["n_clusters"],
        "cluster_items": analysis["clusters"],
        "near_ties_with_top": analysis["near_ties_with_top"],
        "largest_gap": analysis["largest_gap"],
    }

    return ReportOutput(
        markdown=full_markdown,
        key_findings=key_findings,
        artifacts=artifacts,
        hints=list(_HINTS),
        citation_blocks=citation_blocks,
    )
