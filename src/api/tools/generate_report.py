"""Tool: generate_report.

Generates a single-page, publication-ready markdown report with citable blocks,
interleaved figures, and structured analysis following the SOP Deep Research
style contract.
"""

from __future__ import annotations

import hashlib
import json
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
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


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
    """Render an enhanced ranking table with CI width and gap columns."""
    order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])
    top_score = results.theta_hat[order[0]]

    lines = [
        "| Rank | Item | Score (theta_hat) | 95% CI | CI Width | Gap to #1 |",
        "|---:|---|---:|---|---:|---:|",
    ]
    for idx in order:
        item = _escape_table_cell(results.items[idx])
        score = results.theta_hat[idx]
        ci_lo = results.ci_lower[idx]
        ci_hi = results.ci_upper[idx]
        ci_w = ci_hi - ci_lo
        gap = top_score - score
        gap_str = f"{gap:.4f}" if results.ranks[idx] > 1 else "---"

        lines.append(
            f"| {results.ranks[idx]} | {item} | {score:.4f} "
            f"| [{ci_lo:.4f}, {ci_hi:.4f}] | {ci_w:.4f} | {gap_str} |"
        )
    return "\n".join(lines)


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
    top_item, top_score, top_rank = _top_item(results)
    analysis = _analyze_ranking(results)
    order = analysis["order"]

    top_idx = order[0]
    top_ci_lo = results.ci_lower[top_idx]
    top_ci_hi = results.ci_upper[top_idx]
    near_ties = analysis["near_ties_with_top"]
    clusters = analysis["clusters"]

    # ── Executive Summary ────────────────────────────────────────────────
    if near_ties:
        uncertainty = (
            f"However, the 95% bootstrap confidence intervals for **{top_item}** "
            f"overlap with {', '.join(f'**{t}**' for t in near_ties)}, "
            "indicating uncertainty in the exact top position."
        )
    else:
        uncertainty = (
            f"The confidence interval for **{top_item}** does not overlap with "
            "the next-ranked item, suggesting clear separation at the top."
        )

    bullets: list[str] = [
        f"**{top_item}** ranks #1 with score {top_score:.4f} "
        f"(95% CI: [{top_ci_lo:.4f}, {top_ci_hi:.4f}])",
    ]
    if len(clusters) > 1:
        bullets.append(
            f"Items form **{len(clusters)} performance tier(s)** based on "
            "confidence interval overlap"
        )
    if analysis["largest_gap"]:
        lg = analysis["largest_gap"]
        bullets.append(
            f"Largest score gap ({lg['gap']:.4f}) between "
            f"**{lg['from']}** and **{lg['to']}**"
        )
    bullets.append(
        f"CI widths range from {analysis['narrowest_ci']:.4f} "
        f"(**{analysis['narrowest_ci_item']}**) to "
        f"{analysis['widest_ci']:.4f} (**{analysis['widest_ci_item']}**)"
    )

    summary = (
        f"**{top_item}** ranks highest with an estimated preference score "
        f"(theta\\_hat) of **{top_score:.4f}** (Rank #{top_rank}). "
        f"{uncertainty}\n\n"
        "**Key Findings:**\n\n"
        + "\n".join(f"- {b}" for b in bullets)
    )

    # ── Results Narrative ────────────────────────────────────────────────
    parts: list[str] = [
        f"The spectral ranking analysis evaluated **{len(results.items)} items** "
        "from the provided comparison data."
    ]
    for tier_i, cluster in enumerate(clusters):
        if tier_i == 0:
            label = "Top Tier"
        elif tier_i == len(clusters) - 1:
            label = "Bottom Tier"
        else:
            label = f"Tier {tier_i + 1}"

        items_md = ", ".join(f"**{it}**" for it in cluster)
        if len(cluster) == 1:
            ci = results.items.index(cluster[0])
            parts.append(
                f"**{label}**: {items_md} "
                f"(theta\\_hat = {results.theta_hat[ci]:.4f}, "
                f"CI: [{results.ci_lower[ci]:.4f}, {results.ci_upper[ci]:.4f}])."
            )
        else:
            scores = [results.theta_hat[results.items.index(it)] for it in cluster]
            parts.append(
                f"**{label}**: {items_md} "
                f"(scores range {min(scores):.4f} -- {max(scores):.4f}, "
                "overlapping CIs indicate comparable performance within this tier)."
            )
    results_narrative = " ".join(parts)

    # ── Targeted Comparisons ─────────────────────────────────────────────
    targeted = ""
    if len(order) >= 2:
        s_idx = order[1]
        s_item = results.items[s_idx]
        s_score = results.theta_hat[s_idx]
        s_ci_lo = results.ci_lower[s_idx]
        s_ci_hi = results.ci_upper[s_idx]

        overlap = (
            results.ci_upper[s_idx] >= results.ci_lower[top_idx]
            and results.ci_lower[s_idx] <= results.ci_upper[top_idx]
        )
        if overlap:
            targeted = (
                f"**{top_item} vs. {s_item}**: "
                f"{top_item} ({top_score:.4f}, CI [{top_ci_lo:.4f}, {top_ci_hi:.4f}]) "
                f"and {s_item} ({s_score:.4f}, CI [{s_ci_lo:.4f}, {s_ci_hi:.4f}]) "
                "have **overlapping confidence intervals**. While the point "
                f"estimate favours {top_item}, the overlap means the true "
                "ordering remains uncertain at the 95% level."
            )
        else:
            targeted = (
                f"**{top_item} vs. {s_item}**: "
                f"{top_item} ({top_score:.4f}, CI [{top_ci_lo:.4f}, {top_ci_hi:.4f}]) "
                f"and {s_item} ({s_score:.4f}, CI [{s_ci_lo:.4f}, {s_ci_hi:.4f}]) "
                "have **non-overlapping confidence intervals**, suggesting "
                f"measurable separation between these two items."
            )

    # ── Methods ──────────────────────────────────────────────────────────
    B = session_meta.get("B", 2000)
    seed = session_meta.get("seed", 42)
    methods = (
        f"**Estimator**: Spectral ranking estimates latent preference scores "
        f"(theta\\_hat) from the stationary distribution of a Markov chain "
        f"constructed from comparison data. "
        f"**Uncertainty**: 95% bootstrap confidence intervals via Gaussian "
        f"multiplier bootstrap (Fan et al., 2023) with **B = {B}** iterations "
        f"and **seed = {seed}**. "
        f"**Scope**: {len(results.items)} items analyzed."
    )

    # ── Limitations ──────────────────────────────────────────────────────
    limitations = (
        "- CI overlap is **not** a formal hypothesis test; "
        "overlapping CIs indicate uncertainty, not equivalence.\n"
        "- Sparse comparison data can widen intervals and reduce rank precision.\n"
        "- Results assume strong connectivity of the comparison graph "
        "(required for a unique stationary distribution).\n"
        "- Ranks are derived from theta\\_hat ordering; CI overlap affects "
        "rank certainty."
    )

    # ── Reproducibility ──────────────────────────────────────────────────
    fpath = session_meta.get("current_file_path", "N/A")
    rpath = session_meta.get(
        "r_script_path", "src/spectral_ranking/spectral_ranking.R"
    )
    reproducibility = (
        f"- **Input data**: `{fpath}`\n"
        f"- **Engine**: `{rpath}`\n"
        f"- **Bootstrap iterations (B)**: {B}\n"
        f"- **Random seed**: {seed}\n"
        "- **Artifacts**: Ranking tables and SVG visualizations stored in "
        "session artifact directory."
    )

    fallback = {
        "summary": summary,
        "results_narrative": results_narrative,
        "targeted_comparisons": targeted,
        "methods": methods,
        "limitations": limitations,
        "reproducibility": reproducibility,
    }

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
        llm_output = client.generate_json(
            "generate_report", payload=payload, max_completion_tokens=4000
        )
        return {
            k: str(llm_output.get(k) or fallback[k])
            for k in fallback
        }
    except (LLMCallError, ValueError, TypeError, KeyError) as exc:
        raise LLMCallError(f"LLM report generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Hints (static, always attached)
# ---------------------------------------------------------------------------

_HINTS: list[HintSpec] = [
    HintSpec(
        hint_id="hint-theta-hat",
        title="theta_hat (Estimated Preference Score)",
        body=(
            "The estimated latent preference score from spectral ranking, "
            "representing the stationary distribution of a Markov chain "
            "constructed from comparison data. Higher values indicate "
            "stronger preference (when bigbetter=1)."
        ),
        kind=HintKind.DEFINITION,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-ci",
        title="95% Bootstrap Confidence Interval",
        body=(
            "Computed via Gaussian multiplier bootstrap (Fan et al., 2023). "
            "The interval width reflects estimation precision; wider "
            "intervals suggest greater uncertainty."
        ),
        kind=HintKind.DEFINITION,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-ci-overlap",
        title="CI Overlap Caveat",
        body=(
            "Overlapping CIs indicate uncertainty in relative ranking, not "
            "formal equivalence. Non-overlapping intervals suggest measurable "
            "separation, but overlap does not prove items are equivalent."
        ),
        kind=HintKind.CAVEAT,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-spectral-ranking",
        title="Spectral Ranking Method",
        body=(
            "Constructs a Markov chain from comparison data and estimates "
            "preference scores from its stationary distribution. Provides "
            "minimax-optimal ranking inference for multiway comparisons."
        ),
        kind=HintKind.METHOD,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-bootstrap",
        title="Gaussian Multiplier Bootstrap",
        body=(
            "A resampling method using Gaussian multipliers to approximate "
            "the sampling distribution of the spectral estimator, enabling "
            "CI construction without distributional assumptions."
        ),
        kind=HintKind.METHOD,
        sources=[],
    ),
    HintSpec(
        hint_id="hint-rank-interpretation",
        title="Rank Interpretation",
        body=(
            "Ranks are derived from ordering theta_hat values. Rank 1 = "
            "highest preference score. When CIs overlap, rank ordering is "
            "uncertain and should be interpreted cautiously."
        ),
        kind=HintKind.DEFINITION,
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
    narrative = _build_llm_narrative(results, session_meta)
    top_item, top_score, top_rank = _top_item(results)
    analysis = _analyze_ranking(results)
    ranking_table = _render_ranking_table(results)

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
    methods_bid = _stable_block_id(
        "method",
        {"B": session_meta.get("B", 2000), "seed": session_meta.get("seed", 42)},
    )
    limitation_bid = _stable_block_id(
        "limitation", {"n_items": len(results.items)}
    )
    repro_bid = _stable_block_id(
        "repro",
        {
            "current_file_path": str(session_meta.get("current_file_path", "")),
            "r_script_path": str(session_meta.get("r_script_path", "")),
            "B": session_meta.get("B", 2000),
            "seed": session_meta.get("seed", 42),
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
        f"## Ranking Results\n\n{narrative['results_narrative']}",
    )
    table_md = _section(
        table_bid,
        "table",
        f"### Full Ranking Table\n\n{ranking_table}",
    )
    methods_md = _section(
        methods_bid,
        "method",
        f"## Methodology\n\n{narrative['methods']}",
    )
    limitation_md = _section(
        limitation_bid,
        "limitation",
        f"### Limitations and Assumptions\n\n{narrative['limitations']}",
    )
    repro_md = _section(
        repro_bid,
        "repro",
        f"### Reproducibility\n\n{narrative['reproducibility']}",
    )

    # ── Figures (interleaved in the narrative) ───────────────────────────
    figure_mds: list[str] = []
    figure_blocks: list[CitationBlock] = []
    artifacts: list[ArtifactRef] = []

    for idx, plot in enumerate(plots):
        fig_bid = plot.block_id or _stable_block_id(
            "figure",
            {"type": plot.type, "index": idx, "data": plot.data, "config": plot.config},
        )
        cap_plain = plot.caption_plain or plot.type
        cap_acad = plot.caption_academic or plot.type

        fig_body = (
            f"**Figure {idx + 1}: {cap_plain}**\n\n"
            f"![{cap_plain}]({plot.svg_path})\n\n"
            f"*{cap_acad}*"
        )
        fig_md = _section(fig_bid, "figure", fig_body)
        figure_mds.append(fig_md)

        figure_blocks.append(
            CitationBlock(
                block_id=fig_bid,
                kind=CitationKind.FIGURE,
                markdown=fig_md,
                text=f"Figure {idx + 1}: {cap_acad}",
                hint_ids=plot.hint_ids,
                artifact_paths=[plot.svg_path],
            )
        )
        artifacts.append(
            ArtifactRef(
                kind="figure",
                path=plot.svg_path,
                title=plot.type,
                mime_type="image/svg+xml",
            )
        )

    # ── Targeted Comparisons (optional) ──────────────────────────────────
    comparison_bid: str | None = None
    comparison_md = ""
    targeted_text = narrative.get("targeted_comparisons", "").strip()
    if targeted_text:
        comparison_bid = _stable_block_id(
            "comparison",
            {"top_item": top_item, "n_items": len(results.items)},
        )
        comparison_md = _section(
            comparison_bid,
            "comparison",
            f"### Targeted Comparisons\n\n{targeted_text}",
        )

    # ── Assemble full markdown ───────────────────────────────────────────
    parts: list[str] = [
        "# OmniRank Spectral Ranking Report",
        summary_md,
        "---",
        result_md,
        table_md,
        *figure_mds,
    ]
    if comparison_md:
        parts.append(comparison_md)
    parts.append("---")
    parts.extend([methods_md, limitation_md, repro_md])

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
            text=narrative["results_narrative"],
            hint_ids=["hint-theta-hat", "hint-ci", "hint-rank-interpretation"],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=table_bid,
            kind=CitationKind.TABLE,
            markdown=table_md,
            text="Ranking table",
            hint_ids=["hint-theta-hat", "hint-ci", "hint-rank-interpretation"],
            artifact_paths=[],
        ),
        *figure_blocks,
    ]

    if comparison_bid and comparison_md:
        citation_blocks.append(
            CitationBlock(
                block_id=comparison_bid,
                kind=CitationKind.COMPARISON,
                markdown=comparison_md,
                text=targeted_text,
                hint_ids=["hint-ci", "hint-ci-overlap"],
                artifact_paths=[],
            )
        )

    citation_blocks.extend(
        [
            CitationBlock(
                block_id=methods_bid,
                kind=CitationKind.METHOD,
                markdown=methods_md,
                text=narrative["methods"],
                hint_ids=["hint-spectral-ranking", "hint-bootstrap"],
                artifact_paths=[],
            ),
            CitationBlock(
                block_id=limitation_bid,
                kind=CitationKind.LIMITATION,
                markdown=limitation_md,
                text=narrative["limitations"],
                hint_ids=["hint-ci", "hint-ci-overlap"],
                artifact_paths=[],
            ),
            CitationBlock(
                block_id=repro_bid,
                kind=CitationKind.REPRO,
                markdown=repro_md,
                text=narrative["reproducibility"],
                hint_ids=[],
                artifact_paths=[a.path for a in artifacts],
            ),
        ]
    )

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
