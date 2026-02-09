"""Tool: generate_report."""

from __future__ import annotations

import hashlib
import html
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


def _stable_block_id(prefix: str, payload: Any) -> str:
    """Build deterministic block id from stable JSON payload."""
    if isinstance(payload, str):
        serialized = payload
    else:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(serialized.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _escape_text(value: Any) -> str:
    """Escape text content for safe markdown/HTML embedding."""
    text = str(value)
    text = text.replace("\r", "")
    return html.escape(text, quote=False)


def _escape_table_cell(value: Any) -> str:
    """Escape markdown table cell content."""
    text = _escape_text(value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    return text


def _top_item(results: RankingResults) -> tuple[str, float, int]:
    best_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
    return results.items[best_idx], results.theta_hat[best_idx], results.ranks[best_idx]


def _render_ranking_table(results: RankingResults) -> str:
    order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])
    lines = ["| Rank | Item | theta_hat | CI |", "|---:|---|---:|---:|"]
    for idx in order:
        item = _escape_table_cell(results.items[idx])
        lines.append(
            f"| {results.ranks[idx]} | {item} | {results.theta_hat[idx]:.4f} | "
            f"[{results.ci_lower[idx]:.2f}, {results.ci_upper[idx]:.2f}] |"
        )
    return "\n".join(lines)


def _build_llm_narrative(results: RankingResults, session_meta: dict[str, Any]) -> dict[str, str]:
    """Generate report narrative via LLM with deterministic fallback."""
    top_item, top_score, top_rank = _top_item(results)
    fallback = {
        "summary": (
            f"Top-ranked item is {top_item} (rank {top_rank}) with theta_hat={top_score:.4f}. "
            "Use confidence intervals to communicate uncertainty."
        ),
        "results_narrative": (
            "Ranking results are presented as point estimates with uncertainty intervals. "
            "Near-ties should be interpreted cautiously."
        ),
        "methods": (
            f"Spectral ranking with Gaussian multiplier bootstrap was applied using "
            f"B={session_meta.get('B', 2000)} and seed={session_meta.get('seed', 42)}."
        ),
        "limitations": (
            "Confidence interval overlap is not a formal hypothesis test. "
            "Sparse comparisons can widen uncertainty intervals."
        ),
        "reproducibility": (
            f"Input file: {session_meta.get('current_file_path', 'N/A')}. "
            f"Engine script: {session_meta.get('r_script_path', 'src/spectral_ranking/spectral_ranking.R')}."
        ),
    }

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
            "metadata": results.metadata.model_dump() if results.metadata else None,
        },
        "session_meta": session_meta,
    }
    try:
        llm_output = client.generate_json("generate_report", payload=payload, max_completion_tokens=1200)
        return {
            "summary": str(llm_output.get("summary") or fallback["summary"]),
            "results_narrative": str(llm_output.get("results_narrative") or fallback["results_narrative"]),
            "methods": str(llm_output.get("methods") or fallback["methods"]),
            "limitations": str(llm_output.get("limitations") or fallback["limitations"]),
            "reproducibility": str(llm_output.get("reproducibility") or fallback["reproducibility"]),
        }
    except (LLMCallError, ValueError, TypeError, KeyError) as exc:
        raise LLMCallError(f"LLM report generation failed: {exc}") from exc


def generate_report(
    results: RankingResults,
    session_meta: dict[str, Any],
    plots: list[PlotSpec],
) -> ReportOutput:
    """Generate single-page markdown report with citable blocks."""
    narrative = _build_llm_narrative(results, session_meta)
    top_item, top_score, top_rank = _top_item(results)

    summary_text = _escape_text(narrative["summary"])
    results_narrative = _escape_text(narrative["results_narrative"])
    methods_text = _escape_text(narrative["methods"])
    limitations_text = _escape_text(narrative["limitations"])
    reproducibility_text = _escape_text(narrative["reproducibility"])

    ranking_table = _render_ranking_table(results)

    summary_block_id = _stable_block_id(
        "summary",
        {"top_item": top_item, "top_rank": top_rank, "top_score": round(float(top_score), 8)},
    )
    result_block_id = _stable_block_id(
        "result",
        {"items": results.items, "ranks": results.ranks, "theta_hat": [round(x, 8) for x in results.theta_hat]},
    )
    table_block_id = _stable_block_id(
        "table",
        {"rows": len(results.items), "items": results.items, "ranks": results.ranks},
    )
    methods_block_id = _stable_block_id(
        "method",
        {"B": session_meta.get("B", 2000), "seed": session_meta.get("seed", 42)},
    )
    limitation_block_id = _stable_block_id("limitation", {"n_items": len(results.items)})
    reproducibility_block_id = _stable_block_id(
        "repro",
        {
            "current_file_path": str(session_meta.get("current_file_path", "")),
            "r_script_path": str(session_meta.get("r_script_path", "")),
            "B": session_meta.get("B", 2000),
            "seed": session_meta.get("seed", 42),
        },
    )

    figure_sections: list[str] = []
    figure_blocks: list[CitationBlock] = []
    artifacts: list[ArtifactRef] = []
    for index, plot in enumerate(plots):
        plot_block_id = plot.block_id or _stable_block_id(
            "figure",
            {"type": plot.type, "index": index, "data": plot.data, "config": plot.config},
        )
        caption_plain = _escape_text(plot.caption_plain or plot.type)
        caption_academic = _escape_text(plot.caption_academic or plot.type)
        figure_markdown = (
            f"<section data-omni-block-id=\"{plot_block_id}\" data-omni-kind=\"figure\">\n"
            f"**Figure ({_escape_text(plot.type)})**\n\n"
            f"![{caption_plain}]({_escape_text(plot.svg_path)})\n\n"
            f"{caption_academic}\n"
            "</section>"
        )
        figure_sections.append(figure_markdown)
        figure_blocks.append(
            CitationBlock(
                block_id=plot_block_id,
                kind=CitationKind.FIGURE,
                markdown=figure_markdown,
                text=f"Figure {plot.type}: {caption_academic}",
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

    markdown_parts = [
        "# OmniRank Report",
        f"<section data-omni-block-id=\"{summary_block_id}\" data-omni-kind=\"summary\">\n{summary_text}\n</section>",
        f"<section data-omni-block-id=\"{result_block_id}\" data-omni-kind=\"result\">\n{results_narrative}\n</section>",
        f"<section data-omni-block-id=\"{table_block_id}\" data-omni-kind=\"table\">\n{ranking_table}\n</section>",
        *figure_sections,
        f"<section data-omni-block-id=\"{methods_block_id}\" data-omni-kind=\"method\">\n{methods_text}\n</section>",
        f"<section data-omni-block-id=\"{limitation_block_id}\" data-omni-kind=\"limitation\">\n{limitations_text}\n</section>",
        f"<section data-omni-block-id=\"{reproducibility_block_id}\" data-omni-kind=\"repro\">\n{reproducibility_text}\n</section>",
    ]

    hints = [
        HintSpec(
            hint_id="hint-theta-hat",
            title="theta_hat",
            body="Estimated latent preference score from spectral ranking.",
            kind=HintKind.DEFINITION,
            sources=[],
        ),
        HintSpec(
            hint_id="hint-ci",
            title="Confidence Interval",
            body="Bootstrap interval indicating uncertainty of inferred rank position.",
            kind=HintKind.DEFINITION,
            sources=[],
        ),
    ]

    citation_blocks = [
        CitationBlock(
            block_id=summary_block_id,
            kind=CitationKind.SUMMARY,
            markdown=markdown_parts[1],
            text=summary_text,
            hint_ids=[],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=result_block_id,
            kind=CitationKind.RESULT,
            markdown=markdown_parts[2],
            text=results_narrative,
            hint_ids=["hint-theta-hat", "hint-ci"],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=table_block_id,
            kind=CitationKind.TABLE,
            markdown=markdown_parts[3],
            text="Ranking table",
            hint_ids=["hint-theta-hat", "hint-ci"],
            artifact_paths=[],
        ),
        *figure_blocks,
        CitationBlock(
            block_id=methods_block_id,
            kind=CitationKind.METHOD,
            markdown=markdown_parts[-3],
            text=methods_text,
            hint_ids=[],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=limitation_block_id,
            kind=CitationKind.LIMITATION,
            markdown=markdown_parts[-2],
            text=limitations_text,
            hint_ids=["hint-ci"],
            artifact_paths=[],
        ),
        CitationBlock(
            block_id=reproducibility_block_id,
            kind=CitationKind.REPRO,
            markdown=markdown_parts[-1],
            text=reproducibility_text,
            hint_ids=[],
            artifact_paths=[artifact.path for artifact in artifacts],
        ),
    ]

    key_findings: dict[str, Any] = {
        "top_item": top_item,
        "top_rank": top_rank,
        "top_score": top_score,
        "n_items": len(results.items),
    }

    return ReportOutput(
        markdown="\n\n".join(markdown_parts),
        key_findings=key_findings,
        artifacts=artifacts,
        hints=hints,
        citation_blocks=citation_blocks,
    )
