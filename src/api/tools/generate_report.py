"""Tool: generate_report."""

from __future__ import annotations

import hashlib
from typing import Any

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


def _block_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _top_item(results: RankingResults) -> tuple[str, float, int]:
    best_idx = min(range(len(results.ranks)), key=lambda i: results.ranks[i])
    return results.items[best_idx], results.theta_hat[best_idx], results.ranks[best_idx]


def _render_ranking_table(results: RankingResults) -> str:
    order = sorted(range(len(results.ranks)), key=lambda i: results.ranks[i])
    lines = ["| Rank | Item | theta_hat | CI |", "|---:|---|---:|---:|"]
    for idx in order:
        lines.append(
            f"| {results.ranks[idx]} | {results.items[idx]} | {results.theta_hat[idx]:.4f} | [{results.ci_lower[idx]:.2f}, {results.ci_upper[idx]:.2f}] |"
        )
    return "\n".join(lines)


def generate_report(
    results: RankingResults,
    session_meta: dict[str, Any],
    plots: list[PlotSpec],
) -> ReportOutput:
    """Generate single-page markdown report with citable blocks."""
    top_item, top_score, top_rank = _top_item(results)

    summary_text = (
        f"Top-ranked item is **{top_item}** (rank {top_rank}) with theta_hat={top_score:.4f}. "
        "Interpret uncertainty using confidence interval ranges rather than point ranks alone."
    )
    summary_block_id = _block_id("summary", summary_text)

    results_table = _render_ranking_table(results)
    table_block_id = _block_id("table", results_table)

    methods_text = (
        "We use spectral ranking with Gaussian multiplier bootstrap intervals. "
        f"Configuration uses B={session_meta.get('B', 2000)} and seed={session_meta.get('seed', 42)}."
    )
    methods_block_id = _block_id("method", methods_text)

    limitation_text = (
        "Confidence interval overlap should not be interpreted as a formal hypothesis test. "
        "Sparse comparisons may increase uncertainty width."
    )
    limitation_block_id = _block_id("limitation", limitation_text)

    reproducibility_text = (
        f"Input file: `{session_meta.get('current_file_path', 'N/A')}`. "
        f"Engine script: `{session_meta.get('r_script_path', 'src/spectral_ranking/spectral_ranking.R')}`."
    )
    reproducibility_block_id = _block_id("repro", reproducibility_text)

    figure_sections: list[str] = []
    figure_blocks: list[CitationBlock] = []
    artifacts: list[ArtifactRef] = []
    for plot in plots:
        figure_sections.append(
            f"<section data-omni-block-id=\"{plot.block_id}\" data-omni-kind=\"figure\">\n"
            f"**Figure ({plot.type})**\n\n"
            f"![{plot.caption_plain}]({plot.svg_path})\n\n"
            f"{plot.caption_academic}\n"
            "</section>"
        )
        figure_blocks.append(
            CitationBlock(
                block_id=plot.block_id,
                kind=CitationKind.FIGURE,
                markdown=figure_sections[-1],
                text=f"Figure {plot.type}: {plot.caption_academic}",
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
        f"<section data-omni-block-id=\"{table_block_id}\" data-omni-kind=\"table\">\n{results_table}\n</section>",
        *figure_sections,
        f"<section data-omni-block-id=\"{methods_block_id}\" data-omni-kind=\"method\">\n{methods_text}\n</section>",
        f"<section data-omni-block-id=\"{limitation_block_id}\" data-omni-kind=\"limitation\">\n{limitation_text}\n</section>",
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
            block_id=table_block_id,
            kind=CitationKind.TABLE,
            markdown=markdown_parts[2],
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
            text=limitation_text,
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
