"""Unit tests for visualization/report/question tools."""

from pathlib import Path

from core.schemas import PlotSpec, QuotePayload, RankingResults
from tools.answer_question import answer_question
from tools.generate_report import generate_report
from tools.generate_visualizations import generate_visualizations


def _sample_results() -> RankingResults:
    return RankingResults(
        items=["A", "B", "C"],
        theta_hat=[0.6, 0.3, -0.1],
        ranks=[1, 2, 3],
        ci_lower=[1.0, 1.0, 2.0],
        ci_upper=[2.0, 3.0, 3.0],
        indicator_value=None,
    )


def test_generate_visualizations_deterministic_svg(tmp_path: Path):
    results = _sample_results()
    artifact_dir = tmp_path / "artifacts"

    first = generate_visualizations(results=results, viz_types=["ranking_bar", "ci_forest"], artifact_dir=str(artifact_dir))
    second = generate_visualizations(results=results, viz_types=["ranking_bar", "ci_forest"], artifact_dir=str(artifact_dir))

    assert len(first.plots) == 2
    assert len(second.plots) == 2

    for plot_a, plot_b in zip(first.plots, second.plots, strict=True):
        bytes_a = Path(plot_a.svg_path).read_bytes()
        bytes_b = Path(plot_b.svg_path).read_bytes()
        assert bytes_a == bytes_b
        assert plot_a.block_id
        assert plot_a.caption_plain
        assert plot_a.caption_academic


def test_generate_report_contains_required_sections_and_citation_blocks(tmp_path: Path):
    results = _sample_results()
    plot = PlotSpec(
        type="ranking_bar",
        data={},
        config={},
        svg_path=str(tmp_path / "ranking_bar.svg"),
        block_id="figure-ranking-bar-123",
        caption_plain="Plain caption",
        caption_academic="Academic caption",
        hint_ids=["hint-ci"],
    )

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[plot],
    )

    assert "<section data-omni-block-id=" in report.markdown
    assert 'data-omni-kind="summary"' in report.markdown
    assert 'data-omni-kind="table"' in report.markdown
    assert 'data-omni-kind="method"' in report.markdown
    assert 'data-omni-kind="limitation"' in report.markdown
    assert 'data-omni-kind="repro"' in report.markdown
    assert len(report.citation_blocks) > 0
    assert len(report.hints) > 0


def test_answer_question_without_quotes():
    results = _sample_results()
    answer = answer_question(
        question="What is the top item?",
        results=results,
        citation_blocks={},
        quotes=[],
    )

    assert "Top-ranked item is" in answer.answer
    assert len(answer.used_citation_block_ids) == 0


def test_answer_question_with_quotes_uses_block_ids():
    results = _sample_results()
    quote_block_id = "summary-abc"
    answer = answer_question(
        question="Explain this quote.",
        results=results,
        citation_blocks={quote_block_id: "Quoted summary text."},
        quotes=[QuotePayload(quoted_text="Quoted summary text.", block_id=quote_block_id, kind="summary", source="report")],
    )

    assert quote_block_id in answer.used_citation_block_ids
    assert "Quote context considered" in answer.answer
