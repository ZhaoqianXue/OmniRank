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

    assert "# OmniRank Report" in report.markdown
    assert "<section data-omni-block-id=" in report.markdown
    assert 'data-omni-kind="summary"' in report.markdown
    assert 'data-omni-kind="table"' in report.markdown
    assert 'data-omni-kind="method"' in report.markdown
    assert 'data-omni-kind="limitation"' in report.markdown
    assert 'data-omni-kind="repro"' not in report.markdown
    assert "### Full Ranking Table" not in report.markdown
    assert "CI Width" not in report.markdown
    assert "Gap to #1" not in report.markdown
    assert report.markdown.find("## Ranking Results") < report.markdown.find("| Rank | Item | Confidence Interval | Score (θ̂) |")
    assert report.markdown.find("| Rank | Item | Confidence Interval | Score (θ̂) |") < report.markdown.find("### Executive Summary")
    assert "| 1 | A | [1, 2] | 0.6000 |" in report.markdown
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


def test_answer_question_works_before_analysis_results_are_ready():
    answer = answer_question(
        question="Can I ask a methodological question before running ranking?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "uploaded", "has_results": False},
    )

    assert "Conclusion:" not in answer.answer
    assert "Session status: uploaded." in answer.answer


def test_answer_question_integerizes_ci_from_llm_output(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "Model A is ahead with CI [1.0, 6.0].",
                "evidence": ["Model A: CI=[1.0, 6.0]", "Model B: CI=[2.0, 7.0]"],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    results = _sample_results()
    answer = answer_question(
        question="Compare A and B",
        results=results,
        citation_blocks={},
        quotes=[],
    )

    assert "[1, 6]" in answer.answer
    assert "[1.0, 6.0]" not in answer.answer


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


def test_answer_question_drops_spurious_used_ids_without_quotes(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "A is first.",
                "evidence": ["A has rank 1."],
                "used_citation_block_ids": ["summary-abc"],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="What is the top item?",
        results=_sample_results(),
        citation_blocks={"summary-abc": "Some block"},
        quotes=[],
    )

    assert answer.used_citation_block_ids == []


def test_answer_question_omits_unnecessary_references_for_plain_ranking(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "A is first.",
                "evidence": ["A has rank 1."],
                "references": [
                    {
                        "title": "Spectral Ranking Inferences based on General Multiway Comparisons",
                        "url": "https://arxiv.org/html/2308.02918",
                    }
                ],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="Who is first?",
        results=_sample_results(),
        citation_blocks={},
        quotes=[],
    )

    assert "References:" not in answer.answer


def test_answer_question_respects_one_sentence_mode(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "A wins. Additional sentence should be trimmed.",
                "evidence": ["A rank=1", "B rank=2"],
                "references": [
                    {
                        "title": "Spectral Ranking Inferences based on General Multiway Comparisons",
                        "url": "https://arxiv.org/html/2308.02918",
                    }
                ],
                "note": "This is extra note that should be removed.",
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="One sentence only: who wins?",
        results=_sample_results(),
        citation_blocks={},
        quotes=[],
    )

    assert answer.answer.startswith("A wins.")
    assert "Conclusion:" not in answer.answer
    assert "Evidence:" not in answer.answer
    assert "References:" not in answer.answer
    assert "Note:" not in answer.answer


def test_answer_question_shallow_method_does_not_attach_external_reference(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "OmniRank uses spectral ranking with bootstrap uncertainty.",
                "evidence": ["Method overview only."],
                "references": [
                    {
                        "title": "Spectral Ranking Inferences based on General Multiway Comparisons",
                        "url": "https://arxiv.org/html/2308.02918",
                    }
                ],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="What method is used?",
        results=_sample_results(),
        citation_blocks={},
        quotes=[],
    )

    assert "References:" not in answer.answer


def test_answer_question_deep_method_backfills_whitelisted_reference(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "Detailed theoretical explanation of assumptions and asymptotics.",
                "evidence": ["Discusses identifiability and uncertainty quantification."],
                "references": [],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="Provide a detailed theoretical explanation of the method and include the source paper.",
        results=_sample_results(),
        citation_blocks={},
        quotes=[],
    )

    assert "References:" in answer.answer
    assert "https://arxiv.org/html/2308.02918" in answer.answer
