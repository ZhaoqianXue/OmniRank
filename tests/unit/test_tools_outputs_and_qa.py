"""Unit tests for visualization/report/question tools."""

from pathlib import Path

import pandas as pd

from core.schemas import ExecutionResult, ExecutionTrace, PlotSpec, QuotePayload, RankingResults
from core.llm_client import LLMCallError
from tools.answer_question import answer_question
from tools.generate_report import generate_report
from tools.generate_visualizations import _item_rank_matrix_from_csv, generate_visualizations


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

    first = generate_visualizations(results=results, viz_types=["ci_forest"], artifact_dir=str(artifact_dir))
    second = generate_visualizations(results=results, viz_types=["ci_forest"], artifact_dir=str(artifact_dir))

    assert len(first.plots) == 1
    assert len(second.plots) == 1

    for plot_a, plot_b in zip(first.plots, second.plots, strict=True):
        bytes_a = Path(plot_a.svg_path).read_bytes()
        bytes_b = Path(plot_b.svg_path).read_bytes()
        assert bytes_a == bytes_b
        # CI forest is R-generated PNG (not SVG)
        assert plot_a.svg_path.lower().endswith(".png")
        assert bytes_a.startswith(b"\x89PNG")
        assert plot_a.block_id
        assert plot_a.caption_plain
        assert plot_a.caption_academic

    ci_forest = first.plots[0]
    assert ci_forest.type == "ci_forest"
    assert ci_forest.config["x_label"] == "rank"
    assert ci_forest.config["point"] == "rank"
    assert ci_forest.data["rank_point"] == [1, 2, 3]
    assert ci_forest.data["ci_lower"] == [1.0, 1.0, 2.0]
    assert ci_forest.data["ci_upper"] == [2.0, 3.0, 3.0]


def test_generate_visualizations_ci_forest_uses_preferred_prs_method_order(tmp_path: Path):
    results = RankingResults(
        items=["PRS-CS-auto", "LDpred", "AnnoPred", "C+T", "DBSLMM"],
        theta_hat=[0.1, 0.4, 0.0, 0.6, 0.2],
        ranks=[4, 2, 5, 1, 3],
        ci_lower=[3.0, 2.0, 5.0, 1.0, 3.0],
        ci_upper=[5.0, 3.0, 5.0, 2.0, 4.0],
        indicator_value=None,
    )

    artifact_dir = tmp_path / "artifacts"
    output = generate_visualizations(results=results, viz_types=["ci_forest"], artifact_dir=str(artifact_dir))

    assert output.errors == []
    assert [plot.type for plot in output.plots] == ["ci_forest"]
    ci_forest = output.plots[0]
    assert ci_forest.data["names"] == ["C+T", "LDpred", "PRS-CS-auto", "DBSLMM", "AnnoPred"]
    assert ci_forest.data["rank_point"] == [1, 2, 4, 3, 5]
    assert ci_forest.data["theta_hat"] == [0.6, 0.4, 0.1, 0.2, 0.0]
    assert ci_forest.data["ci_lower"] == [1.0, 2.0, 3.0, 3.0, 5.0]
    assert ci_forest.data["ci_upper"] == [2.0, 3.0, 5.0, 4.0, 5.0]


def test_item_rank_matrix_from_csv_uses_preferred_prs_method_order(tmp_path: Path):
    csv_path = tmp_path / "phenotype_pre_ranked.csv"
    csv_path.write_text(
        (
            "phenotype,LDpred2-auto,lassosum,C+T,AnnoPred,LDpred\n"
            "p1,4,3,1,5,2\n"
            "p2,5,2,1,4,3\n"
        ),
        encoding="utf-8",
    )

    item_order, phenotype_order, rank_rows = _item_rank_matrix_from_csv(
        str(csv_path),
        indicator_col="phenotype",
        bigbetter=0,
        pre_ranked=True,
    )

    assert item_order == ["C+T", "LDpred", "lassosum", "LDpred2-auto", "AnnoPred"]
    assert phenotype_order == ["p1", "p2"]
    assert rank_rows == [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 3.0, 2.0, 5.0, 4.0]]


def test_generate_visualizations_deep_mode_generates_indicator_plots(tmp_path: Path, monkeypatch):
    results = _sample_results()
    artifact_dir = tmp_path / "artifacts"
    csv_path = tmp_path / "phenotype.csv"
    csv_path.write_text(
        (
            "phenotype,A,B,C\n"
            "p1,0.9,0.6,0.2\n"
            "p1,0.8,0.5,0.3\n"
            "p2,0.7,0.4,0.6\n"
            "p2,0.6,0.3,0.8\n"
        ),
        encoding="utf-8",
    )

    def fake_executor_run(self, config, session_work_dir):  # noqa: ANN001
        local_df = pd.read_csv(config.csv_path)
        methods = list(local_df.columns)
        means = local_df.mean(axis=0).sort_values(ascending=False)
        ordered_methods = list(means.index)
        rank_lookup = {method: idx + 1 for idx, method in enumerate(ordered_methods)}
        return ExecutionResult(
            success=True,
            results=RankingResults(
                items=methods,
                theta_hat=[float(means.get(method, 0.0)) for method in methods],
                ranks=[rank_lookup[method] for method in methods],
                ci_lower=[float(rank_lookup[method]) for method in methods],
                ci_upper=[float(rank_lookup[method]) for method in methods],
            ),
            trace=ExecutionTrace(
                command="fake",
                stdout="",
                stderr="",
                exit_code=0,
                duration_seconds=0.01,
                timestamp="2026-02-19T00:00:00Z",
            ),
        )

    monkeypatch.setattr("tools.generate_visualizations.RScriptExecutor.run", fake_executor_run)

    output = generate_visualizations(
        results=results,
        viz_types=["ci_forest", "normalized_ranking_over_indicator", "indicator_rankings_heatmap"],
        artifact_dir=str(artifact_dir),
        csv_path=str(csv_path),
        indicator_col="phenotype",
        selected_indicator_values=["p1", "p2"],
        selected_items=["A", "B", "C"],
        bigbetter=1,
        ranking_mode="deep",
        bootstrap_iterations=400,
        seed=42,
    )

    assert output.errors == []
    assert [plot.type for plot in output.plots] == [
        "ci_forest",
        "indicator_rankings_combined",
    ]
    combined = output.plots[1]
    assert combined.data["indicator_col"] == "phenotype"
    assert combined.data["item_order"] == ["A", "B", "C"]
    assert combined.data["phenotype_order"] == ["p1", "p2"]
    assert len(combined.data["rank_rows"]) == 2
    assert combined.config["source"] == "r_spectral"
    for plot in output.plots:
        path = Path(plot.svg_path)
        assert path.exists()
        if path.suffix.lower() == ".png":
            assert path.stat().st_size > 0
        else:
            assert "<svg" in path.read_text(encoding="utf-8")


def test_generate_visualizations_deep_mode_combined_table_uses_raw_integer_ranks(tmp_path: Path, monkeypatch):
    results = _sample_results()
    artifact_dir = tmp_path / "artifacts"
    csv_path = tmp_path / "phenotype_sparse.csv"
    csv_path.write_text(
        (
            "phenotype,A,B,C,D\n"
            "p1,0.90,0.70,0.10,\n"
            "p1,0.80,0.60,0.20,\n"
            "p2,0.88,0.68,0.40,\n"
            "p2,0.78,0.58,0.30,\n"
        ),
        encoding="utf-8",
    )

    def fake_executor_run(self, config, session_work_dir):  # noqa: ANN001
        local_df = pd.read_csv(config.csv_path)
        methods = list(local_df.columns)
        means = local_df.mean(axis=0).sort_values(ascending=False)
        ordered_methods = list(means.index)
        rank_lookup = {method: idx + 1 for idx, method in enumerate(ordered_methods)}
        return ExecutionResult(
            success=True,
            results=RankingResults(
                items=methods,
                theta_hat=[float(means.get(method, 0.0)) for method in methods],
                ranks=[rank_lookup[method] for method in methods],
                ci_lower=[float(rank_lookup[method]) for method in methods],
                ci_upper=[float(rank_lookup[method]) for method in methods],
            ),
            trace=ExecutionTrace(
                command="fake",
                stdout="",
                stderr="",
                exit_code=0,
                duration_seconds=0.01,
                timestamp="2026-02-19T00:00:00Z",
            ),
        )

    monkeypatch.setattr("tools.generate_visualizations.RScriptExecutor.run", fake_executor_run)

    output = generate_visualizations(
        results=results,
        viz_types=["normalized_ranking_over_indicator", "indicator_rankings_heatmap"],
        artifact_dir=str(artifact_dir),
        csv_path=str(csv_path),
        indicator_col="phenotype",
        selected_indicator_values=["p1", "p2"],
        selected_items=["A", "B", "C", "D"],
        bigbetter=1,
        ranking_mode="deep",
        bootstrap_iterations=400,
        seed=42,
    )

    assert output.errors == []
    assert [plot.type for plot in output.plots] == ["indicator_rankings_combined"]
    combined = output.plots[0]
    assert combined.data["item_order"] == ["A", "B", "C", "D"]
    assert combined.data["phenotype_order"] == ["p1", "p2"]
    assert combined.data["rank_rows"] == [[1, 2, 3, None], [1, 2, 3, None]]


def test_generate_report_contains_required_sections_and_citation_blocks(tmp_path: Path):
    results = _sample_results()
    (tmp_path / "ci_forest.svg").write_text("<svg/>", encoding="utf-8")
    plot = PlotSpec(
        type="ci_forest",
        data={},
        config={},
        svg_path=str(tmp_path / "ci_forest.svg"),
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
    assert 'data-omni-kind="method"' not in report.markdown
    assert 'data-omni-kind="limitation"' not in report.markdown
    assert 'data-omni-kind="repro"' not in report.markdown
    assert "### Full Ranking Table" not in report.markdown
    assert "### Ranking Interpretation" not in report.markdown
    assert "## Methodology" not in report.markdown
    assert "### Limitations and Assumptions" not in report.markdown
    assert "CI Width" not in report.markdown
    assert "Gap to #1" not in report.markdown
    assert report.markdown.find("## Ranking Results") < report.markdown.find("| Rank | Item | Confidence Interval | Estimated Score |")
    assert report.markdown.find("| Rank | Item | Confidence Interval | Estimated Score |") < report.markdown.find("## Executive Summary")
    assert "| 1 | A | [1, 2] | 0.6000 |" in report.markdown
    assert "**A** ranks #1 in this run" in report.markdown
    assert len(report.citation_blocks) > 0
    assert len(report.hints) > 0
    assert all(hint.title != "Rank Interpretation" for hint in report.hints)


def test_generate_report_ranking_table_uses_rank_order():
    results = RankingResults(
        items=["LDpred2-inf", "C+T", "PRS-CS-auto", "LDpred", "lassosum"],
        theta_hat=[0.1, 0.6, 0.2, 0.5, 0.4],
        ranks=[5, 1, 4, 2, 3],
        ci_lower=[4.0, 1.0, 3.0, 2.0, 3.0],
        ci_upper=[5.0, 2.0, 5.0, 3.0, 4.0],
        indicator_value=None,
    )

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[],
    )

    table_start = report.markdown.index("| Rank | Item | Confidence Interval | Estimated Score |")
    table_end = report.markdown.index("\n\n## Executive Summary", table_start)
    ranking_table = report.markdown[table_start:table_end]

    assert "| 1 | C+T | [1, 2] | 0.6000 |" in ranking_table
    assert "| 2 | LDpred | [2, 3] | 0.5000 |" in ranking_table
    assert "| 3 | lassosum | [3, 4] | 0.4000 |" in ranking_table
    assert "| 4 | PRS-CS-auto | [3, 5] | 0.2000 |" in ranking_table
    assert "| 5 | LDpred2-inf | [4, 5] | 0.1000 |" in ranking_table


def test_generate_report_retries_llm_when_summary_is_wrong(tmp_path: Path, monkeypatch):
    results = _sample_results()
    calls = {"count": 0}

    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "generate_report"
            calls["count"] += 1
            if calls["count"] == 1:
                return {
                    "summary": "**B** is top-ranked here with a clear lead.",
                    "results_narrative": "Incorrect narrative.",
                    "methods": "Incorrect methods.",
                    "limitations": "- Incorrect limitation.",
                }
            assert "validation_feedback" in payload
            return {
                "summary": (
                    "**A** leads the ranking in this run. "
                    "The ordering near the top still depends on the reported rank intervals.\n\n"
                    "**Key Takeaways:**\n\n"
                    "- **Top result**: **A** is ranked #1.\n"
                    "- **Most uncertainty**: **B** has the widest interval.\n"
                    "- **Largest score gap**: the clearest drop is between **B** and **C**."
                ),
                "results_narrative": (
                    "**Group 1**: **A**, **B** remain close at the top. "
                    "The largest estimated-score gap occurs between **B** and **C**."
                ),
                "methods": "Incorrect methods.",
                "limitations": "- Incorrect limitation.",
            }

    monkeypatch.setattr("tools.generate_report.get_llm_client", lambda: _FakeClient())

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[],
    )

    assert calls["count"] == 2
    assert "**A** leads the ranking in this run." in report.markdown
    assert "**B** is top-ranked here" not in report.markdown


def test_generate_report_falls_back_when_llm_report_generation_fails(tmp_path: Path, monkeypatch):
    results = _sample_results()

    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            raise LLMCallError("synthetic failure")

    monkeypatch.setattr("tools.generate_report.get_llm_client", lambda: _FakeClient())

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[],
    )

    assert "**A** ranks #1 in this run" in report.markdown


def test_generate_report_adds_indicator_ranking_table_for_combined_plot(tmp_path: Path):
    results = _sample_results()
    (tmp_path / "combined.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    plot = PlotSpec(
        type="indicator_rankings_combined",
        data={
            "indicator_col": "phenotype",
            "item_order": ["A", "B", "C"],
            "phenotype_order": ["p1", "p2"],
            "rank_rows": [[1, 2, 3], [2, 1, 3]],
        },
        config={"source": "python", "bigbetter": 1},
        svg_path=str(tmp_path / "combined.png"),
        block_id="figure-indicator-rankings-combined-123",
        caption_plain="(A) ...; (B) ...",
        caption_academic="Combined ranking view.",
        hint_ids=["hint-ci"],
    )

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[plot],
    )

    assert "## Rankings by Phenotype" in report.markdown
    assert "### Phenotype Ranking Table" in report.markdown
    assert "| Phenotype | A | B | C |" in report.markdown
    assert "| p1 | 1 | 2 | 3 |" in report.markdown
    assert "| p2 | 2 | 1 | 3 |" in report.markdown


def test_generate_visualizations_deep_mode_marks_na_on_known_r_failure(tmp_path: Path, monkeypatch):
    results = _sample_results()
    artifact_dir = tmp_path / "artifacts"
    csv_path = tmp_path / "phenotype.csv"
    csv_path.write_text(
        (
            "phenotype,A,B,C\n"
            "p1,0.9,0.6,0.2\n"
            "p2,0.7,0.4,0.6\n"
        ),
        encoding="utf-8",
    )

    def fake_executor_run(self, config, session_work_dir):  # noqa: ANN001
        return ExecutionResult(
            success=False,
            error="R execution failed",
            trace=ExecutionTrace(
                command="fake",
                stdout="",
                stderr="Error: missing values and NaN's not allowed if 'na.rm' is FALSE",
                exit_code=1,
                duration_seconds=0.01,
                timestamp="2026-02-19T00:00:00Z",
            ),
        )

    monkeypatch.setattr("tools.generate_visualizations.RScriptExecutor.run", fake_executor_run)

    output = generate_visualizations(
        results=results,
        viz_types=["normalized_ranking_over_indicator", "indicator_rankings_heatmap"],
        artifact_dir=str(artifact_dir),
        csv_path=str(csv_path),
        indicator_col="phenotype",
        selected_indicator_values=["p1", "p2"],
        selected_items=["A", "B", "C"],
        bigbetter=1,
        ranking_mode="deep",
        bootstrap_iterations=400,
        seed=42,
    )

    assert output.errors == []
    assert [plot.type for plot in output.plots] == ["indicator_rankings_combined"]
    combined = output.plots[0]
    assert combined.data["phenotype_order"] == ["p1", "p2"]
    assert combined.data["item_order"] == ["A", "B", "C"]
    assert combined.data["rank_rows"] == [[None, None, None], [None, None, None]]
    assert combined.config["source"] == "r_spectral"


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


def test_fallback_clustering_answer_when_llm_unavailable():
    """When LLM is unavailable, clustering questions with key_findings use deterministic fallback."""
    results = _sample_results()
    session_context = {
        "status": "completed",
        "key_findings": {
            "cluster_items": [["A", "B"], ["C"]],
            "n_clusters": 2,
        },
    }
    answer = answer_question(
        question="Show me the clustering of items by confidence-interval overlap and explain what it means.",
        results=results,
        citation_blocks={},
        quotes=[],
        session_context=session_context,
    )
    lower = answer.answer.lower()
    assert "ci-overlap" in lower or "group" in lower
    assert "a" in lower and "b" in lower and "c" in lower
    assert "uncertain" in lower or "tied" in lower
    assert "2" in answer.answer  # n_clusters


def test_fallback_clustering_includes_near_ties_when_present():
    """Fallback clustering answer includes near_ties_with_top when present."""
    results = _sample_results()
    session_context = {
        "status": "completed",
        "key_findings": {
            "cluster_items": [["A", "B"], ["C"]],
            "n_clusters": 2,
            "near_ties_with_top": ["B"],
        },
    }
    answer = answer_question(
        question="Show me the clustering of items by confidence-interval overlap and explain what it means.",
        results=results,
        citation_blocks={},
        quotes=[],
        session_context=session_context,
    )
    lower = answer.answer.lower()
    assert "near-ties" in lower or "rank-1" in lower
    assert "b" in lower


def test_answer_question_capability_question_calls_llm(monkeypatch):
    calls = {"count": 0}

    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            calls["count"] += 1
            assert section_key == "answer_question"
            return {
                "conclusion": "You can analyze pairwise comparison data with two items per record.",
                "evidence": ["Use item_a/item_b and winner fields."],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="What types of ranking data can I analyze?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle", "has_results": False, "inferred_format": "pairwise"},
    )

    lower_answer = answer.answer.lower()
    assert "pairwise" in lower_answer
    assert "two items" in lower_answer
    assert calls["count"] == 1


def test_answer_question_capability_question_repairs_wrong_three_plus_claim(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "You need 3+ items per contest for ranking.",
                "evidence": ["3+ items are required in each row."],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="What types of ranking data can I analyze?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle", "has_results": False, "inferred_format": "pairwise"},
    )

    lower_answer = answer.answer.lower()
    assert "3+" not in lower_answer
    assert "two items" in lower_answer


def test_answer_question_capability_question_respects_one_sentence(monkeypatch):
    class _FakeClient:
        def is_available(self) -> bool:
            return True

        def generate_json(self, section_key, payload, max_completion_tokens=0):  # noqa: ANN001
            assert section_key == "answer_question"
            return {
                "conclusion": "Pairwise data with two items is supported. Extra sentence should be removed.",
                "evidence": ["This should be dropped."],
                "used_citation_block_ids": [],
            }

    monkeypatch.setattr("tools.answer_question.get_llm_client", lambda: _FakeClient())

    answer = answer_question(
        question="One sentence: what data formats are supported?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle", "has_results": False},
    )

    assert "\n" not in answer.answer
    assert answer.answer.startswith("Pairwise data with two items is supported.")


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
