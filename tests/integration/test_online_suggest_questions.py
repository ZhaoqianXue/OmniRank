"""Online integration tests for every Suggest Question path.

Each test exercises a real LLM call through answer_question and validates
that the response is concise, English-only, decision-ready, and free of
jargon or structural issues.
"""

from __future__ import annotations

import os
import re

import pytest

from core.schemas import QuotePayload, RankingResults
from tools.answer_question import answer_question


def _require_api_key() -> None:
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is required."


def _sample_results() -> RankingResults:
    return RankingResults(
        items=["Model_A", "Model_B", "Model_C"],
        theta_hat=[0.8, 0.4, 0.1],
        ranks=[1, 2, 3],
        ci_lower=[1.0, 1.0, 2.0],
        ci_upper=[2.0, 3.0, 3.0],
        indicator_value=None,
    )


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_JARGON_TERMS = [
    "indicator_col",
    "ranking_items",
    "bigbetter",
    "theta_hat",
    "awaiting_confirmation",
    "infer_semantic_schema",
    "execute_spectral_ranking",
]
_MAX_ANSWER_LINES = 8
_IDLE_NEXT_STEP_SNIPPET = "Upload a CSV/TSV file, confirm the inferred schema"


def _assert_answer_quality(answer_text: str, *, allow_theta: bool = False) -> None:
    """Shared quality assertions for every suggest-question answer."""
    assert answer_text, "Answer must not be empty."
    assert not _CJK_RE.search(answer_text), f"Answer contains CJK characters: {answer_text!r}"
    for jargon in _JARGON_TERMS:
        if jargon == "theta_hat" and allow_theta:
            continue
        assert jargon not in answer_text, f"Answer leaks internal jargon '{jargon}': {answer_text!r}"
    lines = answer_text.strip().split("\n")
    assert len(lines) <= _MAX_ANSWER_LINES, f"Answer has {len(lines)} lines (max {_MAX_ANSWER_LINES}): {answer_text!r}"
    assert not answer_text.startswith("Conclusion:"), "Answer should not have 'Conclusion:' prefix."


# ---------------------------------------------------------------------------
# Pre-upload / idle stage
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_pre_upload_what_is_omnirank():
    _require_api_key()
    answer = answer_question(
        question="What is OmniRank and how can it help me?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle"},
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "ranking" in lower or "rank" in lower, f"Answer should mention ranking: {answer.answer!r}"
    assert _IDLE_NEXT_STEP_SNIPPET not in answer.answer, (
        f"Overview answer should omit idle upload/schema note; got: {answer.answer!r}"
    )
    print(f"\n[pre-upload] What is OmniRank?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_pre_upload_data_format():
    _require_api_key()
    answer = answer_question(
        question="What ranking data and CSV format does OmniRank support?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle"},
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "csv" in lower or "data" in lower or "format" in lower, f"Should mention data format: {answer.answer!r}"
    assert _IDLE_NEXT_STEP_SNIPPET in answer.answer, (
        f"Data-format answer should include idle next-step note; got: {answer.answer!r}"
    )
    print(f"\n[pre-upload] Data format?\n>>> {answer.answer}")


# ---------------------------------------------------------------------------
# Post-schema / awaiting_confirmation stage
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_post_schema_detected_schema():
    _require_api_key()
    answer = answer_question(
        question="Does this detected schema look right for my data?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={
            "status": "awaiting_confirmation",
            "schema": {
                "bigbetter": 1,
                "ranking_items": ["Model_A", "Model_B", "Model_C"],
                "indicator_col": "Task",
                "indicator_values": ["QA", "Code"],
            },
        },
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[post-schema] Schema right?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_post_schema_direction():
    _require_api_key()
    answer = answer_question(
        question="Is 'higher is better' the correct direction for my metric?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={
            "status": "awaiting_confirmation",
            "schema": {
                "bigbetter": 1,
                "ranking_items": ["Model_A", "Model_B"],
                "indicator_col": None,
                "indicator_values": [],
            },
        },
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[post-schema] Direction correct?\n>>> {answer.answer}")


# ---------------------------------------------------------------------------
# Analyzing / running stage
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_analyzing_what_to_inspect():
    _require_api_key()
    answer = answer_question(
        question="What should I look at first when results are ready?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "running"},
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[analyzing] What to inspect first?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_analyzing_how_to_tell_better():
    _require_api_key()
    answer = answer_question(
        question="How will I know if one item is clearly better than another?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "running"},
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "confidence" in lower or "ci" in lower or "interval" in lower, \
        f"Should mention CI/confidence: {answer.answer!r}"
    print(f"\n[analyzing] How to tell better?\n>>> {answer.answer}")


# ---------------------------------------------------------------------------
# Error stage
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_error_what_went_wrong():
    _require_api_key()
    answer = answer_question(
        question="What went wrong and how can I fix it?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "error", "quality_warnings": ["Disconnected comparison graph detected."]},
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[error] What went wrong?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_error_what_can_i_do():
    _require_api_key()
    answer = answer_question(
        question="What can I still do while this is being resolved?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "error"},
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[error] What can I still do?\n>>> {answer.answer}")


# ---------------------------------------------------------------------------
# Post-analysis / completed stage
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_post_analysis_difference_driver():
    _require_api_key()
    results = _sample_results()
    answer = answer_question(
        question="What drives the difference between Model_A and Model_B?",
        results=results,
        citation_blocks={},
        quotes=[],
        session_context={"status": "completed"},
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "model_a" in lower or "model_b" in lower, f"Should mention item names: {answer.answer!r}"
    print(f"\n[post-analysis] Difference driver?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_post_analysis_clear_vs_tied():
    _require_api_key()
    results = _sample_results()
    answer = answer_question(
        question="Which items are clearly better vs tied based on the confidence intervals (CI)?",
        results=results,
        citation_blocks={},
        quotes=[],
        session_context={"status": "completed"},
    )
    _assert_answer_quality(answer.answer)
    print(f"\n[post-analysis] Clear vs tied?\n>>> {answer.answer}")


@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_post_analysis_clustering_question():
    """Online test for the suggested question: Show me the clustering of items by
    confidence-interval overlap and explain what it means.
    Validates that the LLM returns groups and interprets them."""
    _require_api_key()
    results = _sample_results()
    key_findings = {
        "cluster_items": [["Model_A", "Model_B"], ["Model_C"]],
        "n_clusters": 2,
    }
    answer = answer_question(
        question="Show me the clustering of items by confidence-interval overlap and explain what it means.",
        results=results,
        citation_blocks={},
        quotes=[],
        session_context={
            "status": "completed",
            "key_findings": key_findings,
        },
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "group" in lower or "cluster" in lower, (
        f"Clustering answer should mention groups/clusters: {answer.answer!r}"
    )
    assert "model_a" in lower or "model_b" in lower or "model_c" in lower, (
        f"Clustering answer should mention item names: {answer.answer!r}"
    )
    assert "uncertain" in lower or "tied" in lower or "overlap" in lower or "interval" in lower, (
        f"Clustering answer should explain meaning (uncertainty/tied/overlap): {answer.answer!r}"
    )
    print(f"\n[post-analysis] Clustering?\n>>> {answer.answer}")


# ---------------------------------------------------------------------------
# Method intent
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.online_llm
def test_sq_method_simple_terms():
    _require_api_key()
    answer = answer_question(
        question="How does the spectral ranking method work in simple terms?",
        results=None,
        citation_blocks={},
        quotes=[],
        session_context={"status": "idle"},
    )
    _assert_answer_quality(answer.answer)
    lower = answer.answer.lower()
    assert "eigenvector" not in lower and "matrix" not in lower, \
        f"Simple terms should avoid math jargon: {answer.answer!r}"
    print(f"\n[method] Simple terms?\n>>> {answer.answer}")
