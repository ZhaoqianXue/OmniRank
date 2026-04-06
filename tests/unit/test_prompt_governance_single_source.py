"""Tests for single-source prompt governance."""

from __future__ import annotations

from pathlib import Path

from agents.prompt_loader import PROMPT_PATH, load_prompt_section


def test_prompt_sections_exist_in_single_prompt_file():
    assert PROMPT_PATH.exists()
    content = PROMPT_PATH.read_text(encoding="utf-8")
    assert "TOOL_SECTION:infer_semantic_schema" in content
    assert "TOOL_SECTION:generate_report" in content
    assert "TOOL_SECTION:answer_question" in content


def test_load_prompt_section_returns_expected_sections():
    infer_prompt = load_prompt_section("infer_semantic_schema")
    report_prompt = load_prompt_section("generate_report")
    qa_prompt = load_prompt_section("answer_question")

    assert "strict JSON" in infer_prompt
    assert "strict JSON" in report_prompt
    assert "strict JSON" in qa_prompt


def test_generate_report_prompt_contains_uncertainty_takeaway_guidance():
    report_prompt = load_prompt_section("generate_report")

    assert "Top rank with uncertainty" in report_prompt
    assert "Top group" in report_prompt
    assert "Interpretation of uncertainty" in report_prompt
    assert "Do not use \"largest score gap\" as a required takeaway" in report_prompt


def test_answer_question_prompt_contains_product_overview_guidance():
    qa_prompt = load_prompt_section("answer_question")

    assert "statistical analysis agent" in qa_prompt
    assert "supportive artifacts" in qa_prompt
    assert "semantic schema of each variable" in qa_prompt
    assert "confidence intervals, visualizations, and a single-page reproducible report" in qa_prompt


def test_answer_question_prompt_contains_capability_guidance():
    qa_prompt = load_prompt_section("answer_question")

    assert "supports both pairwise and multiway comparison data in CSV/TSV format" in qa_prompt
    assert "each row compares two items" in qa_prompt
    assert "each row records one comparison among multiple items at once" in qa_prompt
    assert "scores or rank positions" in qa_prompt


def test_agent_has_no_inline_optional_stage_llm_prompt():
    agent_path = Path(__file__).resolve().parents[2] / "src" / "api" / "agents" / "omnirank_agent.py"
    content = agent_path.read_text(encoding="utf-8")

    assert "_optional_llm_stage_note" not in content
    assert "chat.completions.create(" not in content
