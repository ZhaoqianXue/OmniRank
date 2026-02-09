"""Online integration tests for LLM-native OmniRank tools."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.schemas import QuotePayload, RankingResults
from tools.answer_question import answer_question
from tools.generate_report import generate_report
from tools.infer_semantic_schema import infer_semantic_schema
from tools.read_data_file import read_data_file


def _require_api_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is required for online_llm tests."


@pytest.mark.integration
@pytest.mark.online_llm
def test_online_infer_semantic_schema_llm(tmp_path: Path):
    _require_api_key()

    csv_path = tmp_path / "online_pointwise.csv"
    csv_path.write_text("sample,model_a,model_b,task\ns1,0.9,0.8,code\ns2,0.7,0.6,math\n", encoding="utf-8")

    summary_result = read_data_file(str(csv_path))
    assert summary_result.success is True
    assert summary_result.data is not None

    schema_result = infer_semantic_schema(
        data_summary=summary_result.data,
        file_path=str(csv_path),
        user_hints="accuracy metric, higher is better",
    )

    assert schema_result.success is True
    assert schema_result.schema is not None
    assert schema_result.format.value in {"pointwise", "pairwise", "multiway"}
    assert len(schema_result.schema.ranking_items) >= 2
    assert schema_result.schema.bigbetter in {0, 1}


@pytest.mark.integration
@pytest.mark.online_llm
def test_online_generate_report_and_answer_question():
    _require_api_key()

    results = RankingResults(
        items=["Model_A", "Model_B", "Model_C"],
        theta_hat=[0.8, 0.4, 0.1],
        ranks=[1, 2, 3],
        ci_lower=[1.0, 1.0, 2.0],
        ci_upper=[2.0, 3.0, 3.0],
        indicator_value=None,
    )

    report = generate_report(
        results=results,
        session_meta={"B": 2000, "seed": 42, "current_file_path": "/tmp/input.csv"},
        plots=[],
    )
    assert "<section data-omni-block-id=" in report.markdown
    assert len(report.citation_blocks) >= 4

    summary_block = report.citation_blocks[0]
    answer = answer_question(
        question="Can you explain this quoted summary and compare Model_A and Model_B?",
        results=results,
        citation_blocks={block.block_id: block.text for block in report.citation_blocks},
        quotes=[
            QuotePayload(
                quoted_text=summary_block.text,
                block_id=summary_block.block_id,
                kind=summary_block.kind.value,
                source="report",
            )
        ],
    )

    assert answer.answer
    assert summary_block.block_id in answer.used_citation_block_ids
    assert "CI overlap is not a formal hypothesis test" in answer.answer
