"""Unit tests for schema inference and format validation."""

from pathlib import Path

import pytest

from core.schemas import SemanticSchema
from tools.infer_semantic_schema import infer_semantic_schema
from tools.read_data_file import read_data_file
from tools.validate_data_format import validate_data_format


class _FakeLLMClient:
    def __init__(self, payload: dict | list[dict]):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def generate_json(self, section_key: str, payload: dict, max_completion_tokens: int = 1000) -> dict:
        index = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return self.payloads[index]


class _UnavailableLLMClient:
    def is_available(self) -> bool:
        return False


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_infer_semantic_schema_multiway_scores(tmp_path: Path):
    file_path = _write(
        tmp_path / "multiway_scores.csv",
        "sample,model_a,model_b,task\ns1,0.8,0.7,code\ns2,0.9,0.6,math\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert len(result.schema.ranking_items) >= 2


@pytest.mark.parametrize(
    ("filename", "expected_format", "expected_bigbetter"),
    [
        ("example_data_pairwise.csv", "pairwise", 1),
        ("example_data_pairwise_human_logs.csv", "pairwise", 1),
        ("example_data_multiway_scores.csv", "multiway", 1),
        ("example_data_multiway_latency.csv", "multiway", 0),
        ("example_data_multiway_rank_columns.csv", "multiway", 0),
        ("example_data_multiway.csv", "multiway", 0),
    ],
)
def test_infer_semantic_schema_project_examples(filename: str, expected_format: str, expected_bigbetter: int):
    file_path = str(Path(__file__).resolve().parents[2] / "data" / "examples" / filename)
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == expected_format
    assert result.schema is not None
    assert result.schema.bigbetter == expected_bigbetter


def test_infer_semantic_schema_pairwise(tmp_path: Path):
    file_path = _write(
        tmp_path / "pairwise.csv",
        "task,model_a,model_b,model_c\ncode,1,0,\nmath,0,1,\nwriting,,1,0\nqa,1,,0\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pairwise"
    assert result.schema is not None
    assert len(result.schema.ranking_items) >= 2


def test_infer_semantic_schema_pairwise_long_columns(tmp_path: Path):
    file_path = _write(
        tmp_path / "pairwise_long.csv",
        "task,item_a,item_b,winner\ncode,A,B,A\nmath,A,C,C\nqa,B,C,C\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pairwise"
    assert result.schema is not None
    assert result.schema.ranking_items == ["A", "B", "C"]


def test_infer_semantic_schema_pairwise_long_item_value(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "pairwise_long_item_value.csv",
        (
            "comparison_id,task,item,value\n"
            "cmp_1,code,A,1\n"
            "cmp_1,code,B,0\n"
            "cmp_2,math,A,0\n"
            "cmp_2,math,C,1\n"
            "cmp_3,qa,B,1\n"
            "cmp_3,qa,C,0\n"
        ),
    )
    summary = read_data_file(file_path).data
    assert summary is not None
    monkeypatch.setattr("tools.infer_semantic_schema.get_llm_client", lambda: _UnavailableLLMClient())

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pairwise"
    assert result.schema is not None
    assert result.schema.bigbetter == 1
    assert sorted(result.schema.ranking_items) == ["A", "B", "C"]
    assert result.schema.indicator_col == "task"
    assert sorted(result.schema.indicator_values) == ["code", "math", "qa"]


def test_infer_semantic_schema_multiway(tmp_path: Path):
    file_path = _write(
        tmp_path / "multiway.csv",
        "race,rank_1,rank_2,rank_3,track\nr1,A,B,C,grass\nr2,B,C,A,dirt\nr3,C,A,B,grass\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert sorted(result.schema.ranking_items) == ["A", "B", "C"]


def test_infer_semantic_schema_multiway_sets_lower_better(tmp_path: Path):
    file_path = _write(
        tmp_path / "multiway_direction.csv",
        "race,rank_1,rank_2,rank_3\nr1,A,B,C\nr2,B,C,A\nr3,C,A,B\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert result.schema.bigbetter == 0


def test_infer_semantic_schema_uses_llm_format_when_valid(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "pairwise_conflict.csv",
        "task,model_a,model_b,model_c\ncode,1,0,\nmath,0,1,\nwriting,,1,0\nqa,1,,0\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM chose multiway",
                "schema": {
                    "bigbetter": 0,
                    "ranking_items": ["model_a", "model_b", "model_c"],
                    "indicator_col": "task",
                    "indicator_values": ["code", "math", "writing", "qa"],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert "llm chose multiway" in result.format_evidence.lower()


def test_infer_semantic_schema_stabilizes_ambiguous_llm_bigbetter_to_fallback(
    tmp_path: Path,
    monkeypatch,
):
    file_path = _write(
        tmp_path / "multiway_scores_direction.csv",
        "sample,model_a,model_b,model_c\ns1,0.9,0.8,0.7\ns2,0.8,0.7,0.6\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM chose multiway",
                "schema": {
                    "bigbetter": 0,
                    "ranking_items": ["model_a", "model_b", "model_c"],
                    "indicator_col": None,
                    "indicator_values": [],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert result.schema.bigbetter == 1


def test_infer_semantic_schema_preserves_lower_better_with_keyword_evidence(
    tmp_path: Path,
    monkeypatch,
):
    file_path = _write(
        tmp_path / "multiway_latency_direction.csv",
        "sample,latency_a,latency_b,latency_c\ns1,120,130,140\ns2,110,125,138\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM chose multiway",
                "schema": {
                    "bigbetter": 0,
                    "ranking_items": ["latency_a", "latency_b", "latency_c"],
                    "indicator_col": None,
                    "indicator_values": [],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert result.schema.bigbetter == 0


def test_infer_semantic_schema_falls_back_bigbetter_when_llm_bigbetter_is_invalid(
    tmp_path: Path,
    monkeypatch,
):
    file_path = _write(
        tmp_path / "rank_multiway_direction.csv",
        "race,rank_1,rank_2,rank_3\nr1,A,B,C\nr2,B,C,A\nr3,C,A,B\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM chose multiway",
                "schema": {
                    "bigbetter": 2,
                    "ranking_items": ["A", "B", "C"],
                    "indicator_col": None,
                    "indicator_values": [],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert result.schema.bigbetter == 0


def test_infer_semantic_schema_falls_back_format_when_llm_format_is_invalid(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "pairwise_invalid_format.csv",
        "task,model_a,model_b,model_c\ncode,1,0,\nmath,0,1,\nwriting,,1,0\nqa,1,,0\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "unknown",
                "format_evidence": "LLM uncertain",
                "schema": {
                    "bigbetter": 1,
                    "ranking_items": ["model_a", "model_b", "model_c"],
                    "indicator_col": "task",
                    "indicator_values": ["code", "math", "writing", "qa"],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pairwise"


def test_infer_semantic_schema_expands_partial_llm_indicator_values_to_observed(
    tmp_path: Path,
    monkeypatch,
):
    file_path = _write(
        tmp_path / "multiway_partial_indicator_values.csv",
        (
            "sample,model_a,model_b,task\n"
            "s1,0.8,0.7,code\n"
            "s2,0.9,0.6,math\n"
            "s3,0.85,0.65,qa\n"
            "s4,0.82,0.68,code\n"
            "s5,0.88,0.66,math\n"
            "s6,0.84,0.67,qa\n"
        ),
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM proposed partial indicator values",
                "schema": {
                    "bigbetter": 1,
                    "ranking_items": ["model_a", "model_b"],
                    "indicator_col": "task",
                    "indicator_values": ["code"],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert result.schema.indicator_col == "task"
    assert sorted(result.schema.indicator_values) == ["code", "math", "qa"]


def test_infer_semantic_schema_conflict_triggers_llm_self_correction(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "pairwise_conflict_retry.csv",
        "task,model_a,model_b,model_c\ncode,1,0,\nmath,0,1,\nwriting,,1,0\nqa,1,,0\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    client = _FakeLLMClient(
        [
            {
                "format": "multiway",
                "format_evidence": "First pass",
                "schema": {
                    "bigbetter": 1,
                    "ranking_items": ["model_a", "model_b", "model_c"],
                    "indicator_col": "task",
                    "indicator_values": ["code", "math", "writing", "qa"],
                },
            },
            {
                "format": "pairwise",
                "format_evidence": "Revised after consistency feedback",
                "schema": {
                    "bigbetter": 1,
                    "ranking_items": ["model_a", "model_b", "model_c"],
                    "indicator_col": "task",
                    "indicator_values": ["code", "math", "writing", "qa"],
                },
            },
        ]
    )
    monkeypatch.setattr("tools.infer_semantic_schema.get_llm_client", lambda: client)

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pairwise"
    assert "revised" in result.format_evidence.lower()
    assert client.calls == 2


def test_infer_semantic_schema_rank_columns_stabilize_bigbetter_without_retry(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "rank_columns_conflict.csv",
        "match_id,domain,rank_1,rank_2,rank_3\nm1,code,A,B,C\nm2,math,B,C,A\nm3,qa,C,A,B\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    client = _FakeLLMClient(
        [
            {
                "format": "multiway",
                "format_evidence": "First pass",
                "schema": {
                    "bigbetter": 1,
                    "ranking_items": ["A", "B", "C"],
                    "indicator_col": "domain",
                    "indicator_values": ["code", "math", "qa"],
                },
            },
            {
                "format": "multiway",
                "format_evidence": "Revised rank-order direction",
                "schema": {
                    "bigbetter": 0,
                    "ranking_items": ["A", "B", "C"],
                    "indicator_col": "domain",
                    "indicator_values": ["code", "math", "qa"],
                },
            },
        ]
    )
    monkeypatch.setattr("tools.infer_semantic_schema.get_llm_client", lambda: client)

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "multiway"
    assert result.schema is not None
    assert result.schema.bigbetter == 0
    assert client.calls == 1


def test_infer_semantic_schema_long_item_value_ignores_id_column(tmp_path: Path):
    file_path = _write(
        tmp_path / "long_item_value.csv",
        "case_id,item,value\n1,A,0.9\n1,B,0.8\n2,A,0.7\n2,B,0.6\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert result.schema.ranking_items == ["A", "B"]
    assert "case_id" not in result.schema.ranking_items


def test_infer_semantic_schema_rejects_meta_like_indicator_from_llm(tmp_path: Path, monkeypatch):
    file_path = _write(
        tmp_path / "meta_indicator.csv",
        (
            "sample_id,model_a,model_b,description\n"
            "s1,0.90,0.80,description row one\n"
            "s2,0.88,0.79,description row two\n"
            "s3,0.87,0.78,description row three\n"
            "s4,0.86,0.77,description row four\n"
            "s5,0.85,0.76,description row five\n"
            "s6,0.84,0.75,description row six\n"
        ),
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr(
        "tools.infer_semantic_schema.get_llm_client",
        lambda: _FakeLLMClient(
            {
                "format": "multiway",
                "format_evidence": "LLM chose multiway",
                "schema": {
                    "bigbetter": 0,
                    "ranking_items": ["model_a", "model_b"],
                    "indicator_col": "description",
                    "indicator_values": ["description row one", "description row two"],
                },
            }
        ),
    )

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert result.schema.indicator_col is None
    assert result.schema.indicator_values == []


def test_infer_semantic_schema_selects_single_indicator_column(tmp_path: Path):
    file_path = _write(
        tmp_path / "indicator.csv",
        "id,score_a,score_b,task,domain\n1,0.8,0.7,code,nlp\n2,0.9,0.6,math,vision\n3,0.85,0.8,code,nlp\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert isinstance(result.schema.indicator_col, str) or result.schema.indicator_col is None
    if result.schema.indicator_col is not None:
        assert len(result.schema.indicator_values) >= 2


def test_infer_semantic_schema_accepts_high_cardinality_phenotype_indicator(tmp_path: Path, monkeypatch):
    rows = ["Phenotype,model_a,model_b,model_c"]
    for i in range(60):
        phenotype = f"Trait_{i:03d}"
        rows.append(f"{phenotype},0.90,0.80,0.70")
        rows.append(f"{phenotype},0.88,0.79,0.69")

    file_path = _write(tmp_path / "phenotype_high_cardinality.csv", "\n".join(rows) + "\n")
    summary = read_data_file(file_path).data
    assert summary is not None

    monkeypatch.setattr("tools.infer_semantic_schema.get_llm_client", lambda: _UnavailableLLMClient())
    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.schema is not None
    assert result.schema.indicator_col == "Phenotype"
    assert len(result.schema.indicator_values) == 60


def test_validate_data_format_pass(tmp_path: Path):
    file_path = _write(
        tmp_path / "ready.csv",
        "model_a,model_b,task\n0.9,0.7,code\n0.8,0.6,math\n",
    )
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["model_a", "model_b"],
        indicator_col="task",
        indicator_values=["code", "math"],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is True
    assert result.fixable is False


def test_validate_data_format_non_numeric_without_fallback_is_unfixable(tmp_path: Path):
    file_path = _write(
        tmp_path / "non_numeric_unfixable.csv",
        "model_a,model_b\n0.9,low\n0.8,bad\n",
    )
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["model_a", "model_b"],
        indicator_col=None,
        indicator_values=[],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is False
    assert result.fixable is False
    assert len(result.suggested_fixes) > 0


def test_validate_data_format_missing_optional_item_is_ready(tmp_path: Path):
    file_path = _write(
        tmp_path / "missing_optional_item.csv",
        "model_a,model_b\n0.9,0.7\n0.8,0.6\n",
    )
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["model_a", "model_b", "AnnoPred"],
        indicator_col=None,
        indicator_values=[],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is True
    assert result.fixable is False


def test_validate_data_format_pairwise_long_is_fixable(tmp_path: Path):
    file_path = _write(
        tmp_path / "pairwise_fixable.csv",
        "task,item_a,item_b,winner\ncode,A,B,A\nmath,A,C,C\nqa,B,C,C\n",
    )
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["A", "B", "C"],
        indicator_col="task",
        indicator_values=["code", "math", "qa"],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is False
    assert result.fixable is True
    assert any("pairwise" in fix.lower() for fix in result.suggested_fixes)


def test_validate_data_format_long_item_value_is_fixable(tmp_path: Path):
    file_path = _write(
        tmp_path / "pairwise_long_item_value_fixable.csv",
        (
            "comparison_id,task,item,value\n"
            "cmp_1,code,A,1\n"
            "cmp_1,code,B,0\n"
            "cmp_2,math,A,0\n"
            "cmp_2,math,C,1\n"
        ),
    )
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["A", "B", "C"],
        indicator_col="task",
        indicator_values=["code", "math"],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is False
    assert result.fixable is True
    assert any("pivot long item/value" in fix.lower() for fix in result.suggested_fixes)


def test_validate_data_format_unfixable(tmp_path: Path):
    file_path = _write(tmp_path / "unfixable.csv", "only_one_col\n1\n2\n")
    schema = SemanticSchema(
        bigbetter=1,
        ranking_items=["model_a", "model_b"],
        indicator_col=None,
        indicator_values=[],
    )

    result = validate_data_format(file_path, schema)

    assert result.is_ready is False
    assert result.fixable is False
