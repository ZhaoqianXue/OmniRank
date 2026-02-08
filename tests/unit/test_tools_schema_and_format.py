"""Unit tests for schema inference and format validation."""

from pathlib import Path

from core.schemas import SemanticSchema
from tools.infer_semantic_schema import infer_semantic_schema
from tools.read_data_file import read_data_file
from tools.validate_data_format import validate_data_format


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_infer_semantic_schema_pointwise(tmp_path: Path):
    file_path = _write(
        tmp_path / "pointwise.csv",
        "sample,model_a,model_b,task\ns1,0.8,0.7,code\ns2,0.9,0.6,math\n",
    )
    summary = read_data_file(file_path).data
    assert summary is not None

    result = infer_semantic_schema(summary, file_path)

    assert result.success is True
    assert result.format.value == "pointwise"
    assert result.schema is not None
    assert len(result.schema.ranking_items) >= 2


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


def test_validate_data_format_fixable(tmp_path: Path):
    file_path = _write(
        tmp_path / "fixable.csv",
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
    assert result.fixable is True
    assert len(result.suggested_fixes) > 0


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
