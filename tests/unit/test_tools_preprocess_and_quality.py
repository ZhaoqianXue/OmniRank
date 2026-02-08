"""Unit tests for preprocess_data and validate_data_quality."""

from pathlib import Path

import pandas as pd

from core.schemas import SemanticSchema
from tools.preprocess_data import preprocess_data
from tools.validate_data_quality import validate_data_quality


def _write_csv(path: Path, df: pd.DataFrame) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_preprocess_long_to_wide(tmp_path: Path):
    df = pd.DataFrame(
        {
            "case_id": [1, 1, 2, 2, 3, 3],
            "item": ["A", "B", "A", "B", "A", "B"],
            "value": [0.9, 0.8, 0.7, 0.6, 0.8, 0.75],
        }
    )
    file_path = _write_csv(tmp_path / "long.csv", df)
    schema = SemanticSchema(bigbetter=1, ranking_items=["A", "B"], indicator_col=None, indicator_values=[])

    result = preprocess_data(file_path=file_path, schema=schema, output_dir=str(tmp_path))
    output = pd.read_csv(result.preprocessed_csv_path)

    assert "A" in output.columns and "B" in output.columns
    assert result.row_count == len(output)
    assert any("Pivoted long format" in entry for entry in result.transformation_log)


def test_preprocess_multiway_rank_columns_to_wide(tmp_path: Path):
    df = pd.DataFrame(
        {
            "race": ["r1", "r2", "r3"],
            "rank_1": ["A", "B", "C"],
            "rank_2": ["B", "C", "A"],
            "rank_3": ["C", "A", "B"],
        }
    )
    file_path = _write_csv(tmp_path / "multiway_rank.csv", df)
    schema = SemanticSchema(bigbetter=0, ranking_items=["A", "B", "C"], indicator_col=None, indicator_values=[])

    result = preprocess_data(file_path=file_path, schema=schema, output_dir=str(tmp_path))
    output = pd.read_csv(result.preprocessed_csv_path)

    assert set(["A", "B", "C"]).issubset(set(output.columns))
    assert output[["A", "B", "C"]].notna().sum().sum() >= 6
    assert any("multiway" in entry.lower() for entry in result.transformation_log)


def test_preprocess_pairwise_long_to_wide(tmp_path: Path):
    df = pd.DataFrame(
        {
            "task": ["code", "math", "qa", "code"],
            "item_a": ["A", "A", "B", "C"],
            "item_b": ["B", "C", "C", "A"],
            "winner": ["A", "C", "C", "A"],
        }
    )
    file_path = _write_csv(tmp_path / "pairwise_long.csv", df)
    schema = SemanticSchema(bigbetter=1, ranking_items=["A", "B", "C"], indicator_col="task", indicator_values=["code", "math", "qa"])

    result = preprocess_data(file_path=file_path, schema=schema, output_dir=str(tmp_path))
    output = pd.read_csv(result.preprocessed_csv_path)

    assert set(["A", "B", "C"]).issubset(set(output.columns))
    assert any("pairwise long format" in entry.lower() for entry in result.transformation_log)


def test_validate_data_quality_sparse_warning(tmp_path: Path):
    df = pd.DataFrame(
        {
            "A": [1, 0, None, None],
            "B": [0, 1, 1, None],
            "C": [None, None, 0, 1],
            "D": [None, None, None, 0],
            "E": [None, None, None, 1],
        }
    )
    file_path = _write_csv(tmp_path / "sparse_connected.csv", df)
    schema = SemanticSchema(bigbetter=1, ranking_items=["A", "B", "C", "D", "E"], indicator_col=None, indicator_values=[])

    result = validate_data_quality(file_path=file_path, schema=schema)

    assert result.is_valid is True
    assert len(result.warnings) >= 1
    assert len(result.errors) == 0


def test_validate_data_quality_connectivity_blocking(tmp_path: Path):
    df = pd.DataFrame(
        {
            "A": [1, 0, None, None],
            "B": [0, 1, None, None],
            "C": [None, None, 1, 0],
            "D": [None, None, 0, 1],
        }
    )
    file_path = _write_csv(tmp_path / "disconnected.csv", df)
    schema = SemanticSchema(bigbetter=1, ranking_items=["A", "B", "C", "D"], indicator_col=None, indicator_values=[])

    result = validate_data_quality(file_path=file_path, schema=schema)

    assert result.is_valid is False
    assert any("disconnected" in err.lower() for err in result.errors)


def test_validate_data_quality_less_than_two_items_blocking(tmp_path: Path):
    df = pd.DataFrame({"A": [1, 0, 1]})
    file_path = _write_csv(tmp_path / "one_item.csv", df)
    schema = SemanticSchema(bigbetter=1, ranking_items=["A"], indicator_col=None, indicator_values=[])

    result = validate_data_quality(file_path=file_path, schema=schema)

    assert result.is_valid is False
    assert any("fewer than two ranking items" in err.lower() for err in result.errors)
