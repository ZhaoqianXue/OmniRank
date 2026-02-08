"""Unit tests for read_data_file tool."""

from pathlib import Path

from tools.read_data_file import read_data_file


def test_read_data_file_success(tmp_path: Path):
    file_path = tmp_path / "valid.csv"
    file_path.write_text("item_a,item_b\n1,0\n0,1\n", encoding="utf-8")

    result = read_data_file(str(file_path))

    assert result.success is True
    assert result.data is not None
    assert result.data.row_count == 2
    assert result.data.columns == ["item_a", "item_b"]


def test_read_data_file_empty_file(tmp_path: Path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    result = read_data_file(str(file_path))

    assert result.success is False
    assert result.error is not None


def test_read_data_file_encoding_error(tmp_path: Path):
    file_path = tmp_path / "invalid_encoding.csv"
    file_path.write_bytes(b"\xff\xfe\xfa\xfb")

    result = read_data_file(str(file_path))

    assert result.success is False
    assert result.error is not None


def test_read_data_file_non_csv_rejected(tmp_path: Path):
    file_path = tmp_path / "not_csv.txt"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    result = read_data_file(str(file_path))

    assert result.success is False
    assert "csv" in (result.error or "").lower()
