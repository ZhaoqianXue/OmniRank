"""Tool: read_data_file."""

from __future__ import annotations

from .common import infer_column_type, read_table
from core.schemas import DataSummary, ReadDataFileResult


def read_data_file(file_path: str) -> ReadDataFileResult:
    """Read file and return a lightweight summary for reasoning."""
    try:
        df = read_table(file_path)
    except Exception as exc:  # noqa: BLE001
        return ReadDataFileResult(success=False, error=str(exc))

    column_types = {col: infer_column_type(df[col]) for col in df.columns}

    summary = DataSummary(
        columns=df.columns.tolist(),
        sample_rows=df.head(10).where(df.notna(), None).to_dict(orient="records"),
        row_count=len(df),
        column_types=column_types,
    )
    return ReadDataFileResult(success=True, data=summary)
