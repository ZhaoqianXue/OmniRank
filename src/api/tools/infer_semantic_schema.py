"""Tool: infer_semantic_schema."""

from __future__ import annotations

from typing import Optional

from .common import (
    detect_format_from_df,
    infer_bigbetter,
    infer_indicator_column,
    infer_ranking_items,
    read_table,
)
from core.schemas import ComparisonFormat, DataSummary, SemanticSchema, SemanticSchemaResult


def infer_semantic_schema(
    data_summary: DataSummary,
    file_path: str,
    user_hints: Optional[str] = None,
) -> SemanticSchemaResult:
    """Infer comparison format and semantic schema for user verification."""
    try:
        df = read_table(file_path)
    except Exception as exc:  # noqa: BLE001
        return SemanticSchemaResult(
            success=False,
            format=ComparisonFormat.POINTWISE,
            format_evidence="Unable to read file for schema inference.",
            error=str(exc),
        )

    format_name, format_evidence = detect_format_from_df(df)
    ranking_items = infer_ranking_items(df, format_name)
    indicator_col, indicator_values = infer_indicator_column(df, ranking_items)
    bigbetter = infer_bigbetter(df, ranking_items, user_hints=user_hints)

    if format_name == "multiway":
        numeric_subset = [col for col in ranking_items if col in df.columns]
        if numeric_subset:
            values = df[numeric_subset].to_numpy().flatten()
            values = [value for value in values if value == value]
            if values and all(float(value).is_integer() and value >= 1 for value in values):
                bigbetter = 0

    # Fall back to DataSummary column ordering if heuristics did not find enough items.
    if len(ranking_items) < 2:
        numeric_like = [
            col
            for col in data_summary.columns
            if data_summary.column_types.get(col) == "numeric"
        ]
        if len(numeric_like) >= 2:
            ranking_items = numeric_like

    schema = SemanticSchema(
        bigbetter=bigbetter,
        ranking_items=ranking_items,
        indicator_col=indicator_col,
        indicator_values=indicator_values,
    )

    return SemanticSchemaResult(
        success=True,
        format=ComparisonFormat(format_name),
        format_evidence=format_evidence,
        schema=schema,
    )
