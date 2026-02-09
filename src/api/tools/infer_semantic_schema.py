"""Tool: infer_semantic_schema."""

from __future__ import annotations

from typing import Optional

from .common import (
    detect_format_from_df,
    infer_bigbetter,
    infer_indicator_column,
    infer_ranking_items,
    is_meta_column,
    read_table,
)
from core.llm_client import LLMCallError, get_llm_client
from core.schemas import ComparisonFormat, DataSummary, SemanticSchema, SemanticSchemaResult


def _fallback_schema_from_heuristics(
    data_summary: DataSummary,
    file_path: str,
    user_hints: str | None,
) -> SemanticSchemaResult:
    """Heuristic fallback used when LLM is unavailable or returns invalid output."""
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
        bigbetter = 0

    if len(ranking_items) < 2:
        numeric_like = [
            col
            for col in data_summary.columns
            if data_summary.column_types.get(col) == "numeric" and not is_meta_column(col)
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


def _normalize_format(value: object, fallback: ComparisonFormat) -> ComparisonFormat:
    """Normalize model output into supported format enum."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"pointwise", "pairwise", "multiway"}:
            return ComparisonFormat(lowered)
    return fallback


def _normalized_unique_strings(values: object) -> list[str]:
    """Convert arbitrary payload values into unique, non-empty strings."""
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_llm_schema(
    raw_schema: object,
    data_summary: DataSummary,
    fallback_schema: SemanticSchema,
    format_name: ComparisonFormat,
) -> SemanticSchema:
    """Apply deterministic guards to LLM-proposed schema payload."""
    schema_dict = raw_schema if isinstance(raw_schema, dict) else {}

    raw_bigbetter = schema_dict.get("bigbetter")
    bigbetter = fallback_schema.bigbetter
    if isinstance(raw_bigbetter, bool):
        bigbetter = 1 if raw_bigbetter else 0
    elif isinstance(raw_bigbetter, int) and raw_bigbetter in {0, 1}:
        bigbetter = raw_bigbetter
    elif isinstance(raw_bigbetter, str) and raw_bigbetter.strip() in {"0", "1"}:
        bigbetter = int(raw_bigbetter.strip())
    if format_name == ComparisonFormat.MULTIWAY:
        bigbetter = 0

    ranking_items = _normalized_unique_strings(schema_dict.get("ranking_items"))
    if len(ranking_items) < 2:
        ranking_items = list(fallback_schema.ranking_items)
    if len(ranking_items) < 2:
        ranking_items = [
            col
            for col in data_summary.columns
            if data_summary.column_types.get(col) == "numeric" and not is_meta_column(col)
        ]

    indicator_col: str | None = None
    raw_indicator_col = schema_dict.get("indicator_col")
    if isinstance(raw_indicator_col, str):
        candidate = raw_indicator_col.strip()
        if candidate and candidate in data_summary.columns:
            indicator_col = candidate

    if indicator_col is None and fallback_schema.indicator_col in data_summary.columns:
        indicator_col = fallback_schema.indicator_col

    indicator_values: list[str] = []
    if indicator_col is not None:
        indicator_values = _normalized_unique_strings(schema_dict.get("indicator_values"))
        if not indicator_values:
            indicator_values = list(fallback_schema.indicator_values)

    return SemanticSchema(
        bigbetter=bigbetter,
        ranking_items=ranking_items,
        indicator_col=indicator_col,
        indicator_values=indicator_values,
    )


def infer_semantic_schema(
    data_summary: DataSummary,
    file_path: str,
    user_hints: Optional[str] = None,
) -> SemanticSchemaResult:
    """Infer comparison format and semantic schema for user verification."""
    fallback = _fallback_schema_from_heuristics(data_summary, file_path, user_hints)
    if not fallback.success or fallback.schema is None:
        return fallback

    client = get_llm_client()
    if not client.is_available():
        return fallback

    payload = {
        "data_summary": data_summary.model_dump(),
        "user_hints": user_hints,
    }
    try:
        llm_output = client.generate_json("infer_semantic_schema", payload=payload, max_completion_tokens=800)
        inferred_format = _normalize_format(llm_output.get("format"), fallback.format)
        normalized_schema = _normalize_llm_schema(
            raw_schema=llm_output.get("schema"),
            data_summary=data_summary,
            fallback_schema=fallback.schema,
            format_name=inferred_format,
        )
        format_evidence = str(llm_output.get("format_evidence") or fallback.format_evidence).strip()
        if not format_evidence:
            format_evidence = fallback.format_evidence
        return SemanticSchemaResult(
            success=True,
            format=inferred_format,
            format_evidence=format_evidence,
            schema=normalized_schema,
        )
    except (LLMCallError, ValueError, TypeError, KeyError) as exc:
        return SemanticSchemaResult(
            success=False,
            format=fallback.format,
            format_evidence=fallback.format_evidence,
            error=f"LLM semantic inference failed: {exc}",
        )
