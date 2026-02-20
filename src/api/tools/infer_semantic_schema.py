"""Tool: infer_semantic_schema."""

from __future__ import annotations

from typing import Optional

from .common import (
    HIGHER_BETTER_KEYWORDS,
    LOWER_BETTER_KEYWORDS,
    analyze_long_item_value_pairwise,
    detect_format_from_df,
    find_pairwise_long_columns,
    infer_bigbetter,
    infer_indicator_column,
    infer_ranking_items,
    is_reasonable_indicator_column,
    is_meta_column,
    read_table,
)
from core.llm_client import LLMCallError, get_llm_client
from core.schemas import ComparisonFormat, DataSummary, SemanticSchema, SemanticSchemaResult


def _is_rank_position_multiway(df) -> bool:
    """Return True when multiway rows encode rank positions (1, 2, 3...)."""
    lower_cols = [str(c).lower() for c in df.columns]
    if any(col.startswith("rank_") for col in lower_cols):
        return True

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 3:
        return False

    numeric_df = df[numeric_cols].dropna(how="all")
    if numeric_df.empty:
        return False

    def _is_rank_like(row) -> bool:
        values = row.dropna().astype(float).tolist()
        if len(values) < 3:
            return False
        if any(value <= 0 for value in values):
            return False
        if not all(float(value).is_integer() for value in values):
            return False
        rounded = [int(value) for value in values]
        unique = sorted(set(rounded))
        return unique == list(range(1, len(unique) + 1))

    rank_like_ratio = float(numeric_df.apply(_is_rank_like, axis=1).mean())
    return rank_like_ratio >= 0.6


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
            format=ComparisonFormat.MULTIWAY,
            format_evidence="Unable to read file for schema inference.",
            error=str(exc),
        )

    format_name, format_evidence = detect_format_from_df(df)
    ranking_items = infer_ranking_items(df, format_name)
    indicator_col, indicator_values = infer_indicator_column(df, ranking_items)
    bigbetter = infer_bigbetter(df, ranking_items, user_hints=user_hints)

    if format_name == "multiway" and _is_rank_position_multiway(df):
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
        if lowered in {"pairwise", "multiway"}:
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


def _count_direction_keyword_hits(ranking_items: list[str], user_hints: str | None) -> tuple[int, int]:
    """Count higher/lower direction keyword evidence from names and hints."""
    joined = " ".join(str(item).lower() for item in ranking_items)
    if user_hints:
        joined = f"{joined} {user_hints.lower()}".strip()

    high_hits = sum(1 for keyword in HIGHER_BETTER_KEYWORDS if keyword in joined)
    low_hits = sum(1 for keyword in LOWER_BETTER_KEYWORDS if keyword in joined)
    return high_hits, low_hits


def _extract_indicator_values(df, indicator_col: str) -> list[str]:
    """Extract normalized indicator values from dataframe column."""
    if indicator_col not in df.columns:
        return []
    series = df[indicator_col].dropna().astype(str).str.strip()
    return [value for value in sorted(series.unique().tolist()) if value]


def _stabilize_schema_with_data(
    schema: SemanticSchema,
    fallback_schema: SemanticSchema,
    file_path: str,
    structural_signals: dict[str, object],
    user_hints: str | None,
) -> SemanticSchema:
    """Apply deterministic reconciliation using full-data checks."""
    try:
        df = read_table(file_path)
    except Exception:  # noqa: BLE001
        return schema

    ranking_items = list(schema.ranking_items)

    indicator_col = schema.indicator_col
    indicator_values = list(schema.indicator_values)
    if indicator_col and not is_reasonable_indicator_column(df, indicator_col, ranking_items):
        indicator_col = None
        indicator_values = []

    fallback_indicator = fallback_schema.indicator_col
    if indicator_col is None and fallback_indicator and is_reasonable_indicator_column(df, fallback_indicator, ranking_items):
        indicator_col = fallback_indicator
        indicator_values = list(fallback_schema.indicator_values)

    if indicator_col is not None:
        observed_values = _extract_indicator_values(df, indicator_col)
        if observed_values:
            # Keep full observed indicator space by default.
            # LLM-proposed subsets are often incomplete and can degenerate downstream ranking.
            indicator_values = observed_values
        else:
            indicator_col = None
            indicator_values = []

    rank_like_ratio = float(structural_signals.get("rank_like_row_ratio") or 0.0)
    rank_columns = structural_signals.get("rank_columns")
    has_rank_columns = isinstance(rank_columns, list) and len(rank_columns) > 0

    if has_rank_columns or rank_like_ratio >= 0.6:
        bigbetter = 0
    else:
        high_hits, low_hits = _count_direction_keyword_hits(ranking_items, user_hints)
        if low_hits > high_hits:
            bigbetter = 0
        elif high_hits > low_hits:
            bigbetter = 1
        else:
            # Ambiguous semantics: keep deterministic fallback to avoid LLM randomness.
            bigbetter = fallback_schema.bigbetter

    return SemanticSchema(
        bigbetter=bigbetter,
        ranking_items=ranking_items,
        indicator_col=indicator_col,
        indicator_values=indicator_values,
    )


def _reconcile_long_pairwise_schema(
    structural_signals: dict[str, object],
    schema: SemanticSchema,
    fallback_schema: SemanticSchema,
) -> SemanticSchema:
    """Use full-data fallback when long pairwise logs are structurally detected."""
    long_pairwise = structural_signals.get("long_item_value_pairwise")
    if not isinstance(long_pairwise, dict) or not bool(long_pairwise.get("detected")):
        return schema

    ranking_items = list(schema.ranking_items)
    fallback_items = list(fallback_schema.ranking_items)
    if len(fallback_items) >= 2 and len(ranking_items) < len(fallback_items):
        ranking_items = fallback_items

    indicator_col = schema.indicator_col
    indicator_values = list(schema.indicator_values)
    if (
        fallback_schema.indicator_col
        and indicator_col == fallback_schema.indicator_col
        and len(indicator_values) < len(fallback_schema.indicator_values)
    ):
        indicator_values = list(fallback_schema.indicator_values)

    return SemanticSchema(
        bigbetter=schema.bigbetter,
        ranking_items=ranking_items,
        indicator_col=indicator_col,
        indicator_values=indicator_values,
    )


def _build_structural_signals(file_path: str) -> dict[str, object]:
    """Build deterministic structure signals for LLM format/schema reasoning."""
    try:
        df = read_table(file_path)
    except Exception:  # noqa: BLE001
        return {}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_df = df[numeric_cols] if numeric_cols else None
    non_null_counts = numeric_df.notna().sum(axis=1) if numeric_df is not None else None
    share_two_numeric = float((non_null_counts == 2).mean()) if non_null_counts is not None and len(df) else 0.0
    mean_non_null_numeric = float(non_null_counts.mean()) if non_null_counts is not None and len(df) else 0.0

    numeric_values: set[float] = set()
    if numeric_df is not None and not numeric_df.empty:
        flat_values = numeric_df.to_numpy().flatten()
        numeric_values = {float(value) for value in flat_values if value == value}

    rank_like_ratio = 0.0
    if numeric_df is not None and not numeric_df.empty:
        def _is_rank_like(row) -> bool:
            values = row.dropna().astype(float).tolist()
            if len(values) < 3:
                return False
            if any(value <= 0 for value in values):
                return False
            if not all(float(value).is_integer() for value in values):
                return False
            rounded = [int(value) for value in values]
            unique = sorted(set(rounded))
            return unique == list(range(1, len(unique) + 1))

        rank_like_ratio = float(numeric_df.apply(_is_rank_like, axis=1).mean())

    left_col, right_col, winner_col = find_pairwise_long_columns(df)
    long_pairwise = analyze_long_item_value_pairwise(df)
    rank_columns = [col for col in df.columns if str(col).lower().startswith("rank_")]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "rank_columns": rank_columns,
        "pairwise_long_columns": {
            "left": left_col,
            "right": right_col,
            "winner": winner_col,
        },
        "long_item_value_pairwise": long_pairwise,
        "share_rows_with_two_numeric_values": share_two_numeric,
        "mean_non_null_numeric_per_row": mean_non_null_numeric,
        "numeric_values_binary_only": bool(numeric_values) and numeric_values.issubset({0.0, 1.0}),
        "rank_like_row_ratio": rank_like_ratio,
    }


def _detect_strong_conflict(
    structural_signals: dict[str, object],
    inferred_format: ComparisonFormat,
    schema: SemanticSchema,
) -> str | None:
    """Detect high-confidence conflicts and request one LLM self-correction pass."""
    share_two = float(structural_signals.get("share_rows_with_two_numeric_values") or 0.0)
    mean_non_null = float(structural_signals.get("mean_non_null_numeric_per_row") or 0.0)
    rank_like_ratio = float(structural_signals.get("rank_like_row_ratio") or 0.0)
    binary_only = bool(structural_signals.get("numeric_values_binary_only"))
    rank_columns = structural_signals.get("rank_columns")
    has_rank_columns = isinstance(rank_columns, list) and len(rank_columns) > 0
    long_pairwise = structural_signals.get("long_item_value_pairwise")
    long_pairwise_detected = isinstance(long_pairwise, dict) and bool(long_pairwise.get("detected"))

    if binary_only and share_two >= 0.9 and inferred_format != ComparisonFormat.PAIRWISE:
        return (
            "Strong structure conflict: rows are binary-only and almost all rows have exactly "
            "two numeric entries, so the format should be pairwise."
        )

    if long_pairwise_detected and inferred_format != ComparisonFormat.PAIRWISE:
        return (
            "Strong structure conflict: long item/value logs form two-item comparisons "
            "per group, so the format should be pairwise."
        )

    if long_pairwise_detected:
        expected_items = long_pairwise.get("unique_items")
        if isinstance(expected_items, list) and expected_items:
            expected_set = {
                str(value).strip().lower()
                for value in expected_items
                if isinstance(value, str) and value.strip()
            }
            observed_set = {item.strip().lower() for item in schema.ranking_items if item.strip()}
            if expected_set and not expected_set.issubset(observed_set):
                return (
                    "Strong schema conflict: long pairwise logs include additional items in "
                    "the item column that are missing from schema.ranking_items."
                )

    if (not binary_only) and mean_non_null >= 3.0 and share_two <= 0.2 and inferred_format != ComparisonFormat.MULTIWAY:
        return (
            "Strong structure conflict: rows are dense with >=3 numeric entries per row, "
            "so the format should be multiway."
        )

    if has_rank_columns and schema.bigbetter != 0:
        return (
            "Strong schema conflict: rank_* columns encode rank order, so "
            "bigbetter must be 0 (lower rank is better)."
        )

    if rank_like_ratio >= 0.6 and schema.bigbetter != 0:
        return (
            "Strong schema conflict: rows are rank-position style permutations, so "
            "bigbetter must be 0 (lower rank is better)."
        )

    return None


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

    structural_signals = _build_structural_signals(file_path)
    payload = {
        "data_summary": data_summary.model_dump(),
        "user_hints": user_hints,
        "structural_signals": structural_signals,
    }
    try:
        llm_output = client.generate_json("infer_semantic_schema", payload=payload, max_completion_tokens=800)
        inferred_format = _normalize_format(llm_output.get("format"), fallback.format)
        normalized_schema = _normalize_llm_schema(
            raw_schema=llm_output.get("schema"),
            data_summary=data_summary,
            fallback_schema=fallback.schema,
        )
        normalized_schema = _reconcile_long_pairwise_schema(
            structural_signals=structural_signals,
            schema=normalized_schema,
            fallback_schema=fallback.schema,
        )
        normalized_schema = _stabilize_schema_with_data(
            schema=normalized_schema,
            fallback_schema=fallback.schema,
            file_path=file_path,
            structural_signals=structural_signals,
            user_hints=user_hints,
        )

        conflict_reason = _detect_strong_conflict(
            structural_signals=structural_signals,
            inferred_format=inferred_format,
            schema=normalized_schema,
        )
        if conflict_reason:
            retry_payload = {
                **payload,
                "consistency_feedback": {
                    "issue": conflict_reason,
                    "previous_output": llm_output,
                    "action": "Revise output to resolve the conflict while keeping all fields valid.",
                },
            }
            llm_output = client.generate_json("infer_semantic_schema", payload=retry_payload, max_completion_tokens=800)
            inferred_format = _normalize_format(llm_output.get("format"), fallback.format)
            normalized_schema = _normalize_llm_schema(
                raw_schema=llm_output.get("schema"),
                data_summary=data_summary,
                fallback_schema=fallback.schema,
            )
            normalized_schema = _reconcile_long_pairwise_schema(
                structural_signals=structural_signals,
                schema=normalized_schema,
                fallback_schema=fallback.schema,
            )
            normalized_schema = _stabilize_schema_with_data(
                schema=normalized_schema,
                fallback_schema=fallback.schema,
                file_path=file_path,
                structural_signals=structural_signals,
                user_hints=user_hints,
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
