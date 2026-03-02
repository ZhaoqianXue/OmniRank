"""Tool: validate_data_format."""

from __future__ import annotations

from .common import (
    find_long_item_value_columns,
    find_pairwise_long_columns,
    normalize_column_name,
    read_table,
    safe_numeric,
)
from core.schemas import FormatValidationResult, SemanticSchema


def validate_data_format(file_path: str, schema: SemanticSchema) -> FormatValidationResult:
    """Validate structural readiness for R execution."""
    issues: list[str] = []
    suggested_fixes: list[str] = []
    auto_fixable = False

    try:
        df = read_table(file_path)
    except Exception as exc:  # noqa: BLE001
        return FormatValidationResult(
            is_ready=False,
            fixable=False,
            issues=[f"Failed to parse file: {exc}"],
            suggested_fixes=[],
        )

    if len(df.columns) < 2:
        return FormatValidationResult(
            is_ready=False,
            fixable=False,
            issues=["File must contain at least two columns."],
            suggested_fixes=[],
        )

    pairwise_left, pairwise_right, _ = find_pairwise_long_columns(df)
    long_item_col, long_value_col = find_long_item_value_columns(df)
    has_rank_columns = any(col.lower().startswith("rank_") for col in df.columns)

    missing_items: list[str] = []
    ranking_candidates: list[str] = []
    for item in schema.ranking_items:
        if item in df.columns:
            ranking_candidates.append(item)
            continue
        normalized = normalize_column_name(item)
        if normalized in df.columns:
            ranking_candidates.append(normalized)
            continue
        missing_items.append(item)

    if missing_items and len(ranking_candidates) < 2:
        issues.append(f"Missing ranking columns: {', '.join(missing_items[:10])}")
        if has_rank_columns:
            suggested_fixes.append(
                "Convert rank_* multiway columns into wide item-score columns with preprocess_data."
            )
            auto_fixable = True
        elif pairwise_left and pairwise_right:
            suggested_fixes.append(
                "Restructure pairwise long format (item_a/item_b/winner) into wide item-score columns."
            )
            auto_fixable = True
        elif long_item_col and long_value_col:
            suggested_fixes.append(
                "Pivot long item/value rows into wide item-score columns."
            )
            auto_fixable = True
        else:
            suggested_fixes.append(
                "Restructure input into wide format so ranking items become numeric columns."
            )

    if len(ranking_candidates) < 2:
        if pairwise_left and pairwise_right:
            suggested_fixes.append(
                "Restructure pairwise long format (item_a/item_b/winner) into wide item-score columns."
            )
            auto_fixable = True
        elif long_item_col and long_value_col:
            suggested_fixes.append(
                "Pivot long item/value rows into wide item-score columns."
            )
            auto_fixable = True
        elif has_rank_columns:
            suggested_fixes.append(
                "Convert rank_* multiway columns into wide item-score columns with preprocess_data."
            )
            auto_fixable = True
        elif not suggested_fixes:
            suggested_fixes.append(
                "Restructure input into wide format so ranking items become numeric columns."
            )

        issues.append("At least two ranking columns must be available.")
        suggested_fixes = list(dict.fromkeys(suggested_fixes))
        return FormatValidationResult(
            is_ready=False,
            fixable=auto_fixable,
            issues=issues,
            suggested_fixes=suggested_fixes,
        )

    non_numeric_cols: list[str] = []
    no_signal_cols: list[str] = []
    for col in ranking_candidates:
        converted = safe_numeric(df[col])
        if converted.notna().sum() == 0:
            non_numeric_cols.append(col)
            if int(df[col].notna().sum()) == 0:
                no_signal_cols.append(col)

    if non_numeric_cols:
        issues.append(
            "Ranking columns are not numeric/coercible: " + ", ".join(non_numeric_cols[:10])
        )
        remaining_if_dropped = len(ranking_candidates) - len(non_numeric_cols)
        if remaining_if_dropped >= 2:
            auto_fixable = True
            if no_signal_cols:
                suggested_fixes.append(
                    "Drop ranking columns with no numeric signal and continue with the remaining items."
                )
            else:
                suggested_fixes.append("Drop ranking columns that remain non-numeric after coercion.")
        else:
            suggested_fixes.append(
                "Remove or replace non-numeric ranking columns so at least two numeric item columns remain."
            )

    if schema.indicator_col and schema.indicator_col not in df.columns:
        issues.append(f"Indicator column '{schema.indicator_col}' not found.")
        suggested_fixes.append("Clear indicator column or select an existing categorical column.")

    if issues:
        return FormatValidationResult(
            is_ready=False,
            fixable=auto_fixable,
            issues=issues,
            suggested_fixes=suggested_fixes,
        )

    return FormatValidationResult(
        is_ready=True,
        fixable=False,
        issues=[],
        suggested_fixes=[],
    )
