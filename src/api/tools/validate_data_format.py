"""Tool: validate_data_format."""

from __future__ import annotations

from .common import normalize_column_name, read_table, safe_numeric
from core.schemas import FormatValidationResult, SemanticSchema


def validate_data_format(file_path: str, schema: SemanticSchema) -> FormatValidationResult:
    """Validate structural readiness for R execution."""
    issues: list[str] = []
    suggested_fixes: list[str] = []

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

    if missing_items:
        issues.append(f"Missing ranking columns: {', '.join(missing_items[:10])}")
        if any(col.lower().startswith("rank_") for col in df.columns):
            suggested_fixes.append(
                "Convert rank_* multiway columns into wide item-score columns with preprocess_data."
            )
        else:
            suggested_fixes.append(
                "Restructure input into wide format so ranking items become numeric columns."
            )

    if len(ranking_candidates) < 2:
        issues.append("At least two ranking columns must be available.")
        return FormatValidationResult(
            is_ready=False,
            fixable=bool(suggested_fixes),
            issues=issues,
            suggested_fixes=suggested_fixes,
        )

    non_numeric_cols: list[str] = []
    for col in ranking_candidates:
        converted = safe_numeric(df[col])
        if converted.notna().sum() == 0:
            non_numeric_cols.append(col)

    if non_numeric_cols:
        issues.append(
            "Ranking columns are not numeric/coercible: " + ", ".join(non_numeric_cols[:10])
        )
        suggested_fixes.append("Convert ranking columns to numeric and drop invalid rows.")

    if schema.indicator_col and schema.indicator_col not in df.columns:
        issues.append(f"Indicator column '{schema.indicator_col}' not found.")
        suggested_fixes.append("Clear indicator column or select an existing categorical column.")

    if issues:
        return FormatValidationResult(
            is_ready=False,
            fixable=True if suggested_fixes else False,
            issues=issues,
            suggested_fixes=suggested_fixes,
        )

    return FormatValidationResult(
        is_ready=True,
        fixable=False,
        issues=[],
        suggested_fixes=[],
    )
