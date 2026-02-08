"""Tool: request_user_confirmation."""

from __future__ import annotations

from core.schemas import (
    ConfirmationResult,
    FormatValidationResult,
    QualityValidationResult,
    SemanticSchema,
)


def request_user_confirmation(
    proposed_schema: SemanticSchema,
    format_result: FormatValidationResult,
    quality_result: QualityValidationResult,
    confirmed: bool,
    confirmed_schema: SemanticSchema | None = None,
    user_modifications: list[str] | None = None,
    B: int = 2000,
    seed: int = 42,
) -> ConfirmationResult:
    """Apply explicit confirmation payload while preserving validation context."""
    if not format_result.is_ready:
        raise ValueError("Cannot confirm schema: data format is not ready.")
    if not quality_result.is_valid:
        raise ValueError("Cannot confirm schema: data quality has blocking errors.")

    schema = confirmed_schema or proposed_schema

    return ConfirmationResult(
        confirmed=confirmed,
        confirmed_schema=schema,
        user_modifications=user_modifications or [],
        B=B,
        seed=seed,
    )
