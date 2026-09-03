"""Tool: validate_data_quality."""

from __future__ import annotations

import networkx as nx

from .common import build_comparison_graph, normalize_column_name, read_table, sparsity_warning
from core.schemas import QualityValidationResult, SemanticSchema


def validate_data_quality(file_path: str, schema: SemanticSchema) -> QualityValidationResult:
    """Validate statistical quality and blocking conditions."""
    warnings: list[str] = []
    errors: list[str] = []

    try:
        df = read_table(file_path)
    except Exception as exc:  # noqa: BLE001
        return QualityValidationResult(is_valid=False, warnings=[], errors=[str(exc)])

    # Preprocessing may normalize column names for R compatibility, so an item can
    # appear under its normalized name (e.g. 'gemini-2.5-pro' -> 'gemini_25_pro').
    columns = set(df.columns)
    normalized_lookup = {normalize_column_name(col): col for col in df.columns}
    ranking_items = []
    for item in schema.ranking_items:
        if item in columns:
            ranking_items.append(item)
            continue
        resolved = normalized_lookup.get(normalize_column_name(item))
        if resolved is not None:
            ranking_items.append(resolved)
    ranking_items = list(dict.fromkeys(ranking_items))

    if len(ranking_items) < 2:
        errors.append("Fewer than two ranking items are available after preprocessing.")
        return QualityValidationResult(is_valid=False, warnings=warnings, errors=errors)

    graph, comparisons = build_comparison_graph(df, ranking_items)
    if comparisons <= 0:
        errors.append("No valid comparisons were found in the dataset.")

    if len(graph.nodes) < 2:
        errors.append("Comparison graph has fewer than two nodes.")
    else:
        try:
            connected = nx.is_connected(graph)
        except nx.NetworkXPointlessConcept:
            connected = False
        if not connected:
            errors.append(
                "Comparison graph is disconnected. Global ranking is not identifiable across components."
            )

    sparse_msg = sparsity_warning(len(ranking_items), comparisons)
    if sparse_msg:
        warnings.append(sparse_msg)

    unique_rows = df[ranking_items].dropna(how="all").shape[0]
    if unique_rows < 2:
        errors.append("Data has insufficient variation across comparisons.")

    return QualityValidationResult(is_valid=len(errors) == 0, warnings=warnings, errors=errors)
