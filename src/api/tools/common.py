"""Shared utility helpers for OmniRank tools."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


META_KEYWORDS = {
    "id",
    "sample",
    "case",
    "row",
    "index",
    "description",
    "note",
    "text",
}

HIGHER_BETTER_KEYWORDS = {
    "score",
    "accuracy",
    "f1",
    "auc",
    "precision",
    "recall",
    "win",
    "reward",
    "success",
}

LOWER_BETTER_KEYWORDS = {
    "error",
    "loss",
    "time",
    "latency",
    "distance",
    "cost",
    "rank",
    "penalty",
}

INDICATOR_KEYWORDS = {"task", "category", "type", "group", "class", "domain", "segment", "benchmark"}
PAIRWISE_STRUCTURAL_COLUMNS = {
    "item_a",
    "item_b",
    "item1",
    "item2",
    "left",
    "right",
    "winner",
    "preferred",
    "outcome",
}
PAIRWISE_LEFT_COLUMNS = ("item_a", "item1", "left", "player_a")
PAIRWISE_RIGHT_COLUMNS = ("item_b", "item2", "right", "player_b")
PAIRWISE_WINNER_COLUMNS = ("winner", "preferred", "outcome", "label", "result")
LONG_ITEM_COLUMNS = ("item", "item_name", "method", "model", "candidate", "algorithm")
LONG_VALUE_COLUMNS = ("value", "score", "metric", "rating", "measurement")


def is_meta_column(name: str) -> bool:
    """Return True when column name is likely metadata."""
    lower = name.lower()
    return any(keyword in lower for keyword in META_KEYWORDS)


def find_pairwise_long_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Find pairwise long-format columns (left, right, winner)."""
    lower_map = {col.lower(): col for col in df.columns}

    left_col = next((lower_map[key] for key in PAIRWISE_LEFT_COLUMNS if key in lower_map), None)
    right_col = next((lower_map[key] for key in PAIRWISE_RIGHT_COLUMNS if key in lower_map), None)
    winner_col = next((lower_map[key] for key in PAIRWISE_WINNER_COLUMNS if key in lower_map), None)
    return left_col, right_col, winner_col


def find_long_item_value_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Find long-format item/value columns."""
    lower_map = {col.lower(): col for col in df.columns}
    item_col = next((lower_map[key] for key in LONG_ITEM_COLUMNS if key in lower_map), None)
    value_col = next((lower_map[key] for key in LONG_VALUE_COLUMNS if key in lower_map), None)
    return item_col, value_col


def read_table(file_path: str) -> pd.DataFrame:
    """Read a CSV file with strict path validation."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported")
    return pd.read_csv(path)


def infer_column_type(series: pd.Series) -> str:
    """Infer high-level column type for DataSummary."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def normalize_column_name(name: str) -> str:
    """Normalize one column name for R compatibility."""
    normalized = name.strip().replace(" ", "_").replace("-", "_")
    normalized = re.sub(r"[^A-Za-z0-9_]", "", normalized)
    return normalized or "column"


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric with coercion."""
    return pd.to_numeric(series, errors="coerce")


def detect_format_from_df(df: pd.DataFrame) -> tuple[str, str]:
    """Detect pointwise/pairwise/multiway using structural patterns."""
    lower_cols = [c.lower() for c in df.columns]
    if any(col.startswith("rank_") for col in lower_cols):
        return "multiway", "Detected rank_* columns indicating multiway comparisons."

    left_col, right_col, _winner_col = find_pairwise_long_columns(df)
    if left_col and right_col:
        left_values = df[left_col].dropna().astype(str).str.strip()
        right_values = df[right_col].dropna().astype(str).str.strip()
        unique_items = sorted({*left_values.tolist(), *right_values.tolist()} - {""})
        if len(unique_items) >= 2:
            return "pairwise", "Detected pairwise long format columns (item_a/item_b style)."

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) >= 3:
        numeric_df = df[numeric_cols].dropna(how="all")
        if not numeric_df.empty:
            def _is_rank_like(row: pd.Series) -> bool:
                values = pd.to_numeric(row, errors="coerce").dropna().astype(float).tolist()
                if len(values) < 3:
                    return False
                if any(value <= 0 for value in values):
                    return False
                rounded = [int(v) for v in values if float(v).is_integer()]
                if len(rounded) != len(values):
                    return False
                unique = sorted(set(rounded))
                return unique == list(range(1, len(unique) + 1))

            rank_like_ratio = float(numeric_df.apply(_is_rank_like, axis=1).mean())
            if rank_like_ratio >= 0.6:
                return "multiway", "Detected rank-like row permutations across numeric item columns."

    if len(numeric_cols) >= 2:
        numeric_df = df[numeric_cols]
        non_null_counts = numeric_df.notna().sum(axis=1)
        share_two = float((non_null_counts == 2).mean()) if len(df) else 0.0
        values = numeric_df.to_numpy().flatten()
        values = values[~pd.isna(values)]
        value_set = set(values.tolist())
        if share_two >= 0.7 and value_set.issubset({0, 1, 0.0, 1.0}):
            return "pairwise", "Rows mostly contain two 0/1 comparison entries, indicating pairwise data."

    return "pointwise", "Detected dense numeric score matrix consistent with pointwise data."


def infer_bigbetter(df: pd.DataFrame, ranking_items: list[str], user_hints: str | None = None) -> int:
    """Infer direction where 1 means larger values are better."""
    joined = " ".join([c.lower() for c in ranking_items])
    if user_hints:
        joined += f" {user_hints.lower()}"

    high_hits = sum(1 for kw in HIGHER_BETTER_KEYWORDS if kw in joined)
    low_hits = sum(1 for kw in LOWER_BETTER_KEYWORDS if kw in joined)

    if high_hits > low_hits:
        return 1
    if low_hits > high_hits:
        return 0

    numeric_cols = [col for col in ranking_items if col in df.columns]
    if not numeric_cols:
        return 1

    numeric_values = []
    for col in numeric_cols:
        numeric_values.extend(pd.to_numeric(df[col], errors="coerce").dropna().tolist())

    if not numeric_values:
        return 1

    min_v = min(numeric_values)
    max_v = max(numeric_values)
    if min_v >= 0 and max_v <= 1:
        return 1
    return 1


def infer_ranking_items(df: pd.DataFrame, format_name: str) -> list[str]:
    """Infer ranking item columns from dataframe schema."""
    if format_name == "multiway":
        rank_columns = [c for c in df.columns if c.lower().startswith("rank_")]
        if rank_columns:
            try:
                sample = df[rank_columns].stack(future_stack=True).dropna()
            except TypeError:
                sample = df[rank_columns].stack(dropna=True)
            if not sample.empty and not pd.api.types.is_numeric_dtype(sample):
                inferred_items = sorted({str(value) for value in sample.astype(str).tolist() if str(value).strip()})
                if len(inferred_items) >= 2:
                    return inferred_items
            return rank_columns

    if format_name == "pairwise":
        left_col, right_col, _winner_col = find_pairwise_long_columns(df)
        if left_col and right_col:
            values = pd.concat([df[left_col], df[right_col]], axis=0).dropna().astype(str).tolist()
            unique_items = sorted({value.strip() for value in values if value.strip()})
            if len(unique_items) >= 2:
                return unique_items

    item_col, value_col = find_long_item_value_columns(df)
    if item_col and value_col:
        item_values = df[item_col].dropna().astype(str).str.strip()
        unique_items = sorted({value for value in item_values.tolist() if value})
        if len(unique_items) >= 2:
            return unique_items

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    ranking_items: list[str] = []
    for col in numeric_cols:
        if is_meta_column(col):
            continue
        ranking_items.append(col)

    return ranking_items


def infer_indicator_column(df: pd.DataFrame, ranking_items: list[str]) -> tuple[str | None, list[str]]:
    """Infer at most one indicator column."""
    pairwise_left, pairwise_right, pairwise_winner = find_pairwise_long_columns(df)
    long_item_col, long_value_col = find_long_item_value_columns(df)
    structural_exclusions = {
        pairwise_left,
        pairwise_right,
        pairwise_winner,
        long_item_col,
        long_value_col,
    }
    candidates = [
        col
        for col in df.columns
        if col not in ranking_items
        and col.lower() not in PAIRWISE_STRUCTURAL_COLUMNS
        and col not in structural_exclusions
        and not col.lower().startswith("rank_")
        and not pd.api.types.is_numeric_dtype(df[col])
    ]

    def cardinality_ok(col: str) -> bool:
        unique_count = int(df[col].dropna().nunique())
        return 2 <= unique_count <= 50

    prioritized = [
        col
        for col in candidates
        if cardinality_ok(col) and any(keyword in col.lower() for keyword in INDICATOR_KEYWORDS)
    ]

    selected: str | None = None
    if prioritized:
        selected = prioritized[0]
    else:
        fallback = [col for col in candidates if cardinality_ok(col)]
        selected = fallback[0] if fallback else None

    if selected is None:
        return None, []

    values = [str(v) for v in sorted(df[selected].dropna().unique().tolist())]
    return selected, values


def build_comparison_graph(df: pd.DataFrame, ranking_items: list[str]) -> tuple[nx.Graph, float]:
    """Build graph and return estimated comparison count M."""
    valid_items = [col for col in ranking_items if col in df.columns]
    graph = nx.Graph()
    graph.add_nodes_from(valid_items)

    comparisons = 0.0
    for _, row in df[valid_items].iterrows():
        present = [item for item in valid_items if pd.notna(row[item])]
        if len(present) < 2:
            continue
        k = len(present)
        comparisons += (k * (k - 1)) / 2
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                graph.add_edge(present[i], present[j])

    return graph, comparisons


def sparsity_warning(n_items: int, comparisons: float) -> str | None:
    """Return sparsity warning when M < n log n."""
    if n_items < 2:
        return None

    threshold = n_items * math.log(n_items)
    if comparisons < threshold:
        return (
            f"Sparse comparison graph: M={comparisons:.2f} is below n*log(n)={threshold:.2f}. "
            "Uncertainty intervals may be wider."
        )
    return None
