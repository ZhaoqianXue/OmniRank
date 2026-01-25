"""
Data Agent

Responsible for:
- Data format detection (pointwise vs pairwise vs multiway)
- Schema inference (bigbetter, ranking_items, indicator_col)
- Data validation (connectivity, sparsity checks)
"""

import logging
import io
from typing import Optional

import pandas as pd
import networkx as nx

from core.schemas import (
    DataFormat,
    InferredSchema,
    ValidationWarning,
)

logger = logging.getLogger(__name__)


def detect_format(df: pd.DataFrame) -> DataFormat:
    """
    Detect the data format from a DataFrame.
    
    Heuristics:
    - Pairwise: Sparse matrix with exactly 2 non-null values per row (0 and 1)
    - Pointwise: Dense numeric matrix (most cells have values)
    - Multiway: Reserved for future extension
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Detected DataFormat
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numeric_cols) < 2:
        # Fallback to pointwise
        return DataFormat.POINTWISE
    
    numeric_df = df[numeric_cols]
    
    # Check for pairwise pattern: exactly 2 non-null values per row
    non_null_counts = numeric_df.notna().sum(axis=1)
    
    # Pairwise: most rows have exactly 2 non-null values
    pairwise_rows = (non_null_counts == 2).sum()
    total_rows = len(df)
    
    if total_rows > 0 and pairwise_rows / total_rows > 0.8:
        # Additionally check for 0/1 pattern
        values = numeric_df.values.flatten()
        values = values[~pd.isna(values)]
        unique_values = set(values)
        
        if unique_values.issubset({0, 1, 0.0, 1.0}):
            return DataFormat.PAIRWISE
    
    # Default to pointwise
    return DataFormat.POINTWISE


def infer_bigbetter(df: pd.DataFrame, data_format: DataFormat) -> tuple[int, float]:
    """
    Infer whether higher scores are better.
    
    For pairwise data: 1 typically indicates the winner
    For pointwise data: Heuristic based on column names and value distribution
    
    Returns:
        tuple of (bigbetter, confidence)
    """
    # Pairwise format: 1 is always "winner"
    if data_format == DataFormat.PAIRWISE:
        return 1, 0.95
    
    # Pointwise: check for common patterns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Check column names for hints
    lower_cols = [c.lower() for c in numeric_cols]
    
    # Patterns suggesting "lower is better" (error, loss, time, distance)
    lower_better_keywords = ["error", "loss", "time", "distance", "cost", "latency"]
    # Patterns suggesting "higher is better" (score, accuracy, f1, auc)
    higher_better_keywords = ["score", "accuracy", "f1", "auc", "precision", "recall", "reward"]
    
    lower_count = sum(any(kw in col for kw in lower_better_keywords) for col in lower_cols)
    higher_count = sum(any(kw in col for kw in higher_better_keywords) for col in lower_cols)
    
    if higher_count > lower_count:
        return 1, 0.8
    elif lower_count > higher_count:
        return 0, 0.8
    
    # Default: assume higher is better with lower confidence
    return 1, 0.6


def extract_ranking_items(df: pd.DataFrame, data_format: DataFormat) -> list[str]:
    """
    Extract the items to be ranked.
    
    For pointwise: column names (models)
    For pairwise: column names excluding indicator column
    """
    # Known non-ranking columns to exclude
    exclude_patterns = [
        "sample", "case", "id", "description", "task", "category", "indicator"
    ]
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Filter out columns matching exclude patterns
    ranking_items = []
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(pattern in col_lower for pattern in exclude_patterns):
            ranking_items.append(col)
    
    # If we filtered everything, use all numeric columns
    if not ranking_items:
        ranking_items = numeric_cols
    
    return ranking_items


def detect_indicator_column(df: pd.DataFrame) -> tuple[Optional[str], list[str]]:
    """
    Detect indicator column (e.g., Task type) for stratified analysis.
    
    Returns:
        tuple of (indicator_col, indicator_values)
    """
    # Look for categorical columns that could be indicators
    string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Common indicator column names
    indicator_keywords = ["task", "category", "type", "group", "class", "domain"]
    
    for col in string_cols:
        col_lower = col.lower()
        if any(kw in col_lower for kw in indicator_keywords):
            values = df[col].dropna().unique().tolist()
            if 2 <= len(values) <= 50:  # Reasonable number of categories
                return col, values
    
    # Check first string column if it looks like an indicator
    for col in string_cols:
        values = df[col].dropna().unique().tolist()
        if 2 <= len(values) <= 20:
            return col, values
    
    return None, []


def infer_schema(content: bytes, filename: str) -> InferredSchema:
    """
    Infer complete schema from file content.
    
    Args:
        content: Raw file content
        filename: Original filename for format hints
        
    Returns:
        InferredSchema with all inferred fields
    """
    # Parse CSV
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        # Return minimal schema
        return InferredSchema(
            format=DataFormat.POINTWISE,
            bigbetter=1,
            ranking_items=[],
            indicator_col=None,
            indicator_values=[],
            confidence=0.0,
        )
    
    # Detect format
    data_format = detect_format(df)
    logger.info(f"Detected format: {data_format}")
    
    # Infer bigbetter
    bigbetter, bb_confidence = infer_bigbetter(df, data_format)
    
    # Extract ranking items
    ranking_items = extract_ranking_items(df, data_format)
    
    # Detect indicator column
    indicator_col, indicator_values = detect_indicator_column(df)
    
    # Calculate overall confidence
    confidence = bb_confidence * (0.9 if len(ranking_items) >= 2 else 0.5)
    
    return InferredSchema(
        format=data_format,
        bigbetter=bigbetter,
        ranking_items=ranking_items,
        indicator_col=indicator_col,
        indicator_values=indicator_values,
        confidence=round(confidence, 2),
    )


def check_connectivity(df: pd.DataFrame, ranking_items: list[str]) -> tuple[bool, str]:
    """
    Check if the comparison graph is connected.
    
    For valid spectral ranking, all items should be in the same connected component.
    
    Returns:
        tuple of (is_connected, message)
    """
    # Build comparison graph
    G = nx.Graph()
    G.add_nodes_from(ranking_items)
    
    # Extract comparisons based on format
    numeric_df = df[ranking_items]
    
    for _, row in numeric_df.iterrows():
        non_null_items = [col for col in ranking_items if pd.notna(row[col])]
        # Add edges between all compared items in this row
        for i in range(len(non_null_items)):
            for j in range(i + 1, len(non_null_items)):
                G.add_edge(non_null_items[i], non_null_items[j])
    
    is_connected = nx.is_connected(G)
    
    if is_connected:
        return True, "Comparison graph is connected"
    else:
        n_components = nx.number_connected_components(G)
        components = list(nx.connected_components(G))
        isolated = [c for c in components if len(c) == 1]
        return False, f"Graph has {n_components} components. Isolated items: {isolated}"


def check_sparsity(
    df: pd.DataFrame,
    ranking_items: list[str]
) -> tuple[float, str, str]:
    """
    Check data sparsity and return warnings if problematic.
    
    Returns:
        tuple of (sparsity_ratio, severity, message)
    """
    n_items = len(ranking_items)
    n_samples = len(df)
    
    # Expected comparisons for dense data: n_samples * C(n_items, 2)
    max_comparisons = n_samples * (n_items * (n_items - 1) / 2)
    
    # Count actual comparisons
    numeric_df = df[ranking_items]
    actual_comparisons = 0
    
    for _, row in numeric_df.iterrows():
        non_null = row.notna().sum()
        actual_comparisons += non_null * (non_null - 1) / 2
    
    sparsity_ratio = actual_comparisons / max_comparisons if max_comparisons > 0 else 0
    
    # Determine severity
    if sparsity_ratio < 0.1:
        return sparsity_ratio, "error", f"Data is extremely sparse ({sparsity_ratio:.1%}). Results may be unreliable."
    elif sparsity_ratio < 0.3:
        return sparsity_ratio, "warning", f"Data is sparse ({sparsity_ratio:.1%}). Consider collecting more comparisons."
    else:
        return sparsity_ratio, "info", f"Sparsity ratio: {sparsity_ratio:.1%}"


def validate_data(
    content: bytes,
    schema: InferredSchema
) -> list[ValidationWarning]:
    """
    Validate data against inferred schema.
    
    Checks:
    - Connectivity of comparison graph
    - Sparsity of data
    - Minimum sample size
    
    Args:
        content: Raw file content
        schema: Inferred schema
        
    Returns:
        List of ValidationWarning objects
    """
    warnings = []
    
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        warnings.append(ValidationWarning(
            type="format",
            message=f"Failed to parse file: {str(e)}",
            severity="error",
        ))
        return warnings
    
    ranking_items = schema.ranking_items
    
    if len(ranking_items) < 2:
        warnings.append(ValidationWarning(
            type="format",
            message="At least 2 ranking items are required",
            severity="error",
        ))
        return warnings
    
    # Check minimum sample size
    if len(df) < 10:
        warnings.append(ValidationWarning(
            type="sparsity",
            message=f"Only {len(df)} samples. Consider collecting more data for reliable results.",
            severity="warning",
        ))
    
    # Check connectivity
    is_connected, conn_msg = check_connectivity(df, ranking_items)
    if not is_connected:
        warnings.append(ValidationWarning(
            type="connectivity",
            message=conn_msg,
            severity="error",
        ))
    
    # Check sparsity
    sparsity, severity, sparsity_msg = check_sparsity(df, ranking_items)
    if severity in ["warning", "error"]:
        warnings.append(ValidationWarning(
            type="sparsity",
            message=sparsity_msg,
            severity=severity,
        ))
    
    return warnings


class DataAgent:
    """
    Data Agent: Handles data upload, schema inference, and validation.
    
    This agent is the first step in the OmniRank workflow.
    """
    
    def __init__(self):
        """Initialize Data Agent."""
        self.name = "data"
    
    def process(
        self,
        content: bytes,
        filename: str
    ) -> tuple[InferredSchema, list[ValidationWarning]]:
        """
        Process uploaded data file.
        
        Args:
            content: Raw file content
            filename: Original filename
            
        Returns:
            tuple of (InferredSchema, list of ValidationWarnings)
        """
        logger.info(f"DataAgent processing file: {filename}")
        
        # Infer schema
        schema = infer_schema(content, filename)
        logger.info(f"Inferred schema: format={schema.format}, items={len(schema.ranking_items)}")
        
        # Validate data
        warnings = validate_data(content, schema)
        
        if warnings:
            logger.warning(f"Validation warnings: {[w.message for w in warnings]}")
        
        return schema, warnings
