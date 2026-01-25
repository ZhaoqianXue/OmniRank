"""
Data Agent

Responsible for:
- Data format detection (pointwise vs pairwise vs multiway)
- Schema inference (bigbetter, ranking_items, indicator_col)
- Data validation (connectivity, sparsity checks)

This agent uses LLM reasoning combined with statistical analysis
to intelligently understand user data semantics.
"""

import logging
import io
import json
import math
import os
from typing import Optional

import pandas as pd
import networkx as nx
from openai import OpenAI

from core.schemas import (
    DataFormat,
    InferredSchema,
    ValidationWarning,
    AgentType,
)
from core.session_memory import SessionMemory, TraceType

logger = logging.getLogger(__name__)


class DataAgent:
    """
    Data Agent: Handles data upload, schema inference, and validation.
    
    Uses LLM to:
    1. Intelligently detect data format from structure and content
    2. Infer semantic schema (bigbetter, items, indicators)
    3. Generate human-readable validation feedback
    """
    
    # =========================================================================
    # System Prompt: Structured System Instructions (OpenAI Recommended)
    # =========================================================================
    # Following the pattern from todo.md: "Structured System Instructions"
    # This separates ROLE + KNOWLEDGE (system) from TASK + DATA (user)
    # =========================================================================
    
    SYSTEM_PROMPT = """
# Role

You are a Data Schema Analyst, an expert in understanding and interpreting structured data for spectral ranking analysis. Your expertise includes:
- Identifying data formats (pointwise, pairwise, multiway)
- Inferring semantic meaning from column names and value distributions
- Detecting preference directions and stratification dimensions

# Knowledge Base

## Data Formats

### Pointwise Format
- Dense numeric matrix where each row is a sample/observation
- Columns represent items being ranked (e.g., models, products)
- Cell values are scores/metrics for that item on that sample
- Example: LLM benchmark scores, product ratings across criteria

### Pairwise Format
- Sparse matrix encoding head-to-head comparisons
- Each row is one comparison between exactly 2 items
- Values: 1 = winner, 0 = loser, NaN = not compared
- May have indicator column (e.g., "Task") for stratified analysis

### Multiway Format
- Encodes rankings where multiple items compete simultaneously
- Example: Horse race results, multi-player game outcomes
- Columns may be Rank_1 (winner), Rank_2, ..., Rank_k

## BigBetter Inference Rules

### Higher is Better (bigbetter=1)
- Column name patterns: score, accuracy, f1, auc, precision, recall, win, reward
- Distribution patterns: bounded [0,1], unbounded positive, increasing metrics
- Context: performance metrics, success rates, quality scores

### Lower is Better (bigbetter=0)
- Column name patterns: error, loss, time, latency, distance, cost, rank
- Distribution patterns: non-negative with right-skew, time/duration values
- Context: error rates, response times, costs, rank positions

## Indicator Column Rules

An indicator column is a categorical dimension for stratified analysis:
- Examples: "Task" (code/math/writing), "Category" (sports/news/tech)
- CRITICAL: Select AT MOST ONE indicator column
- Prefer columns with semantic meaning over arbitrary IDs
- Good cardinality: 2-20 unique values (not too few, not too many)

# Output Format

Always respond in valid JSON format with no markdown formatting.
"""
    
    # Separate system prompt for validation explanation
    VALIDATION_SYSTEM_PROMPT = """
# Role

You are a Data Validation Advisor who explains technical validation results in plain language.

# Communication Style

- Be concise (2-3 sentences maximum)
- Use non-technical language
- Be helpful and reassuring where appropriate
- Clearly state whether analysis can proceed
"""
    
    def __init__(self):
        """Initialize Data Agent with LLM capabilities."""
        self.name = "data"
        
        # Initialize OpenAI client (following AnalystAgent pattern)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
            self.enabled = True
        else:
            logger.warning("OPENAI_API_KEY not set, Data Agent using fallback mode")
            self.enabled = False
    
    # =========================================================================
    # Main Process Method
    # =========================================================================
    
    def process(
        self,
        content: bytes,
        filename: str,
        session: Optional[SessionMemory] = None
    ) -> tuple[InferredSchema, list[ValidationWarning], str]:
        """
        Process uploaded data file with intelligent schema inference.
        
        Args:
            content: Raw file content
            filename: Original filename
            session: Optional session for trace logging
            
        Returns:
            tuple of (InferredSchema, list of ValidationWarnings, explanation)
        """
        logger.info(f"DataAgent processing file: {filename}")
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            return (
                InferredSchema(
                    format=DataFormat.POINTWISE,
                    bigbetter=1,
                    ranking_items=[],
                    indicator_col=None,
                    indicator_values=[],
                    confidence=0.0,
                ),
                [ValidationWarning(
                    type="format",
                    message=f"Failed to parse file: {str(e)}",
                    severity="error",
                )],
                "Unable to parse the uploaded file. Please ensure it is a valid CSV.",
            )
        
        # Infer schema using LLM or fallback
        if self.enabled:
            schema = self._infer_schema_with_llm(df, filename, session)
            logger.info(f"LLM inferred schema: format={schema.format}, confidence={schema.confidence}")
        else:
            schema = self._infer_schema_fallback(df, filename)
            logger.info(f"Fallback inferred schema: format={schema.format}")
        
        # Validate data
        warnings = self._validate_data(content, schema, df)
        
        if warnings:
            logger.warning(f"Validation warnings: {[w.message for w in warnings]}")
        
        # Generate explanation
        explanation = self._generate_validation_explanation(warnings, df, schema)
        
        return schema, warnings, explanation
    
    # =========================================================================
    # LLM-Powered Schema Inference
    # =========================================================================
    
    def _build_data_summary(self, df: pd.DataFrame) -> str:
        """Build a comprehensive data summary for LLM analysis."""
        summary_lines = []
        
        # Basic info
        summary_lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        summary_lines.append(f"Column names: {list(df.columns)}")
        
        # Column type analysis
        summary_lines.append("\nColumn Analysis:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            unique = df[col].nunique()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                summary_lines.append(
                    f"  - {col}: numeric, {non_null} non-null, unique={unique}, "
                    f"range=[{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}"
                )
            else:
                sample_values = df[col].dropna().unique()[:5].tolist()
                summary_lines.append(
                    f"  - {col}: categorical, {non_null} non-null, unique={unique}, "
                    f"samples={sample_values}"
                )
        
        # Sparsity analysis for pairwise detection
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) >= 2:
            numeric_df = df[numeric_cols]
            non_null_per_row = numeric_df.notna().sum(axis=1)
            avg_non_null = non_null_per_row.mean()
            rows_with_exactly_2 = (non_null_per_row == 2).sum()
            summary_lines.append(f"\nSparsity: avg {avg_non_null:.1f} non-null values per row")
            summary_lines.append(f"Rows with exactly 2 non-null numeric values: {rows_with_exactly_2}")
            
            # Check for 0/1 pattern
            all_values = numeric_df.values.flatten()
            all_values = all_values[~pd.isna(all_values)]
            unique_vals = set(all_values)
            if unique_vals.issubset({0, 1, 0.0, 1.0}):
                summary_lines.append("Value pattern: Only 0 and 1 values (winner/loser encoding)")
        
        # Sample rows
        summary_lines.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
        
        return "\n".join(summary_lines)

    def _infer_schema_with_llm(
        self,
        df: pd.DataFrame,
        filename: str,
        session: Optional[SessionMemory] = None
    ) -> InferredSchema:
        """
        Use LLM to intelligently infer the complete data schema.
        
        This method combines macro-level (column names, structure) and
        micro-level (value distributions) analysis.
        """
        data_summary = self._build_data_summary(df)
        
        # User prompt: Task + Data (separated from system role/knowledge)
        user_prompt = f"""Analyze the following data and infer the semantic schema for spectral ranking analysis.

## Input

**Filename**: {filename}

**Data Summary**:
{data_summary}

## Task

Infer the complete schema by analyzing:
1. FORMAT: Check sparsity pattern, value types (0/1 vs continuous), structure
2. BIGBETTER: Analyze BOTH column name semantics AND value distributions
3. RANKING_ITEMS: Identify numeric columns representing entities to rank (exclude IDs, descriptions)
4. INDICATOR: Select AT MOST ONE categorical column suitable for stratification
5. CONFIDENCE: How certain are you about this inference (consider ambiguity)

## Required Output

Respond in EXACTLY this JSON format:
{{
    "format": "pointwise" | "pairwise" | "multiway",
    "format_reasoning": "Brief explanation of why this format was detected",
    "bigbetter": 1 | 0,
    "bigbetter_reasoning": "Brief explanation combining column name patterns AND value distribution analysis",
    "ranking_items": ["item1", "item2", ...],
    "ranking_items_reasoning": "Brief explanation of which columns are items to be ranked",
    "indicator_col": "column_name" | null,
    "indicator_reasoning": "Brief explanation of indicator selection (remember: AT MOST ONE)",
    "confidence": 0.0-1.0
}}"""

        try:
            # Use Structured System Instructions (OpenAI Recommended)
            # System message: Role + Knowledge
            # User message: Task + Data
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=4096,  # Large token limit for complete response
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response (handle potential markdown code blocks)
            if result_text.startswith("```"):
                # Remove markdown code block markers
                lines = result_text.split("\n")
                result_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            result = json.loads(result_text)
            
            # Log trace if session available
            if session:
                session.add_trace(
                    TraceType.SCHEMA_INFERENCE,
                    {
                        "status": "completed",
                        "method": "llm",
                        "format": result.get("format"),
                        "bigbetter": result.get("bigbetter"),
                        "confidence": result.get("confidence"),
                        "tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    agent=AgentType.DATA,
                )
            
            # Convert to InferredSchema
            format_map = {
                "pointwise": DataFormat.POINTWISE,
                "pairwise": DataFormat.PAIRWISE,
                "multiway": DataFormat.MULTIWAY,
            }
            
            indicator_values = []
            if result.get("indicator_col") and result["indicator_col"] in df.columns:
                indicator_values = df[result["indicator_col"]].dropna().unique().tolist()
            
            return InferredSchema(
                format=format_map.get(result.get("format", "pointwise"), DataFormat.POINTWISE),
                bigbetter=result.get("bigbetter", 1),
                ranking_items=result.get("ranking_items", []),
                indicator_col=result.get("indicator_col"),
                indicator_values=indicator_values,
                confidence=result.get("confidence", 0.7),
            )
            
        except Exception as e:
            logger.error(f"LLM schema inference failed: {e}")
            if session:
                session.add_trace(
                    TraceType.SCHEMA_INFERENCE,
                    {"status": "failed", "error": str(e), "method": "llm"},
                    agent=AgentType.DATA,
                    success=False,
                    error_message=str(e),
                )
            # Fallback to heuristic method
            return self._infer_schema_fallback(df, filename)
    
    # =========================================================================
    # Fallback Schema Inference (Heuristic-based)
    # =========================================================================
    
    def _infer_schema_fallback(self, df: pd.DataFrame, filename: str) -> InferredSchema:
        """
        Fallback schema inference using heuristics when LLM is unavailable.
        
        This preserves the original hardcoded logic for robustness.
        """
        logger.info("Using fallback schema inference (no LLM)")
        
        # Detect format (existing logic)
        data_format = self._detect_format_heuristic(df)
        
        # Infer bigbetter (existing logic)
        bigbetter, bb_confidence = self._infer_bigbetter_heuristic(df, data_format)
        
        # Extract ranking items (existing logic)
        ranking_items = self._extract_ranking_items_heuristic(df, data_format)
        
        # Detect indicator column (existing logic)
        indicator_col, indicator_values = self._detect_indicator_heuristic(df)
        
        # Calculate overall confidence (lower for fallback)
        confidence = bb_confidence * 0.7  # Penalize for using fallback
        
        return InferredSchema(
            format=data_format,
            bigbetter=bigbetter,
            ranking_items=ranking_items,
            indicator_col=indicator_col,
            indicator_values=indicator_values,
            confidence=round(confidence, 2),
        )

    def _detect_format_heuristic(self, df: pd.DataFrame) -> DataFormat:
        """Heuristic-based format detection (fallback)."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return DataFormat.POINTWISE
        
        numeric_df = df[numeric_cols]
        non_null_counts = numeric_df.notna().sum(axis=1)
        pairwise_rows = (non_null_counts == 2).sum()
        total_rows = len(df)
        
        if total_rows > 0 and pairwise_rows / total_rows > 0.8:
            values = numeric_df.values.flatten()
            values = values[~pd.isna(values)]
            unique_values = set(values)
            if unique_values.issubset({0, 1, 0.0, 1.0}):
                return DataFormat.PAIRWISE
        
        return DataFormat.POINTWISE

    def _infer_bigbetter_heuristic(
        self,
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> tuple[int, float]:
        """Heuristic-based bigbetter inference (fallback)."""
        if data_format == DataFormat.PAIRWISE:
            return 1, 0.95
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        lower_cols = [c.lower() for c in numeric_cols]
        
        lower_better_keywords = ["error", "loss", "time", "distance", "cost", "latency", "rank"]
        higher_better_keywords = ["score", "accuracy", "f1", "auc", "precision", "recall", "reward", "win"]
        
        lower_count = sum(any(kw in col for kw in lower_better_keywords) for col in lower_cols)
        higher_count = sum(any(kw in col for kw in higher_better_keywords) for col in lower_cols)
        
        if higher_count > lower_count:
            return 1, 0.8
        elif lower_count > higher_count:
            return 0, 0.8
        
        # Micro-level: Check if values are bounded [0, 1] -> likely accuracy/probability
        numeric_df = df[numeric_cols]
        all_vals = numeric_df.values.flatten()
        all_vals = all_vals[~pd.isna(all_vals)]
        
        if len(all_vals) > 0:
            min_val, max_val = all_vals.min(), all_vals.max()
            if 0 <= min_val and max_val <= 1:
                return 1, 0.7  # Bounded [0,1] usually means higher is better
        
        return 1, 0.6

    def _extract_ranking_items_heuristic(
        self,
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> list[str]:
        """Heuristic-based ranking items extraction (fallback)."""
        exclude_patterns = [
            "sample", "case", "id", "description", "task", "category", 
            "indicator", "index", "row", "unnamed"
        ]
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        
        ranking_items = []
        for col in numeric_cols:
            col_lower = col.lower()
            if not any(pattern in col_lower for pattern in exclude_patterns):
                ranking_items.append(col)
        
        if not ranking_items:
            ranking_items = numeric_cols
        
        return ranking_items

    def _detect_indicator_heuristic(
        self,
        df: pd.DataFrame
    ) -> tuple[Optional[str], list[str]]:
        """Heuristic-based indicator column detection (fallback)."""
        string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Priority keywords for indicator columns
        indicator_keywords = ["task", "category", "type", "group", "class", "domain"]
        
        # First pass: look for keyword matches
        for col in string_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in indicator_keywords):
                values = df[col].dropna().unique().tolist()
                if 2 <= len(values) <= 50:
                    return col, values  # Return FIRST match (at most ONE)
        
        # Second pass: any categorical with reasonable cardinality
        for col in string_cols:
            values = df[col].dropna().unique().tolist()
            if 2 <= len(values) <= 20:
                return col, values
        
        return None, []
    
    # =========================================================================
    # Data Validation
    # =========================================================================
    
    def _validate_data(
        self,
        content: bytes,
        schema: InferredSchema,
        df: pd.DataFrame
    ) -> list[ValidationWarning]:
        """
        Validate data against inferred schema.
        
        Checks (per writing.md):
        1. Sparsity: M < n * log(n) threshold
        2. Connectivity: Graph must be connected
        3. Critical: Required columns must exist
        """
        warnings = []
        ranking_items = schema.ranking_items
        
        # Critical: Check minimum items
        if len(ranking_items) < 2:
            warnings.append(ValidationWarning(
                type="format",
                message="At least 2 ranking items are required for comparison",
                severity="error",
            ))
            return warnings
        
        # Critical: Check columns exist
        missing_cols = [col for col in ranking_items if col not in df.columns]
        if missing_cols:
            warnings.append(ValidationWarning(
                type="format",
                message=f"Missing columns in data: {missing_cols}",
                severity="error",
            ))
            return warnings
        
        # Warning: Check minimum sample size
        if len(df) < 10:
            warnings.append(ValidationWarning(
                type="sparsity",
                message=f"Only {len(df)} samples. Results may be unreliable with small datasets.",
                severity="warning",
            ))
        
        # Warning: Check sparsity (CORRECTED FORMULA: M < n * log(n))
        sparsity_ratio, severity, sparsity_msg = self._check_sparsity_correct(df, ranking_items)
        if severity in ["warning", "error"]:
            warnings.append(ValidationWarning(
                type="sparsity",
                message=sparsity_msg,
                severity=severity,
            ))
        
        # Warning: Check connectivity
        is_connected, conn_msg = self._check_connectivity(df, ranking_items)
        if not is_connected:
            warnings.append(ValidationWarning(
                type="connectivity",
                message=conn_msg,
                severity="warning",  # Changed from error to warning per writing.md tiered feedback
            ))
        
        return warnings
    
    def _check_sparsity_correct(
        self,
        df: pd.DataFrame,
        ranking_items: list[str]
    ) -> tuple[float, str, str]:
        """
        Check data sparsity using the correct theoretical threshold.
        
        Per writing.md: Warn if M < n * log(n)
        where M = number of comparisons, n = number of items
        """
        n_items = len(ranking_items)
        
        if n_items < 2:
            return 0, "error", "At least 2 ranking items required"
        
        # Count total comparisons (M)
        valid_cols = [col for col in ranking_items if col in df.columns]
        if not valid_cols:
            return 0, "error", "No valid ranking columns found"
        
        numeric_df = df[valid_cols]
        
        total_comparisons = 0
        for _, row in numeric_df.iterrows():
            non_null = row.notna().sum()
            if non_null >= 2:
                # Each row with k non-null values contributes C(k,2) pairwise comparisons
                total_comparisons += non_null * (non_null - 1) / 2
        
        # Theoretical threshold: n * log(n) - CORRECTED per writing.md
        threshold = n_items * math.log(n_items) if n_items > 1 else 1
        
        # Sparsity ratio for reporting
        sparsity_ratio = total_comparisons / threshold if threshold > 0 else 0
        
        if total_comparisons < threshold:
            return sparsity_ratio, "warning", (
                f"Data may be too sparse for reliable inference. "
                f"Comparisons: {int(total_comparisons)}, Threshold (n*log(n)): {threshold:.0f}. "
                f"Consider collecting more data."
            )
        else:
            return sparsity_ratio, "info", (
                f"Data density is adequate. "
                f"Comparisons: {int(total_comparisons)}, Threshold: {threshold:.0f}."
            )

    def _check_connectivity(
        self,
        df: pd.DataFrame,
        ranking_items: list[str]
    ) -> tuple[bool, str]:
        """
        Check if the comparison graph is connected using networkx.
        
        Per writing.md: Issues Connectivity Warnings if the comparison graph is disjoint.
        """
        G = nx.Graph()
        G.add_nodes_from(ranking_items)
        
        # Build comparison graph - filter to valid columns first
        valid_cols = [col for col in ranking_items if col in df.columns]
        if not valid_cols:
            return False, "No valid ranking columns found in data"
        
        numeric_df = df[valid_cols]
        
        for _, row in numeric_df.iterrows():
            non_null_items = [col for col in valid_cols if pd.notna(row.get(col))]
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
            isolated = [list(c) for c in components if len(c) == 1]
            return False, (
                f"Comparison graph has {n_components} disconnected components. "
                f"Rankings will be relative within each component. "
                f"Isolated items: {isolated[:3]}{'...' if len(isolated) > 3 else ''}"
            )
    
    # =========================================================================
    # LLM-Powered Validation Explanation
    # =========================================================================
    
    def _generate_validation_explanation(
        self,
        warnings: list[ValidationWarning],
        df: pd.DataFrame,
        schema: InferredSchema
    ) -> str:
        """
        Generate natural language explanation of validation results.
        Uses Structured System Instructions pattern.
        """
        if not self.enabled:
            return ""
        
        if not warnings:
            return "Data validation passed. Your data is ready for analysis."
        
        warning_text = "\n".join([f"- {w.severity.upper()}: {w.message}" for w in warnings])
        
        # User prompt: Task + Data
        user_prompt = f"""Explain these validation results to a non-technical user.

## Data Context
- Rows: {len(df)}
- Ranking Items: {len(schema.ranking_items)}
- Format: {schema.format.value}

## Validation Results
{warning_text}

## Required Output
Provide a brief explanation covering:
1. What these warnings mean in plain language
2. Whether analysis can proceed and what limitations to expect"""

        try:
            # Use Structured System Instructions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=4096,  # gpt-5-nano requires >= 500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Validation explanation failed: {e}")
            return ""


# =============================================================================
# Legacy Function Exports (for backward compatibility)
# =============================================================================
# These functions are kept for any external code that might call them directly

def detect_format(df: pd.DataFrame) -> DataFormat:
    """
    Legacy function for format detection.
    For new code, use DataAgent._detect_format_heuristic() instead.
    """
    agent = DataAgent()
    return agent._detect_format_heuristic(df)


def infer_bigbetter(df: pd.DataFrame, data_format: DataFormat) -> tuple[int, float]:
    """
    Legacy function for bigbetter inference.
    For new code, use DataAgent._infer_bigbetter_heuristic() instead.
    """
    agent = DataAgent()
    return agent._infer_bigbetter_heuristic(df, data_format)


def extract_ranking_items(df: pd.DataFrame, data_format: DataFormat) -> list[str]:
    """
    Legacy function for ranking items extraction.
    For new code, use DataAgent._extract_ranking_items_heuristic() instead.
    """
    agent = DataAgent()
    return agent._extract_ranking_items_heuristic(df, data_format)


def detect_indicator_column(df: pd.DataFrame) -> tuple[Optional[str], list[str]]:
    """
    Legacy function for indicator column detection.
    For new code, use DataAgent._detect_indicator_heuristic() instead.
    """
    agent = DataAgent()
    return agent._detect_indicator_heuristic(df)


def infer_schema(content: bytes, filename: str) -> InferredSchema:
    """
    Legacy function for schema inference.
    For new code, use DataAgent().process() instead.
    """
    agent = DataAgent()
    schema, _, _ = agent.process(content, filename)
    return schema


def check_connectivity(df: pd.DataFrame, ranking_items: list[str]) -> tuple[bool, str]:
    """
    Legacy function for connectivity check.
    For new code, use DataAgent._check_connectivity() instead.
    """
    agent = DataAgent()
    return agent._check_connectivity(df, ranking_items)


def check_sparsity(
    df: pd.DataFrame,
    ranking_items: list[str]
) -> tuple[float, str, str]:
    """
    Legacy function for sparsity check.
    For new code, use DataAgent._check_sparsity_correct() instead.
    """
    agent = DataAgent()
    return agent._check_sparsity_correct(df, ranking_items)


def validate_data(content: bytes, schema: InferredSchema) -> list[ValidationWarning]:
    """
    Legacy function for data validation.
    For new code, use DataAgent().process() instead.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return [ValidationWarning(
            type="format",
            message=f"Failed to parse file: {str(e)}",
            severity="error",
        )]
    
    agent = DataAgent()
    return agent._validate_data(content, schema, df)
