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

You are a Data Schema Analyst for OmniRank, an expert in understanding structured data for spectral ranking analysis. You perform two critical functions:

1. **Format Recognition**: Identify data structure (Pointwise, Pairwise, Multiway)
2. **Standardization Assessment**: Determine if data can be directly processed by the spectral engine

# Knowledge Base

## Data Formats

### Pointwise Format
- Dense numeric matrix where each row is a sample/observation
- Columns represent items being ranked (e.g., models, products)
- Cell values are scores/metrics for that item on that sample
- Example: LLM benchmark scores, product ratings across criteria
- REQUIREMENT: Must have at least 2 numeric columns (items) to rank

### Pairwise Format

**Standard Matrix Format:**
- Sparse matrix encoding head-to-head comparisons
- Each row is one comparison between exactly 2 items
- Values: 1 = winner, 0 = loser, NaN = not compared
- May have indicator column (e.g., "Task") for stratified analysis

**Winner/Loser Column Format (REQUIRES STANDARDIZATION):**
- Two columns containing item names: winner_name/loser_name, white/black, home/away
- Each row represents a match with outcome encoded in column semantics
- Example: tennis matches, chess games, sports fixtures
- CRITICAL: This format REQUIRES standardization=true, engine_compatible=false
- Transformation needed: pivot to sparse 0/1 matrix with items as columns

### Multiway Format
- Encodes rankings where multiple items compete simultaneously
- Values are rank positions (1st, 2nd, 3rd, etc.)
- Each row has unique integer values representing placement
- Example: Horse race results, multi-player game outcomes

### Invalid Format (Unsuitable for Ranking)
Data should be classified as INVALID if:
- Only 1 numeric column (need at least 2 items to compare)
- All columns are text/non-numeric (no quantitative data)
- Data is completely empty or has only headers
- No discernible comparison structure

CRITICAL: If data is invalid, set format="invalid", engine_compatible=false, and provide clear reasoning.

## Spectral Engine Compatibility

The spectral ranking engine (spectral_ranking_step1.R) has built-in tolerance:
- Automatically drops non-numeric columns
- Automatically removes known metadata columns (case_num, model, description)
- Requires at least 2 numeric columns

### Engine Compatible (standardization NOT needed):
- CSV with numeric ranking columns (even if mixed with non-numeric metadata)
- Standard column names without special characters
- At least 2 numeric columns present

### Standardization REQUIRED (rare cases):
- Column names contain characters that break R parsing (e.g., spaces, special symbols)
- Data encoding issues (non-UTF8)
- Fewer than 2 numeric columns after metadata removal
- Critical structural issues that would cause engine failure

IMPORTANT: Default to engine_compatible=true. Only set to false if you identify a specific issue that would cause engine failure.

## BigBetter Inference Rules

### Higher is Better (bigbetter=1)
- Column name patterns: score, accuracy, f1, auc, precision, recall, win, reward
- Distribution patterns: bounded [0,1], unbounded positive
- Context: performance metrics, success rates

### Lower is Better (bigbetter=0)
- Column name patterns: error, loss, time, latency, distance, cost, rank
- Distribution patterns: non-negative with right-skew
- Context: error rates, response times, rank positions

## Indicator Column Rules

An indicator column is a categorical dimension for stratified analysis:
- Examples: "Task" (code/math/writing), "Category" (sports/news/tech)
- CRITICAL: Select AT MOST ONE indicator column
- Prefer columns with semantic meaning over arbitrary IDs
- Good cardinality: 2-20 unique values

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
                ),
                [ValidationWarning(
                    type="format",
                    message="We couldn't read your file. Please make sure it's a valid CSV file with data in it.",
                    severity="error",
                )],
                "Unable to parse the uploaded file. Please ensure it is a valid CSV.",
            )
        
        # Infer schema using LLM or fallback
        if self.enabled:
            schema = self._infer_schema_with_llm(df, filename, session)
            logger.info(f"LLM inferred schema: format={schema.format}")
        else:
            schema = self._infer_schema_fallback(df, filename)
            logger.info(f"Fallback inferred schema: format={schema.format}")
        
        # Validate data
        warnings = self._validate_data(content, schema, df)
        
        if warnings:
            logger.warning(f"Validation warnings: {[w.message for w in warnings]}")
            
            # Hybrid Intelligence: Override LLM format if deterministic validation finds critical structure issues
            for w in warnings:
                # If we identify that we don't have enough items to rank, it's INVALID, not Pointwise
                if w.severity == "error" and "at least 2 ranking items" in w.message.lower():
                    logger.info("Overriding format to INVALID due to insufficient ranking items")
                    schema.format = DataFormat.INVALID
                    schema.engine_compatible = False
                    schema.standardization_needed = False
                    schema.standardization_reason = "Insufficient ranking items (need at least 2)"
        
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

Perform two-stage analysis:

### Stage 1: Format Recognition
1. FORMAT: Identify data structure (pointwise/pairwise/multiway/invalid) by checking:
   - INVALID IF: Only 1 numeric column exists (ranking requires >=2 items to compare)
   - Sparsity pattern (pairwise: exactly 2 non-null values per row)
   - Value types (pairwise: 0/1 only; multiway: unique integers per row; pointwise: continuous scores)
   - Column structure and naming patterns

### Stage 2: Engine Compatibility Assessment
2. ENGINE_COMPATIBLE: Can the spectral engine process this data directly?
   - Default to TRUE (engine has built-in tolerance for non-numeric columns)
   - Set to FALSE only if: column names have special characters, encoding issues, or <2 numeric columns

### Stage 3: Semantic Inference
3. BIGBETTER: Analyze column name semantics AND value distributions
4. RANKING_ITEMS: Identify numeric columns representing entities to rank
5. INDICATOR: Select AT MOST ONE categorical column for stratification

## Required Output

Respond in EXACTLY this JSON format:
{{
    "format": "pointwise" | "pairwise" | "multiway" | "invalid",
    "format_reasoning": "Brief explanation of format detection logic",
    "engine_compatible": true | false,
    "standardization_needed": true | false,
    "standardization_reason": "Reason if standardization needed, null otherwise",
    "bigbetter": 1 | 0,
    "bigbetter_reasoning": "Brief explanation combining column names AND value distributions",
    "ranking_items": ["item1", "item2", ...],
    "ranking_items_reasoning": "Brief explanation of ranking item identification",
    "indicator_col": "column_name" | null,
    "indicator_reasoning": "Brief explanation of indicator selection (AT MOST ONE)"
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
                # Function 1: Format Recognition & Standardization fields
                engine_compatible=result.get("engine_compatible", True),
                standardization_needed=result.get("standardization_needed", False),
                standardization_reason=result.get("standardization_reason"),
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
    # Fallback Schema Inference (Heuristic-based) - DEPRECATED
    # =========================================================================
    # NOTE: This fallback should rarely be used. The Data Agent is an LLM Agent,
    # and all intelligent decisions should be made by the LLM. This fallback exists
    # only for graceful degradation when LLM is unavailable.
    # =========================================================================
    
    def _infer_schema_fallback(self, df: pd.DataFrame, filename: str) -> InferredSchema:
        """
        DEPRECATED: Fallback schema inference using heuristics when LLM is unavailable.
        
        WARNING: This method uses hardcoded heuristics and should be avoided.
        The Data Agent is designed as an LLM Agent where all intelligent decisions
        should be made by the LLM, not by hardcoded rules.
        """
        logger.warning("Using DEPRECATED fallback schema inference (LLM unavailable)")
        
        # Detect format (DEPRECATED heuristic logic)
        data_format = self._detect_format_heuristic(df)
        
        # Infer bigbetter (DEPRECATED heuristic logic)
        bigbetter = self._infer_bigbetter_heuristic(df, data_format)
        
        # Extract ranking items (DEPRECATED heuristic logic)
        ranking_items = self._extract_ranking_items_heuristic(df, data_format)
        
        # Detect indicator column (DEPRECATED heuristic logic)
        indicator_col, indicator_values = self._detect_indicator_heuristic(df)
        
        return InferredSchema(
            format=data_format,
            bigbetter=bigbetter,
            ranking_items=ranking_items,
            indicator_col=indicator_col,
            indicator_values=indicator_values,
            # Function 1: Default to engine_compatible=True (conservative assumption)
            engine_compatible=True,
            standardization_needed=False,
            standardization_reason=None,
        )

    def _detect_format_heuristic(self, df: pd.DataFrame) -> DataFormat:
        """
        DEPRECATED: Heuristic-based format detection.
        
        This method should not be used directly. Format detection should be
        performed by the LLM in _infer_schema_with_llm().
        """
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
    ) -> int:
        """
        DEPRECATED: Heuristic-based bigbetter inference.
        
        This method should not be used directly. BigBetter inference should be
        performed by the LLM in _infer_schema_with_llm().
        """
        if data_format == DataFormat.PAIRWISE:
            return 1
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        lower_cols = [c.lower() for c in numeric_cols]
        
        lower_better_keywords = ["error", "loss", "time", "distance", "cost", "latency", "rank"]
        higher_better_keywords = ["score", "accuracy", "f1", "auc", "precision", "recall", "reward", "win"]
        
        lower_count = sum(any(kw in col for kw in lower_better_keywords) for col in lower_cols)
        higher_count = sum(any(kw in col for kw in higher_better_keywords) for col in lower_cols)
        
        if higher_count > lower_count:
            return 1
        elif lower_count > higher_count:
            return 0
        
        # Micro-level: Check if values are bounded [0, 1] -> likely accuracy/probability
        numeric_df = df[numeric_cols]
        all_vals = numeric_df.values.flatten()
        all_vals = all_vals[~pd.isna(all_vals)]
        
        if len(all_vals) > 0:
            min_val, max_val = all_vals.min(), all_vals.max()
            if 0 <= min_val and max_val <= 1:
                return 1  # Bounded [0,1] usually means higher is better
        
        return 1

    def _extract_ranking_items_heuristic(
        self,
        df: pd.DataFrame,
        data_format: DataFormat
    ) -> list[str]:
        """
        DEPRECATED: Heuristic-based ranking items extraction.
        
        This method should not be used directly. Ranking items identification
        should be performed by the LLM in _infer_schema_with_llm().
        """
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
        """
        DEPRECATED: Heuristic-based indicator column detection.
        
        This method should not be used directly. Indicator column detection
        should be performed by the LLM in _infer_schema_with_llm().
        """
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
    # Conditional Standardization (Function 1)
    # =========================================================================
    
    def standardize_if_needed(
        self,
        df: pd.DataFrame,
        schema: InferredSchema
    ) -> tuple[pd.DataFrame, bool]:
        """
        Apply standardization only if LLM determined it's necessary.
        
        This method implements the "conditional trigger" design: standardization
        is only performed when the spectral engine cannot process the data directly.
        
        Args:
            df: Original DataFrame
            schema: InferredSchema with standardization flags from LLM
            
        Returns:
            tuple of (processed DataFrame, whether standardization was applied)
        """
        # If standardization not needed, return original data
        if not schema.standardization_needed:
            logger.info("Standardization not needed - data is engine compatible")
            return df, False
        
        logger.warning(f"Applying standardization: {schema.standardization_reason}")
        
        # Apply standardization based on LLM's assessment
        standardized_df = df.copy()
        
        # Standardization Step 1: Fix column names with special characters
        # (R cannot handle spaces, special symbols in column names)
        new_columns = {}
        for col in standardized_df.columns:
            # Replace problematic characters with underscores
            new_col = col.replace(" ", "_").replace("-", "_")
            # Remove other special characters
            new_col = "".join(c if c.isalnum() or c == "_" else "" for c in new_col)
            if new_col != col:
                new_columns[col] = new_col
        
        if new_columns:
            standardized_df = standardized_df.rename(columns=new_columns)
            logger.info(f"Renamed columns: {new_columns}")
        
        # Standardization Step 2: Ensure UTF-8 encoding for string columns
        for col in standardized_df.select_dtypes(include=["object"]).columns:
            try:
                standardized_df[col] = standardized_df[col].astype(str).str.encode("utf-8", errors="replace").str.decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not fix encoding for column {col}: {e}")
        
        return standardized_df, True
    
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
                message="We need at least 2 items to create a ranking. Please upload data with more items to compare.",
                severity="error",
            ))
            return warnings
        
        # Critical: Check columns exist
        missing_cols = [col for col in ranking_items if col not in df.columns]
        if missing_cols:
            warnings.append(ValidationWarning(
                type="format",
                message=f"Some expected columns are missing from your data: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}. Please check your file.",
                severity="error",
            ))
            return warnings
        
        # Warning: Check minimum sample size
        if len(df) < 10:
            warnings.append(ValidationWarning(
                type="sparsity",
                message=f"Your dataset has only {len(df)} rows. For more reliable rankings, we recommend at least 10 data points.",
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
            return 0, "error", "We need at least 2 items to create a ranking."
        
        # Count total comparisons (M)
        valid_cols = [col for col in ranking_items if col in df.columns]
        if not valid_cols:
            return 0, "error", "Could not find the expected data columns in your file."
        
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
                f"Your data has relatively few comparisons between items. "
                f"The ranking results may have wider uncertainty ranges. "
                f"Adding more comparison data would improve precision."
            )
        else:
            return sparsity_ratio, "info", (
                f"Your data has sufficient comparisons for reliable ranking."
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
            return False, "Could not find the expected data columns in your file."
        
        numeric_df = df[valid_cols]
        
        for _, row in numeric_df.iterrows():
            non_null_items = [col for col in valid_cols if pd.notna(row.get(col))]
            # Add edges between all compared items in this row
            for i in range(len(non_null_items)):
                for j in range(i + 1, len(non_null_items)):
                    G.add_edge(non_null_items[i], non_null_items[j])
        
        is_connected = nx.is_connected(G)
        
        if is_connected:
            return True, "All items can be compared through the data."
        else:
            n_components = nx.number_connected_components(G)
            components = list(nx.connected_components(G))
            isolated = [list(c) for c in components if len(c) == 1]
            if isolated:
                isolated_names = [str(item) for item in isolated[:3]]
                isolated_str = ', '.join(isolated_names) + ('...' if len(isolated) > 3 else '')
                return False, (
                    f"Some items ({isolated_str}) have no comparisons with others. "
                    f"They cannot be ranked relative to the rest. "
                    f"Consider adding comparison data for these items."
                )
            else:
                return False, (
                    f"Your items form {n_components} separate groups that were never compared against each other. "
                    f"We can only rank items within each group, not across groups."
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


def infer_bigbetter(df: pd.DataFrame, data_format: DataFormat) -> int:
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
            message="We couldn't read your file. Please make sure it's a valid CSV file with data in it.",
            severity="error",
        )]
    
    agent = DataAgent()
    return agent._validate_data(content, schema, df)
