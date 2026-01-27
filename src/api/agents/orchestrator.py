"""
Engine Orchestrator Agent

Responsible for:
- Coordinating R script execution (step1 and step2)
- Making decisions about when to trigger step2 refinement
- Managing execution flow and error handling
- Preprocessing data based on user selections (items, indicator values)
"""

import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI

from core.r_executor import (
    RScriptExecutor,
    Step1Params,
    Step2Params,
    Step1Result,
    Step2Result,
    should_run_step2,
    RExecutorError,
)
from core.schemas import (
    AnalysisConfig,
    RankingResults,
    RankingItem,
    RankingMetadata,
    PairwiseComparison,
    DataFormat,
)
from core.session_memory import SessionMemory, TraceType

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorDecision:
    """Decision made by the orchestrator."""
    run_step2: bool
    reason: str
    confidence: float


class EngineOrchestrator:
    """
    Engine Orchestrator Agent: Coordinates R script execution.
    
    Workflow:
    1. Run Step 1 (vanilla spectral ranking)
    2. Analyze metadata (heterogeneity, spectral gap, CI width)
    3. Decide whether to run Step 2 (refined estimation)
    4. Convert results to frontend-compatible format
    """
    
    def __init__(
        self,
        r_executor: Optional[RScriptExecutor] = None,
        use_llm_decision: bool = False,
    ):
        """
        Initialize Engine Orchestrator.
        
        Args:
            r_executor: RScriptExecutor instance (creates default if None)
            use_llm_decision: Whether to use LLM for step2 decision (default: rule-based)
        """
        self.name = "orchestrator"
        self.r_executor = r_executor or RScriptExecutor()
        self.use_llm_decision = use_llm_decision
        
        # Initialize OpenAI client if using LLM decision
        if use_llm_decision:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
            else:
                logger.warning("OPENAI_API_KEY not set, falling back to rule-based decision")
                self.use_llm_decision = False
    
    def _make_llm_decision(self, step1_result: Step1Result) -> OrchestratorDecision:
        """
        Use LLM to decide whether to run Step 2.
        
        This provides more nuanced reasoning than simple thresholds.
        Decision logic follows architecture.md:
        - Gatekeeper: sparsity_ratio >= 1.0 (data sufficiency)
        - Trigger A: heterogeneity_index > 0.5
        - Trigger B: mean_ci_width_top_5 > 5.0
        """
        metadata = step1_result.metadata
        
        n_items = metadata.get('k_methods', 1)
        ci_width = metadata.get('mean_ci_width_top_5', 0)
        ci_ratio = ci_width / n_items if n_items > 0 else 0
        
        prompt = f"""You are an expert in spectral ranking inference. Decide whether Step 2 (refined estimation with optimal weights) should be executed.

## Step 1 Metadata
- Sparsity Ratio: {metadata.get('sparsity_ratio', 'N/A')}
- Heterogeneity Index: {metadata.get('heterogeneity_index', 'N/A')}
- Number of Items (n): {n_items}
- Mean CI Width (Top 5): {ci_width}
- CI Width Ratio (CI/n): {ci_ratio:.2%}

## Decision Rules (FOLLOW STRICTLY)

| Condition | Threshold | Role |
|-----------|-----------|------|
| Gatekeeper | sparsity_ratio >= 1.0 | MUST pass, otherwise STOP |
| Trigger A | heterogeneity_index > 0.5 | At least ONE trigger needed |
| Trigger B | CI_width / n > 0.2 (20%) | At least ONE trigger needed |

## Decision Logic
```
IF sparsity_ratio < 1.0:
    DECISION = NO (Data too sparse, Step 2 unstable)
ELSE IF heterogeneity_index > 0.5 OR (CI_width / n) > 0.2:
    DECISION = YES (Refinement beneficial)
ELSE:
    DECISION = NO (Step 1 sufficient)
```

## Your Task
Apply the rules above to the metadata. Respond in exactly this format:
DECISION: [YES/NO]
REASON: [One sentence explanation referencing the specific rule that applies]
CONFIDENCE: [0.0-1.0]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=500,  # gpt-5-mini requires >= 500
            )
            
            text = response.choices[0].message.content.strip()
            
            # Parse response
            lines = text.split("\n")
            decision = "YES" in lines[0].upper()
            reason = lines[1].replace("REASON:", "").strip() if len(lines) > 1 else "LLM decision"
            confidence = 0.8
            
            if len(lines) > 2:
                try:
                    confidence = float(lines[2].replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            
            return OrchestratorDecision(
                run_step2=decision,
                reason=reason,
                confidence=confidence,
            )
            
        except Exception as e:
            logger.error(f"LLM decision failed: {e}, falling back to rule-based")
            # Fallback to rule-based
            run, reason = should_run_step2(step1_result)
            return OrchestratorDecision(
                run_step2=run,
                reason=f"Rule-based decision (LLM failed): {reason}",
                confidence=0.7,
            )
    
    def _decide_step2(self, step1_result: Step1Result) -> OrchestratorDecision:
        """
        Decide whether to run Step 2.
        
        Uses either LLM or rule-based decision depending on configuration.
        
        Decision Logic:
        1. GATEKEEPER: sparsity_ratio >= 1.0 (data sufficiency)
        2. TRIGGERS: heterogeneity > 0.5 OR CI_width/n > 0.2 (20%)
        """
        if self.use_llm_decision:
            return self._make_llm_decision(step1_result)
        
        # Rule-based decision (returns tuple: (bool, str))
        run, reason = should_run_step2(step1_result)
        
        return OrchestratorDecision(
            run_step2=run,
            reason=reason,
            confidence=0.90,  # High confidence for rule-based deterministic decision
        )
    
    def _preprocess_data(
        self,
        session: SessionMemory,
        config: AnalysisConfig,
    ) -> str:
        """
        Preprocess data based on user selections (items and indicator values).
        
        Returns the path to the preprocessed CSV file (original or filtered).
        """
        # Check if any filtering is needed
        needs_filtering = (
            (config.selected_items is not None and len(config.selected_items) > 0) or
            (config.selected_indicator_values is not None and len(config.selected_indicator_values) > 0)
        )
        
        if not needs_filtering:
            # No filtering needed, use original file
            return session.file_path
        
        logger.info(f"Preprocessing data with filters: items={config.selected_items}, indicators={config.selected_indicator_values}")
        
        # Read the original data
        df = pd.read_csv(session.file_path)
        original_rows = len(df)
        
        # Get schema info
        schema = session.inferred_schema
        data_format = schema.format if schema else DataFormat.POINTWISE
        
        if data_format == DataFormat.POINTWISE:
            # Pointwise: columns are items, filter columns
            if config.selected_items:
                # Keep only selected item columns (preserve non-item columns like indicators)
                all_items = schema.ranking_items if schema else []
                non_item_cols = [col for col in df.columns if col not in all_items]
                selected_cols = non_item_cols + [item for item in config.selected_items if item in df.columns]
                df = df[selected_cols]
                logger.info(f"Filtered columns: {len(df.columns)} columns retained")
        
        elif data_format == DataFormat.PAIRWISE:
            # Pairwise: items are column names (except indicator columns)
            if config.selected_items:
                all_items = schema.ranking_items if schema else []
                indicator_col = schema.indicator_col if schema else None
                
                # For pairwise data, item names ARE the column names
                # Keep: indicator column + selected items
                selected_cols = []
                for col in df.columns:
                    # Keep indicator column
                    if col == indicator_col:
                        selected_cols.append(col)
                    # Keep if it's a selected item
                    elif col in config.selected_items:
                        selected_cols.append(col)
                    # Keep if it's not an item column (e.g., other metadata)
                    elif col not in all_items:
                        selected_cols.append(col)
                
                df = df[selected_cols]
                logger.info(f"Filtered pairwise columns: {len(df.columns)} columns retained")
        
        # Filter by indicator values
        if config.selected_indicator_values and schema and schema.indicator_col:
            indicator_col = schema.indicator_col
            if indicator_col in df.columns:
                df = df[df[indicator_col].isin(config.selected_indicator_values)]
                logger.info(f"Filtered by indicator values: {len(df)} rows retained from {original_rows}")
        
        # Save to temporary file
        temp_dir = Path(tempfile.gettempdir()) / "omnirank" / session.session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "filtered_data.csv"
        df.to_csv(temp_path, index=False)
        
        logger.info(f"Preprocessed data saved to: {temp_path}")
        return str(temp_path)
    
    def _convert_to_ranking_results(
        self,
        step1_result: Step1Result,
        step2_result: Optional[Step2Result],
        step2_triggered: bool,
    ) -> RankingResults:
        """
        Convert R script output to frontend-compatible RankingResults.
        """
        # Use step2 results if available, otherwise step1
        result = step2_result if step2_result else step1_result
        
        # Convert methods to RankingItem list
        items = []
        for method in result.methods:
            ci_two_sided = method.get("ci_two_sided", [1, len(result.methods)])
            items.append(RankingItem(
                name=method["name"],
                theta_hat=method["theta_hat"],
                rank=method["rank"],
                ci_lower=method.get("ci_left", ci_two_sided[0]),
                ci_upper=method.get("ci_uniform_left", ci_two_sided[1]),
                ci_two_sided=tuple(ci_two_sided),
            ))
        
        # Sort by rank
        items.sort(key=lambda x: x.rank)
        
        # Build metadata
        metadata = RankingMetadata(
            n_items=len(items),
            n_comparisons=result.metadata.get("n_samples", 0),
            heterogeneity_index=step1_result.metadata.get("heterogeneity_index", 0),
            sparsity_ratio=step1_result.metadata.get("sparsity_ratio", 0),
            step2_triggered=step2_triggered,
            runtime_sec=result.metadata.get("runtime_sec", 0),
        )
        
        return RankingResults(
            items=items,
            metadata=metadata,
            pairwise_matrix=[],  # TODO: Generate pairwise matrix for heatmap
        )
    
    def execute(
        self,
        session: SessionMemory,
        config: AnalysisConfig,
    ) -> RankingResults:
        """
        Execute the full analysis workflow.
        
        Args:
            session: Session containing uploaded file
            config: Analysis configuration
            
        Returns:
            RankingResults
            
        Raises:
            RExecutorError: If R script execution fails
        """
        logger.info(f"Orchestrator executing analysis for session {session.session_id}")
        
        if not session.file_path:
            raise ValueError("No file uploaded in session")
        
        # Preprocess data based on user selections
        processed_csv_path = self._preprocess_data(session, config)
        
        # Step 1: Vanilla spectral ranking
        session.add_trace(
            TraceType.STEP1_EXECUTION,
            {
                "status": "started",
                "config": config.model_dump(),
                "filtered": processed_csv_path != session.file_path,
            },
            agent=None,
        )
        
        step1_params = Step1Params(
            csv_path=processed_csv_path,  # Use preprocessed data
            bigbetter=config.bigbetter,
            bootstrap_iterations=config.bootstrap_iterations,
            random_seed=config.random_seed,
        )
        
        try:
            step1_result = self.r_executor.run_step1(
                step1_params,
                session.session_id,
            )
            session.step1_json_path = step1_result.json_path
            
            session.add_trace(
                TraceType.STEP1_EXECUTION,
                {
                    "status": "completed",
                    "runtime_sec": step1_result.metadata.get("runtime_sec"),
                    "n_items": len(step1_result.methods),
                },
                agent=None,
                duration_sec=step1_result.metadata.get("runtime_sec"),
            )
            
        except RExecutorError as e:
            session.add_trace(
                TraceType.STEP1_EXECUTION,
                {"status": "failed", "error": str(e)},
                agent=None,
                success=False,
                error_message=str(e),
            )
            raise
        
        # Step 2 Decision
        decision = self._decide_step2(step1_result)
        
        session.add_trace(
            TraceType.STEP2_DECISION,
            {
                "run_step2": decision.run_step2,
                "reason": decision.reason,
                "confidence": decision.confidence,
            },
            agent=None,
        )
        
        step2_result = None
        
        if decision.run_step2:
            # Step 2: Refined estimation
            session.add_trace(
                TraceType.STEP2_EXECUTION,
                {"status": "started"},
                agent=None,
            )
            
            step2_params = Step2Params(
                csv_path=processed_csv_path,  # Use same preprocessed data
                step1_json_path=step1_result.json_path,
            )
            
            try:
                step2_result = self.r_executor.run_step2(
                    step2_params,
                    session.session_id,
                )
                session.step2_json_path = step2_result.json_path
                
                session.add_trace(
                    TraceType.STEP2_EXECUTION,
                    {
                        "status": "completed",
                        "runtime_sec": step2_result.metadata.get("runtime_sec"),
                        "convergence": step2_result.metadata.get("convergence_diff_l2"),
                    },
                    agent=None,
                    duration_sec=step2_result.metadata.get("runtime_sec"),
                )
                
            except RExecutorError as e:
                session.add_trace(
                    TraceType.STEP2_EXECUTION,
                    {"status": "failed", "error": str(e)},
                    agent=None,
                    success=False,
                    error_message=str(e),
                )
                # Continue with step1 results
                logger.warning(f"Step 2 failed, using Step 1 results: {e}")
        
        # Convert to RankingResults
        results = self._convert_to_ranking_results(
            step1_result,
            step2_result,
            step2_triggered=decision.run_step2,
        )
        
        return results
