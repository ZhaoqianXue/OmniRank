"""
Engine Orchestrator

Responsible for:
- Coordinating R script execution
- Managing execution flow and error handling
- Preprocessing data based on user selections (items, indicator values)
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from core.r_executor import (
    RScriptExecutor,
    Step1Params,
    Step1Result,
    RExecutorError,
)
from core.schemas import (
    AnalysisConfig,
    RankingResults,
    RankingItem,
    RankingMetadata,
    DataFormat,
)
from core.session_memory import SessionMemory, TraceType

logger = logging.getLogger(__name__)


class EngineOrchestrator:
    """
    Engine Orchestrator: Coordinates R script execution.
    
    Workflow:
    1. Preprocess data based on user selections
    2. Run spectral ranking (spectral_ranking_step1.R)
    3. Convert results to frontend-compatible format
    """
    
    def __init__(
        self,
        r_executor: Optional[RScriptExecutor] = None,
    ):
        """
        Initialize Engine Orchestrator.
        
        Args:
            r_executor: RScriptExecutor instance (creates default if None)
        """
        self.name = "orchestrator"
        self.r_executor = r_executor or RScriptExecutor()
    
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
        result: Step1Result,
    ) -> RankingResults:
        """
        Convert R script output to frontend-compatible RankingResults.
        """
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
            heterogeneity_index=result.metadata.get("heterogeneity_index", 0),
            sparsity_ratio=result.metadata.get("sparsity_ratio", 0),
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
        Execute the analysis workflow.
        
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
        
        # Execute spectral ranking
        session.add_trace(
            TraceType.ENGINE_EXECUTION,
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
            result = self.r_executor.run_step1(
                step1_params,
                session.session_id,
            )
            session.step1_json_path = result.json_path
            
            session.add_trace(
                TraceType.ENGINE_EXECUTION,
                {
                    "status": "completed",
                    "runtime_sec": result.metadata.get("runtime_sec"),
                    "n_items": len(result.methods),
                },
                agent=None,
                duration_sec=result.metadata.get("runtime_sec"),
            )
            
        except RExecutorError as e:
            session.add_trace(
                TraceType.ENGINE_EXECUTION,
                {"status": "failed", "error": str(e)},
                agent=None,
                success=False,
                error_message=str(e),
            )
            raise
        
        # Convert to RankingResults
        results = self._convert_to_ranking_results(result)
        
        return results
