"""
Engine Orchestrator Agent

Responsible for:
- Coordinating R script execution (step1 and step2)
- Making decisions about when to trigger step2 refinement
- Managing execution flow and error handling
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

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
                self.model = os.getenv("OPENAI_MODEL", "gpt-5-nano")
            else:
                logger.warning("OPENAI_API_KEY not set, falling back to rule-based decision")
                self.use_llm_decision = False
    
    def _make_llm_decision(self, step1_result: Step1Result) -> OrchestratorDecision:
        """
        Use LLM to decide whether to run Step 2.
        
        This provides more nuanced reasoning than simple thresholds.
        """
        metadata = step1_result.metadata
        
        prompt = f"""You are an expert in spectral ranking inference. Analyze the following Step 1 results and decide whether Step 2 (refined estimation with optimal weights) should be executed.

Step 1 Metadata:
- Heterogeneity Index: {metadata.get('heterogeneity_index', 'N/A')}
- Spectral Gap: {metadata.get('spectral_gap', 'N/A')}
- Sparsity Ratio: {metadata.get('sparsity_ratio', 'N/A')}
- Mean CI Width (Top 5): {metadata.get('mean_ci_width_top_5', 'N/A')}
- Number of Items: {metadata.get('k_methods', 'N/A')}
- Number of Samples: {metadata.get('n_samples', 'N/A')}

Criteria for Step 2:
1. High heterogeneity (>0.3) suggests unbalanced comparison counts
2. Small spectral gap (<0.1) suggests slow mixing
3. Wide CI for top items (>3) suggests low inference precision

Respond in exactly this format:
DECISION: [YES/NO]
REASON: [One sentence explanation]
CONFIDENCE: [0.0-1.0]"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
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
            run = should_run_step2(step1_result)
            return OrchestratorDecision(
                run_step2=run,
                reason="Rule-based decision (LLM failed)",
                confidence=0.7,
            )
    
    def _decide_step2(self, step1_result: Step1Result) -> OrchestratorDecision:
        """
        Decide whether to run Step 2.
        
        Uses either LLM or rule-based decision depending on configuration.
        """
        if self.use_llm_decision:
            return self._make_llm_decision(step1_result)
        
        # Rule-based decision
        run = should_run_step2(step1_result)
        
        metadata = step1_result.metadata
        reasons = []
        
        if metadata.get("heterogeneity_index", 0) > 0.3:
            reasons.append("high heterogeneity")
        if metadata.get("spectral_gap", 1) < 0.1:
            reasons.append("small spectral gap")
        if metadata.get("mean_ci_width_top_5", 0) > 3:
            reasons.append("wide confidence intervals")
        
        reason = ", ".join(reasons) if reasons else "metrics within normal range"
        
        return OrchestratorDecision(
            run_step2=run,
            reason=f"Step 2 {'triggered' if run else 'skipped'}: {reason}",
            confidence=0.85,
        )
    
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
        
        # Step 1: Vanilla spectral ranking
        session.add_trace(
            TraceType.STEP1_EXECUTION,
            {"status": "started", "config": config.model_dump()},
            agent=None,
        )
        
        step1_params = Step1Params(
            csv_path=session.file_path,
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
                csv_path=session.file_path,
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
