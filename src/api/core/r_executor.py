"""
R Script Executor

Manages subprocess execution of R spectral ranking scripts with robust error handling,
timeout management, and JSON output parsing.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to R scripts (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "src" / "spectral_ranking"
STEP1_SCRIPT = SCRIPT_DIR / "spectral_ranking_step1.R"
STEP2_SCRIPT = SCRIPT_DIR / "spectral_ranking_step2.R"


class RExecutorError(Exception):
    """Base exception for R executor errors."""
    pass


class RScriptNotFoundError(RExecutorError):
    """R script file not found."""
    pass


class RExecutionError(RExecutorError):
    """R script execution failed."""

    def __init__(self, message: str, stderr: str = "", exit_code: int = -1):
        super().__init__(message)
        self.stderr = stderr
        self.exit_code = exit_code


class RTimeoutError(RExecutorError):
    """R script execution timed out."""
    pass


class ROutputParseError(RExecutorError):
    """Failed to parse R script output."""
    pass


@dataclass
class Step1Params:
    """Parameters for Step 1 (Vanilla Spectral Ranking)."""
    csv_path: str
    bigbetter: int  # 0 or 1
    bootstrap_iterations: int = 2000
    random_seed: int = 42
    output_dir: Optional[str] = None


@dataclass
class Step2Params:
    """Parameters for Step 2 (Refined Spectral Ranking)."""
    csv_path: str
    step1_json_path: str
    output_dir: Optional[str] = None
    # Optional overrides (defaults from step1 JSON)
    bigbetter: Optional[int] = None
    bootstrap_iterations: Optional[int] = None
    random_seed: Optional[int] = None


@dataclass
class Step1Result:
    """Result from Step 1 execution."""
    job_id: str
    params: dict
    methods: list[dict]
    metadata: dict
    json_path: str
    csv_path: str


@dataclass
class Step2Result:
    """Result from Step 2 execution."""
    job_id: str
    params: dict
    methods: list[dict]
    metadata: dict
    json_path: str
    csv_path: str


@dataclass
class ExecutionTrace:
    """Trace of R script execution for debugging."""
    script: str
    args: list[str]
    exit_code: int
    stdout: str
    stderr: str
    runtime_sec: float
    success: bool
    error_message: Optional[str] = None


class RScriptExecutor:
    """
    Executes R spectral ranking scripts via subprocess.
    
    Usage:
        executor = RScriptExecutor()
        result = executor.run_step1(params)
        if should_refine(result):
            result2 = executor.run_step2(params2)
    """

    def __init__(
        self,
        rscript_path: str = "Rscript",
        default_timeout: int = 300,
        work_dir: Optional[Path] = None,
    ):
        """
        Initialize the R script executor.
        
        Args:
            rscript_path: Path to Rscript executable
            default_timeout: Default timeout in seconds
            work_dir: Working directory for temporary files (defaults to system temp)
        """
        self.rscript_path = rscript_path
        self.default_timeout = default_timeout
        self.work_dir = work_dir or Path(tempfile.gettempdir()) / "omnirank"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Execution history for debugging
        self.traces: list[ExecutionTrace] = []
    
    def _verify_scripts(self) -> None:
        """Verify that R scripts exist."""
        if not STEP1_SCRIPT.exists():
            raise RScriptNotFoundError(f"Step 1 script not found: {STEP1_SCRIPT}")
        if not STEP2_SCRIPT.exists():
            raise RScriptNotFoundError(f"Step 2 script not found: {STEP2_SCRIPT}")
    
    def _create_output_dir(self, session_id: str, step: str) -> Path:
        """Create output directory for a session/step."""
        output_dir = self.work_dir / session_id / step
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _execute(
        self,
        script_path: Path,
        args: list[str],
        timeout: Optional[int] = None,
    ) -> tuple[str, str, int, float]:
        """
        Execute an R script with arguments.
        
        Returns:
            tuple of (stdout, stderr, exit_code, runtime_sec)
        """
        import time
        
        cmd = [self.rscript_path, str(script_path)] + args
        logger.info(f"Executing: {' '.join(cmd)}")
        
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
            runtime = time.time() - start_time
            return result.stdout, result.stderr, result.returncode, runtime
            
        except subprocess.TimeoutExpired as e:
            runtime = time.time() - start_time
            raise RTimeoutError(
                f"R script timed out after {timeout} seconds"
            ) from e
    
    def _parse_json_output(self, output_dir: Path, filename: str) -> dict:
        """Parse JSON output from R script."""
        json_path = output_dir / filename
        
        if not json_path.exists():
            raise ROutputParseError(f"Expected output file not found: {json_path}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ROutputParseError(f"Failed to parse JSON output: {e}") from e
    
    def run_step1(
        self,
        params: Step1Params,
        session_id: str,
        timeout: Optional[int] = None,
    ) -> Step1Result:
        """
        Execute Step 1: Vanilla Spectral Ranking.
        
        Args:
            params: Step 1 parameters
            session_id: Session identifier for output organization
            timeout: Execution timeout in seconds
            
        Returns:
            Step1Result with ranking results and metadata
        """
        self._verify_scripts()
        
        # Create output directory
        output_dir = params.output_dir or str(
            self._create_output_dir(session_id, "step1")
        )
        
        # Build arguments
        args = [
            "--csv", params.csv_path,
            "--bigbetter", str(params.bigbetter),
            "--B", str(params.bootstrap_iterations),
            "--seed", str(params.random_seed),
            "--out", output_dir,
        ]
        
        # Execute
        stdout, stderr, exit_code, runtime = self._execute(
            STEP1_SCRIPT, args, timeout
        )
        
        # Record trace
        trace = ExecutionTrace(
            script="step1",
            args=args,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            runtime_sec=runtime,
            success=(exit_code == 0),
            error_message=stderr if exit_code != 0 else None,
        )
        self.traces.append(trace)
        
        # Check for errors
        if exit_code != 0:
            logger.error(f"Step 1 failed: {stderr}")
            raise RExecutionError(
                f"Step 1 execution failed",
                stderr=stderr,
                exit_code=exit_code,
            )
        
        # Parse output
        output_path = Path(output_dir)
        data = self._parse_json_output(output_path, "ranking_results.json")
        
        return Step1Result(
            job_id=data.get("job_id", session_id),
            params=data.get("params", {}),
            methods=data.get("methods", []),
            metadata=data.get("metadata", {}),
            json_path=str(output_path / "ranking_results.json"),
            csv_path=str(output_path / "ranking_results.csv"),
        )
    
    def run_step2(
        self,
        params: Step2Params,
        session_id: str,
        timeout: Optional[int] = None,
    ) -> Step2Result:
        """
        Execute Step 2: Refined Spectral Ranking with optimal weights.
        
        Args:
            params: Step 2 parameters (includes path to Step 1 JSON)
            session_id: Session identifier
            timeout: Execution timeout in seconds
            
        Returns:
            Step2Result with refined ranking results
        """
        self._verify_scripts()
        
        # Create output directory
        output_dir = params.output_dir or str(
            self._create_output_dir(session_id, "step2")
        )
        
        # Build arguments
        args = [
            "--csv", params.csv_path,
            "--json_step1", params.step1_json_path,
            "--out", output_dir,
        ]
        
        # Add optional overrides
        if params.bigbetter is not None:
            args.extend(["--bigbetter", str(params.bigbetter)])
        if params.bootstrap_iterations is not None:
            args.extend(["--B", str(params.bootstrap_iterations)])
        if params.random_seed is not None:
            args.extend(["--seed", str(params.random_seed)])
        
        # Execute
        stdout, stderr, exit_code, runtime = self._execute(
            STEP2_SCRIPT, args, timeout
        )
        
        # Record trace
        trace = ExecutionTrace(
            script="step2",
            args=args,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            runtime_sec=runtime,
            success=(exit_code == 0),
            error_message=stderr if exit_code != 0 else None,
        )
        self.traces.append(trace)
        
        # Check for errors
        if exit_code != 0:
            logger.error(f"Step 2 failed: {stderr}")
            raise RExecutionError(
                f"Step 2 execution failed",
                stderr=stderr,
                exit_code=exit_code,
            )
        
        # Parse output
        output_path = Path(output_dir)
        data = self._parse_json_output(output_path, "ranking_results_step2.json")
        
        return Step2Result(
            job_id=data.get("job_id", session_id),
            params=data.get("params", {}),
            methods=data.get("methods", []),
            metadata=data.get("metadata", {}),
            json_path=str(output_path / "ranking_results_step2.json"),
            csv_path=str(output_path / "ranking_results_step2.csv"),
        )
    
    def get_traces(self) -> list[ExecutionTrace]:
        """Get all execution traces for debugging."""
        return self.traces
    
    def clear_traces(self) -> None:
        """Clear execution traces."""
        self.traces.clear()


def should_run_step2(step1_result: Step1Result) -> tuple[bool, str]:
    """
    Determine whether Step 2 refinement should be triggered based on Step 1 metadata.
    
    Decision Logic:
    1. GATEKEEPER: sparsity_ratio >= 1.0 (data sufficiency)
    2. TRIGGERS: heterogeneity > 0.5 OR CI_width/n > 0.2 (20%)
    
    Args:
        step1_result: Results from Step 1 execution
        
    Returns:
        Tuple of (should_run: bool, reason: str)
    """
    metadata = step1_result.metadata
    
    sparsity_ratio = metadata.get("sparsity_ratio", 0.0)
    heterogeneity = metadata.get("heterogeneity_index", 0.0)
    ci_width_top5 = metadata.get("mean_ci_width_top_5", 0.0)
    n_items = metadata.get("k_methods", 1)  # Number of items being ranked
    
    # 1. GATEKEEPER: Check data sufficiency
    if sparsity_ratio < 1.0:
        reason = f"Data too sparse (sparsity_ratio={sparsity_ratio:.2f} < 1.0). Step 2 would be unstable."
        logger.info(f"Step 2 blocked by gatekeeper: {reason}")
        return False, reason
    
    # 2. CHECK TRIGGERS
    triggers_activated = []
    
    # Trigger A: Heterogeneity (CV > 0.5)
    if heterogeneity > 0.5:
        triggers_activated.append(f"high heterogeneity ({heterogeneity:.3f} > 0.5)")
    
    # Trigger B: Uncertainty (CI width > 20% of total items)
    ci_width_ratio = ci_width_top5 / n_items if n_items > 0 else 0.0
    if ci_width_ratio > 0.2:
        triggers_activated.append(f"wide CI for top-5 ({ci_width_top5:.1f}/{n_items} = {ci_width_ratio:.1%} > 20%)")
    
    # 3. FINAL DECISION
    if triggers_activated:
        reason = f"Data sufficient (sparsity_ratio={sparsity_ratio:.2f}). Triggers: {', '.join(triggers_activated)}"
        logger.info(f"Triggering Step 2: {reason}")
        return True, reason
    
    reason = f"Step 1 sufficient. No triggers activated (heterogeneity={heterogeneity:.3f}, CI_ratio={ci_width_ratio:.1%})"
    logger.info(f"Step 2 skipped: {reason}")
    return False, reason
