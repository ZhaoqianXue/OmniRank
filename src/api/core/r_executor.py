"""
R Script Executor

Manages subprocess execution of R spectral ranking scripts with robust error handling,
timeout management, and JSON output parsing.
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to R scripts (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "src" / "spectral_ranking"
STEP1_SCRIPT = SCRIPT_DIR / "spectral_ranking.R"


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
    """Parameters for Spectral Ranking execution."""
    csv_path: str
    bigbetter: int  # 0 or 1
    bootstrap_iterations: int = 2000
    random_seed: int = 42
    output_dir: Optional[str] = None


@dataclass
class Step1Result:
    """Result from spectral ranking execution."""
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
            raise RScriptNotFoundError(f"Spectral ranking script not found: {STEP1_SCRIPT}")
    
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
        Execute spectral ranking.
        
        Args:
            params: Spectral ranking parameters
            session_id: Session identifier for output organization
            timeout: Execution timeout in seconds
            
        Returns:
            Step1Result with ranking results and metadata
        """
        self._verify_scripts()
        
        # Create output directory
        output_dir = params.output_dir or str(
            self._create_output_dir(session_id, "ranking")
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
            script="spectral_ranking",
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
            logger.error(f"Spectral ranking failed: {stderr}")
            raise RExecutionError(
                f"Spectral ranking execution failed",
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
    
    def get_traces(self) -> list[ExecutionTrace]:
        """Get all execution traces for debugging."""
        return self.traces
    
    def clear_traces(self) -> None:
        """Clear execution traces."""
        self.traces.clear()
