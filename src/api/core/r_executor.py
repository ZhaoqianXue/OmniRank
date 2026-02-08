"""R spectral engine executor for OmniRank."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .schemas import EngineConfig, ExecutionResult, ExecutionTrace, RankingResults


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCRIPT_PATH = PROJECT_ROOT / "src" / "spectral_ranking" / "spectral_ranking.R"


class RExecutorError(RuntimeError):
    """Raised when R execution fails."""


@dataclass
class PreparedInput:
    """Prepared CSV input for the R engine."""

    csv_path: str
    cleanup_paths: list[Path]


class RScriptExecutor:
    """Execute spectral_ranking.R in subprocess."""

    def __init__(self, rscript_binary: str = "Rscript", timeout_seconds: int = 300):
        self.rscript_binary = rscript_binary
        self.timeout_seconds = timeout_seconds

    def _prepare_filtered_input(self, config: EngineConfig, work_dir: Path) -> PreparedInput:
        """Apply selected item / indicator filters and materialize temp csv if needed."""
        cleanup_paths: list[Path] = []
        csv_path = Path(config.csv_path)

        if not config.selected_items and not config.selected_indicator_values:
            return PreparedInput(csv_path=str(csv_path), cleanup_paths=cleanup_paths)

        df = pd.read_csv(csv_path)

        if config.selected_items:
            keep_cols = [col for col in df.columns if col in set(config.selected_items)]
            non_numeric_cols = [
                col
                for col in df.columns
                if col not in keep_cols and not pd.api.types.is_numeric_dtype(df[col])
            ]
            keep_cols = non_numeric_cols + keep_cols
            keep_cols = [col for col in keep_cols if col in df.columns]
            df = df[keep_cols]

        if config.selected_indicator_values:
            # Find candidate indicator column from non-numeric columns.
            indicator_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            if indicator_cols:
                indicator_col = indicator_cols[0]
                df = df[df[indicator_col].isin(config.selected_indicator_values)]

        filtered_path = work_dir / "engine_input_filtered.csv"
        df.to_csv(filtered_path, index=False)
        cleanup_paths.append(filtered_path)
        return PreparedInput(csv_path=str(filtered_path), cleanup_paths=cleanup_paths)

    def run(self, config: EngineConfig, session_work_dir: Path) -> ExecutionResult:
        """Execute R script and parse ranking JSON output."""
        script_path = Path(config.r_script_path)
        if not script_path.is_absolute():
            script_path = PROJECT_ROOT / script_path
        if not script_path.exists():
            raise RExecutorError(f"R script not found: {script_path}")

        output_dir = session_work_dir / "engine_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        prepared = self._prepare_filtered_input(config, session_work_dir)

        command = [
            self.rscript_binary,
            str(script_path),
            "--csv",
            prepared.csv_path,
            "--bigbetter",
            str(config.bigbetter),
            "--B",
            str(config.B),
            "--seed",
            str(config.seed),
            "--out",
            str(output_dir),
        ]

        started = time.time()
        try:
            proc = subprocess.run(  # noqa: S603
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            trace = ExecutionTrace(
                command=" ".join(command),
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=-1,
                duration_seconds=duration,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            return ExecutionResult(success=False, error="R execution timed out", trace=trace)

        duration = time.time() - started
        trace = ExecutionTrace(
            command=" ".join(command),
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_seconds=duration,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        result_path = output_dir / "ranking_results.json"
        if proc.returncode != 0:
            return ExecutionResult(success=False, error="R execution failed", trace=trace)

        if not result_path.exists():
            return ExecutionResult(success=False, error="ranking_results.json not produced", trace=trace)

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ExecutionResult(success=False, error=f"Failed to parse output JSON: {exc}", trace=trace)

        methods = payload.get("methods", [])
        if not methods:
            return ExecutionResult(success=False, error="Engine output has no methods", trace=trace)

        ranking = RankingResults(
            items=[str(method.get("name", "")) for method in methods],
            theta_hat=[float(method.get("theta_hat", 0.0)) for method in methods],
            ranks=[int(method.get("rank", i + 1)) for i, method in enumerate(methods)],
            ci_lower=[float(method.get("ci_left", method.get("ci_two_sided", [1, 1])[0])) for method in methods],
            ci_upper=[float(method.get("ci_uniform_left", method.get("ci_two_sided", [1, 1])[1])) for method in methods],
            indicator_value=None,
        )

        return ExecutionResult(success=True, results=ranking, trace=trace)
