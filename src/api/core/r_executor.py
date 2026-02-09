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

from .schemas import EngineConfig, ExecutionResult, ExecutionTrace, RankingMetadata, RankingResults


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
            selected_values = {str(value) for value in config.selected_indicator_values}

            # Choose the most likely indicator column by value overlap.
            indicator_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
            best_col: Optional[str] = None
            best_overlap = 0
            for col in indicator_cols:
                column_values = {str(value) for value in df[col].dropna().astype(str).tolist()}
                overlap = len(column_values.intersection(selected_values))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_col = col

            if best_col and best_overlap > 0:
                normalized = df[best_col].where(df[best_col].notna(), None).map(
                    lambda value: str(value) if value is not None else None
                )
                df = df[normalized.isin(selected_values)]

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

        metadata_payload = payload.get("metadata", {})
        metadata: RankingMetadata | None = None
        if isinstance(metadata_payload, dict):
            metadata = RankingMetadata(
                n_samples=int(metadata_payload.get("n_samples", 0) or 0),
                k_methods=int(metadata_payload.get("k_methods", 0) or 0),
                runtime_sec=float(metadata_payload.get("runtime_sec", 0.0) or 0.0),
                heterogeneity_index=float(metadata_payload.get("heterogeneity_index", 0.0) or 0.0),
                spectral_gap=float(metadata_payload.get("spectral_gap", 0.0) or 0.0),
                sparsity_ratio=float(metadata_payload.get("sparsity_ratio", 0.0) or 0.0),
                mean_ci_width_top_5=float(metadata_payload.get("mean_ci_width_top_5", 0.0) or 0.0),
                n_comparisons=(
                    int(metadata_payload["n_comparisons"])
                    if "n_comparisons" in metadata_payload and metadata_payload["n_comparisons"] is not None
                    else None
                ),
            )

        ci_bounds: list[tuple[float, float]] = []
        for method in methods:
            ci_two_sided = method.get("ci_two_sided")
            if isinstance(ci_two_sided, (list, tuple)) and len(ci_two_sided) >= 2:
                ci_bounds.append((float(ci_two_sided[0]), float(ci_two_sided[1])))
                continue

            ci_left = method.get("ci_left", method.get("ci_two_left", 1.0))
            ci_right = method.get("ci_uniform_left", method.get("ci_two_right", ci_left))
            ci_bounds.append((float(ci_left), float(ci_right)))

        ranking = RankingResults(
            items=[str(method.get("name", "")) for method in methods],
            theta_hat=[float(method.get("theta_hat", 0.0)) for method in methods],
            ranks=[int(method.get("rank", i + 1)) for i, method in enumerate(methods)],
            ci_lower=[bounds[0] for bounds in ci_bounds],
            ci_upper=[bounds[1] for bounds in ci_bounds],
            indicator_value=None,
            metadata=metadata,
        )

        return ExecutionResult(success=True, results=ranking, trace=trace)
