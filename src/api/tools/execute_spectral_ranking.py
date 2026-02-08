"""Tool: execute_spectral_ranking."""

from __future__ import annotations

from pathlib import Path

from core.r_executor import RScriptExecutor
from core.schemas import EngineConfig, ExecutionResult


def execute_spectral_ranking(config: EngineConfig, session_work_dir: str) -> ExecutionResult:
    """Execute spectral_ranking.R using deterministic subprocess invocation."""
    executor = RScriptExecutor()
    return executor.run(config=config, session_work_dir=Path(session_work_dir))
