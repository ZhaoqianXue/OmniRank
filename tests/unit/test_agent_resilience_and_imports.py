"""Resilience and import smoke tests for OmniRank backend modules."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

from agents.omnirank_agent import OmniRankAgent
from core.llm_client import resolve_reasoning_effort
from core.session_memory import SessionMemory
from core.schemas import SessionStatus


def test_agent_infer_runs_without_optional_stage_note_hook(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("A,B\n1,0\n0,1\n", encoding="utf-8")

    session = SessionMemory(
        session_id="test-session",
        status=SessionStatus.UPLOADED,
        original_file_path=str(csv_path),
        current_file_path=str(csv_path),
    )

    response = OmniRankAgent().infer(session=session, user_hints=None)

    assert response.success is True
    assert response.requires_confirmation is True


def test_websocket_module_import_smoke():
    module = importlib.import_module("api.websocket")
    assert module is not None


def test_resolve_reasoning_effort_matches_model_family(monkeypatch):
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    assert resolve_reasoning_effort("gpt-5-mini") == "minimal"
    assert resolve_reasoning_effort("gpt-5.4-nano") == "none"
    assert resolve_reasoning_effort("gpt-5-nano") == "none"


def test_resolve_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    assert resolve_reasoning_effort("gpt-5-mini") == "high"


def test_import_core_schemas_has_no_field_shadow_warning():
    repo_root = Path(__file__).resolve().parents[2]
    api_path = repo_root / "src" / "api"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(api_path)

    command = [
        sys.executable,
        "-W",
        "always",
        "-c",
        "import core.schemas",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)  # noqa: S603

    assert completed.returncode == 0
    assert 'Field name "schema"' not in completed.stderr
