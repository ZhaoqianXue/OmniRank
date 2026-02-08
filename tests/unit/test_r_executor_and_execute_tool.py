"""Unit tests for R executor and execute_spectral_ranking tool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.r_executor import RScriptExecutor
from core.schemas import EngineConfig
from tools.execute_spectral_ranking import execute_spectral_ranking


def _make_config(tmp_path: Path, script_path: Path) -> EngineConfig:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("A,B\n1,0\n0,1\n", encoding="utf-8")
    return EngineConfig(
        csv_path=str(csv_path),
        bigbetter=1,
        selected_items=["A", "B"],
        selected_indicator_values=None,
        B=2000,
        seed=42,
        r_script_path=str(script_path),
    )


def test_execute_spectral_ranking_success(monkeypatch, tmp_path: Path):
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")
    config = _make_config(tmp_path, script)

    def fake_run(command, **kwargs):  # noqa: ANN001
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "methods": [
                {"name": "A", "theta_hat": 0.25, "rank": 1, "ci_left": 1, "ci_uniform_left": 2},
                {"name": "B", "theta_hat": -0.25, "rank": 2, "ci_left": 2, "ci_uniform_left": 2},
            ]
        }
        (out_dir / "ranking_results.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("core.r_executor.subprocess.run", fake_run)
    result = execute_spectral_ranking(config=config, session_work_dir=str(tmp_path))

    assert result.success is True
    assert result.results is not None
    assert result.results.items == ["A", "B"]
    assert result.trace.exit_code == 0


def test_execute_spectral_ranking_non_zero_exit(monkeypatch, tmp_path: Path):
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")
    config = _make_config(tmp_path, script)

    def fake_run(command, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="failure")

    monkeypatch.setattr("core.r_executor.subprocess.run", fake_run)
    executor = RScriptExecutor()
    result = executor.run(config=config, session_work_dir=tmp_path)

    assert result.success is False
    assert result.error == "R execution failed"
    assert result.trace.exit_code == 1


def test_execute_spectral_ranking_missing_json(monkeypatch, tmp_path: Path):
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")
    config = _make_config(tmp_path, script)

    def fake_run(command, **kwargs):  # noqa: ANN001
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("core.r_executor.subprocess.run", fake_run)
    executor = RScriptExecutor()
    result = executor.run(config=config, session_work_dir=tmp_path)

    assert result.success is False
    assert result.error == "ranking_results.json not produced"


def test_execute_spectral_ranking_timeout(monkeypatch, tmp_path: Path):
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")
    config = _make_config(tmp_path, script)

    def fake_run(command, **kwargs):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=command, timeout=1, output="partial", stderr="timeout")

    monkeypatch.setattr("core.r_executor.subprocess.run", fake_run)
    executor = RScriptExecutor()
    result = executor.run(config=config, session_work_dir=tmp_path)

    assert result.success is False
    assert result.error == "R execution timed out"
    assert result.trace.exit_code == -1


def test_execute_spectral_ranking_ci_prefers_two_sided(monkeypatch, tmp_path: Path):
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")
    config = _make_config(tmp_path, script)

    def fake_run(command, **kwargs):  # noqa: ANN001
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "methods": [
                {
                    "name": "A",
                    "theta_hat": 0.25,
                    "rank": 1,
                    "ci_two_sided": [11, 19],
                    "ci_left": 3,
                    "ci_uniform_left": 5,
                },
                {
                    "name": "B",
                    "theta_hat": -0.25,
                    "rank": 2,
                    "ci_two_sided": [21, 29],
                    "ci_left": 4,
                    "ci_uniform_left": 6,
                },
            ]
        }
        (out_dir / "ranking_results.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("core.r_executor.subprocess.run", fake_run)
    result = execute_spectral_ranking(config=config, session_work_dir=str(tmp_path))

    assert result.success is True
    assert result.results is not None
    assert result.results.ci_lower == [11.0, 21.0]
    assert result.results.ci_upper == [19.0, 29.0]


def test_prepare_filtered_input_selects_matching_indicator_column(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "region,task,A,B\nus,code,1,0\neu,math,0,1\nus,math,1,0\n",
        encoding="utf-8",
    )
    script = tmp_path / "spectral_ranking.R"
    script.write_text("# placeholder", encoding="utf-8")

    config = EngineConfig(
        csv_path=str(csv_path),
        bigbetter=1,
        selected_items=["A", "B"],
        selected_indicator_values=["code"],
        B=2000,
        seed=42,
        r_script_path=str(script),
    )

    executor = RScriptExecutor()
    prepared = executor._prepare_filtered_input(config=config, work_dir=tmp_path)  # noqa: SLF001
    filtered = Path(prepared.csv_path).read_text(encoding="utf-8")

    assert "code" in filtered
    assert "math" not in filtered
