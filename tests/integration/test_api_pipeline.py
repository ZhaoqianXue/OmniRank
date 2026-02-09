"""Integration tests for OmniRank staged HTTP API."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from api import routes as api_routes
from core.schemas import ExecutionResult, ExecutionTrace, RankingResults, RunResponse, SessionStatus
from main import app


def _mock_execution_result() -> ExecutionResult:
    return ExecutionResult(
        success=True,
        results=RankingResults(
            items=["model_1", "model_2", "model_3"],
            theta_hat=[0.5, 0.1, -0.2],
            ranks=[1, 2, 3],
            ci_lower=[1.0, 1.0, 2.0],
            ci_upper=[2.0, 3.0, 3.0],
            indicator_value=None,
        ),
        error=None,
        trace=ExecutionTrace(
            command="Rscript mock",
            stdout="ok",
            stderr="",
            exit_code=0,
            duration_seconds=0.12,
            timestamp="2026-02-08T00:00:00Z",
        ),
    )


def test_full_pipeline_upload_infer_confirm_run_question(monkeypatch):
    def fake_run(self, config, session_work_dir):  # noqa: ANN001
        return _mock_execution_result()

    monkeypatch.setattr("core.r_executor.RScriptExecutor.run", fake_run)
    client = TestClient(app)

    upload = client.post("/api/upload/example/pointwise")
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    preview = client.get(f"/api/preview/{session_id}")
    assert preview.status_code == 200
    assert len(preview.json()["columns"]) >= 2

    qa_early = client.post(
        f"/api/sessions/{session_id}/question",
        json={"question": "What can I ask before ranking is complete?", "quotes": []},
    )
    assert qa_early.status_code == 200
    assert "Conclusion:" in qa_early.json()["answer"]["answer"]

    infer = client.post(f"/api/sessions/{session_id}/infer", json={"user_hints": None})
    assert infer.status_code == 200
    infer_body = infer.json()
    assert infer_body["success"] is True
    assert infer_body["requires_confirmation"] is True

    confirmed_schema = infer_body["schema_result"]["schema"]
    confirm = client.post(
        f"/api/sessions/{session_id}/confirm",
        json={
            "confirmed": True,
            "confirmed_schema": confirmed_schema,
            "user_modifications": [],
            "B": 2000,
            "seed": 42,
        },
    )
    assert confirm.status_code == 200
    assert confirm.json()["session_status"] == "confirmed"

    run = client.post(
        f"/api/sessions/{session_id}/run",
        json={"selected_items": None, "selected_indicator_values": None},
    )
    assert run.status_code == 200
    run_body = run.json()
    assert run_body["success"] is True
    assert len(run_body["visualizations"]["plots"]) == 2
    assert run_body["report"]["markdown"]

    snapshot = client.get(f"/api/sessions/{session_id}")
    assert snapshot.status_code == 200
    snapshot_body = snapshot.json()
    assert snapshot_body["session"]["status"] == "completed"
    assert len(snapshot_body["artifacts"]) >= 2

    artifact_id = snapshot_body["artifacts"][0]["artifact_id"]
    artifact = client.get(f"/api/sessions/{session_id}/artifacts/{artifact_id}")
    assert artifact.status_code == 200
    assert artifact.content

    quote_block_id = run_body["report"]["citation_blocks"][0]["block_id"]
    qa = client.post(
        f"/api/sessions/{session_id}/question",
        json={
            "question": "Please explain this quoted section.",
            "quotes": [
                {
                    "quoted_text": "Top-ranked item",
                    "block_id": quote_block_id,
                    "kind": "summary",
                    "source": "report",
                }
            ],
        },
    )
    assert qa.status_code == 200
    answer = qa.json()["answer"]
    assert "answer" in answer
    assert quote_block_id in answer["used_citation_block_ids"]


def test_reject_confirmation_then_reinfer_with_hints(monkeypatch):
    def fake_run(self, config, session_work_dir):  # noqa: ANN001
        return _mock_execution_result()

    monkeypatch.setattr("core.r_executor.RScriptExecutor.run", fake_run)
    client = TestClient(app)

    upload = client.post("/api/upload/example/pairwise")
    session_id = upload.json()["session_id"]

    infer = client.post(f"/api/sessions/{session_id}/infer", json={"user_hints": None})
    infer_body = infer.json()
    schema = infer_body["schema_result"]["schema"]

    reject = client.post(
        f"/api/sessions/{session_id}/confirm",
        json={
            "confirmed": False,
            "confirmed_schema": schema,
            "user_modifications": ["User wants to provide hints"],
            "B": 2000,
            "seed": 42,
        },
    )
    assert reject.status_code == 200
    assert reject.json()["session_status"] == "awaiting_confirmation"

    reinfer = client.post(
        f"/api/sessions/{session_id}/infer",
        json={"user_hints": "Treat score columns as pairwise wins."},
    )
    assert reinfer.status_code == 200
    assert reinfer.json()["success"] is True


def test_global_question_without_session():
    client = TestClient(app)
    response = client.post(
        "/api/question",
        json={"question": "What is spectral ranking in OmniRank?", "quotes": []},
    )
    assert response.status_code == 200
    assert "Conclusion:" in response.json()["answer"]["answer"]


def test_pairwise_long_upload_infer_requires_confirmation():
    client = TestClient(app)
    csv_content = "task,item_a,item_b,winner\ncode,A,B,A\nmath,A,C,C\nqa,B,C,C\n"

    upload = client.post(
        "/api/upload",
        files={"file": ("pairwise_long.csv", csv_content, "text/csv")},
    )
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    infer = client.post(f"/api/sessions/{session_id}/infer", json={"user_hints": None})
    assert infer.status_code == 200
    body = infer.json()
    assert body["success"] is True
    assert body["requires_confirmation"] is True


def test_async_run_start_and_status(monkeypatch):
    def fake_async_run(session, selected_items, selected_indicator_values, progress_callback=None):  # noqa: ANN001
        session.status = SessionStatus.RUNNING
        if progress_callback:
            progress_callback(0.2, "Executing spectral ranking engine...")
            progress_callback(0.8, "Generating report...")
            progress_callback(1.0, "Ranking completed.")
        session.status = SessionStatus.COMPLETED
        return RunResponse(success=True)

    monkeypatch.setattr(api_routes.agent, "run", fake_async_run)
    client = TestClient(app)

    upload = client.post("/api/upload/example/pointwise")
    assert upload.status_code == 200
    session_id = upload.json()["session_id"]

    infer = client.post(f"/api/sessions/{session_id}/infer", json={"user_hints": None})
    assert infer.status_code == 200
    schema = infer.json()["schema_result"]["schema"]

    confirm = client.post(
        f"/api/sessions/{session_id}/confirm",
        json={
            "confirmed": True,
            "confirmed_schema": schema,
            "user_modifications": [],
            "B": 2000,
            "seed": 42,
        },
    )
    assert confirm.status_code == 200

    start = client.post(
        f"/api/sessions/{session_id}/run/start",
        json={"selected_items": None, "selected_indicator_values": None},
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]
    assert start.json()["status"] == "queued"

    latest_status = None
    for _ in range(20):
        status_resp = client.get(f"/api/sessions/{session_id}/run/{job_id}")
        assert status_resp.status_code == 200
        latest_status = status_resp.json()
        if latest_status["status"] == "completed":
            break
        time.sleep(0.05)

    assert latest_status is not None
    assert latest_status["status"] == "completed"
    assert latest_status["progress"] == 1.0
    assert latest_status["result"]["success"] is True
