"""OmniRank HTTP API routes (single-agent SOP pipeline)."""

from __future__ import annotations

import csv
import io
import logging
import mimetypes
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from agents.omnirank_agent import OmniRankAgent
from core.schemas import (
    ConfirmResponse,
    ConfirmRequest,
    DataPreview,
    DailyUsageResponse,
    InferRequest,
    InferResponse,
    QuestionResponse,
    QuestionRequest,
    RunJobStatus,
    RunJobStatusResponse,
    RunRequest,
    RunResponse,
    RunStartResponse,
    SessionStatus,
    SessionSnapshotResponse,
    UploadResponse,
)
from core.session_memory import get_session_store
from core.usage_tracker import get_usage_tracker, usage_user_scope
from tools.answer_question import answer_question as answer_question_tool

logger = logging.getLogger(__name__)
router = APIRouter(tags=["omnirank"])


EXAMPLE_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
EXAMPLE_DATASETS: dict[str, str] = {
    "pairwise": "example_data_pairwise.csv",
    "pointwise": "example_data_pointwise.csv",
    "multiway": "example_data_multiway.csv",
}

agent = OmniRankAgent()
run_executor = ThreadPoolExecutor(max_workers=2)
_run_jobs_lock = Lock()
_run_jobs: dict[str, dict[str, Any]] = {}
_ACTIVE_RUN_JOB_STATUSES = {RunJobStatus.QUEUED, RunJobStatus.RUNNING}
USER_SUB_HEADER = "x-omnirank-user-sub"


def _set_run_job_state(job_id: str, **updates: Any) -> None:
    with _run_jobs_lock:
        if job_id not in _run_jobs:
            return
        _run_jobs[job_id].update(updates)


def _get_run_job_state(job_id: str) -> dict[str, Any] | None:
    with _run_jobs_lock:
        current = _run_jobs.get(job_id)
        if current is None:
            return None
        return dict(current)


def _has_active_run_job(session_id: str) -> bool:
    with _run_jobs_lock:
        return any(
            job["session_id"] == session_id and job["status"] in _ACTIVE_RUN_JOB_STATUSES
            for job in _run_jobs.values()
        )


def _extract_user_sub(http_request: Request) -> str | None:
    raw = http_request.headers.get(USER_SUB_HEADER, "").strip()
    if not raw:
        return None
    return raw


def _resolve_session_user_sub(session: Any, http_request: Request) -> str | None:
    header_user_sub = _extract_user_sub(http_request)
    session_user_sub = getattr(session, "user_sub", None)

    if session_user_sub and header_user_sub and session_user_sub != header_user_sub:
        raise HTTPException(status_code=403, detail="Session is bound to a different user.")

    if session_user_sub:
        return session_user_sub

    if header_user_sub:
        session.user_sub = header_user_sub
        return header_user_sub

    return None


def _run_session_job(session_id: str, job_id: str, request: RunRequest) -> None:
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        _set_run_job_state(
            job_id,
            status=RunJobStatus.FAILED,
            progress=1.0,
            message="Run failed: session not found.",
            error="Session not found",
        )
        return

    def update_progress(progress: float, message: str) -> None:
        _set_run_job_state(
            job_id,
            status=RunJobStatus.RUNNING,
            progress=min(1.0, max(0.0, float(progress))),
            message=message,
            error=None,
        )
        store.update_session(session)

    try:
        update_progress(0.01, "Preparing run task...")
        with usage_user_scope(session.user_sub):
            response = agent.run(
                session=session,
                selected_items=request.selected_items,
                selected_indicator_values=request.selected_indicator_values,
                progress_callback=update_progress,
            )
        store.update_session(session)

        if response.success:
            _set_run_job_state(
                job_id,
                status=RunJobStatus.COMPLETED,
                progress=1.0,
                message="Ranking completed.",
                result=response,
                error=None,
            )
            return

        _set_run_job_state(
            job_id,
            status=RunJobStatus.FAILED,
            progress=1.0,
            message=response.error or "Run failed.",
            error=response.error or "Run failed",
            result=response,
        )
    except Exception as exc:  # noqa: BLE001
        session.status = SessionStatus.ERROR
        session.error = str(exc)
        store.update_session(session)
        _set_run_job_state(
            job_id,
            status=RunJobStatus.FAILED,
            progress=1.0,
            message=f"Run failed: {exc}",
            error=str(exc),
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(http_request: Request, file: UploadFile = File(...)):
    """Upload CSV file and create session."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    normalized_name = file.filename.replace("\\", "/")
    if Path(normalized_name).name != normalized_name or ".." in normalized_name.split("/"):
        raise HTTPException(status_code=400, detail="Unsafe filename path is not allowed")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    content = await file.read()
    store = get_session_store()
    session = store.create_session()
    session.user_sub = _extract_user_sub(http_request)

    saved_path = store.save_file(session.session_id, file.filename, content)
    session.filename = Path(saved_path).name
    session.original_file_path = saved_path
    session.current_file_path = saved_path
    session.status = SessionStatus.UPLOADED
    store.update_session(session)

    return UploadResponse(session_id=session.session_id, filename=session.filename)


@router.post("/upload/example/{example_id}", response_model=UploadResponse)
async def upload_example(example_id: str, http_request: Request):
    """Upload one built-in example dataset into a new session."""
    if example_id not in EXAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Unknown example id: {example_id}")

    filename = EXAMPLE_DATASETS[example_id]
    source_path = EXAMPLE_DATA_DIR / filename
    if not source_path.exists():
        raise HTTPException(status_code=500, detail=f"Example file missing: {filename}")

    content = source_path.read_bytes()
    store = get_session_store()
    session = store.create_session()
    session.user_sub = _extract_user_sub(http_request)
    saved_path = store.save_file(session.session_id, filename, content)

    session.filename = Path(saved_path).name
    session.original_file_path = saved_path
    session.current_file_path = saved_path
    session.status = SessionStatus.UPLOADED
    store.update_session(session)

    return UploadResponse(session_id=session.session_id, filename=session.filename)


@router.get("/usage/daily", response_model=DailyUsageResponse)
async def get_daily_usage(http_request: Request):
    """Get today's backend-tracked usage for the current user."""
    user_sub = _extract_user_sub(http_request)
    snapshot = get_usage_tracker().get_daily_snapshot(user_sub=user_sub)
    return DailyUsageResponse(**snapshot)


@router.get("/preview/{session_id}", response_model=DataPreview)
async def get_preview(session_id: str, http_request: Request):
    """Get full CSV rows for client-side pagination preview."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)
    if not session.current_file_path:
        raise HTTPException(status_code=400, detail="Session has no uploaded file")

    content = Path(session.current_file_path).read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(content))
    columns = reader.fieldnames or []

    rows: list[dict[str, str | float | int | None]] = []
    for row in reader:
        normalized: dict[str, str | float | int | None] = {}
        for key, value in row.items():
            if value is None or value == "":
                normalized[key] = None
                continue
            try:
                if "." in value:
                    normalized[key] = float(value)
                else:
                    normalized[key] = int(value)
            except ValueError:
                normalized[key] = value
        rows.append(normalized)

    return DataPreview(columns=columns, rows=rows, totalRows=len(rows))


@router.post("/sessions/{session_id}/infer", response_model=InferResponse)
async def infer_session(session_id: str, payload: InferRequest, http_request: Request):
    """Run infer phase: read -> infer -> format loop -> quality validation."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    user_sub = _resolve_session_user_sub(session, http_request)
    with usage_user_scope(user_sub):
        response = agent.infer(session=session, user_hints=payload.user_hints)
    store.update_session(session)
    return response


@router.post("/sessions/{session_id}/confirm", response_model=ConfirmResponse)
async def confirm_session(session_id: str, payload: ConfirmRequest, http_request: Request):
    """Persist user confirmation and schema adjustments."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)

    try:
        confirmation = agent.confirm(
            session=session,
            confirmed=payload.confirmed,
            confirmed_schema=payload.confirmed_schema,
            user_modifications=payload.user_modifications,
            B=payload.B,
            seed=payload.seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.update_session(session)
    return ConfirmResponse(confirmation=confirmation, session_status=session.status)


@router.post("/sessions/{session_id}/run", response_model=RunResponse)
async def run_session(session_id: str, payload: RunRequest, http_request: Request):
    """Run confirmed session through engine + visualization + report."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    user_sub = _resolve_session_user_sub(session, http_request)

    with usage_user_scope(user_sub):
        response = agent.run(
            session=session,
            selected_items=payload.selected_items,
            selected_indicator_values=payload.selected_indicator_values,
        )
    store.update_session(session)

    if not response.success:
        raise HTTPException(status_code=400, detail=response.error or "Run failed")
    return response


@router.post("/sessions/{session_id}/run/start", response_model=RunStartResponse)
async def start_run_session(session_id: str, payload: RunRequest, http_request: Request):
    """Start run in a background worker and return job id for polling."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)

    if session.status not in {SessionStatus.CONFIRMED, SessionStatus.COMPLETED}:
        raise HTTPException(status_code=400, detail=f"Session is not runnable in state {session.status.value}.")

    if _has_active_run_job(session_id):
        raise HTTPException(status_code=409, detail="A run job is already in progress for this session.")

    job_id = str(uuid.uuid4())
    with _run_jobs_lock:
        _run_jobs[job_id] = {
            "job_id": job_id,
            "session_id": session_id,
            "status": RunJobStatus.QUEUED,
            "progress": 0.0,
            "message": "Run job queued.",
            "result": None,
            "error": None,
        }

    run_executor.submit(_run_session_job, session_id, job_id, payload)
    return RunStartResponse(job_id=job_id, status=RunJobStatus.QUEUED, progress=0.0, message="Run job queued.")


@router.get("/sessions/{session_id}/run/{job_id}", response_model=RunJobStatusResponse)
async def get_run_status(session_id: str, job_id: str, http_request: Request):
    """Get async run job status and optional final result."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)

    job = _get_run_job_state(job_id)
    if job is None or job.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Run job not found")

    return RunJobStatusResponse(
        job_id=job["job_id"],
        session_id=job["session_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        result=job.get("result"),
        error=job.get("error"),
    )


@router.post("/sessions/{session_id}/question", response_model=QuestionResponse)
async def ask_question(session_id: str, payload: QuestionRequest, http_request: Request):
    """Answer follow-up question with optional quote payloads."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    user_sub = _resolve_session_user_sub(session, http_request)

    try:
        with usage_user_scope(user_sub):
            answer = agent.answer(session=session, question=payload.question, quotes=payload.quotes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.update_session(session)
    return QuestionResponse(answer=answer)


@router.post("/question", response_model=QuestionResponse)
async def ask_question_global(payload: QuestionRequest, http_request: Request):
    """Answer question with optional session context; supports no-session chat."""
    user_sub = _extract_user_sub(http_request)

    if payload.session_id:
        store = get_session_store()
        session = store.get_session(payload.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        user_sub = _resolve_session_user_sub(session, http_request)
        try:
            with usage_user_scope(user_sub):
                answer = agent.answer(session=session, question=payload.question, quotes=payload.quotes)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        store.update_session(session)
        return QuestionResponse(answer=answer)

    with usage_user_scope(user_sub):
        answer = answer_question_tool(
            question=payload.question,
            results=None,
            citation_blocks={},
            quotes=payload.quotes,
            session_context={"status": "idle", "has_results": False},
        )
    return QuestionResponse(answer=answer)


@router.get("/sessions/{session_id}/artifacts/{artifact_id}")
async def get_artifact(session_id: str, artifact_id: str, http_request: Request):
    """Download artifact by id."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)

    meta = session.artifacts.get(artifact_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(meta["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media_type = meta["mime_type"] or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path=path, filename=path.name, media_type=media_type)


@router.get("/sessions/{session_id}", response_model=SessionSnapshotResponse)
async def get_session_snapshot(session_id: str, http_request: Request):
    """Return full session snapshot with artifact descriptors."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _resolve_session_user_sub(session, http_request)

    return SessionSnapshotResponse(session=session.to_snapshot(), artifacts=session.artifact_descriptors())


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, http_request: Request):
    """Delete session and associated temporary files."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is not None:
        _resolve_session_user_sub(session, http_request)
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
