"""OmniRank HTTP API routes (single-agent SOP pipeline)."""

from __future__ import annotations

import csv
import io
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agents.omnirank_agent import OmniRankAgent
from core.schemas import (
    ConfirmResponse,
    ConfirmRequest,
    DataPreview,
    InferRequest,
    InferResponse,
    QuestionResponse,
    QuestionRequest,
    RunRequest,
    RunResponse,
    SessionStatus,
    SessionSnapshotResponse,
    UploadResponse,
)
from core.session_memory import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["omnirank"])


EXAMPLE_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "examples"
EXAMPLE_DATASETS: dict[str, str] = {
    "pairwise": "example_data_pairwise.csv",
    "pointwise": "example_data_pointwise.csv",
    "multiway": "example_data_multiway.csv",
}

agent = OmniRankAgent()


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
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

    saved_path = store.save_file(session.session_id, file.filename, content)
    session.filename = Path(saved_path).name
    session.original_file_path = saved_path
    session.current_file_path = saved_path
    session.status = SessionStatus.UPLOADED
    store.update_session(session)

    return UploadResponse(session_id=session.session_id, filename=session.filename)


@router.post("/upload/example/{example_id}", response_model=UploadResponse)
async def upload_example(example_id: str):
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
    saved_path = store.save_file(session.session_id, filename, content)

    session.filename = Path(saved_path).name
    session.original_file_path = saved_path
    session.current_file_path = saved_path
    session.status = SessionStatus.UPLOADED
    store.update_session(session)

    return UploadResponse(session_id=session.session_id, filename=session.filename)


@router.get("/preview/{session_id}", response_model=DataPreview)
async def get_preview(session_id: str):
    """Get full CSV rows for client-side pagination preview."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
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
async def infer_session(session_id: str, request: InferRequest):
    """Run infer phase: read -> infer -> format loop -> quality validation."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    response = agent.infer(session=session, user_hints=request.user_hints)
    store.update_session(session)
    return response


@router.post("/sessions/{session_id}/confirm", response_model=ConfirmResponse)
async def confirm_session(session_id: str, request: ConfirmRequest):
    """Persist user confirmation and schema adjustments."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        confirmation = agent.confirm(
            session=session,
            confirmed=request.confirmed,
            confirmed_schema=request.confirmed_schema,
            user_modifications=request.user_modifications,
            B=request.B,
            seed=request.seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.update_session(session)
    return ConfirmResponse(confirmation=confirmation, session_status=session.status)


@router.post("/sessions/{session_id}/run", response_model=RunResponse)
async def run_session(session_id: str, request: RunRequest):
    """Run confirmed session through engine + visualization + report."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    response = agent.run(
        session=session,
        selected_items=request.selected_items,
        selected_indicator_values=request.selected_indicator_values,
    )
    store.update_session(session)

    if not response.success:
        raise HTTPException(status_code=400, detail=response.error or "Run failed")
    return response


@router.post("/sessions/{session_id}/question", response_model=QuestionResponse)
async def ask_question(session_id: str, request: QuestionRequest):
    """Answer follow-up question with optional quote payloads."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        answer = agent.answer(session=session, question=request.question, quotes=request.quotes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.update_session(session)
    return QuestionResponse(answer=answer)


@router.get("/sessions/{session_id}/artifacts/{artifact_id}")
async def get_artifact(session_id: str, artifact_id: str):
    """Download artifact by id."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    meta = session.artifacts.get(artifact_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(meta["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media_type = meta["mime_type"] or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(path=path, filename=path.name, media_type=media_type)


@router.get("/sessions/{session_id}", response_model=SessionSnapshotResponse)
async def get_session_snapshot(session_id: str):
    """Return full session snapshot with artifact descriptors."""
    store = get_session_store()
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionSnapshotResponse(session=session.to_snapshot(), artifacts=session.artifact_descriptors())


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session and associated temporary files."""
    store = get_session_store()
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}
