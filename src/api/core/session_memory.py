"""Session-scoped memory for OmniRank single-agent pipeline."""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Optional

from .schemas import (
    ArtifactDescriptor,
    CitationBlock,
    DataSummary,
    EngineConfig,
    ExecutionTrace,
    FormatValidationResult,
    QualityValidationResult,
    RankingResults,
    ReportOutput,
    SemanticSchema,
    SessionSnapshot,
    SessionStatus,
    ToolCallRecord,
    VisualizationOutput,
)


SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")


def _now_iso() -> str:
    """Return unix timestamp as ISO-like string with millisecond precision."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time() % 1)*1000):03d}Z"


def sanitize_uploaded_filename(filename: str) -> str:
    """Normalize uploaded filename to basename + whitelist."""
    base = Path(filename).name
    normalized = SAFE_FILENAME_PATTERN.sub("_", base).strip()
    if not normalized:
        normalized = "upload.csv"
    return normalized


@dataclass
class SessionMemory:
    """In-memory state for one analysis session."""

    session_id: str
    status: SessionStatus = SessionStatus.IDLE
    filename: Optional[str] = None
    original_file_path: Optional[str] = None
    current_file_path: Optional[str] = None

    data_summary: Optional[DataSummary] = None
    inferred_schema: Optional[SemanticSchema] = None
    format_validation_result: Optional[FormatValidationResult] = None
    quality_validation_result: Optional[QualityValidationResult] = None
    confirmed_schema: Optional[SemanticSchema] = None

    config: Optional[EngineConfig] = None
    current_results: Optional[RankingResults] = None
    execution_trace: list[ExecutionTrace] = field(default_factory=list)

    report_output: Optional[ReportOutput] = None
    visualization_output: Optional[VisualizationOutput] = None
    citation_blocks: list[CitationBlock] = field(default_factory=list)

    tool_call_history: list[ToolCallRecord] = field(default_factory=list)
    artifacts: dict[str, dict[str, str]] = field(default_factory=dict)

    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    error: Optional[str] = None

    def touch(self) -> None:
        """Update modified timestamp."""
        self.updated_at = _now_iso()

    def add_tool_call(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Append one immutable tool call record."""
        self.tool_call_history.append(
            ToolCallRecord(
                tool_name=tool_name,
                inputs=inputs,
                outputs=outputs,
                timestamp=_now_iso(),
                success=success,
                error=error,
            )
        )
        self.touch()

    def add_execution_trace(self, trace: ExecutionTrace) -> None:
        """Append execution trace record."""
        self.execution_trace.append(trace)
        self.touch()

    def register_artifact(self, kind: str, path: str, title: str, mime_type: str) -> str:
        """Register a filesystem artifact and return stable artifact id."""
        artifact_id = str(uuid.uuid4())
        self.artifacts[artifact_id] = {
            "kind": kind,
            "path": path,
            "title": title,
            "mime_type": mime_type,
        }
        self.touch()
        return artifact_id

    def to_snapshot(self) -> SessionSnapshot:
        """Serialize session for API responses."""
        return SessionSnapshot(
            session_id=self.session_id,
            status=self.status,
            original_file_path=self.original_file_path,
            current_file_path=self.current_file_path,
            filename=self.filename,
            data_summary=self.data_summary,
            inferred_schema=self.inferred_schema,
            format_validation_result=self.format_validation_result,
            quality_validation_result=self.quality_validation_result,
            confirmed_schema=self.confirmed_schema,
            config=self.config,
            current_results=self.current_results,
            execution_trace=self.execution_trace,
            tool_call_history=self.tool_call_history,
            report_output=self.report_output,
            visualization_output=self.visualization_output,
            citation_blocks=self.citation_blocks,
            created_at=self.created_at,
            updated_at=self.updated_at,
            error=self.error,
        )

    def artifact_descriptors(self) -> list[ArtifactDescriptor]:
        """Return artifact metadata list."""
        descriptors: list[ArtifactDescriptor] = []
        for artifact_id, meta in self.artifacts.items():
            descriptors.append(
                ArtifactDescriptor(
                    artifact_id=artifact_id,
                    kind=meta["kind"],
                    title=meta["title"],
                    mime_type=meta["mime_type"],
                )
            )
        return descriptors


class SessionStore:
    """In-memory session store with per-session work directories."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self._sessions: dict[str, SessionMemory] = {}
        self._temp_dir = temp_dir or Path(tempfile.gettempdir()) / "omnirank_sessions"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self) -> SessionMemory:
        """Create and register a new session."""
        session_id = str(uuid.uuid4())
        session = SessionMemory(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session by id."""
        return self._sessions.get(session_id)

    def update_session(self, session: SessionMemory) -> None:
        """Upsert session."""
        session.touch()
        self._sessions[session.session_id] = session

    def delete_session(self, session_id: str) -> bool:
        """Delete one session and workspace."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False

        work_dir = self.get_session_work_dir(session_id)
        if work_dir.exists():
            for child in sorted(work_dir.glob("**/*"), reverse=True):
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError:
                        pass
                elif child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            try:
                work_dir.rmdir()
            except OSError:
                pass

        return True

    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        """Persist uploaded file in session work directory."""
        work_dir = self.get_session_work_dir(session_id)
        safe_filename = sanitize_uploaded_filename(filename)
        target_path = (work_dir / safe_filename).resolve()
        work_dir_resolved = work_dir.resolve()
        if work_dir_resolved not in target_path.parents and target_path != work_dir_resolved:
            raise ValueError("Unsafe upload path resolved outside the session workspace.")
        target_path.write_bytes(content)
        return str(target_path)

    def get_session_work_dir(self, session_id: str) -> Path:
        """Return session work directory path."""
        session_dir = self._temp_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir


_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get global session store singleton."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
