"""
Session Memory Management

Manages session state, execution history, and context for agent interactions.
Provides in-memory storage for MVP with Redis-compatible interface for production.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path
import tempfile

from .schemas import (
    SessionStatus,
    InferredSchema,
    AnalysisConfig,
    RankingResults,
    ChatMessage,
    AgentType,
)


class TraceType(str, Enum):
    """Types of execution traces."""
    UPLOAD = "upload"
    SCHEMA_INFERENCE = "schema_inference"
    VALIDATION = "validation"
    ENGINE_EXECUTION = "engine_execution"
    REPORT_GENERATION = "report_generation"
    ERROR = "error"
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"


@dataclass
class ExecutionTraceEntry:
    """Single entry in execution trace."""
    trace_type: TraceType
    timestamp: float
    agent: Optional[AgentType]
    data: dict
    duration_sec: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SessionMemory:
    """
    In-memory session state and context.
    
    Attributes:
        session_id: Unique session identifier
        status: Current session status
        filename: Uploaded filename
        file_content: Raw file content (bytes)
        file_path: Temporary file path on disk
        inferred_schema: Schema inferred by Data Agent
        config: User-confirmed analysis configuration
        results: Final ranking results
        messages: Chat history
        traces: Execution trace for debugging
        created_at: Session creation timestamp
        updated_at: Last update timestamp
    """
    session_id: str
    status: SessionStatus = SessionStatus.IDLE
    
    # Data state
    filename: Optional[str] = None
    file_content: Optional[bytes] = None
    file_path: Optional[str] = None
    data_agent_started: bool = False
    
    # Schema and config
    inferred_schema: Optional[InferredSchema] = None
    config: Optional[AnalysisConfig] = None
    
    # Results
    results: Optional[RankingResults] = None
    step1_json_path: Optional[str] = None
    
    # Chat history
    messages: list[ChatMessage] = field(default_factory=list)
    
    # Execution traces
    traces: list[ExecutionTraceEntry] = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_trace(
        self,
        trace_type: TraceType,
        data: dict,
        agent: Optional[AgentType] = None,
        duration_sec: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Add an execution trace entry."""
        entry = ExecutionTraceEntry(
            trace_type=trace_type,
            timestamp=time.time(),
            agent=agent,
            data=data,
            duration_sec=duration_sec,
            success=success,
            error_message=error_message,
        )
        self.traces.append(entry)
        self.updated_at = time.time()
    
    def add_message(
        self,
        role: str,
        content: str,
        agent: Optional[AgentType] = None,
    ) -> ChatMessage:
        """Add a chat message."""
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            agent=agent,
        )
        self.messages.append(message)
        self.updated_at = time.time()
        return message
    
    def get_recent_context(self, n_messages: int = 10) -> list[ChatMessage]:
        """Get recent chat messages for context."""
        return self.messages[-n_messages:]
    
    def get_traces_by_type(self, trace_type: TraceType) -> list[ExecutionTraceEntry]:
        """Get traces filtered by type."""
        return [t for t in self.traces if t.trace_type == trace_type]
    
    def get_error_traces(self) -> list[ExecutionTraceEntry]:
        """Get all error traces."""
        return [t for t in self.traces if not t.success]
    
    def to_context_dict(self) -> dict:
        """
        Convert session state to a context dictionary for LLM prompts.
        
        Returns minimal relevant information to avoid token bloat.
        """
        ctx = {
            "session_id": self.session_id,
            "status": self.status.value,
        }
        
        if self.filename:
            ctx["filename"] = self.filename
        
        if self.inferred_schema:
            ctx["schema"] = {
                "format": self.inferred_schema.format.value,
                "bigbetter": self.inferred_schema.bigbetter,
                "n_items": len(self.inferred_schema.ranking_items),
                "items": self.inferred_schema.ranking_items[:10],  # First 10
            }
        
        if self.config:
            ctx["config"] = {
                "bigbetter": self.config.bigbetter,
                "bootstrap": self.config.bootstrap_iterations,
            }
        
        if self.results:
            ctx["results_summary"] = {
                "n_items": self.results.metadata.n_items,
                "n_comparisons": self.results.metadata.n_comparisons,
                "top_3": [
                    {"name": item.name, "rank": item.rank}
                    for item in sorted(self.results.items, key=lambda x: x.rank)[:3]
                ],
            }
        
        # Recent errors
        errors = self.get_error_traces()
        if errors:
            ctx["recent_errors"] = [
                {
                    "type": e.trace_type.value,
                    "message": e.error_message,
                }
                for e in errors[-3:]  # Last 3 errors
            ]
        
        return ctx


class SessionStore:
    """
    In-memory session store.
    
    Thread-safe storage for session memories.
    For production, replace with Redis-backed implementation.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize session store."""
        self._sessions: dict[str, SessionMemory] = {}
        self._temp_dir = temp_dir or Path(tempfile.gettempdir()) / "omnirank_sessions"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> SessionMemory:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = SessionMemory(session_id=session_id)
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionMemory:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session()
    
    def update_session(self, session: SessionMemory) -> None:
        """Update session in store."""
        session.updated_at = time.time()
        self._sessions[session.session_id] = session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session by ID."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            # Clean up temp files
            if session.file_path and Path(session.file_path).exists():
                try:
                    Path(session.file_path).unlink()
                except Exception:
                    pass
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> list[str]:
        """List all session IDs."""
        return list(self._sessions.keys())
    
    def save_file(self, session_id: str, filename: str, content: bytes) -> str:
        """
        Save uploaded file to temp directory.
        
        Returns path to saved file.
        """
        session_dir = self._temp_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = session_dir / filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        return str(file_path)
    
    def get_session_work_dir(self, session_id: str) -> Path:
        """Get working directory for a session."""
        work_dir = self._temp_dir / session_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    
    def cleanup_expired_sessions(self, max_age_sec: int = 3600) -> int:
        """
        Clean up sessions older than max_age_sec.
        
        Returns number of sessions cleaned up.
        """
        now = time.time()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.updated_at > max_age_sec
        ]
        
        for sid in expired:
            self.delete_session(sid)
        
        return len(expired)


# Global session store instance
_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get or create global session store."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
