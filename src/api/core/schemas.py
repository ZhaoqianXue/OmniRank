"""
OmniRank API Schemas (Pydantic Models)
Contract-First: These schemas mirror shared/types/api.ts for Python backend.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Common Types
# ============================================================================

class DataFormat(str, Enum):
    """Supported data formats for ranking analysis."""
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    MULTIWAY = "multiway"


class InferredSchema(BaseModel):
    """Schema inferred by the Data Agent from uploaded data."""
    format: DataFormat
    bigbetter: int = Field(ge=0, le=1, description="1=higher is better, 0=lower is better")
    ranking_items: list[str]
    indicator_col: Optional[str] = None
    indicator_values: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationWarning(BaseModel):
    """Warning or error from data validation."""
    type: str  # "sparsity" | "connectivity" | "format"
    message: str
    severity: str  # "warning" | "error"


# ============================================================================
# Upload API
# ============================================================================

class UploadResponse(BaseModel):
    """Response from file upload endpoint."""
    session_id: str
    filename: str
    inferred_schema: InferredSchema
    warnings: list[ValidationWarning] = []


# ============================================================================
# Data Preview API
# ============================================================================

class DataPreview(BaseModel):
    """Preview of uploaded data for display."""
    columns: list[str]
    rows: list[dict]  # List of row dictionaries
    totalRows: int


# ============================================================================
# Analyze API
# ============================================================================

class AnalysisConfig(BaseModel):
    """User-configurable analysis parameters."""
    bigbetter: int = Field(ge=0, le=1)
    selected_items: Optional[list[str]] = None
    selected_indicator_values: Optional[list[str]] = None
    bootstrap_iterations: int = Field(default=2000, ge=100, le=10000)
    random_seed: int = Field(default=42)


class AnalyzeRequest(BaseModel):
    """Request to trigger ranking analysis."""
    session_id: str
    config: AnalysisConfig


class AnalysisStatus(str, Enum):
    """Status of an analysis job."""
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class AnalyzeResponse(BaseModel):
    """Response from analysis endpoint."""
    status: AnalysisStatus
    results: Optional["RankingResults"] = None
    error: Optional[str] = None


# ============================================================================
# Ranking Results
# ============================================================================

class RankingItem(BaseModel):
    """Individual item in ranking results."""
    name: str
    theta_hat: float
    rank: int
    ci_lower: int
    ci_upper: int
    ci_two_sided: tuple[int, int]


class RankingMetadata(BaseModel):
    """Metadata about the ranking analysis."""
    n_items: int
    n_comparisons: int
    heterogeneity_index: float
    sparsity_ratio: float
    step2_triggered: bool
    runtime_sec: float


class PairwiseComparison(BaseModel):
    """Pairwise comparison statistics for heatmap visualization."""
    item_a: str
    item_b: str
    win_rate_a: float = Field(ge=0.0, le=1.0)
    n_comparisons: int


class RankingResults(BaseModel):
    """Complete ranking analysis results."""
    items: list[RankingItem]
    metadata: RankingMetadata
    pairwise_matrix: list[PairwiseComparison] = []


# ============================================================================
# WebSocket Messages
# ============================================================================

class AgentType(str, Enum):
    """Types of agents in the system."""
    DATA = "data"
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"


class WSMessageType(str, Enum):
    """Types of WebSocket messages."""
    PROGRESS = "progress"
    AGENT_MESSAGE = "agent_message"
    RESULT = "result"
    ERROR = "error"


class WSProgressPayload(BaseModel):
    """Payload for progress updates."""
    progress: float = Field(ge=0.0, le=1.0)
    message: str


class WSAgentMessagePayload(BaseModel):
    """Payload for agent messages."""
    agent: AgentType
    message: str
    thinking: Optional[str] = None


class WSResultPayload(BaseModel):
    """Payload for result messages."""
    results: RankingResults


class WSErrorPayload(BaseModel):
    """Payload for error messages."""
    error: str
    agent: Optional[AgentType] = None


class WSMessage(BaseModel):
    """WebSocket message structure."""
    type: WSMessageType
    payload: dict  # Union type handled at runtime


# ============================================================================
# Session State
# ============================================================================

class SessionStatus(str, Enum):
    """Status of a user session."""
    IDLE = "idle"
    UPLOADING = "uploading"
    CONFIGURING = "configuring"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"


class ChatMessage(BaseModel):
    """A message in the chat history."""
    id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float
    agent: Optional[AgentType] = None


class SessionState(BaseModel):
    """Complete session state."""
    session_id: str
    filename: Optional[str] = None
    schema_: Optional[InferredSchema] = Field(None, alias="schema")
    config: Optional[AnalysisConfig] = None
    results: Optional[RankingResults] = None
    messages: list[ChatMessage] = []
    status: SessionStatus = SessionStatus.IDLE

    class Config:
        populate_by_name = True


# Update forward references
AnalyzeResponse.model_rebuild()
