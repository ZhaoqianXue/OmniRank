"""OmniRank API Schemas (contract-first)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ComparisonFormat(str, Enum):
    """Supported comparison formats."""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    MULTIWAY = "multiway"


class SessionStatus(str, Enum):
    """Session lifecycle states."""

    IDLE = "idle"
    UPLOADED = "uploaded"
    INFERRED = "inferred"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class DataSummary(BaseModel):
    """Lightweight dataset summary for JIT context loading."""

    columns: list[str]
    sample_rows: list[dict[str, Any]]
    row_count: int
    column_types: dict[str, str]


class ReadDataFileResult(BaseModel):
    """Result from read_data_file tool."""

    success: bool
    data: Optional[DataSummary] = None
    error: Optional[str] = None


class SemanticSchema(BaseModel):
    """Inferred schema used by downstream tools and engine."""

    bigbetter: int = Field(ge=0, le=1)
    ranking_items: list[str]
    indicator_col: Optional[str] = None
    indicator_values: list[str] = Field(default_factory=list)


class SemanticSchemaResult(BaseModel):
    """Result from infer_semantic_schema tool."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    success: bool
    format: ComparisonFormat
    format_evidence: str
    schema_: Optional[SemanticSchema] = Field(default=None, alias="schema")
    error: Optional[str] = None

    @property
    def schema(self) -> Optional[SemanticSchema]:
        """Backward-compatible accessor for schema payload."""
        return self.schema_


class FormatValidationResult(BaseModel):
    """Result from validate_data_format tool."""

    is_ready: bool
    fixable: bool
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class QualityValidationResult(BaseModel):
    """Result from validate_data_quality tool."""

    is_valid: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PreprocessResult(BaseModel):
    """Result from preprocess_data tool."""

    preprocessed_csv_path: str
    transformation_log: list[str] = Field(default_factory=list)
    row_count: int
    dropped_rows: int


class EngineConfig(BaseModel):
    """Input config for execute_spectral_ranking."""

    csv_path: str
    bigbetter: int = Field(ge=0, le=1)
    selected_items: Optional[list[str]] = None
    selected_indicator_values: Optional[list[str]] = None
    B: int = Field(default=2000, ge=100, le=100000)
    seed: int = 42
    r_script_path: str = "src/spectral_ranking/spectral_ranking.R"


class RankingMetadata(BaseModel):
    """Engine metadata propagated from R output."""

    n_samples: int = 0
    k_methods: int = 0
    runtime_sec: float = 0.0
    heterogeneity_index: float = 0.0
    spectral_gap: float = 0.0
    sparsity_ratio: float = 0.0
    mean_ci_width_top_5: float = 0.0
    n_comparisons: Optional[int] = None


class RankingResults(BaseModel):
    """Ranking output from spectral engine."""

    items: list[str]
    theta_hat: list[float]
    ranks: list[int]
    ci_lower: list[float]
    ci_upper: list[float]
    indicator_value: Optional[str] = None
    metadata: Optional[RankingMetadata] = None


class ExecutionTrace(BaseModel):
    """Structured execution trace for reproducibility/debugging."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timestamp: str


class ExecutionResult(BaseModel):
    """Result from execute_spectral_ranking tool."""

    success: bool
    results: Optional[RankingResults] = None
    error: Optional[str] = None
    trace: ExecutionTrace


class ToolCallRecord(BaseModel):
    """Append-only record of tool invocations."""

    tool_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    timestamp: str
    success: bool = True
    error: Optional[str] = None


class PlotSpec(BaseModel):
    """Deterministic visualization artifact specification."""

    type: str
    data: dict[str, Any]
    config: dict[str, Any]
    svg_path: str
    block_id: str = ""
    caption_plain: str = ""
    caption_academic: str = ""
    hint_ids: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    """Artifact metadata."""

    kind: str
    path: str
    title: str = ""
    mime_type: str = "text/plain"


class HintKind(str, Enum):
    """Hint categories in report."""

    DEFINITION = "definition"
    ASSUMPTION = "assumption"
    CAVEAT = "caveat"
    METHOD = "method"


class HintSpec(BaseModel):
    """Inline micro-explanation."""

    hint_id: str
    title: str
    body: str
    kind: HintKind = HintKind.DEFINITION
    sources: list[str] = Field(default_factory=list)


class CitationKind(str, Enum):
    """Citable report block categories."""

    SUMMARY = "summary"
    RESULT = "result"
    COMPARISON = "comparison"
    METHOD = "method"
    LIMITATION = "limitation"
    REPRO = "repro"
    FIGURE = "figure"
    TABLE = "table"


class CitationBlock(BaseModel):
    """Citable block wrapper for quote UX."""

    block_id: str
    kind: CitationKind
    markdown: str
    text: str
    hint_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)


class QuoteSource(str, Enum):
    """Source of quoted text."""

    REPORT = "report"
    USER_UPLOAD = "user_upload"
    EXTERNAL = "external"


class QuotePayload(BaseModel):
    """Quoted text payload sent from frontend."""

    quoted_text: str
    block_id: Optional[str] = None
    kind: Optional[str] = None
    source: QuoteSource = QuoteSource.REPORT


class VisualizationOutput(BaseModel):
    """Result from generate_visualizations tool."""

    plots: list[PlotSpec]
    errors: list[str] = Field(default_factory=list)


class ReportOutput(BaseModel):
    """Result from generate_report tool."""

    markdown: str
    key_findings: dict[str, Any]
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    hints: list[HintSpec] = Field(default_factory=list)
    citation_blocks: list[CitationBlock] = Field(default_factory=list)


class AnswerOutput(BaseModel):
    """Result from answer_question tool."""

    answer: str
    supporting_evidence: list[str] = Field(default_factory=list)
    used_citation_block_ids: list[str] = Field(default_factory=list)


class ConfirmationResult(BaseModel):
    """Result from request_user_confirmation tool."""

    confirmed: bool
    confirmed_schema: SemanticSchema
    user_modifications: list[str] = Field(default_factory=list)
    B: int = 2000
    seed: int = 42


class DataPreview(BaseModel):
    """Data preview payload."""

    columns: list[str]
    rows: list[dict[str, Any]]
    totalRows: int


class UploadResponse(BaseModel):
    """Upload response payload."""

    session_id: str
    filename: str


class InferRequest(BaseModel):
    """Infer request payload."""

    user_hints: Optional[str] = None


class InferResponse(BaseModel):
    """Infer response payload."""

    success: bool
    data_summary: Optional[DataSummary] = None
    schema_result: Optional[SemanticSchemaResult] = None
    format_result: Optional[FormatValidationResult] = None
    quality_result: Optional[QualityValidationResult] = None
    preprocessed_path: Optional[str] = None
    requires_confirmation: bool = True
    error: Optional[str] = None


class ConfirmRequest(BaseModel):
    """Confirmation request payload."""

    confirmed: bool
    confirmed_schema: SemanticSchema
    user_modifications: list[str] = Field(default_factory=list)
    B: int = 2000
    seed: int = 42


class ConfirmResponse(BaseModel):
    """Confirmation response payload."""

    confirmation: ConfirmationResult
    session_status: SessionStatus


class RunRequest(BaseModel):
    """Run request payload."""

    selected_items: Optional[list[str]] = None
    selected_indicator_values: Optional[list[str]] = None


class RunResponse(BaseModel):
    """Execution response payload."""

    success: bool
    config: Optional[EngineConfig] = None
    execution: Optional[ExecutionResult] = None
    visualizations: Optional[VisualizationOutput] = None
    report: Optional[ReportOutput] = None
    error: Optional[str] = None


class RunJobStatus(str, Enum):
    """Async run job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunStartResponse(BaseModel):
    """Async run start response payload."""

    job_id: str
    status: RunJobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str


class RunJobStatusResponse(BaseModel):
    """Async run job status payload."""

    job_id: str
    session_id: str
    status: RunJobStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str
    result: Optional[RunResponse] = None
    error: Optional[str] = None


class QuestionRequest(BaseModel):
    """Question request payload."""

    question: str
    quotes: list[QuotePayload] = Field(default_factory=list)


class QuestionResponse(BaseModel):
    """Question response payload."""

    answer: AnswerOutput


class SessionSnapshot(BaseModel):
    """Serializable session state returned by API."""

    session_id: str
    status: SessionStatus
    original_file_path: Optional[str] = None
    current_file_path: Optional[str] = None
    filename: Optional[str] = None
    data_summary: Optional[DataSummary] = None
    inferred_schema: Optional[SemanticSchema] = None
    format_validation_result: Optional[FormatValidationResult] = None
    quality_validation_result: Optional[QualityValidationResult] = None
    confirmed_schema: Optional[SemanticSchema] = None
    config: Optional[EngineConfig] = None
    current_results: Optional[RankingResults] = None
    execution_trace: list[ExecutionTrace] = Field(default_factory=list)
    tool_call_history: list[ToolCallRecord] = Field(default_factory=list)
    report_output: Optional[ReportOutput] = None
    visualization_output: Optional[VisualizationOutput] = None
    citation_blocks: list[CitationBlock] = Field(default_factory=list)
    created_at: str
    updated_at: str
    error: Optional[str] = None


class ArtifactDescriptor(BaseModel):
    """Artifact listing entry."""

    artifact_id: str
    kind: str
    title: str
    mime_type: str


class SessionSnapshotResponse(BaseModel):
    """Session snapshot endpoint response."""

    session: SessionSnapshot
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    version: str
    r_available: bool


class WSMessageType(str, Enum):
    """WebSocket message categories."""

    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    PROGRESS = "progress"
    ERROR = "error"
    ECHO = "echo"
    PONG = "pong"


class WSProgressPayload(BaseModel):
    """Progress payload for websocket notifications."""

    progress: float
    message: str


class WSMessage(BaseModel):
    """Generic websocket message container."""

    type: WSMessageType
    payload: dict[str, Any]
