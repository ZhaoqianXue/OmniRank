"""
OmniRank API Core Module

Core components for spectral ranking execution and session management.
"""

from .schemas import (
    DataFormat,
    InferredSchema,
    ValidationWarning,
    UploadResponse,
    AnalysisConfig,
    AnalyzeRequest,
    AnalysisStatus,
    AnalyzeResponse,
    RankingItem,
    RankingMetadata,
    PairwiseComparison,
    RankingResults,
    AgentType,
    WSMessageType,
    WSProgressPayload,
    WSAgentMessagePayload,
    WSResultPayload,
    WSErrorPayload,
    WSMessage,
    SessionStatus,
    ChatMessage,
    SessionState,
)

from .r_executor import (
    RScriptExecutor,
    RExecutorError,
    RScriptNotFoundError,
    RExecutionError,
    RTimeoutError,
    ROutputParseError,
    Step1Params,
    Step2Params,
    Step1Result,
    Step2Result,
    ExecutionTrace,
    should_run_step2,
)

from .session_memory import (
    SessionMemory,
    SessionStore,
    TraceType,
    ExecutionTraceEntry,
    get_session_store,
)

__all__ = [
    # Schemas
    "DataFormat",
    "InferredSchema",
    "ValidationWarning",
    "UploadResponse",
    "AnalysisConfig",
    "AnalyzeRequest",
    "AnalysisStatus",
    "AnalyzeResponse",
    "RankingItem",
    "RankingMetadata",
    "PairwiseComparison",
    "RankingResults",
    "AgentType",
    "WSMessageType",
    "WSProgressPayload",
    "WSAgentMessagePayload",
    "WSResultPayload",
    "WSErrorPayload",
    "WSMessage",
    "SessionStatus",
    "ChatMessage",
    "SessionState",
    # R Executor
    "RScriptExecutor",
    "RExecutorError",
    "RScriptNotFoundError",
    "RExecutionError",
    "RTimeoutError",
    "ROutputParseError",
    "Step1Params",
    "Step2Params",
    "Step1Result",
    "Step2Result",
    "ExecutionTrace",
    "should_run_step2",
    # Session Memory
    "SessionMemory",
    "SessionStore",
    "TraceType",
    "ExecutionTraceEntry",
    "get_session_store",
]
