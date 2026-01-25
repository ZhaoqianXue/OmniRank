"""
LangGraph Workflow Definition

Defines the agent orchestration workflow using LangGraph.
"""

import logging
from typing import TypedDict, Annotated, Literal, Optional
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

from core.schemas import (
    InferredSchema,
    ValidationWarning,
    AnalysisConfig,
    RankingResults,
    SessionStatus,
)
from core.session_memory import SessionMemory, TraceType, get_session_store
from agents.data_agent import DataAgent
from agents.orchestrator import EngineOrchestrator
from agents.analyst_agent import AnalystAgent

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

class WorkflowState(TypedDict):
    """State passed between workflow nodes."""
    session_id: str
    
    # Upload phase
    file_content: Optional[bytes]
    filename: Optional[str]
    
    # Schema phase
    schema: Optional[InferredSchema]
    warnings: list[ValidationWarning]
    
    # Config phase
    config: Optional[AnalysisConfig]
    
    # Analysis phase
    results: Optional[RankingResults]
    
    # Report phase
    report: Optional[str]
    
    # Error handling
    error: Optional[str]
    status: str


# ============================================================================
# Node Functions
# ============================================================================

def process_upload(state: WorkflowState) -> WorkflowState:
    """
    Node: Process uploaded file and infer schema.
    """
    logger.info(f"[process_upload] Session: {state['session_id']}")
    
    store = get_session_store()
    session = store.get_session(state["session_id"])
    
    if not session:
        return {**state, "error": "Session not found", "status": "error"}
    
    if not state.get("file_content"):
        return {**state, "error": "No file content", "status": "error"}
    
    # Run Data Agent
    data_agent = DataAgent()
    schema, warnings = data_agent.process(
        state["file_content"],
        state.get("filename", "data.csv"),
    )
    
    # Update session
    session.inferred_schema = schema
    session.status = SessionStatus.CONFIGURING
    store.update_session(session)
    
    # Check for blocking errors
    has_errors = any(w.severity == "error" for w in warnings)
    
    return {
        **state,
        "schema": schema,
        "warnings": warnings,
        "status": "error" if has_errors else "configuring",
        "error": "Validation errors detected" if has_errors else None,
    }


def run_analysis(state: WorkflowState) -> WorkflowState:
    """
    Node: Execute spectral ranking analysis.
    """
    logger.info(f"[run_analysis] Session: {state['session_id']}")
    
    store = get_session_store()
    session = store.get_session(state["session_id"])
    
    if not session:
        return {**state, "error": "Session not found", "status": "error"}
    
    if not state.get("config"):
        return {**state, "error": "No analysis config", "status": "error"}
    
    # Update session status
    session.status = SessionStatus.ANALYZING
    session.config = state["config"]
    store.update_session(session)
    
    # Run Engine Orchestrator
    orchestrator = EngineOrchestrator()
    
    try:
        results = orchestrator.execute(session, state["config"])
        
        # Update session
        session.results = results
        session.status = SessionStatus.COMPLETED
        store.update_session(session)
        
        return {
            **state,
            "results": results,
            "status": "completed",
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        session.status = SessionStatus.ERROR
        store.update_session(session)
        
        return {
            **state,
            "error": str(e),
            "status": "error",
        }


def generate_report(state: WorkflowState) -> WorkflowState:
    """
    Node: Generate analysis report.
    """
    logger.info(f"[generate_report] Session: {state['session_id']}")
    
    if not state.get("results"):
        return {**state, "error": "No results to report", "status": "error"}
    
    store = get_session_store()
    session = store.get_session(state["session_id"])
    
    if not session:
        return {**state, "error": "Session not found", "status": "error"}
    
    # Run Analyst Agent
    analyst = AnalystAgent()
    report = analyst.generate_report(state["results"], session)
    
    return {
        **state,
        "report": report,
        "status": "completed",
    }


def handle_error(state: WorkflowState) -> WorkflowState:
    """
    Node: Handle errors and generate diagnostic message.
    """
    logger.info(f"[handle_error] Session: {state['session_id']}, Error: {state.get('error')}")
    
    store = get_session_store()
    session = store.get_session(state["session_id"])
    
    if session:
        analyst = AnalystAgent()
        # Create a simple exception for diagnosis
        error = Exception(state.get("error", "Unknown error"))
        diagnosis = analyst.diagnose_error(error, session)
        
        session.add_trace(
            TraceType.ERROR,
            {"error": state.get("error"), "diagnosis": diagnosis},
            agent=None,
            success=False,
            error_message=state.get("error"),
        )
        store.update_session(session)
        
        return {**state, "report": diagnosis}
    
    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_upload(state: WorkflowState) -> Literal["run_analysis", "handle_error"]:
    """Route after upload based on validation results."""
    if state.get("error") or state.get("status") == "error":
        return "handle_error"
    return "run_analysis"


def route_after_analysis(state: WorkflowState) -> Literal["generate_report", "handle_error"]:
    """Route after analysis based on results."""
    if state.get("error") or state.get("status") == "error":
        return "handle_error"
    return "generate_report"


# ============================================================================
# Graph Construction
# ============================================================================

def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow.
    
    Workflow:
        process_upload -> (route) -> run_analysis -> (route) -> generate_report
                                  |                          |
                                  v                          v
                            handle_error              handle_error
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("process_upload", process_upload)
    workflow.add_node("run_analysis", run_analysis)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("handle_error", handle_error)
    
    # Set entry point
    workflow.set_entry_point("process_upload")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "process_upload",
        route_after_upload,
        {
            "run_analysis": "run_analysis",
            "handle_error": "handle_error",
        }
    )
    
    workflow.add_conditional_edges(
        "run_analysis",
        route_after_analysis,
        {
            "generate_report": "generate_report",
            "handle_error": "handle_error",
        }
    )
    
    # Terminal edges
    workflow.add_edge("generate_report", END)
    workflow.add_edge("handle_error", END)
    
    return workflow


def compile_workflow():
    """
    Compile the workflow for execution.
    
    Returns:
        Compiled graph that can be invoked
    """
    workflow = create_workflow()
    return workflow.compile()


# Global compiled workflow
_compiled_workflow = None


def get_workflow():
    """Get or create compiled workflow."""
    global _compiled_workflow
    if _compiled_workflow is None:
        _compiled_workflow = compile_workflow()
    return _compiled_workflow


# ============================================================================
# Execution Helper
# ============================================================================

async def run_full_analysis(
    session_id: str,
    file_content: bytes,
    filename: str,
    config: AnalysisConfig,
) -> WorkflowState:
    """
    Run the complete analysis workflow.
    
    Args:
        session_id: Session identifier
        file_content: Raw file content
        filename: Original filename
        config: Analysis configuration
        
    Returns:
        Final workflow state
    """
    workflow = get_workflow()
    
    initial_state: WorkflowState = {
        "session_id": session_id,
        "file_content": file_content,
        "filename": filename,
        "schema": None,
        "warnings": [],
        "config": config,
        "results": None,
        "report": None,
        "error": None,
        "status": "uploading",
    }
    
    # Run workflow
    result = await workflow.ainvoke(initial_state)
    
    return result
