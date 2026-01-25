"""
OmniRank REST API Routes
Handles file upload, analysis triggering, and results retrieval.
"""

import asyncio
import csv
import io
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks

from pydantic import BaseModel

from core.schemas import (
    UploadResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisStatus,
    SessionStatus,
    WSMessage,
    WSMessageType,
    AgentType,
    DataPreview,
)

# Example data configuration
EXAMPLE_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "examples"

EXAMPLE_DATASETS = {
    "pairwise": {
        "filename": "example_data_pairwise.csv",
        "title": "LLM Pairwise Comparison",
    },
    "pointwise": {
        "filename": "example_data_pointwise.csv",
        "title": "Model Performance Scores",
    },
}


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    session_id: str
    question: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    answer: str
    agent: str = "analyst"
from core.session_memory import get_session_store
from agents.data_agent import DataAgent
from agents.orchestrator import EngineOrchestrator
from agents.analyst_agent import AnalystAgent
from api.websocket import get_connection_manager

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ranking"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: Annotated[UploadFile, File(description="CSV or JSON file")]):
    """
    Upload a comparison data file for analysis.
    
    The Data Agent will:
    1. Detect the data format (pointwise/pairwise/multiway)
    2. Infer the schema (bigbetter, ranking_items, indicators)
    3. Validate the data and return warnings
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not (file.filename.endswith(".csv") or file.filename.endswith(".json")):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and JSON files are supported",
        )
    
    # Read file content
    content = await file.read()
    
    # Create session
    store = get_session_store()
    session = store.create_session()
    
    # Save file
    file_path = store.save_file(session.session_id, file.filename, content)
    session.filename = file.filename
    session.file_content = content
    session.file_path = file_path
    session.status = SessionStatus.UPLOADING
    
    # Run Data Agent
    data_agent = DataAgent()
    schema, warnings = data_agent.process(content, file.filename)
    
    # Update session
    session.inferred_schema = schema
    session.status = SessionStatus.CONFIGURING
    store.update_session(session)
    
    return UploadResponse(
        session_id=session.session_id,
        filename=file.filename,
        inferred_schema=schema,
        warnings=warnings,
    )


@router.post("/upload/example/{example_id}", response_model=UploadResponse)
async def upload_example(example_id: str):
    """
    Load example data for analysis.
    
    Available examples:
    - pairwise: LLM pairwise comparison data
    - pointwise: Model performance scores
    """
    if example_id not in EXAMPLE_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown example dataset: {example_id}. Available: {list(EXAMPLE_DATASETS.keys())}",
        )
    
    example_info = EXAMPLE_DATASETS[example_id]
    file_path = EXAMPLE_DATA_DIR / example_info["filename"]
    
    if not file_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Example data file not found: {example_info['filename']}",
        )
    
    # Read file content
    content = file_path.read_bytes()
    filename = example_info["filename"]
    
    # Create session
    store = get_session_store()
    session = store.create_session()
    
    # Save file
    saved_path = store.save_file(session.session_id, filename, content)
    session.filename = filename
    session.file_content = content
    session.file_path = saved_path
    session.status = SessionStatus.UPLOADING
    
    # Run Data Agent
    data_agent = DataAgent()
    schema, warnings = data_agent.process(content, filename)
    
    # Update session
    session.inferred_schema = schema
    session.status = SessionStatus.CONFIGURING
    store.update_session(session)
    
    return UploadResponse(
        session_id=session.session_id,
        filename=filename,
        inferred_schema=schema,
        warnings=warnings,
    )


@router.get("/preview/{session_id}", response_model=DataPreview)
async def get_data_preview(session_id: str):
    """
    Get full data for preview and pagination in the UI.
    
    Returns all rows - pagination is handled client-side for better UX.
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.file_content:
        raise HTTPException(status_code=400, detail="No file uploaded for this session")
    
    try:
        # Decode content
        content_str = session.file_content.decode("utf-8")
        
        # Parse CSV
        reader = csv.DictReader(io.StringIO(content_str))
        columns = reader.fieldnames or []
        
        rows = []
        for row in reader:
            # Convert numeric strings to numbers for cleaner display
            processed_row = {}
            for k, v in row.items():
                try:
                    # Try float conversion
                    if v and "." in v:
                        processed_row[k] = float(v)
                    elif v:
                        processed_row[k] = int(v)
                    else:
                        processed_row[k] = v
                except (ValueError, TypeError):
                    processed_row[k] = v
            rows.append(processed_row)
        
        return DataPreview(
            columns=columns,
            rows=rows,
            totalRows=len(rows),
        )
    except Exception as e:
        logger.error(f"Failed to parse data preview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse data: {str(e)}")


async def _send_ws_progress(session_id: str, progress: float, message: str, agent: str = None):
    """Send progress update via WebSocket."""
    try:
        manager = get_connection_manager()
        ws_message = WSMessage(
            type=WSMessageType.PROGRESS,
            payload={"progress": progress, "message": message, "agent": agent},
        )
        await manager.send_message(session_id, ws_message)
    except Exception as e:
        logger.debug(f"WebSocket send failed (no clients?): {e}")


async def _send_ws_agent_message(session_id: str, agent: AgentType, message: str):
    """Send agent message via WebSocket."""
    try:
        manager = get_connection_manager()
        ws_message = WSMessage(
            type=WSMessageType.AGENT_MESSAGE,
            payload={"agent": agent.value, "message": message},
        )
        await manager.send_message(session_id, ws_message)
    except Exception as e:
        logger.debug(f"WebSocket send failed: {e}")


async def _send_ws_result(session_id: str, results):
    """Send final results via WebSocket."""
    try:
        manager = get_connection_manager()
        ws_message = WSMessage(
            type=WSMessageType.RESULT,
            payload={"results": results.model_dump() if results else None},
        )
        await manager.send_message(session_id, ws_message)
    except Exception as e:
        logger.debug(f"WebSocket send failed: {e}")


async def _send_ws_error(session_id: str, error: str):
    """Send error via WebSocket."""
    try:
        manager = get_connection_manager()
        ws_message = WSMessage(
            type=WSMessageType.ERROR,
            payload={"error": error},
        )
        await manager.send_message(session_id, ws_message)
    except Exception as e:
        logger.debug(f"WebSocket send failed: {e}")


async def _run_analysis_async(session_id: str, config):
    """Async analysis execution with WebSocket progress updates."""
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        return
    
    try:
        session.status = SessionStatus.ANALYZING
        session.config = config
        store.update_session(session)
        
        # Progress: Starting
        await _send_ws_progress(session_id, 0.1, "Starting analysis...")
        await _send_ws_agent_message(session_id, AgentType.ORCHESTRATOR, "Preparing spectral ranking computation...")
        
        # Run Engine Orchestrator (blocking call in thread pool)
        orchestrator = EngineOrchestrator()
        
        await _send_ws_progress(session_id, 0.3, "Running Step 1: Vanilla spectral ranking...")
        await _send_ws_agent_message(session_id, AgentType.ORCHESTRATOR, "Executing R script for initial ranking estimation...")
        
        # Execute in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, orchestrator.execute, session, config)
        
        await _send_ws_progress(session_id, 0.7, "Analysis complete. Generating report...")
        await _send_ws_agent_message(session_id, AgentType.ANALYST, "Creating summary report...")
        
        # Generate report
        analyst = AnalystAgent()
        report = await loop.run_in_executor(None, analyst.generate_report, results, session)
        
        await _send_ws_progress(session_id, 0.9, "Finalizing results...")
        
        # Update session
        session.results = results
        session.status = SessionStatus.COMPLETED
        session.add_message("assistant", report, agent=None)
        store.update_session(session)
        
        # Send final results
        await _send_ws_progress(session_id, 1.0, "Complete!")
        await _send_ws_result(session_id, results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        session.status = SessionStatus.ERROR
        session.add_message("system", f"Error: {str(e)}", agent=None)
        store.update_session(session)
        await _send_ws_error(session_id, str(e))


def _run_analysis_sync(session_id: str, config):
    """Synchronous wrapper for async analysis execution."""
    asyncio.run(_run_analysis_async(session_id, config))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Trigger ranking analysis with user-confirmed configuration.
    
    The Engine Orchestrator will:
    1. Execute spectral_ranking_step1.R
    2. Check diagnostics (heterogeneity, sparsity)
    3. Conditionally execute spectral_ranking_step2.R
    """
    store = get_session_store()
    session = store.get_session(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.file_path:
        raise HTTPException(status_code=400, detail="No file uploaded for this session")
    
    # Start analysis in background
    background_tasks.add_task(_run_analysis_sync, request.session_id, request.config)
    
    return AnalyzeResponse(
        status=AnalysisStatus.PROCESSING,
        results=None,
        error=None,
    )


@router.get("/results/{session_id}", response_model=AnalyzeResponse)
async def get_results(session_id: str):
    """
    Retrieve analysis results for a session.
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status == SessionStatus.COMPLETED:
        return AnalyzeResponse(
            status=AnalysisStatus.COMPLETED,
            results=session.results,
            error=None,
        )
    elif session.status == SessionStatus.ERROR:
        # Get last error message
        error_msgs = [m for m in session.messages if m.role == "system" and "Error" in m.content]
        error = error_msgs[-1].content if error_msgs else "Unknown error"
        return AnalyzeResponse(
            status=AnalysisStatus.ERROR,
            results=None,
            error=error,
        )
    else:
        return AnalyzeResponse(
            status=AnalysisStatus.PROCESSING,
            results=None,
            error=None,
        )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its associated data.
    """
    store = get_session_store()
    
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "deleted", "session_id": session_id}


@router.get("/session/{session_id}/messages")
async def get_messages(session_id: str):
    """
    Get chat messages for a session.
    """
    store = get_session_store()
    session = store.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": [m.model_dump() for m in session.messages],
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask a follow-up question about the analysis results.
    
    The Analyst Agent will answer using:
    - Session context (data schema, execution trace)
    - Ranking results (if available)
    - Spectral ranking domain knowledge
    """
    store = get_session_store()
    session = store.get_session(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.results:
        raise HTTPException(
            status_code=400,
            detail="No analysis results available. Please complete the analysis first.",
        )
    
    # Add user question to session messages
    session.add_message("user", request.question, agent=None)
    
    # Get answer from Analyst Agent
    analyst = AnalystAgent()
    answer = analyst.answer_question(request.question, session.results, session)
    
    # Add answer to session messages
    session.add_message("assistant", answer, agent=AgentType.ANALYST)
    store.update_session(session)
    
    return ChatResponse(answer=answer, agent="analyst")
