"""
OmniRank REST API Routes
Handles file upload, analysis triggering, and results retrieval.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends

from core.schemas import (
    UploadResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    AnalysisStatus,
    InferredSchema,
    DataFormat,
    RankingResults,
)

router = APIRouter(tags=["ranking"])

# In-memory session storage (will be replaced with proper storage later)
sessions: dict = {}


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
    
    # Generate session ID
    session_id = str(uuid.uuid4())
    
    # Read file content
    content = await file.read()
    
    # Store session data
    sessions[session_id] = {
        "filename": file.filename,
        "content": content,
        "status": "uploaded",
    }
    
    # TODO: Invoke Data Agent for schema inference
    # For now, return placeholder schema
    inferred_schema = InferredSchema(
        format=DataFormat.PAIRWISE,
        bigbetter=1,
        ranking_items=["item_1", "item_2", "item_3"],  # Placeholder
        indicator_col=None,
        indicator_values=[],
        confidence=0.85,
    )
    
    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        inferred_schema=inferred_schema,
        warnings=[],
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Trigger ranking analysis with user-confirmed configuration.
    
    The Engine Orchestrator will:
    1. Execute spectral_ranking_step1.R
    2. Check diagnostics (heterogeneity, sparsity)
    3. Conditionally execute spectral_ranking_step2.R
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update session with config
    sessions[session_id]["config"] = request.config
    sessions[session_id]["status"] = "analyzing"
    
    # TODO: Invoke Engine Orchestrator
    # For now, return processing status
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    status = session.get("status", "unknown")
    
    if status == "completed":
        return AnalyzeResponse(
            status=AnalysisStatus.COMPLETED,
            results=session.get("results"),
            error=None,
        )
    elif status == "error":
        return AnalyzeResponse(
            status=AnalysisStatus.ERROR,
            results=None,
            error=session.get("error"),
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
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "deleted", "session_id": session_id}
