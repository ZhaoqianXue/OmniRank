"""
OmniRank FastAPI Backend
Main application entry point.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.websocket import websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print("OmniRank API starting up...")
    yield
    # Shutdown
    print("OmniRank API shutting down...")


app = FastAPI(
    title="OmniRank API",
    description="LLM Agent Platform for Spectral Ranking Inference",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api")
app.include_router(websocket_router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "omnirank-api"}


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "r_available": _check_r_available(),
    }


def _check_r_available() -> bool:
    """Check if R is available in the system."""
    import subprocess
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
