"""OmniRank FastAPI entrypoint (single-agent SOP implementation)."""

from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from core.schemas import HealthResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _load_cors_origins() -> list[str]:
    """Parse CORS origins from env var or fallback to local defaults."""
    raw = os.getenv("CORS_ORIGINS", "")
    if not raw.strip():
        return list(DEFAULT_CORS_ORIGINS)
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    # Preserve insertion order while removing duplicates.
    return list(dict.fromkeys(origins)) or list(DEFAULT_CORS_ORIGINS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Service lifecycle hooks."""
    logger.info("OmniRank API starting up")
    yield
    logger.info("OmniRank API shutting down")


app = FastAPI(
    title="OmniRank API",
    description="Single-agent spectral ranking platform",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception):
    """Global error handler."""
    logger.error("Unhandled exception: %s", exc)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred.",
            "error_type": type(exc).__name__,
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"status": "ok", "service": "omnirank-api"}


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health endpoint with R availability."""
    return HealthResponse(status="healthy", version="0.2.0", r_available=_check_r_available())


def _check_r_available() -> bool:
    """Check if Rscript binary is available."""
    import subprocess

    try:
        result = subprocess.run(["Rscript", "--version"], capture_output=True, timeout=5, check=False)
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
