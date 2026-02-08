"""Prompt loader for single-source system prompt governance."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_PATH = Path(__file__).parent / "prompts" / "omnirank_system_prompt.md"


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Load and cache the OmniRank system prompt from the single source file."""
    return PROMPT_PATH.read_text(encoding="utf-8")
