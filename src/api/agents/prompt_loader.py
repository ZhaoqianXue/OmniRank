"""Prompt loader for single-source system prompt governance."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re


PROMPT_PATH = Path(__file__).parent / "prompts" / "omnirank_system_prompt.md"
SECTION_PATTERN = re.compile(
    r"<!-- TOOL_SECTION:(?P<key>[a-zA-Z0-9_-]+) -->\n(?P<body>.*?)\n<!-- END_TOOL_SECTION:(?P=key) -->",
    re.DOTALL,
)


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Load and cache the OmniRank system prompt from the single source file."""
    return PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_prompt_sections() -> dict[str, str]:
    """Load all tool-specific prompt sections from the same single source file."""
    prompt_text = load_system_prompt()
    sections: dict[str, str] = {}
    for match in SECTION_PATTERN.finditer(prompt_text):
        key = match.group("key").strip()
        body = match.group("body").strip()
        sections[key] = body
    return sections


def load_prompt_section(section_key: str) -> str:
    """Load one tool-specific prompt section by key."""
    sections = _load_prompt_sections()
    if section_key not in sections:
        raise KeyError(f"Prompt section not found: {section_key}")
    return sections[section_key]
