"""Immutable registry of OmniRank tool implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .answer_question import answer_question
from .execute_spectral_ranking import execute_spectral_ranking
from .generate_report import generate_report
from .generate_visualizations import generate_visualizations
from .infer_semantic_schema import infer_semantic_schema
from .preprocess_data import preprocess_data
from .read_data_file import read_data_file
from .request_user_confirmation import request_user_confirmation
from .validate_data_format import validate_data_format
from .validate_data_quality import validate_data_quality


ToolFn = Callable[..., object]


@dataclass(frozen=True)
class ToolRegistry:
    """Fixed registry for the ten OmniRank tools."""

    tools: Mapping[str, ToolFn]

    def get(self, name: str) -> ToolFn:
        """Fetch tool callable by exact name."""
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def list_names(self) -> list[str]:
        """Return stable tool list."""
        return list(self.tools.keys())


def build_tool_registry() -> ToolRegistry:
    """Construct immutable tool registry."""
    tools: dict[str, ToolFn] = {
        "read_data_file": read_data_file,
        "infer_semantic_schema": infer_semantic_schema,
        "validate_data_format": validate_data_format,
        "validate_data_quality": validate_data_quality,
        "preprocess_data": preprocess_data,
        "request_user_confirmation": request_user_confirmation,
        "execute_spectral_ranking": execute_spectral_ranking,
        "generate_report": generate_report,
        "generate_visualizations": generate_visualizations,
        "answer_question": answer_question,
    }
    return ToolRegistry(tools=tools)
