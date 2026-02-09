"""OmniRank agent package."""

from typing import TYPE_CHECKING

__all__ = ["OmniRankAgent"]

if TYPE_CHECKING:
    from .omnirank_agent import OmniRankAgent


def __getattr__(name: str):
    """Lazy import to avoid package-level circular imports."""
    if name == "OmniRankAgent":
        from .omnirank_agent import OmniRankAgent

        return OmniRankAgent
    raise AttributeError(name)
