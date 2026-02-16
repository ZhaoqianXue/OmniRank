"""Per-user daily usage tracker based on OpenAI response token usage."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
from threading import Lock
import time
from typing import Any


_ACTIVE_USER_SUB: ContextVar[str | None] = ContextVar("omnirank_usage_user_sub", default=None)
DEFAULT_DAILY_LIMIT_USD = 0.25


def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _to_float(value: Any, fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if parsed < 0:
        return fallback
    return parsed


def _default_model_prices() -> dict[str, dict[str, float]]:
    """Fallback prices (USD per 1M tokens). Can be overridden by env JSON."""
    return {
        "gpt-5": {"input_per_1m": 1.25, "output_per_1m": 10.0},
        "gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.0},
        "gpt-5-nano": {"input_per_1m": 0.05, "output_per_1m": 0.4},
    }


def _load_model_prices() -> dict[str, dict[str, float]]:
    raw = os.getenv("OMNIRANK_MODEL_PRICES_JSON", "").strip()
    if not raw:
        return _default_model_prices()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _default_model_prices()

    if not isinstance(parsed, dict):
        return _default_model_prices()

    prices: dict[str, dict[str, float]] = {}
    for model, value in parsed.items():
        if not isinstance(model, str) or not isinstance(value, dict):
            continue
        input_per_1m = _to_float(value.get("input_per_1m"), -1.0)
        output_per_1m = _to_float(value.get("output_per_1m"), -1.0)
        if input_per_1m < 0 or output_per_1m < 0:
            continue
        prices[model.strip()] = {
            "input_per_1m": input_per_1m,
            "output_per_1m": output_per_1m,
        }

    return prices or _default_model_prices()


@dataclass
class DailyUsageTotals:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    used_usd: float = 0.0
    unpriced_calls: int = 0
    updated_at_unix: float = 0.0


class UsageTracker:
    """Thread-safe in-memory usage ledger."""

    def __init__(self):
        self._daily_limit_usd = _to_float(os.getenv("OMNIRANK_DAILY_LIMIT_USD"), DEFAULT_DAILY_LIMIT_USD)
        self._model_prices = _load_model_prices()
        self._lock = Lock()
        self._daily_usage: dict[str, dict[str, DailyUsageTotals]] = {}

    @property
    def daily_limit_usd(self) -> float:
        return self._daily_limit_usd

    def _resolve_model_pricing(self, model: str) -> dict[str, float] | None:
        direct = self._model_prices.get(model)
        if direct is not None:
            return direct

        for key, value in self._model_prices.items():
            if model.startswith(f"{key}-"):
                return value
        return None

    def _cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
        pricing = self._resolve_model_pricing(model)
        if pricing is None:
            return 0.0, False

        input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
        return max(0.0, input_cost + output_cost), True

    def record_response_usage(self, user_sub: str | None, model: str, usage: Any) -> None:
        if not user_sub:
            return

        if isinstance(usage, dict):
            input_tokens = _to_int(usage.get("input_tokens"))
            output_tokens = _to_int(usage.get("output_tokens"))
            total_tokens = _to_int(usage.get("total_tokens"))
        else:
            input_tokens = _to_int(getattr(usage, "input_tokens", 0))
            output_tokens = _to_int(getattr(usage, "output_tokens", 0))
            total_tokens = _to_int(getattr(usage, "total_tokens", 0))

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
            return

        cost_usd, priced = self._cost_usd(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        day_key = _today_key()
        now = time.time()

        with self._lock:
            user_bucket = self._daily_usage.setdefault(user_sub, {})
            totals = user_bucket.setdefault(day_key, DailyUsageTotals())
            totals.input_tokens += input_tokens
            totals.output_tokens += output_tokens
            totals.total_tokens += total_tokens
            totals.used_usd += cost_usd
            if not priced:
                totals.unpriced_calls += 1
            totals.updated_at_unix = now

            # Keep only recent days to bound memory.
            if len(user_bucket) > 60:
                for stale_day in sorted(user_bucket.keys())[:-60]:
                    user_bucket.pop(stale_day, None)

    def get_daily_snapshot(self, user_sub: str | None) -> dict[str, Any]:
        day_key = _today_key()
        zero_snapshot = {
            "date": day_key,
            "limit_usd": self._daily_limit_usd,
            "used_usd": 0.0,
            "progress_percent": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "pricing_configured": True,
        }
        if not user_sub:
            return zero_snapshot

        with self._lock:
            totals = self._daily_usage.get(user_sub, {}).get(day_key)
            if totals is None:
                return zero_snapshot

            used_usd = max(0.0, totals.used_usd)
            if self._daily_limit_usd > 0:
                progress_percent = min(100.0, (used_usd / self._daily_limit_usd) * 100.0)
            else:
                progress_percent = 0.0

            return {
                "date": day_key,
                "limit_usd": self._daily_limit_usd,
                "used_usd": used_usd,
                "progress_percent": progress_percent,
                "input_tokens": max(0, totals.input_tokens),
                "output_tokens": max(0, totals.output_tokens),
                "total_tokens": max(0, totals.total_tokens),
                "pricing_configured": totals.unpriced_calls == 0,
            }


@contextmanager
def usage_user_scope(user_sub: str | None):
    token = _ACTIVE_USER_SUB.set(user_sub)
    try:
        yield
    finally:
        _ACTIVE_USER_SUB.reset(token)


def get_active_usage_user_sub() -> str | None:
    return _ACTIVE_USER_SUB.get()


_tracker: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker

