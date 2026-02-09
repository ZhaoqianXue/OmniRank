"""LLM client wrapper for OmniRank LLM-native tools."""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.prompt_loader import load_prompt_section, load_system_prompt


JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*\})\s*```", re.DOTALL)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails or returns invalid content."""


class OmniLLMClient:
    """Thin wrapper around OpenAI chat completions for JSON-first tool calls."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ):
        configured_model = model or os.getenv("OPENAI_MODEL")
        self.model = configured_model or "gpt-5-mini"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key, timeout=timeout_seconds) if self.api_key else None

    def is_available(self) -> bool:
        """Return True when API key is available."""
        return self.client is not None

    def generate_json(
        self,
        section_key: str,
        payload: dict[str, Any],
        max_completion_tokens: int = 1000,
    ) -> dict[str, Any]:
        """Call model and return parsed JSON payload."""
        if self.client is None:
            raise LLMCallError("OPENAI_API_KEY is required for LLM-native tools.")

        section_prompt = load_prompt_section(section_key)
        system_prompt = load_system_prompt()
        user_prompt = (
            "Follow this tool section exactly and return strict JSON only.\n\n"
            f"[Tool Section: {section_key}]\n{section_prompt}\n\n"
            f"[Input Payload JSON]\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=max_completion_tokens,
                    reasoning={"effort": "minimal"},
                )
                content = self._extract_content(response)
                parsed = self._parse_json(content)
                if not isinstance(parsed, dict):
                    raise LLMCallError("Model response JSON must be an object.")
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(0.6 * (attempt + 1))

        message = self._normalize_error(last_error) if last_error is not None else "Unknown LLM error."
        raise LLMCallError(message)

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract assistant content from OpenAI response object."""
        response_text = getattr(response, "output_text", None)
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()

        output_items = getattr(response, "output", None)
        if isinstance(output_items, list):
            for item in output_items:
                if getattr(item, "type", None) != "message":
                    continue
                content_items = getattr(item, "content", None)
                if isinstance(content_items, list):
                    for content_item in content_items:
                        text = None
                        if isinstance(content_item, dict):
                            text = content_item.get("text")
                        else:
                            text = getattr(content_item, "text", None)
                        if isinstance(text, str) and text.strip():
                            return text.strip()

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMCallError("LLM response has no choices.")

        message = choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for entry in content:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(entry, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            joined = "".join(parts).strip()
            if joined:
                return joined

        raise LLMCallError("LLM response content is empty.")

    @staticmethod
    def _parse_json(content: str) -> Any:
        """Parse model content as JSON with fenced-block fallback."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        fenced = JSON_BLOCK_PATTERN.search(content)
        if fenced:
            return json.loads(fenced.group(1))

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])

        raise LLMCallError("LLM response is not valid JSON.")

    @staticmethod
    def _normalize_error(exc: Exception) -> str:
        """Normalize common OpenAI/API errors into one concise message."""
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return f"LLM request failed (status {status_code}): {exc}"
        return f"LLM request failed: {exc}"


@lru_cache(maxsize=1)
def get_llm_client() -> OmniLLMClient:
    """Get process-level LLM client singleton."""
    return OmniLLMClient()
