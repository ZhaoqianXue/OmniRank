"""Tests for runtime/deployment-related environment behavior."""

from __future__ import annotations

from pathlib import Path

from core.session_memory import SESSION_DIR_ENV_KEY, SessionStore
from main import DEFAULT_CORS_ORIGINS, _load_cors_origins


def test_load_cors_origins_uses_defaults_when_env_missing(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert _load_cors_origins() == list(DEFAULT_CORS_ORIGINS)


def test_load_cors_origins_parses_and_deduplicates(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        " https://a.example.com/ , https://b.example.com,https://a.example.com ",
    )
    assert _load_cors_origins() == ["https://a.example.com", "https://b.example.com"]


def test_session_store_uses_env_session_dir(tmp_path, monkeypatch):
    configured = tmp_path / "render_sessions"
    monkeypatch.setenv(SESSION_DIR_ENV_KEY, str(configured))

    store = SessionStore()
    assert store._temp_dir == configured
    assert configured.exists()


def test_session_store_explicit_temp_dir_overrides_env(tmp_path, monkeypatch):
    configured = tmp_path / "from_env"
    explicit = tmp_path / "from_arg"
    monkeypatch.setenv(SESSION_DIR_ENV_KEY, str(configured))

    store = SessionStore(temp_dir=explicit)
    assert store._temp_dir == explicit
    assert explicit.exists()
    assert not Path(configured).exists()
