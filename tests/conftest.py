"""Pytest fixtures — shared mock data store with deterministic seed."""

from __future__ import annotations


import pytest

from analyst.data import get_store


@pytest.fixture(autouse=True)
def _no_anthropic_key(monkeypatch):
    """Force template narrator in tests — no network calls allowed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture()
def store():
    return get_store()
