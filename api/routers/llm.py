"""
LLM discovery routes (no chat inference).

Exposes:
  GET /llm/config          — active provider, resolved model, Ollama base URL
  GET /llm/ollama/models   — tags from the local Ollama daemon (``/api/tags``)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.llm_helpers import llm_config_snapshot
from core.ollama_models import list_local_ollama_models

router = APIRouter(tags=["llm"])


@router.get("/llm/config")
def llm_config() -> dict[str, Any]:
    """Return non-secret LLM settings so clients can match the server (and probe Ollama)."""
    return llm_config_snapshot()


@router.get("/llm/ollama/models")
def ollama_models() -> dict[str, Any]:
    """
    List models reported by Ollama at ``OLLAMA_BASE_URL``.

    Does not require ``LLM_PROVIDER=ollama`` — useful for UIs that let operators pick a local model.
    """
    return list_local_ollama_models()
