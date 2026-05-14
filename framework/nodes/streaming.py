"""
SSE-oriented async token streaming for grounded answers and static fallbacks.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from core.llm_helpers import get_llm
from core.text_ar import normalize_arabic_answer_text
from framework.state import AgentState

from .answer_prompt import build_answer_prompt
from .fallback_text import fallback_answer_text


async def astream_answer_tokens(state: AgentState) -> AsyncIterator[str]:
    """Stream model token deltas for the grounded answer (SSE /chat)."""
    prescripted = state.get("prescripted_answer")
    if prescripted:
        for line in str(prescripted).splitlines(keepends=True):
            if line:
                yield line
                await asyncio.sleep(0.02)
        return

    llm = get_llm()
    messages = [HumanMessage(content=build_answer_prompt(state))]
    pending = ""
    lang = state.get("language", "en")

    def _fix_ar(s: str) -> str:
        if lang in ("ar", "mixed"):
            return normalize_arabic_answer_text(s)
        return s

    async for chunk in llm.astream(messages):
        c = getattr(chunk, "content", None)
        parts: list[str] = []
        if isinstance(c, str) and c:
            parts = [c]
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    t = str(p.get("text", ""))
                    if t:
                        parts.append(t)

        for part in parts:
            if not part:
                continue
            pending += part
            last_ws = max(pending.rfind(" "), pending.rfind("\n"), pending.rfind("\r"))
            if last_ws >= 0:
                to_emit = pending[: last_ws + 1]
                pending = pending[last_ws + 1 :]
                yield _fix_ar(to_emit)

    if pending:
        yield _fix_ar(pending)


async def astream_fallback_tokens(state: AgentState) -> AsyncIterator[str]:
    """Stream the fallback / greeting message line by line (SSE /chat)."""
    text = fallback_answer_text(state)
    for line in text.splitlines(keepends=True):
        if line.strip():
            yield line
            await asyncio.sleep(0.04)
