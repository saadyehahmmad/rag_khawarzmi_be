"""
Al-Khawarzmi prescript provider — implements ``framework.profile.PrescriptProvider``.

Wraps ``alkawarzmi.payload_context.payload_context_step`` so it can be composed into an ``RAGProfile``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alkawarzmi.payload_context import payload_context_step as _payload_context_step

if TYPE_CHECKING:
    from framework.state import AgentState


class AlKhawarzmiPrescripts:
    """
    Zero-LLM fast-path handler for the Al-Khawarzmi platform.

    Resolves where-am-I, designer navigation, and named-greeting turns
    entirely from the HTTP payload — no retrieval or LLM call needed.
    """

    def run(self, state: "AgentState") -> dict[str, str]:
        """
        Inspect the client payload context and return a state update.

        Returns:
            ``{"prescripted_answer": "<text>"}`` when a fast-path answer is available,
            or ``{}`` when normal pipeline processing should continue.
        """
        return _payload_context_step(state)
