"""
Post-LLM output safety layer for the RAG answer pipeline.

Performs lightweight, rule-based checks on the generated answer before it is
returned to the user.  All checks are fast (no LLM calls) and non-blocking by
default: findings are emitted as warnings so operators can monitor them.

Set ``SAFETY_STRICT=true`` to substitute the fallback message whenever a flag
is raised.  Leave unset (default) for monitoring-only mode where the answer is
still returned but the flag is recorded in the application log.

Checks performed:
  1. Hallucination self-disclosure — known LLM self-disclosure phrases.
  2. Cross-system leakage — answer makes authoritative claims about a different
     system than the one retrieved for.
  3. Citation overflow — answer cites ``[n]`` where n > number of retrieved chunks.
  4. Empty answer — the LLM returned a blank string.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.env_utils import env_bool

logger = logging.getLogger(__name__)

# When true, answer_node substitutes the fallback message on any safety flag.
SAFETY_STRICT: bool = env_bool("SAFETY_STRICT", False)

_VALID_SYSTEMS = frozenset({"designer", "runtime", "callcenter", "admin"})

# Phrases that signal the LLM admitted it lacked grounding.
_HALLUCINATION_SIGNALS = (
    "i made up",
    "i invented",
    "i'm not sure this is accurate",
    "i cannot verify",
    "this is hypothetical",
    "i don't actually know",
    "fabricated",
)

# Action verbs used to detect authoritative cross-system instructions.
_ACTION_VERBS = ("you must", "you should", "you need to", "يجب", "ينبغي", "تحتاج")

# Regex: citation references like [1], [12] in the answer.
_RE_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class SafetyResult:
    """Result of a post-answer safety check."""

    safe: bool
    flags: list[str] = field(default_factory=list)

    def log(self, request_id: str = "") -> None:
        """Emit a structured warning for every flag found."""
        for flag in self.flags:
            logger.warning("output_safety flag [rid=%s]: %s", request_id or "-", flag)


def check_answer(
    answer: str,
    *,
    system: str | None,
    chunks: list[str],
    request_id: str = "",
) -> SafetyResult:
    """
    Run all post-LLM safety checks and return a consolidated result.

    Args:
        answer: The full answer string produced by the LLM.
        system: The routed system name (e.g. ``"designer"``); ``None`` for fallback path.
        chunks: Retrieved chunk texts, parallel to citation indices ``[1]``, ``[2]``, …
        request_id: Correlation ID forwarded to log lines.

    Returns:
        :class:`SafetyResult` with ``safe=True`` when no issues are found.
        Flags are always logged as warnings regardless of ``SAFETY_STRICT``.
    """
    flags: list[str] = []
    lower = answer.lower()

    # 1. Hallucination self-disclosure
    for signal in _HALLUCINATION_SIGNALS:
        if signal in lower:
            flags.append(f"hallucination_signal: matched phrase '{signal}'")

    # 2. Cross-system leakage — only flag when the routed system is known and the
    #    answer makes an *authoritative claim* about a different system (not a
    #    mere cross-reference).  Heuristic: "in the <other> system" + action verb
    #    within a 70-character window.
    if system and system in _VALID_SYSTEMS:
        for other in _VALID_SYSTEMS - {system}:
            pattern = f"in the {other}"
            pos = lower.find(pattern)
            if pos >= 0:
                window = lower[max(0, pos - 10): pos + 70]
                if any(v in window for v in _ACTION_VERBS):
                    flags.append(
                        f"cross_system_leakage: answer gives instructions for '{other}' "
                        f"while routed to '{system}'"
                    )

    # 3. Citation overflow — [n] where n > len(chunks)
    cited_indices = {int(m) for m in _RE_CITATION.findall(answer)}
    max_valid = len(chunks)
    overflow = {idx for idx in cited_indices if idx > max_valid}
    if overflow:
        flags.append(
            f"citation_overflow: answer cites indices {sorted(overflow)} but only "
            f"{max_valid} chunk(s) were retrieved"
        )

    # 4. Empty answer
    if not answer.strip():
        flags.append("empty_answer: LLM returned a blank response")

    result = SafetyResult(safe=not flags, flags=flags)
    if flags:
        result.log(request_id)
    return result
