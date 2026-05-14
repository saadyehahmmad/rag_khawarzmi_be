"""
Retrieval relevance gate from chunk count and optional top-1 L2 cap.
"""

from __future__ import annotations

from .config import _RELEVANCE_MAX_L2_CAP, _RELEVANCE_MIN_CHUNKS


def relevance_from_dense_distance(
    chunks: list[str],
    best_distance: float | None,
) -> str:
    """
    Determine retrieval relevance using two complementary signals.

    Signal 1 — chunk count: if the hybrid pipeline returned at least
    ``_RELEVANCE_MIN_CHUNKS`` non-empty chunks, the corpus has enough signal
    to attempt an answer regardless of the top-1 L2 distance.

    Signal 2 — top-1 L2 distance (optional; only applied when chunk count is
    below the minimum threshold). Disabled when RETRIEVAL_RELEVANCE_MAX_L2 is unset.
    """
    if not chunks:
        return "irrelevant"
    if len(chunks) >= _RELEVANCE_MIN_CHUNKS:
        return "relevant"
    if _RELEVANCE_MAX_L2_CAP is None:
        return "relevant"
    if best_distance is None:
        return "relevant"
    return "relevant" if best_distance <= _RELEVANCE_MAX_L2_CAP else "irrelevant"
