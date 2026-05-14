"""
Semantic image selection from docs/screens.json.

Matches the LLM-rewritten question against pre-computed flow embeddings to find
the most relevant UI screenshots for a given system.  Uses the same embedding
singleton as Chroma retrieval so no additional model overhead is incurred.

Cache invalidation: screens.json mtime is checked on every call (one cheap
syscall).  Both caches are rebuilt whenever the file is saved on disk, so new
flows are picked up without a server restart.
"""

from __future__ import annotations

import logging

from core.env_utils import env_default_float
from core.paths import SCREENS_JSON_PATH, load_screens_json, resolve_flow_images

logger = logging.getLogger(__name__)


def _float_env_or_default(name: str, default: float) -> float:
    """Parse ``name`` as float with ``env_default_float``; on invalid value log, print, and return ``default``."""
    try:
        return env_default_float(name, default)
    except ValueError:
        msg = f"Invalid {name} - using {default!r}"
        logger.warning(msg)
        print(f"[env] {msg}", flush=True)
        return default


# Minimum cosine similarity (0–1) to accept a semantic flow match.
# Normalised E5 embeddings → dot-product == cosine similarity.
SEMANTIC_FLOW_THRESHOLD: float = _float_env_or_default("SEMANTIC_FLOW_THRESHOLD", 0.65)

# Minimum margin between best and runner-up similarity.
# Helps avoid picking an unrelated flow when multiple flows are similarly "kinda close"
# (e.g. generic "language" / "settings" phrasing).
SEMANTIC_FLOW_MARGIN: float = _float_env_or_default("SEMANTIC_FLOW_MARGIN", 0.03)

# Module-level caches — rebuilt on mtime change without a server restart.
_SCREENS_DATA: dict | None = None
_SCREENS_MTIME: float = 0.0
_FLOW_EMBEDDINGS: dict[str, list[float]] | None = None


def _build_flow_text(flow: dict) -> str:
    """
    Build the embedding text for a flow from its bilingual description.

    Both ``desc.en`` and ``desc.ar`` are concatenated so the resulting vector
    covers user queries in either language without maintaining a keyword list.
    """
    desc = flow.get("desc", {})
    if isinstance(desc, dict):
        return f"{desc.get('en', '')} {desc.get('ar', '')}".strip()
    return str(desc).strip()


def _ensure_flow_embeddings(flows: list[dict]) -> None:
    """
    Build and cache a semantic embedding for every flow (once per process).

    Uses a deferred import of ``get_embeddings`` to avoid a circular dependency
    (image_selection is imported by nodes, which is imported by llm_helpers init).
    Embeddings are batched in a single model pass.
    """
    global _FLOW_EMBEDDINGS
    if _FLOW_EMBEDDINGS is not None:
        return

    from core.llm_helpers import get_embeddings  # deferred to avoid circular import

    emb = get_embeddings()
    _FLOW_EMBEDDINGS = {}
    texts = [_build_flow_text(f) for f in flows]
    vectors = emb.embed_documents(texts)
    for flow, vector in zip(flows, vectors):
        _FLOW_EMBEDDINGS[flow["id"]] = vector
    logger.info("Built semantic embeddings for %d flow(s)", len(_FLOW_EMBEDDINGS))


def _load_screens_data() -> dict:
    """
    Load and cache docs/screens.json, auto-invalidating when the file changes.

    Returns an empty dict when the file is absent or malformed; that result is
    intentionally not cached so the next request retries.
    """
    global _SCREENS_DATA, _SCREENS_MTIME, _FLOW_EMBEDDINGS

    try:
        current_mtime = SCREENS_JSON_PATH.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    if _SCREENS_DATA is not None and current_mtime == _SCREENS_MTIME:
        return _SCREENS_DATA

    if _SCREENS_DATA is not None:
        logger.info("screens.json changed on disk — reloading caches.")
    _SCREENS_DATA = None
    _FLOW_EMBEDDINGS = None

    data = load_screens_json()
    if data:
        logger.info("Loaded screens.json from %s (mtime=%.0f)", SCREENS_JSON_PATH, current_mtime)
        _SCREENS_DATA = data
        _SCREENS_MTIME = current_mtime
    else:
        return {}
    return _SCREENS_DATA


def select_images_for_question(
    rewritten_question: str,
    system: str,
    language: str = "en",
) -> list[str]:
    """
    Return image filenames for the semantically closest flow in screens.json.

    Only the LLM-rewritten question is embedded for matching.  The rewrite node
    always produces a fully resolved standalone query, so the raw user text is
    intentionally excluded to avoid noise from short follow-up phrases like
    "please in English" or "what was step 1?".

    Args:
        rewritten_question: LLM-resolved standalone question used for retrieval.
        system: Active system scope (e.g. ``"designer"``).
        language: Detected prompt language forwarded to ``resolve_flow_images``
                  for language-keyed image paths.

    Returns:
        List of image filenames to be served under ``/images/``.
    """
    from core.llm_helpers import get_embeddings  # deferred to avoid circular import

    screens = _load_screens_data()
    if not screens:
        return []

    flows: list[dict] = screens.get("flows", [])
    image_registry: dict = screens.get("images", {})
    if not flows or not image_registry:
        return []

    try:
        _ensure_flow_embeddings(flows)
        if not _FLOW_EMBEDDINGS:
            return []

        q_vec = get_embeddings().embed_query(rewritten_question)
        best_flow: dict | None = None
        best_sim = 0.0
        second_sim = 0.0

        for flow in flows:
            flow_system = flow.get("system", "")
            if flow_system and flow_system != system:
                continue
            fvec = _FLOW_EMBEDDINGS.get(flow["id"])
            if fvec is None:
                continue
            # Dot-product of two L2-normalised vectors == cosine similarity.
            sim = sum(a * b for a, b in zip(q_vec, fvec))
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_flow = flow
            elif sim > second_sim:
                second_sim = sim

        if best_flow is None or best_sim < SEMANTIC_FLOW_THRESHOLD:
            logger.debug(
                "  Image selection: no match (best_sim=%.3f threshold=%.3f)",
                best_sim,
                SEMANTIC_FLOW_THRESHOLD,
            )
            return []

        margin = best_sim - second_sim
        if margin < SEMANTIC_FLOW_MARGIN:
            logger.debug(
                "  Image selection: ambiguous match (best_sim=%.3f second_sim=%.3f margin=%.3f < %.3f)",
                best_sim,
                second_sim,
                margin,
                SEMANTIC_FLOW_MARGIN,
            )
            return []

        result = resolve_flow_images(best_flow, image_registry, language=language)
        logger.info(
            "  Image selection: flow=%s sim=%.3f lang=%s images=%s",
            best_flow.get("id"),
            best_sim,
            language,
            result,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning("Image selection failed: %s", exc)
        return []
