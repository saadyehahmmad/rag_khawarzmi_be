"""
Shared configuration, constants, and logging for LangGraph pipeline nodes.

Env-driven retrieval knobs and JSON-fence regexes for rewrite+route parsing.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.env_utils import (
    env_bool,
    env_default_float,
    env_default_int,
    env_default_str,
    notify_env_fallback_once,
)
from framework.state import AgentState

load_dotenv()

# Preserve historical logger name so log filters and docs stay aligned.
logger = logging.getLogger("agent.nodes")

LOG_PATH = Path(env_default_str("LOG_PATH", "./logs"))
LOG_PATH.mkdir(parents=True, exist_ok=True)

RETRIEVAL_TOP_K = max(1, env_default_int("RETRIEVAL_TOP_K", 8))
# When listing every survey question, merge overview + all page_summary chunks (can be large).
SURVEY_CATALOG_MAX_CHUNKS = max(32, env_default_int("SURVEY_CATALOG_MAX_CHUNKS", 160))
SURVEY_CATALOG_PAGE_GET_LIMIT = max(50, env_default_int("SURVEY_CATALOG_PAGE_GET_LIMIT", 500))
_default_fetch_k = max(RETRIEVAL_TOP_K * 3, 12)
_fk_raw = os.environ.get("RETRIEVAL_FETCH_K")
if _fk_raw is None or not str(_fk_raw).strip():
    notify_env_fallback_once("RETRIEVAL_FETCH_K", _default_fetch_k)
    RETRIEVAL_FETCH_K = max(RETRIEVAL_TOP_K, _default_fetch_k)
else:
    RETRIEVAL_FETCH_K = max(RETRIEVAL_TOP_K, int(str(_fk_raw).strip(), 10))
RETRIEVAL_MMR_LAMBDA = env_default_float("RETRIEVAL_MMR_LAMBDA", 0.5)
RETRIEVAL_USE_MMR = env_bool("RETRIEVAL_USE_MMR", False)
RETRIEVAL_NORMALIZE_AR = env_bool("RETRIEVAL_NORMALIZE_AR", True)
HYBRID_DENSE_POOL = max(RETRIEVAL_TOP_K, env_default_int("HYBRID_DENSE_POOL", 24))
_RELEVANCE_MAX_L2_RAW = os.getenv("RETRIEVAL_RELEVANCE_MAX_L2", "").strip()
_RELEVANCE_MIN_CHUNKS = max(1, env_default_int("RETRIEVAL_MIN_RELEVANT_CHUNKS", 3))

_RE_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
_RE_FENCE_OPEN = re.compile(r"^```[^\n]*\n?")
_RE_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def _parse_l2_cap(raw: str) -> float | None:
    """Parse RETRIEVAL_RELEVANCE_MAX_L2 once at startup; returns None to disable the gate."""
    if not raw:
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except ValueError:
        logger.warning("Invalid RETRIEVAL_RELEVANCE_MAX_L2=%r; ignoring gate", raw)
        return None


_RELEVANCE_MAX_L2_CAP: float | None = _parse_l2_cap(_RELEVANCE_MAX_L2_RAW)

# Keyword bundle for ``hybrid_retrieve`` — single definition avoids drift across call sites.
HYBRID_RETRIEVE_KWARGS: dict[str, Any] = {
    "top_k": RETRIEVAL_TOP_K,
    "dense_pool_size": HYBRID_DENSE_POOL,
    "use_mmr_dense_pool": RETRIEVAL_USE_MMR,
    "mmr_fetch_k": RETRIEVAL_FETCH_K,
    "mmr_lambda": RETRIEVAL_MMR_LAMBDA,
}

NodeFn = Callable[[AgentState], dict[str, Any]]
