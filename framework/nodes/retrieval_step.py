"""
Hybrid retrieval node: survey-scoped merge, image URLs, relevance metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core import session_store
from core.llm_helpers import get_vector_store
from framework.survey_store import get_survey_vector_store
from core.retrieval import (
    RETRIEVAL_HYBRID,
    documents_to_chunks_and_refs,
    hybrid_retrieve,
)
from core.text_ar import normalize_arabic_question
from framework.state import AgentState

from .answer_prompt import effective_retrieval_query
from core.nodes.config import (
    HYBRID_RETRIEVE_KWARGS,
    RETRIEVAL_NORMALIZE_AR,
    SURVEY_CATALOG_MAX_CHUNKS,
    SURVEY_CATALOG_PAGE_GET_LIMIT,
    logger,
)
from core.nodes.relevance import relevance_from_dense_distance

if TYPE_CHECKING:
    from framework.profile import SurveyRetrievalHooks

_survey_hooks: SurveyRetrievalHooks | None = None


def configure_retrieval_hooks(hooks: SurveyRetrievalHooks | None) -> None:
    """Bind survey/designer hooks from :func:`configure_pipeline`."""
    global _survey_hooks
    _survey_hooks = hooks


def _survey_retrieval() -> SurveyRetrievalHooks:
    """Return configured hooks, defaulting to Al-Khawarzmi when ``build_graph(None)``."""
    global _survey_hooks
    if _survey_hooks is None:
        from alkawarzmi.survey_retrieval import AlKhawarzmiSurveyRetrievalHooks

        _survey_hooks = AlKhawarzmiSurveyRetrievalHooks()
    return _survey_hooks

# Chroma ``chunk_type`` values for docs under ``vector_stores/surveys/<id>/`` (not designer manuals).
_SURVEY_INDEX_CHUNK_TYPES = frozenset({"survey_overview", "page_summary", "question", "rule"})


def _empty_retrieval(
    relevance: str,
    *,
    survey_context_missing: bool = False,
    survey_ingesting: bool = False,
) -> dict[str, Any]:
    """Shared shape for early-exit retrieval updates (prescripted, off-topic, missing survey)."""
    return {
        "retrieved_chunks": [],
        "retrieved_source_refs": [],
        "retrieval_best_distance": None,
        "relevance": relevance,
        "image_urls": [],
        "survey_context_missing": survey_context_missing,
        "survey_ingesting": survey_ingesting,
        "survey_vector_context_used": False,
        "survey_index_absent": False,
    }


def _finalize_retrieval(
    state: AgentState,
    docs: list[Any],
    best_d: float | None,
    rq: str,
    lang: str,
    image_system: str,
    *,
    log_survey_merge: bool = False,
) -> dict[str, Any]:
    """Chunks + relevance + image URLs + logging shared by survey-merge and standard paths."""
    chunks, refs = documents_to_chunks_and_refs(docs)
    relevance = relevance_from_dense_distance(chunks, best_d)
    image_query = rq or (state.get("rewritten_question") or "")
    h = _survey_retrieval()
    selected_fnames = h.select_images_for_question(image_query, image_system, language=lang)
    image_urls = [f"/images/{fname}" for fname in selected_fnames]

    if log_survey_merge:
        logger.info(
            "  Retrieved %s chunks (survey+designer) best_l2=%s relevance=%s images=%s",
            len(chunks),
            best_d,
            relevance,
            len(image_urls),
        )
    else:
        logger.info(
            "  Retrieved %s chunks hybrid=%s best_l2=%s relevance=%s images=%s",
            len(chunks),
            RETRIEVAL_HYBRID,
            best_d,
            relevance,
            len(image_urls),
        )

    out: dict[str, Any] = {
        "retrieved_chunks": chunks,
        "retrieved_source_refs": refs,
        "retrieval_best_distance": best_d,
        "relevance": relevance,
        "image_urls": image_urls,
        "survey_context_missing": False,
        "survey_ingesting": False,
        "survey_vector_context_used": False,
        "survey_index_absent": False,
    }
    return out


def retrieval_node(state: AgentState) -> dict[str, Any]:
    """Hybrid retrieve + top-1 dense distance for relevance gating (no LLM grader)."""
    if state.get("prescripted_answer"):
        logger.info("Node: retrieval skipped — prescripted answer (payload context)")
        return _empty_retrieval("relevant", survey_context_missing=False, survey_ingesting=False)

    system = state.get("system") or "designer"
    if system == "none":
        logger.info("Node: retrieval skipped — question routed as off-topic")
        return _empty_retrieval("irrelevant", survey_context_missing=False, survey_ingesting=False)

    logger.info("Node: retrieval + relevance (%s)", system)
    rq = effective_retrieval_query(state)
    lang = state.get("language", "en")
    if RETRIEVAL_NORMALIZE_AR and lang in ("ar", "mixed"):
        rq = normalize_arabic_question(rq)

    survey_id = state.get("survey_id")
    if survey_id and system == "designer":
        from alkawarzmi.ingestion.survey_session import survey_store_has_embeddings, survey_store_is_ingesting

        h = _survey_retrieval()
        st = session_store.get_status(survey_id)
        # Check both in-memory state (single-worker) AND disk marker (multi-worker / post-restart).
        ingesting = bool(st and st.get("status") == "ingesting") or survey_store_is_ingesting(survey_id)
        ready = session_store.is_ready(survey_id)
        disk = survey_store_has_embeddings(survey_id)

        if ingesting:
            logger.info(
                "Node: survey %s still ingesting — wait path (empty retrieval until ready)",
                survey_id,
            )
            return _empty_retrieval(
                "irrelevant",
                survey_context_missing=True,
                survey_ingesting=True,
            )

        if not ready and not disk:
            logger.info(
                "Node: survey %s not in session and no on-disk Chroma — designer-only retrieval (survey_index_absent)",
                survey_id,
            )
            designer_store = get_vector_store("designer")
            designer_docs, designer_best_d = hybrid_retrieve(designer_store, rq, **HYBRID_RETRIEVE_KWARGS)
            out = _finalize_retrieval(
                state,
                designer_docs,
                designer_best_d,
                rq,
                lang,
                "designer",
                log_survey_merge=False,
            )
            out["survey_index_absent"] = True
            return out

        if not ready and disk:
            logger.info(
                "Node: survey %s session not ready but on-disk index present — retrieving",
                survey_id,
            )

        logger.info("Node: retrieving from survey collection (survey_id=%s)", survey_id)
        survey_store = get_survey_vector_store(survey_id)

        if h.wants_survey_rules_catalog(state):
            catalog = h.fetch_survey_all_rules(
                survey_store,
                limit=SURVEY_CATALOG_PAGE_GET_LIMIT,
            )
            if len(catalog) >= 1:
                survey_docs = catalog
                survey_best_d = 0.25
                logger.info(
                    "Node: survey rules-catalog mode — %s rule chunks",
                    len(survey_docs),
                )
            else:
                survey_docs, survey_best_d = hybrid_retrieve(survey_store, rq, **HYBRID_RETRIEVE_KWARGS)
                logger.info("Node: rules catalog get empty; falling back to hybrid for survey")
        elif h.wants_survey_question_catalog(state):
            catalog = h.fetch_survey_overview_and_pages(
                survey_store,
                limit=SURVEY_CATALOG_PAGE_GET_LIMIT,
            )
            if len(catalog) >= 2:
                survey_docs = catalog
                survey_best_d = 0.25
                logger.info(
                    "Node: survey question-catalog mode — %s layout chunks (overview+all pages)",
                    len(survey_docs),
                )
            else:
                survey_docs, survey_best_d = hybrid_retrieve(survey_store, rq, **HYBRID_RETRIEVE_KWARGS)
                logger.info(
                    "Node: catalog get returned <%s chunks; falling back to hybrid for survey",
                    2,
                )
        elif h.wants_survey_overview(state):
            # Fetch the overview chunk first, then append hybrid results so the model sees
            # the top-level survey description before any page/question detail.
            overview_docs = h.fetch_survey_overview_only(survey_store, limit=5)
            hybrid_docs, hybrid_best_d = hybrid_retrieve(survey_store, rq, **HYBRID_RETRIEVE_KWARGS)
            if overview_docs:
                seen_ids = {id(d) for d in overview_docs}
                extra = [d for d in hybrid_docs if id(d) not in seen_ids]
                survey_docs = overview_docs + extra
                survey_best_d = 0.25
                logger.info(
                    "Node: survey overview mode — overview=%s + hybrid=%s chunks",
                    len(overview_docs),
                    len(extra),
                )
            else:
                survey_docs, survey_best_d = hybrid_docs, hybrid_best_d
                logger.info("Node: overview get empty; falling back to hybrid for survey")
        else:
            survey_docs, survey_best_d = hybrid_retrieve(survey_store, rq, **HYBRID_RETRIEVE_KWARGS)

        designer_store = get_vector_store("designer")
        designer_docs, designer_best_d = hybrid_retrieve(designer_store, rq, **HYBRID_RETRIEVE_KWARGS)

        seen_ids = {id(d) for d in survey_docs}
        designer_tail = [d for d in designer_docs if id(d) not in seen_ids]
        room = max(0, SURVEY_CATALOG_MAX_CHUNKS - len(survey_docs))
        d_take = max(HYBRID_RETRIEVE_KWARGS["top_k"], room)
        merged = survey_docs + designer_tail[:d_take]
        if len(merged) > SURVEY_CATALOG_MAX_CHUNKS:
            keep_designer = max(0, SURVEY_CATALOG_MAX_CHUNKS - len(survey_docs))
            merged = survey_docs + designer_tail[:keep_designer]
        docs = merged
        best_d = survey_best_d if survey_best_d is not None else designer_best_d
        survey_vector_used = any(
            (d.metadata or {}).get("chunk_type") in _SURVEY_INDEX_CHUNK_TYPES for d in docs
        )

        out = _finalize_retrieval(
            state,
            docs,
            best_d,
            rq,
            lang,
            "designer",
            log_survey_merge=True,
        )
        out["survey_vector_context_used"] = survey_vector_used
        return out

    store = get_vector_store(system)
    docs, best_d = hybrid_retrieve(store, rq, **HYBRID_RETRIEVE_KWARGS)
    return _finalize_retrieval(state, docs, best_d, rq, lang, system, log_survey_merge=False)
