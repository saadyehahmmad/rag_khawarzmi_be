"""Survey-scoped Chroma client (session collections under ``vector_stores/surveys/``)."""

from __future__ import annotations

from langchain_chroma import Chroma

from core.llm_helpers import get_embeddings


def get_survey_vector_store(survey_id: str | int) -> Chroma:
    """Return a Chroma client for the given survey's session collection."""
    from alkawarzmi.ingestion.survey_session import collection_name, survey_store_path

    sid = str(survey_id)
    path = survey_store_path(sid)
    return Chroma(
        persist_directory=str(path),
        embedding_function=get_embeddings(),
        collection_name=collection_name(sid),
    )
