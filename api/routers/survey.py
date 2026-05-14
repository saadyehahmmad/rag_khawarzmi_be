"""
Survey ingestion routes.

  POST /read-survey
    Called by App BE on every Builder / Logic / Design save.
    Parses the structured survey payload, detects content changes, and
    triggers background embedding into a per-survey Chroma collection.
    Returns 202 when ingestion starts, 200 when content is unchanged.

  GET /ingest/survey/{survey_id}/status
    Debug / monitoring endpoint that exposes the current ingestion state.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from core import session_store
from alkawarzmi.ingestion.survey_session import build_session_metadata, ingest_survey
from api.deps import verify_rag_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class LocalizedText(BaseModel):
    ar: str = ""
    en: str = ""


class ControlOption(BaseModel):
    id: str = ""
    label: LocalizedText


class ScaleLabels(BaseModel):
    min: LocalizedText = Field(default_factory=LocalizedText)
    max: LocalizedText = Field(default_factory=LocalizedText)


class ScaleProps(BaseModel):
    min: int = 1
    max: int = 5
    labels: ScaleLabels = Field(default_factory=ScaleLabels)


class ControlProps(BaseModel):
    options: list[ControlOption] = Field(default_factory=list)
    scale: ScaleProps | None = None


class ControlValidations(BaseModel):
    required: bool = False
    type: str = ""
    min_value: int | None = None
    max_value: int | None = None


class ControlSettings(BaseModel):
    validations: ControlValidations = Field(default_factory=ControlValidations)
    props: ControlProps = Field(default_factory=ControlProps)


class SurveyControl(BaseModel):
    id: str = ""
    name: str = ""
    label: LocalizedText
    type: str
    settings: ControlSettings = Field(default_factory=ControlSettings)
    children: list["SurveyControl"] = Field(default_factory=list)


SurveyControl.model_rebuild()  # required for self-referencing children


class SurveyPage(BaseModel):
    id: str = ""
    name: LocalizedText = Field(default_factory=LocalizedText)
    title: LocalizedText = Field(default_factory=LocalizedText)
    controls: list[SurveyControl] = Field(default_factory=list)


class ReadSurveyRequest(BaseModel):
    survey_id: int
    surveyVersion: str = ""
    title: LocalizedText
    pages: list[SurveyPage] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)


class ReadSurveyResponse(BaseModel):
    status: str   # "accepted" | "skipped"
    survey_id: int
    message: str = ""


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------


def _run_ingest(data: dict[str, Any]) -> None:
    """Blocking ingest — called in a background thread via FastAPI BackgroundTasks."""
    survey_id = str(data["survey_id"])
    try:
        count = ingest_survey(data)
        meta = build_session_metadata(data)
        session_store.set_status(
            survey_id,
            "ready",
            question_count=count,
            version_hash=data.get("_hash", ""),
            **meta,
        )
        logger.info("Survey %s ingestion complete: %d chunks", survey_id, count)
    except Exception as exc:  # noqa: BLE001
        session_store.set_status(survey_id, "failed", error=str(exc))
        logger.error("Survey %s ingestion failed: %s", survey_id, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/read-survey", response_model=ReadSurveyResponse)
async def read_survey(
    body: ReadSurveyRequest,
    background_tasks: BackgroundTasks,
    _auth: None = Depends(verify_rag_api_key),
) -> ReadSurveyResponse:
    """
    Ingest structured survey content into a session-scoped Chroma collection.

    App BE parses HOLD_JSON → this clean structure and calls this endpoint on
    every Builder / Logic / Design save.

    - Returns 200 status='skipped' when content hash is unchanged (idempotent).
    - Returns 200 status='accepted' and starts background embedding otherwise.
    """
    data = body.model_dump()
    survey_id = str(body.survey_id)

    new_hash = session_store.compute_hash(data)
    data["_hash"] = new_hash

    current = session_store.get_status(survey_id)
    if (
        current
        and current.get("status") == "ready"
        and current.get("version_hash") == new_hash
    ):
        return ReadSurveyResponse(
            status="skipped",
            survey_id=body.survey_id,
            message="No changes detected, ingestion skipped.",
        )

    session_store.set_status(survey_id, "ingesting", version_hash=new_hash)
    background_tasks.add_task(_run_ingest, data)

    return ReadSurveyResponse(
        status="accepted",
        survey_id=body.survey_id,
        message="Ingestion started in background.",
    )


@router.get("/ingest/survey/{survey_id}/status")
async def survey_ingest_status(
    survey_id: int,
    _auth: None = Depends(verify_rag_api_key),
) -> dict[str, Any]:
    """Return current ingestion state for a survey (debug / monitoring use only)."""
    state = session_store.get_status(survey_id)
    if state is None:
        return {
            "survey_id": survey_id,
            "status": "not_found",
            "message": "This survey has not been ingested in the current server session.",
        }
    return {"survey_id": survey_id, **state}
