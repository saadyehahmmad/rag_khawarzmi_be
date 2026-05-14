"""
Real-time ingestion of structured survey data into a per-survey Chroma collection.

Called by POST /read-survey whenever App BE saves a survey (Builder/Logic/Design changes).
Each survey gets its own collection: survey_{survey_id} under vector_stores/surveys/.

The input is the clean structured JSON produced by App BE (not raw HOLD_JSON).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.governance import redact_prompt_injection_spans
from core.llm_helpers import get_embeddings
from core.paths import vector_store_root

logger = logging.getLogger(__name__)

_SURVEY_COLLECTION_PREFIX = "survey_"
_INGEST_BATCH_SIZE = 50


def collection_name(survey_id: str | int) -> str:
    return f"{_SURVEY_COLLECTION_PREFIX}{survey_id}"


def survey_store_path(survey_id: str | int) -> Path:
    return vector_store_root() / "surveys" / str(survey_id)


def survey_chroma_db_path(survey_id: str | int) -> Path:
    """Path to the persisted Chroma SQLite file for a survey collection (may be absent)."""
    return survey_store_path(survey_id) / "chroma.sqlite3"


def survey_ingesting_marker_path(survey_id: str | int) -> Path:
    """
    Sibling file that exists ONLY while ``ingest_survey`` is actively running.

    Stored outside the survey directory (``surveys/<id>.ingesting``) so it
    survives ``shutil.rmtree`` on the collection folder and is visible to all
    worker processes that share the same filesystem.
    """
    return vector_store_root() / "surveys" / f"{survey_id}.ingesting"


def survey_store_is_ingesting(survey_id: str | int) -> bool:
    """
    True when a disk-level ingestion marker exists for this survey.

    Unlike ``session_store`` (which is process-local and resets on restart),
    this marker is written before the Chroma collection is modified and deleted
    only after ingestion completes, making it reliable across multiple workers
    and server restarts.
    """
    try:
        return survey_ingesting_marker_path(survey_id).exists()
    except OSError:
        return False


def survey_store_has_embeddings(survey_id: str | int) -> bool:
    """
    True when an on-disk Chroma database exists for this survey (from a prior ``ingest_survey``).

    ``session_store`` is process-local and resets on restart; this check still allows retrieval
    from ``vector_stores/surveys/<id>/`` when the in-memory status is missing.
    """
    p = survey_chroma_db_path(survey_id)
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def get_survey_store(survey_id: str | int) -> Chroma:
    """Return the Chroma client for the given survey's collection (must already be ingested)."""
    path = survey_store_path(survey_id)
    return Chroma(
        persist_directory=str(path),
        embedding_function=get_embeddings(),
        collection_name=collection_name(survey_id),
    )


def _get_localized(obj: dict[str, Any], key: str, prefer: str = "ar") -> str:
    """Return the preferred language string from a localized dict, falling back to the other."""
    val = obj.get(key, {})
    if not isinstance(val, dict):
        return str(val)
    alt = "en" if prefer == "ar" else "ar"
    return val.get(prefer, "") or val.get(alt, "")


def _control_to_lines(
    ctrl: dict[str, Any],
    page_title_ar: str,
    survey_name_ar: str,
    survey_id: str,
    page_id: str,
    seq: int,
) -> list[str]:
    """Render one control (question) as human-readable lines for embedding."""
    q_type = ctrl.get("type", "")
    q_name = ctrl.get("name", "") or ctrl.get("id", "")
    label_ar = _get_localized(ctrl.get("label", {}), "ar")
    label_en = _get_localized(ctrl.get("label", {}), "en")

    settings = ctrl.get("settings", {})
    validations = settings.get("validations", {})
    props = settings.get("props", {})
    mandatory = validations.get("required", False)

    lines = [
        f"Survey: {survey_name_ar} (ID: {survey_id})",
        f"Page {page_id}: {page_title_ar}",
        f"Question {seq} | Type: {q_type}",
        f"Text AR: {label_ar}",
    ]
    if label_en:
        lines.append(f"Text EN: {label_en}")
    lines.append(f"Mandatory: {mandatory}")

    # Radio / checkbox options
    options = props.get("options", [])
    if options:
        lines.append("Options:")
        for opt in options:
            opt_label = opt.get("label", {})
            opt_ar = opt_label.get("ar", "") if isinstance(opt_label, dict) else ""
            opt_en = opt_label.get("en", "") if isinstance(opt_label, dict) else ""
            label = opt_ar or opt_en
            if opt_ar and opt_en:
                label = f"{opt_ar} | {opt_en}"
            lines.append(f"  - {label}")

    # Scale range
    scale = props.get("scale")
    if scale:
        s_min = scale.get("min", validations.get("min_value", 1))
        s_max = scale.get("max", validations.get("max_value", 5))
        scale_labels = scale.get("labels", {})
        min_label = _get_localized(scale_labels.get("min", {}), "ar")
        max_label = _get_localized(scale_labels.get("max", {}), "ar")
        lines.append(f"Scale range: {s_min} to {s_max}")
        if min_label:
            lines.append(f"  {s_min} = {min_label}")
        if max_label:
            lines.append(f"  {s_max} = {max_label}")
    elif validations.get("min_value") is not None and validations.get("max_value") is not None:
        lines.append(f"Scale range: {validations['min_value']} to {validations['max_value']}")

    return lines


def _survey_to_documents(data: dict[str, Any]) -> list[Document]:
    """
    Convert the structured survey dict into embeddable Document chunks.

    Produces three tiers of chunks for balanced retrieval granularity:
      1. Survey overview   — one doc with survey name, id, page/question/rule counts.
      2. Page summary      — one doc per page listing all its controls briefly.
      3. Control detail    — one doc per control with full text and options.
    """
    docs: list[Document] = []
    survey_id = str(data.get("survey_id", ""))
    title = data.get("title", {})
    name_ar = title.get("ar", "") if isinstance(title, dict) else ""
    name_en = title.get("en", "") if isinstance(title, dict) else ""
    pages = data.get("pages", [])
    rules = data.get("rules") or []
    num_rules = len(rules) if isinstance(rules, list) else 0

    # ── 1. Survey overview ────────────────────────────────────────────────────
    total_controls = sum(len(p.get("controls", [])) for p in pages)
    overview_lines = [
        f"Survey: {name_ar}",
        f"Survey EN: {name_en}",
        f"Survey ID: {survey_id}",
        f"Total pages: {len(pages)}",
        f"Total questions: {total_controls}",
        f"Total rules: {num_rules}",
        f"إجمالي القواعد: {num_rules}",
    ]
    docs.append(Document(
        page_content=redact_prompt_injection_spans("\n".join(overview_lines)),
        metadata={"survey_id": survey_id, "chunk_type": "survey_overview", "system": "designer"},
    ))

    for page_idx, page in enumerate(pages, start=1):
        page_id = str(page.get("id", page_idx))
        page_title_ar = page.get("title", {}).get("ar", "") or page.get("name", {}).get("ar", "")
        page_title_en = page.get("title", {}).get("en", "") or page.get("name", {}).get("en", "")
        controls = page.get("controls", [])

        # ── 2. Page summary ───────────────────────────────────────────────────
        page_lines = [
            f"Page {page_id}: {page_title_ar}",
            f"Page EN: {page_title_en}",
            f"Survey: {name_ar} (ID: {survey_id})",
            f"Questions on this page: {len(controls)}",
        ]
        for seq, ctrl in enumerate(controls, start=1):
            label_ar = ctrl.get("label", {}).get("ar", "") or ctrl.get("label", {}).get("en", "")
            q_type = ctrl.get("type", "")
            mandatory = ctrl.get("settings", {}).get("validations", {}).get("required", False)
            mandatory_label = "إلزامي" if mandatory else "اختياري"
            page_lines.append(f"  Q{seq} [{q_type}] ({mandatory_label}): {label_ar}")

        docs.append(Document(
            page_content=redact_prompt_injection_spans("\n".join(page_lines)),
            metadata={
                "survey_id": survey_id,
                "chunk_type": "page_summary",
                "page_id": page_id,
                "page_title_ar": page_title_ar,
                "system": "designer",
            },
        ))

        # ── 3. Per-control detail ──────────────────────────────────────────────
        for seq, ctrl in enumerate(controls, start=1):
            lines = _control_to_lines(
                ctrl, page_title_ar, name_ar, survey_id, page_id, seq
            )
            q_name = ctrl.get("name", "") or ctrl.get("id", "")
            docs.append(Document(
                page_content=redact_prompt_injection_spans("\n".join(lines)),
                metadata={
                    "survey_id": survey_id,
                    "chunk_type": "question",
                    "page_id": page_id,
                    "question_seq": seq,
                    "question_name": q_name,
                    "question_type": ctrl.get("type", ""),
                    "system": "designer",
                },
            ))

    # ── 4. Rules ──────────────────────────────────────────────────────────────
    docs.extend(_rules_to_documents(data))

    return docs


# ---------------------------------------------------------------------------
# Operator display labels
# ---------------------------------------------------------------------------

_OPERATOR_LABELS: dict[str, str] = {
    "equal_to": "=",
    "not_equal_to": "≠",
    "greater_than": ">",
    "less_than": "<",
    "greater_than_or_equal": "≥",
    "less_than_or_equal": "≤",
    "contains": "contains",
    "not_contains": "not contains",
    "is_empty": "is empty",
    "is_not_empty": "is not empty",
}


def _rules_to_documents(data: dict[str, Any]) -> list[Document]:
    """
    Convert survey logic rules into embeddable Document chunks (one per rule).

    Each document describes bilingual description, conditions, and actions in
    plain readable text (no internal rule id in ``page_content`` — ``rule_id``
    stays in metadata only). Helps the assistant answer logic questions without
    surfacing ids like ``R_5`` to end users.
    """
    docs: list[Document] = []
    survey_id = str(data.get("survey_id", ""))
    title = data.get("title", {})
    name_ar = title.get("ar", "") if isinstance(title, dict) else ""
    name_en = title.get("en", "") if isinstance(title, dict) else ""
    rules = data.get("rules", [])

    for rule in rules:
        rule_id = rule.get("id", "")
        desc = rule.get("description", {})
        desc_en = desc.get("en", "") if isinstance(desc, dict) else ""
        desc_ar = desc.get("ar", "") if isinstance(desc, dict) else ""
        de = str(desc_en).strip()
        da = str(desc_ar).strip()

        if_block = rule.get("if", {})
        when: list[dict] = if_block.get("when", [])
        logic_op: str = if_block.get("logicalOperator", "AND")
        then: list[dict] = if_block.get("then", [])

        lines: list[str] = []
        if de:
            lines.append(de)
        if da:
            lines.append(da)
        if not de and not da:
            lines.append("(Logic rule — no short description in survey JSON.)")
        lines.extend(
            [
                f"Survey: {name_ar} / {name_en} (ID: {survey_id})",
                "",
            ]
        )

        # Conditions ───────────────────────────────────────────────────────────
        if when:
            lines.append(f"Condition ({logic_op}):")
            for cond in when:
                left = cond.get("leftOperand", {})
                right = cond.get("rightOperand", {})
                left_val = left.get("value", "")
                left_type = left.get("type", "")
                right_val = right.get("value", "")
                op_raw = cond.get("operator", "")
                op = _OPERATOR_LABELS.get(op_raw, op_raw)
                prefix = "Question" if left_type == "question" else "Value"
                lines.append(f"  - {prefix} [{left_val}] {op} \"{right_val}\"")

        # Actions ──────────────────────────────────────────────────────────────
        if then:
            lines.append("Actions:")
            for action in then:
                a_type = action.get("type", "")
                target = action.get("target", {})
                ids_str = ", ".join(target.get("ids", []))
                if a_type == "show_question":
                    lines.append(f"  - Show: [{ids_str}]")
                elif a_type == "warning_message":
                    msg = action.get("message", {})
                    msg_en = msg.get("en", "") if isinstance(msg, dict) else ""
                    msg_ar = msg.get("ar", "") if isinstance(msg, dict) else ""
                    lines.append(f"  - Warning on [{ids_str}]: {msg_en}")
                    if msg_ar:
                        lines.append(f"    AR: {msg_ar}")
                elif a_type == "error_message":
                    msg = action.get("message", {})
                    msg_en = msg.get("en", "") if isinstance(msg, dict) else ""
                    msg_ar = msg.get("ar", "") if isinstance(msg, dict) else ""
                    lines.append(f"  - Error on [{ids_str}]: {msg_en}")
                    if msg_ar:
                        lines.append(f"    AR: {msg_ar}")
                else:
                    lines.append(f"  - {a_type} on [{ids_str}]")

        # Index of referenced question IDs for discoverability ─────────────────
        referenced: set[str] = set()
        for cond in when:
            left = cond.get("leftOperand", {})
            if left.get("type") == "question" and left.get("value"):
                referenced.add(left["value"])
        for action in then:
            for qid in action.get("target", {}).get("ids", []):
                referenced.add(qid)
        docs.append(Document(
            page_content=redact_prompt_injection_spans("\n".join(lines)),
            metadata={
                "survey_id": survey_id,
                "chunk_type": "rule",
                "rule_id": rule_id,
                "system": "designer",
                "referenced_question_ids": ",".join(sorted(referenced)) if referenced else "",
            },
        ))

    return docs


def _summarize_rules_for_designer_session(rules: list[Any]) -> tuple[str, str]:
    """One short line per language from rule descriptions only (for session_store metadata)."""
    if not rules:
        return (
            "There are no rules on this survey yet.",
            "لا توجد قواعد في هذا الاستبيان حتى الآن.",
        )
    parts_en: list[str] = []
    parts_ar: list[str] = []
    for rule in rules[:12]:
        if not isinstance(rule, dict):
            continue
        desc = rule.get("description") or {}
        if isinstance(desc, dict):
            e = str(desc.get("en", "") or "").strip()
            a = str(desc.get("ar", "") or "").strip()
            if e:
                parts_en.append(e[:140] + ("…" if len(e) > 140 else ""))
            if a:
                parts_ar.append(a[:140] + ("…" if len(a) > 140 else ""))
    if not parts_en and not parts_ar:
        return (
            f"{len(rules)} rule(s) are defined; add short descriptions in the Logic designer "
            "to make them easier to scan.",
            f"تم تعريف {len(rules)} قاعدة؛ يُفضّل إضافة وصف مختصر في مصمّم المنطق لتسهيل المراجعة.",
        )
    join_en = " · ".join(parts_en[:6])
    join_ar = " · ".join(parts_ar[:6]) if parts_ar else join_en
    if len(rules) > 6 and parts_en:
        join_en += f" (+{len(rules) - 6} more)"
    if len(rules) > 6 and parts_ar:
        join_ar += f" (+{len(rules) - 6} أخرى)"
    return join_en, join_ar


def build_session_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """
    Derive bilingual title, counts, and short rule-design summaries for session_store.

    Called on each successful survey ingest for session status and client UX
    (not from chat history).

    Args:
        data: Structured survey dict (same shape as POST /read-survey body).

    Returns:
        Flat dict merged into ``session_store.set_status(..., **kwargs)``.
    """
    pages = data.get("pages") or []
    num_questions = sum(len(p.get("controls") or []) for p in pages if isinstance(p, dict))
    rules = data.get("rules") or []
    num_rules = len(rules) if isinstance(rules, list) else 0

    title = data.get("title") or {}
    if not isinstance(title, dict):
        title = {}
    title_en = str(title.get("en", "") or "").strip()
    title_ar = str(title.get("ar", "") or "").strip()

    rules_summary_en, rules_summary_ar = _summarize_rules_for_designer_session(rules)

    return {
        "title_en": title_en,
        "title_ar": title_ar,
        "num_questions": num_questions,
        "num_rules": num_rules,
        "rules_summary_en": rules_summary_en,
        "rules_summary_ar": rules_summary_ar,
    }


def ingest_survey(data: dict[str, Any]) -> int:
    """
    Embed structured survey data into its dedicated Chroma collection.

    Always performs a clean re-ingest (deletes the old collection first) so
    re-saves after edits produce a consistent, up-to-date index.

    A sibling marker file (``<id>.ingesting``) is written before any work
    begins and removed in a ``finally`` block, so the ingesting state is
    visible to all worker processes and survives server restarts.

    Returns:
        Number of document chunks ingested.
    """
    survey_id = str(data["survey_id"])
    docs = _survey_to_documents(data)

    # ── Write disk marker before touching the collection ─────────────────────
    marker = survey_ingesting_marker_path(survey_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")

    try:
        path = survey_store_path(survey_id)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        store = Chroma(
            persist_directory=str(path),
            embedding_function=get_embeddings(),
            collection_name=collection_name(survey_id),
        )

        for i in range(0, len(docs), _INGEST_BATCH_SIZE):
            store.add_documents(docs[i : i + _INGEST_BATCH_SIZE])

        logger.info(
            "Survey ingestion complete: survey_id=%s chunks=%d collection=%s",
            survey_id, len(docs), collection_name(survey_id),
        )
        return len(docs)
    finally:
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
