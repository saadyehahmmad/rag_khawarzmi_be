"""
Detect user intent to list every question or every rule on the attached survey (Designer context),
ask for a survey overview/summary, and broader phrasing that refers to **this** survey/form
(used when the per-survey index is absent).

Used when ``system`` is designer and ``survey_id`` is present: full-catalog retrieval bypasses
hybrid ``top_k`` so all page summaries or rule chunks can reach the answer model.
"""

from __future__ import annotations

import re

from core.client_locale import question_with_rewrite
from framework.state import AgentState

_RULES_CATALOG_INTENT = re.compile(
    r"\b("
    r"all rules|every rule|each rule|list (?:of |all |every )?rules|complete list of rules"
    r"|full rules|rules inventory|show (?:me )?all rules|display (?:all )?rules"
    r"|جميع القواعد|كافة القواعد|كل القواعد|جميع القواعس|كافة القواعس|كل القواعس"
    r"|قائمة (?:كاملة )?(?:بال|ب)?القواعد|قائمة القواعد|قائمة القواعس|سرد القواعد|سرد القواعس"
    r"|استعراض (?:كل|جميع) القواعد|استعراض (?:كل|جميع) القواعس"
    r"|اعرض (?:لي )?(?:كل|جميع) القواعد|اعرض (?:لي )?(?:كل|جميع) القواعس"
    r"|عرض (?:كل|جميع) القواعد|عرض (?:كل|جميع) القواعس"
    r"|اذكر (?:لي )?(?:كل|جميع) القواعد|اذكر (?:لي )?(?:كل|جميع) القواعس"
    r"|ما هي (?:جميع|كل|كافة) القواعد|ما هي (?:جميع|كل|كافة) القواعس"
    r")\b",
    re.I,
)

# User is referring to the concrete form attached to the session (not only generic Designer docs).
_BODY_FOCUS_EN = re.compile(
    r"\b("
    r"this survey|my survey|the survey|current survey|attached survey|our survey"
    r"|this form|my form|the form|current form|attached form"
    r"|questions in (?:this|my|the) (?:survey|form)|pages in (?:this|my|the) (?:survey|form)"
    r"|structure of (?:this|my|the) (?:survey|form)|in my form|on this form|for this survey"
    r"|tell me about (?:this|my|the|our) (?:survey|form)"
    r"|what(?:'s| is) (?:in|inside) (?:this|my|the|our) (?:survey|form)"
    r")\b",
    re.I,
)

_BODY_FOCUS_AR = re.compile(
    r"(?:"
    r"هذا الاستبيان|هذه الاستمارة|استمارتي|الاستمارة الحالية|الاستبيان الحالي"
    r"|في استمارتي|في هذه الاستمارة|في هذا الاستبيان|عن هذه الاستمارة|عن استمارتي"
    r"|أسئلة (?:هذه )?الاستمارة|صفحات (?:هذه )?الاستمارة|محتوى (?:هذه )?الاستمارة"
    r"|الاستمارة المرفقة|الاستبيان المرفق|هذه الاستمارة المفتوحة|هذا الاستبيان المفتوح"
    r")",
    re.I,
)

# User is asking for a high-level summary or description of the survey itself (not a full catalog).
_SURVEY_OVERVIEW_INTENT = re.compile(
    r"\b(?:"
    # EN: "what is this/my/the survey/form"
    r"what(?:'s| is) (?:this|my|the|our|the current) (?:survey|form)"
    r"|what(?:'s| is) (?:this|the) (?:survey|form) (?:about|for)"
    # EN: summary / overview / description of the survey
    r"|(?:give (?:me )?(?:a |an )?)?(?:summary|overview|description|abstract) of (?:this|my|the|our) (?:survey|form)"
    r"|(?:summarize|describe|explain) (?:this|my|the|our) (?:survey|form)"
    r"|(?:this|my|the|our) (?:survey|form) (?:summary|overview|description)"
    r"|about (?:this|my|the|our) (?:survey|form)"
    # EN: purpose / scope / coverage
    r"|(?:purpose|objective|goal|scope) of (?:this|my|the|our) (?:survey|form)"
    r"|what does (?:this|my|the|our) (?:survey|form) (?:cover|measure|ask)"
    r"|(?:tell me|walk me through) (?:about )?(?:this|my|the|our) (?:survey|form)"
    r"|brief(?:ly)? (?:on|about) (?:this|my|the|our) (?:survey|form)"
    r")\b"
    r"|(?:"
    # AR: "ما هذه الاستمارة / ما هي هذه الاستمارة / ما هو هذا الاستبيان"
    r"ما (?:هي|هو|هذه|هذا)\s*(?:هذه|هذا|ال)?\s*(?:الاستمارة|الاستبيان|النموذج)"
    # AR: ملخص / وصف / نظرة عامة
    r"|ملخص (?:هذه|هذا|ال)?(?:الاستمارة|الاستبيان|النموذج)"
    r"|وصف (?:هذه|هذا|ال)?(?:الاستمارة|الاستبيان|النموذج)"
    r"|نظرة عامة (?:على|عن) (?:هذه|هذا|ال)?(?:الاستمارة|الاستبيان|النموذج)"
    r"|(?:لخّص|لخص|صف|اشرح) (?:هذه|هذا) (?:الاستمارة|الاستبيان|النموذج)"
    r"|عن (?:هذه|هذا) (?:الاستمارة|الاستبيان|النموذج)"
    r"|نبذة (?:عن|على) (?:هذه|هذا|ال)?(?:الاستمارة|الاستبيان|النموذج)"
    r"|(?:الغرض|الهدف) (?:من|لـ?) (?:هذه|هذا|ال)?(?:الاستمارة|الاستبيان|النموذج)"
    r"|ما (?:الذي|الذى) (?:يقيس|يغطي|يشمل) (?:هذه|هذا) (?:الاستمارة|الاستبيان|النموذج)"
    r")",
    re.I,
)

_CATALOG_INTENT = re.compile(
    r"\b("
    r"all questions|every question|each question|list (?:of |all |every )?questions|complete list"
    r"|full list|entire form|whole survey|question inventory|all fields|every field"
    r"|جميع الأسئلة|كافة الأسئلة|كل الأسئلة|كل اسئلة|كل الاسئلة"
    r"|قائمة (?:كاملة )?(?:بال|ب)?أسئلة|قائمة الأسئلة|سرد الأسئلة|استعراض (?:كل|جميع) الأسئلة"
    r"|اذكر (?:لي )?(?:كل|جميع) الأسئلة|ما هي (?:جميع|كل|كافة) الأسئلة"
    r"|أسئلة الاستمارة(?: كاملة)?|جميع(?: بيانات)? الاستمارة|كل ما في الاستمارة"
    r")\b",
    re.I,
)


def wants_survey_question_catalog(state: AgentState) -> bool:
    """
    True when the user is asking for a complete inventory of questions on the attached survey.

    In that case we bypass survey hybrid top-k and load every ``survey_overview`` +
    ``page_summary`` chunk from Chroma so the model sees all pages at once.
    """
    blob = question_with_rewrite(state, sep="\n")
    return bool(_CATALOG_INTENT.search(blob))


def wants_survey_rules_catalog(state: AgentState) -> bool:
    """
    True when the user wants every logic rule on the attached survey listed.

    Loads all ``chunk_type == "rule"`` documents from Chroma (metadata ``get``),
    same rationale as question catalog: hybrid ``top_k`` cannot return every rule.
    """
    blob = question_with_rewrite(state, sep="\n")
    return bool(_RULES_CATALOG_INTENT.search(blob))


def wants_survey_overview(state: AgentState) -> bool:
    """
    True when the user is asking what the attached survey is about, or requesting a summary /
    description / overview of the form — without asking for a full question catalog.

    In that case we fetch only the ``survey_overview`` chunk (much lighter than the full catalog)
    so the model can answer decisively from the top-level survey description.
    """
    if wants_survey_question_catalog(state) or wants_survey_rules_catalog(state):
        return False
    blob = question_with_rewrite(state, sep="\n")
    return bool(_SURVEY_OVERVIEW_INTENT.search(blob))


def wants_survey_structure_help(state: AgentState) -> bool:
    """
    True when the user likely needs **this survey's** embedded index (questions, pages, rules),
    not only general Designer product documentation.

    Used when the per-survey vector store is absent: we still run manual retrieval first, but
    the fallback message should explain indexing only for these intents.
    """
    if wants_survey_question_catalog(state) or wants_survey_rules_catalog(state):
        return True
    if wants_survey_overview(state):
        return True
    blob = question_with_rewrite(state, sep="\n")
    return bool(_BODY_FOCUS_EN.search(blob) or _BODY_FOCUS_AR.search(blob))
