"""
Al-Khawarzmi RAGProfile — the single concrete product instance for this deployment.

This is the **only** file that needs to change when deploying the RAG engine
for a different product. Import this ``PROFILE`` in ``api/deps.py`` and pass it
to ``build_graph(PROFILE)`` — nothing else in ``framework/`` or ``core/`` changes.

To add a second product (e.g. "Qatar Statistics Portal"):
    1. Create ``qatarportal/`` with equivalent adapter classes.
    2. Create ``qatarportal/profile.py`` with a ``PROFILE`` instance.
    3. In ``api/deps.py``, swap: ``from qatarportal.profile import PROFILE``.
"""

from __future__ import annotations

from alkawarzmi.greeting_reply import GREETING_MESSAGE_AR, GREETING_MESSAGE_EN
from alkawarzmi.fallback import AlKhawarzmiFallback
from alkawarzmi.intents import AlKhawarzmiIntentDetector
from alkawarzmi.prescripts import AlKhawarzmiPrescripts
from alkawarzmi.prompts import AlKhawarzmiPrompts
from alkawarzmi.survey_retrieval import AlKhawarzmiSurveyRetrievalHooks
from framework.profile import RAGProfile

PROFILE = RAGProfile(
    platform_name="Al-Khawarzmi",
    systems=["designer", "runtime", "callcenter", "admin"],
    prompts=AlKhawarzmiPrompts(),
    fallback=AlKhawarzmiFallback(),
    intent_detector=AlKhawarzmiIntentDetector(),
    prescripts=AlKhawarzmiPrescripts(),
    greeting_reply_en=GREETING_MESSAGE_EN,
    greeting_reply_ar=GREETING_MESSAGE_AR,
    survey_retrieval=AlKhawarzmiSurveyRetrievalHooks(),
)
