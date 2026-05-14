"""
RAGProfile — the binding contract between framework/ (Layer 2) and business packages (Layer 3).

A business package (e.g. ``alkawarzmi/``) implements these Protocol interfaces and
assembles a single ``RAGProfile`` instance. ``framework/`` depends only on this contract —
never on any concrete business package directly.

To swap products, change one import in ``api/deps.py``:
    from alkawarzmi.profile import PROFILE  →  from myproduct.profile import PROFILE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from framework.state import AgentState


@runtime_checkable
class PromptProvider(Protocol):
    """Provides the prompt templates used by the RAG pipeline."""

    def rewrite_and_route(self) -> str:
        """Return the rewrite-and-route JSON prompt template string."""
        ...

    def answer(self) -> str:
        """Return the final answer system prompt template string."""
        ...


@runtime_checkable
class FallbackProvider(Protocol):
    """Provides bilingual fallback text when retrieval is not usable."""

    def get(self, state: "AgentState") -> str:
        """Return the appropriate fallback string for the given state."""
        ...


@runtime_checkable
class IntentDetector(Protocol):
    """Detects platform-specific intent types."""

    def is_platform_overview(self, question: str) -> bool:
        """Return True when the user is asking what the platform is (no specific system)."""
        ...


@runtime_checkable
class PrescriptProvider(Protocol):
    """Provides zero-LLM fast-path answers from client payload context."""

    def run(self, state: "AgentState") -> dict[str, str]:
        """
        Inspect the client payload and return a state update dict.

        Returns:
            ``{"prescripted_answer": "<text>"}`` when a fast-path answer is available,
            or ``{}`` when normal pipeline processing should continue.
        """
        ...


@runtime_checkable
class SurveyRetrievalHooks(Protocol):
    """
    Product-specific survey catalog, layout retrieval, and screenshot selection.

    Injected into the retrieval node so ``framework`` / ``core`` never import a
    concrete designer package.
    """

    def wants_survey_rules_catalog(self, state: "AgentState") -> bool:
        """Return True when the user is asking for the full survey rules inventory."""
        ...

    def wants_survey_question_catalog(self, state: "AgentState") -> bool:
        """Return True when the user is asking for the full question/page inventory."""
        ...

    def wants_survey_overview(self, state: "AgentState") -> bool:
        """Return True when the user is asking what the survey is about or for a summary/description."""
        ...

    def fetch_survey_overview_only(self, survey_store: Any, *, limit: int) -> list[Any]:
        """Return only the ``survey_overview`` chunk(s) for lightweight summary queries."""
        ...

    def fetch_survey_all_rules(self, survey_store: Any, *, limit: int) -> list[Any]:
        """Return rule chunks for catalog mode (LangChain ``Document`` list)."""
        ...

    def fetch_survey_overview_and_pages(self, survey_store: Any, *, limit: int) -> list[Any]:
        """Return overview + page_summary chunks for catalog mode."""
        ...

    def select_images_for_question(self, question: str, image_system: str, *, language: str) -> list[str]:
        """Return image filenames (no URL prefix) for the answer UI."""
        ...


@dataclass
class RAGProfile:
    """
    All business-specific configuration for a single RAG product deployment.

    Assemble one instance per product and inject it into ``build_graph(profile)``.
    The ``framework/`` and ``core/`` layers depend only on this dataclass — never
    on any concrete business package.

    Attributes:
        platform_name:     Human-readable product name (used in API title, logs).
        systems:           Ordered list of Chroma collection slugs (e.g. ["designer", "runtime"]).
        prompts:           Template provider for rewrite+route and answer prompts.
        fallback:          Provider for bilingual fallback messages.
        intent_detector:   Detects platform-overview and similar product-specific intents.
        prescripts:        Zero-LLM fast-path handler (page context, navigation, named greetings).
        greeting_reply_en: English generic greeting response text.
        greeting_reply_ar: Arabic generic greeting response text.
        survey_retrieval: Optional survey/designer hooks for merged survey + manual retrieval.
    """

    platform_name: str
    systems: list[str]
    prompts: PromptProvider
    fallback: FallbackProvider
    intent_detector: IntentDetector
    prescripts: PrescriptProvider
    greeting_reply_en: str
    greeting_reply_ar: str
    survey_retrieval: SurveyRetrievalHooks | None = None
