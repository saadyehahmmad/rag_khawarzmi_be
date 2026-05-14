"""
LangGraph StateGraph: linear pipeline plus conditional edge on retrieval relevance.

Linear edges are derived from ``framework.nodes.PRE_ANSWER_PIPELINE`` so the graph matches SSE /chat.

Profile injection
-----------------
Pass a ``RAGProfile`` to ``build_graph(profile)`` to configure all business-layer
dependencies (prompts, systems, prescripts, fallback, intent detection) before the
graph is compiled.  When ``None``, the active product :data:`alkawarzmi.profile.PROFILE`
is used (same as production).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from framework.nodes import (
    PRE_ANSWER_PIPELINE,
    answer_node,
    configure_pipeline,
    fallback_node,
    route_by_relevance,
)
from framework.state import AgentState

if TYPE_CHECKING:
    from framework.profile import RAGProfile


def build_graph(profile: "RAGProfile | None" = None):
    """
    Build and compile the LangGraph agent (no checkpointing).

    Args:
        profile: Business-layer configuration.  When ``None``, uses
                 :data:`alkawarzmi.profile.PROFILE`.
    """
    if profile is None:
        from alkawarzmi.profile import PROFILE

        profile = PROFILE
    configure_pipeline(profile)

    graph = StateGraph(AgentState)

    for name, fn in PRE_ANSWER_PIPELINE:
        graph.add_node(name, fn)

    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    first, _ = PRE_ANSWER_PIPELINE[0]
    graph.add_edge(START, first)

    for i in range(len(PRE_ANSWER_PIPELINE) - 1):
        a, _ = PRE_ANSWER_PIPELINE[i]
        b, _ = PRE_ANSWER_PIPELINE[i + 1]
        graph.add_edge(a, b)

    last, _ = PRE_ANSWER_PIPELINE[-1]
    graph.add_conditional_edges(
        last,
        route_by_relevance,
        {"answer": "answer", "fallback": "fallback"},
    )

    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "question": "How do I add skip logic to a question in the survey designer?",
            "language": "en",
            "rewritten_question": "",
            "system": None,
            "retrieved_chunks": [],
            "retrieved_source_refs": [],
            "relevance": "unknown",
            "answer": "",
            "thread_id": str(uuid.uuid4()),
            "conversation_history": [],
        }
    )
    print("\n=== ANSWER ===")
    print(result.get("answer", ""))
    print("\nRouted to system:", result.get("system"))
    print("Language:", result.get("language"))
