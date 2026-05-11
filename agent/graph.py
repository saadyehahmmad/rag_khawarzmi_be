"""
LangGraph StateGraph: linear pipeline plus conditional edge on retrieval relevance.

Linear edges are derived from agent.nodes.PRE_ANSWER_PIPELINE so the graph matches SSE /chat.
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    PRE_ANSWER_PIPELINE,
    answer_node,
    fallback_node,
    route_by_relevance,
)
from agent.state import AgentState


def build_graph():
    """Build and compile the LangGraph agent (no checkpointing)."""
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
