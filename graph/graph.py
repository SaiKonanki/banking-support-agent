"""
graph.py

Step 3 of the build: graph assembly.

Imports the stub nodes and router functions from nodes.py and wires them
into an actual LangGraph StateGraph. This file is the only place that
touches the langgraph package directly — nodes.py and its router functions
stay importable and unit-testable even in an environment without langgraph
installed.
"""

from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.nodes import (
    intake_node,
    authenticate_node,
    account_lookup_node,
    transaction_lookup_node,
    diagnostics_node,
    resolution_fraud_node,
    resolution_duplicate_node,
    resolution_failed_node,
    resolution_pending_node,
    verification_failed_node,
    out_of_scope_node,
    route_after_intake,
    route_after_auth,
    route_after_lookup,
    route_after_diagnostics,
)


def build_graph(checkpointer=None):
    if checkpointer is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        except ImportError:
            checkpointer = None

    graph = StateGraph(AgentState)


    graph.add_node("intake", intake_node)
    graph.add_node("authenticate", authenticate_node)
    graph.add_node("account_lookup", account_lookup_node)
    graph.add_node("transaction_lookup", transaction_lookup_node)
    graph.add_node("diagnostics", diagnostics_node)
    graph.add_node("resolution_fraud", resolution_fraud_node)
    graph.add_node("resolution_duplicate", resolution_duplicate_node)
    graph.add_node("resolution_failed", resolution_failed_node)
    graph.add_node("resolution_pending", resolution_pending_node)
    graph.add_node("verification_failed", verification_failed_node)
    graph.add_node("out_of_scope", out_of_scope_node)

    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "authenticate": "authenticate",
            "out_of_scope": "out_of_scope",
        },
    )


    graph.add_conditional_edges(
        "authenticate",
        route_after_auth,
        {
            "account_lookup": "account_lookup",
            "authenticate": "authenticate",
            "verification_failed": "verification_failed",
        },
    )
    graph.add_edge("account_lookup", "transaction_lookup")

    graph.add_conditional_edges(
        "transaction_lookup",
        route_after_lookup,
        {
            "diagnostics": "diagnostics",
            "transaction_lookup": "transaction_lookup",
            "out_of_scope": "out_of_scope",
        },
    )

    graph.add_conditional_edges(
        "diagnostics",
        route_after_diagnostics,
        {
            "resolution_fraud": "resolution_fraud",
            "resolution_duplicate": "resolution_duplicate",
            "resolution_failed": "resolution_failed",
            "resolution_pending": "resolution_pending",
            "out_of_scope": "out_of_scope",
        },
    )

    graph.add_edge("resolution_fraud", END)
    graph.add_edge("resolution_duplicate", END)
    graph.add_edge("resolution_failed", END)
    graph.add_edge("resolution_pending", END)
    graph.add_edge("verification_failed", END)
    graph.add_edge("out_of_scope", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()

