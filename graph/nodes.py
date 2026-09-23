"""
nodes.py

Step 3 of the build: stub node functions and the router functions.

Deliberately has NO dependency on langgraph — everything here is plain
Python (state dict in, dict or string out). That's not an accident: the
router functions are the auditable policy decisions in this whole project,
so they need to be testable on their own, without a compiled graph, a
model, or even langgraph installed. graph.py imports from here and does
the actual LangGraph wiring.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only needed for type hints — avoids a hard langgraph dependency
    # at import time, so this module (and its router functions) stays
    # unit-testable even without langgraph installed
    from graph.state import AgentState
from graph.config import (
    MAX_AUTH_ATTEMPTS,
    MAX_LOOKUP_ATTEMPTS,
    MAX_FIX_ATTEMPTS,
    MAX_PENDING_REROUTES,
)


# ---------------------------------------------------------------------------
# Live nodes (LLM-backed)
#
# These are no longer stubs — they call the LLM wrapper in graph/llm.py.
# The import is done inside the function body (not at module top) so
# nodes.py, and the router functions below, stay importable — and
# node-level unit tests stay mockable — without the anthropic package
# installed. See graph/llm.py's docstring for the same reasoning.
# ---------------------------------------------------------------------------

def intake_node(state: "AgentState") -> dict:
    """Greets the customer and captures why they're calling.

    Calls the LLM (graph/llm.py:classify_intake) to turn the customer's
    free-text description into a structured call_reason, plus whatever
    transaction hints (date/location/amount) it can pull out along the way.
    Those hints are optional — the transaction_lookup node still does the
    real lookup — this just gives it a head start when the customer
    volunteers detail up front.
    """
    from graph.llm import classify_intake

    description = state.get("customer_description", "")
    result = classify_intake(description)

    update = {
        "customer_description": description,
        "call_reason": result["call_reason"],
        "agent_notes": [f"intake classified call_reason={result['call_reason']}"],
    }
    # Only set these if the LLM actually found something — don't clobber a
    # value with None just because this particular utterance didn't mention it.
    if result.get("transaction_date"):
        update["transaction_date"] = result["transaction_date"]
    if result.get("transaction_location"):
        update["transaction_location"] = result["transaction_location"]
    if result.get("transaction_amount") is not None:
        update["transaction_amount"] = result["transaction_amount"]
    return update


# ---------------------------------------------------------------------------
# Stub nodes
#
# Each node is a plain function: (state) -> dict of fields to update.
# Real versions (step 4+) will call an LLM and/or the read/action tools.
# These stubs just do the minimum to make state progress realistically
# enough to unit-test routing.
# ---------------------------------------------------------------------------


def authenticate_node(state: "AgentState") -> dict:
    """Verifies the caller's identity (auth_code lookup in the real version).
    Stub just increments the attempt counter and trusts a pre-set
    `authenticated` flag so tests can drive both branches.
    """
    return {
        "auth_attempts": state.get("auth_attempts", 0) + 1,
        "authenticated": state.get("authenticated", False),
        "agent_notes": ["authentication attempted"],
    }


def account_lookup_node(state: "AgentState") -> dict:
    """Pulls the customer's accounts and checks standing.
    Real version: read tool against accounts.json.
    """
    return {
        "account_status": state.get("account_status", "active"),
        "account_standing_ok": state.get("account_status", "active") == "active",
        "agent_notes": ["account standing checked"],
    }


def transaction_lookup_node(state: "AgentState") -> dict:
    """Finds the transaction the customer is calling about.
    Real version: read tool against transactions.json, LLM-assisted matching
    when the customer's description is ambiguous.
    """
    return {
        "lookup_status": state.get("lookup_status", "found"),
        "lookup_attempts": state.get("lookup_attempts", 0) + 1,
        "agent_notes": ["transaction lookup attempted"],
    }


def diagnostics_node(state: "AgentState") -> dict:
    """Reads the diagnostics table for the found transaction.
    Real version: read tool against diagnostics.json.
    """
    return {
        "fraud_flag": state.get("fraud_flag", False),
        "duplicate_match_id": state.get("duplicate_match_id"),
        "transaction_status": state.get("transaction_status", "posted"),
        "block_reason": state.get("block_reason"),
        "agent_notes": ["diagnostics run"],
    }


def resolution_fraud_node(state: "AgentState") -> dict:
    """Always escalates. Optionally freezes the card with confirmation."""
    return {
        "resolution_type": "escalated",
        "resolution_status": "escalated",
        "escalated": True,
        "case_summary": "Fraud case pre-built for handoff.",
        "resolution_notes": ["fraud path: escalated to human agent"],
    }


def resolution_duplicate_node(state: "AgentState") -> dict:
    """Auto-resolves under the $50 / 1-year-lookback policy, else escalates."""
    return {
        "resolution_type": "auto_resolved",
        "resolution_status": "resolved",
        "resolution_notes": ["duplicate path: resolved per policy"],
    }


def resolution_failed_node(state: "AgentState") -> dict:
    """Explains the block reason; offers retry or escalation."""
    return {
        "resolution_type": "explained",
        "resolution_status": "resolved",
        "fix_attempts": state.get("fix_attempts", 0) + 1,
        "resolution_notes": ["failed-transaction path: explained to customer"],
    }


def resolution_pending_node(state: "AgentState") -> dict:
    """Explain-only, no action tools. May reroute to diagnostics once if the
    pending charge looks abnormally stuck.
    """
    return {
        "resolution_type": "explained",
        "resolution_status": "resolved",
        "pending_reroutes": state.get("pending_reroutes", 0),
        "resolution_notes": ["pending path: explained to customer"],
    }


def verification_failed_node(state: "AgentState") -> dict:
    """Terminal node: caller could not be verified within MAX_AUTH_ATTEMPTS."""
    return {
        "resolution_status": "closed",
        "resolution_notes": ["call ended: identity could not be verified"],
    }


def out_of_scope_node(state: "AgentState") -> dict:
    """Terminal-ish node: call reason isn't one of the four handled here."""
    return {
        "resolution_type": "explained",
        "resolution_status": "closed",
        "resolution_notes": ["out of scope: routed to general support"],
    }


# ---------------------------------------------------------------------------
# Router functions
#
# Each one reads the state a node just produced and returns the *name* of
# the next node as a string. This is where the actual branching policy
# lives — plain, auditable, unit-testable Python. No LLM judgment calls.
# ---------------------------------------------------------------------------

def route_after_auth(state: "AgentState") -> str:
    if state["authenticated"]:
        return "account_lookup"
    if state["auth_attempts"] < MAX_AUTH_ATTEMPTS:
        return "authenticate"
    return "verification_failed"


def route_after_lookup(state: "AgentState") -> str:
    if state["lookup_status"] == "found":
        return "diagnostics"
    if state["lookup_attempts"] < MAX_LOOKUP_ATTEMPTS:
        return "transaction_lookup"
    # exhausted retries without a clean match — treat as out of scope for now;
    # step 4 will likely replace this with a "clarify with customer" branch
    return "out_of_scope"


def route_after_diagnostics(state: "AgentState") -> str:
    """Diagnostic evidence overrides the customer's stated call_reason, in
    priority order: fraud > duplicate > failed > pending > posted-clean.
    """
    if state["fraud_flag"]:
        return "resolution_fraud"
    if state["duplicate_match_id"]:
        return "resolution_duplicate"
    if state["transaction_status"] == "failed":
        return "resolution_failed"
    if state["transaction_status"] == "pending":
        return "resolution_pending"
    # posted, clean, no flags — nothing to resolve
    return "out_of_scope"
