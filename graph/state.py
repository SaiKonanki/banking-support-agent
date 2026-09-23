"""
state.py

The shared state schema for the Agentic Banking Support Agent graph.

This is step 2 of the build. No LLM calls and no LangGraph graph wiring live
here — just the state definition every node reads from and writes to.

Field groups, in the order they get populated as a call progresses:
    1. Identity & session
    2. Account standing
    3. Intake (why the customer is calling)
    4. Transaction lookup
    5. Diagnostics
    6. Resolution
    7. Cross-cutting (conversation history, audit log)

Two list fields (agent_notes, resolution_notes) and the audit log use
Annotated[..., operator.add] so that when multiple nodes append to them in
the same graph "step," LangGraph merges the additions instead of one
overwriting the other. `messages` uses LangGraph's own add_messages reducer
for the same reason, and because it understands LangChain message objects.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # -- 1. Identity & session --------------------------------------------
    customer_id: str
    session_id: str
    authenticated: bool
    auth_attempts: int  # capped at 3 (see review: was 0-retry, now industry-standard 3)

    # -- 2. Account standing ------------------------------------------------
    account_status: Literal["active", "frozen", "on_hold"]
    accounts: dict[str, dict]  # account_id -> account record, all of a customer's accounts
    account_tenure_years: float  # tenure of the specific account in play; context only,
    # deliberately NOT used in any auto-resolve decision
    account_standing_ok: bool  # explicit flag so edge logic doesn't re-derive this

    # -- 3. Intake ------------------------------------------------------
    # Populated by the intake node (added per the design review — nothing
    # upstream of it previously produced call_reason or customer_description).
    customer_description: str  # raw customer words describing why they're calling
    call_reason: Literal["failed", "fraud", "duplicate", "pending", "out_of_scope"]

    # -- 4. Transaction lookup ------------------------------------------------
    transaction_id: Optional[str]
    transaction_date: Optional[str]
    transaction_location: Optional[str]
    transaction_account: Optional[str]
    transaction_amount: Optional[float]
    lookup_status: Literal["found", "not_found", "ambiguous"]
    lookup_attempts: int  # bounds the not_found/ambiguous retry loop (review addition)
    candidate_transactions: list[dict]  # populated when lookup_status == "ambiguous",
    # so the customer can be presented options instead of the loop just failing blind

    # -- 5. Diagnostics ------------------------------------------------
    fraud_flag: bool
    duplicate_match_id: Optional[str]
    dispute_history: list[dict]  # past disputes on this account, checked by the
    # duplicate resolution node against the 1-year auto-resolve rule
    transaction_status: Literal["pending", "failed", "posted"]
    block_reason: Optional[str]  # insufficient_funds / card_frozen / system_error / none
    agent_notes: Annotated[list[str], operator.add]

    # -- 6. Resolution ------------------------------------------------
    resolution_type: Optional[str]  # auto_resolved / escalated / explained
    resolution_status: Literal["in_progress", "resolved", "escalated", "closed"]
    resolution_notes: Annotated[list[str], operator.add]
    fix_attempts: int  # hard ceiling of 2 distinct failed fix attempts before escalation
    escalated: bool
    case_summary: Optional[str]  # pre-built structured case file, always populated
    # before a fraud escalation
    card_frozen: bool  # set when the fraud path freezes the card with confirmation
    pending_reroutes: int  # bounds the "abnormally stuck pending" loop back into
    # deeper diagnostics (edge case from the Resolution C node design)

    # -- 7. Cross-cutting ------------------------------------------------
    messages: Annotated[list, add_messages]  # the actual conversation, for the LLM
    # nodes (intake, lookup matching, customer-facing explanations) and for
    # interrupt()/checkpointer replay
    tool_audit_log: Annotated[list[dict], operator.add]  # every tool call made during
    # the session (name, args, result, timestamp) — supports the auditability
    # goal from the design review, independent of the conversational messages
