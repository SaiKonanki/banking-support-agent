"""
scenarios.py

Scenario definitions for the Step 6 evaluation suite. Each scenario drives
the real compiled graph (real LLM calls, real interrupts) through one of the
four call-reason paths, using a real customer/transaction from data/.

Four scenarios are scripted (fixed interrupt responses) — cheap, deterministic
regression coverage of the routing wiring end-to-end. One (fraud) uses a
simulated customer instead of fixed responses, since that's the path with the
richest interrupt/guardrail surface (see eval/simulated_customer.py).

`guardrail_rules` are copied verbatim from the relevant _EXPLAIN_*_SYSTEM_PROMPT
in graph/llm.py — if those prompts change, update these too.
"""

from dataclasses import dataclass, field
from typing import Optional


def base_state(customer_id: str, transaction_id: str, customer_description: str, session_id: str) -> dict:
    """A fully-populated initial AgentState. Everything downstream of intake/auth
    gets overwritten by the real nodes; this just avoids KeyErrors on unset fields.
    """
    return {
        "customer_id": customer_id,
        "session_id": session_id,
        "auth_code_provided": None,
        "authenticated": False,
        "auth_attempts": 0,
        "customer_description": customer_description,
        "call_reason": "",
        "transaction_id": transaction_id,  # direct-ID fast path — keeps focus on
        # resolution behavior rather than re-exercising lookup matching, which
        # Gate 2's unit tests (tests/test_lookup_node.py) already cover.
        "lookup_status": "",
        "lookup_attempts": 0,
        "candidate_transactions": [],
        "fraud_flag": False,
        "duplicate_match_id": None,
        "dispute_history": [],
        "transaction_status": "",
        "block_reason": None,
        "agent_notes": [],
        "resolution_type": None,
        "resolution_status": "in_progress",
        "resolution_notes": [],
        "fix_attempts": 0,
        "escalated": False,
        "case_summary": None,
        "card_frozen": False,
        "pending_reroutes": 0,
        "messages": [],
        "tool_audit_log": [],
    }


@dataclass
class Scenario:
    id: str
    call_reason: str
    initial_state: dict
    expected: dict  # state key -> expected value, checked with ==
    guardrail_rules: list  # verbatim rules from the node's system prompt
    ground_truth_facts: dict  # facts the judge checks the explanation against
    scripted_responses: Optional[dict] = None  # interrupt "action" -> response
    simulated_customer_persona: Optional[str] = None  # set instead of scripted_responses


SCENARIOS = [
    Scenario(
        id="duplicate_auto_resolve",
        call_reason="duplicate",
        initial_state=base_state(
            "CUST001", "TXN0008",
            "I think I got charged twice for groceries at H-E-B, both for $19.28",
            "eval-duplicate-auto",
        ),
        scripted_responses={"otp_request": "2824"},
        expected={
            "resolution_type": "auto_resolved",
            "resolution_status": "resolved",
            "escalated": False,
        },
        guardrail_rules=[
            "Follow the provided guidance exactly for what outcome to communicate.",
            "If a refund was issued, confirm the amount and say 3-5 business days.",
            "If escalated, explain that a specialist will review and follow up within 24-48 hours.",
            "Never make up a different outcome than what the guidance says.",
            "Keep your response to 2-3 sentences.",
            "Use a warm, professional tone.",
        ],
        ground_truth_facts={"outcome": "auto_resolved (refund)", "amount": "$19.28"},
    ),
    Scenario(
        id="duplicate_escalate",
        call_reason="duplicate",
        initial_state=base_state(
            "CUST002", "TXN0009",
            "I was charged twice at the Shell gas station for $98.89",
            "eval-duplicate-escalate",
        ),
        scripted_responses={"otp_request": "5506"},
        expected={
            "resolution_type": "escalated",
            "resolution_status": "escalated",
            "escalated": True,
        },
        guardrail_rules=[
            "Follow the provided guidance exactly for what outcome to communicate.",
            "If a refund was issued, confirm the amount and say 3-5 business days.",
            "If escalated, explain that a specialist will review and follow up within 24-48 hours.",
            "Never make up a different outcome than what the guidance says.",
            "Keep your response to 2-3 sentences.",
            "Use a warm, professional tone.",
        ],
        ground_truth_facts={"outcome": "escalated (over the $50 auto-resolve threshold)"},
    ),
    Scenario(
        id="failed_insufficient_funds",
        call_reason="failed",
        initial_state=base_state(
            "CUST001", "TXN0007",
            "My Planet Fitness membership payment was declined",
            "eval-failed",
        ),
        scripted_responses={"otp_request": "2824"},
        expected={
            "resolution_type": "explained",
            "resolution_status": "resolved",
            "escalated": False,
            "fix_attempts": 1,
        },
        guardrail_rules=[
            "Never reveal internal system codes, flag names, or diagnostic IDs.",
            "Never promise the transaction will succeed on retry.",
            "Keep your response to 2-3 sentences.",
            "Use a warm, professional tone with no banking jargon.",
            'Address the customer directly ("your transaction").',
        ],
        ground_truth_facts={"block_reason": "insufficient_funds"},
    ),
    Scenario(
        id="pending_normal",
        call_reason="pending",
        initial_state=base_state(
            "CUST004", "TXN0042",
            "My Amazon order charge has been pending for a few days",
            "eval-pending",
        ),
        scripted_responses={"otp_request": "2679"},
        expected={
            "resolution_type": "explained",
            "resolution_status": "resolved",
            "pending_reroutes": 0,
        },
        guardrail_rules=[
            "Explain what a pending/authorization hold is in simple terms.",
            "Never promise the charge will drop off by a specific date.",
            "Keep your response to 2-3 sentences.",
            "Use a patient, professional tone.",
        ],
        ground_truth_facts={"variant": "normal (first visit, not a reroute)"},
    ),
    Scenario(
        id="fraud_simulated_customer",
        call_reason="fraud",
        initial_state=base_state(
            "CUST003", "TXN0029",
            "I see a charge I don't recognize",
            "eval-fraud-simulated",
        ),
        simulated_customer_persona=(
            "You are Aisha Khan, a bank customer. You just noticed a $330.70 charge "
            "at a Shell Gas Station in San Antonio, TX on your account that you don't "
            "recall making, and you're mildly anxious about it. Your 4-digit "
            "verification code, if asked, is 4657 — give it plainly when asked to "
            "verify your identity. If asked whether you'd like your card frozen as a "
            "precaution, you're cautious about fraud so you agree, but express a "
            "little hesitation about the inconvenience first. Answer in 1-2 short, "
            "natural sentences, in character — do not break character or mention "
            "that you are an AI."
        ),
        expected={
            "resolution_type": "escalated",
            "resolution_status": "escalated",
            "escalated": True,
            "fraud_flag": True,
        },
        guardrail_rules=[
            "Never confirm or deny that fraud actually occurred — that is for the investigator.",
            "Never promise a specific refund amount or timeline.",
            "Mention that the case has been escalated to a specialist who will follow up.",
            "Follow the provided guidance exactly for what to say about the customer's card.",
            "Keep your response to 2-4 sentences.",
            "Use a calm, reassuring, professional tone.",
        ],
        ground_truth_facts={"merchant": "Shell Gas Station", "amount": "$330.70"},
    ),
]
