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

import json
from pathlib import Path

# AgentState must be resolvable at runtime, not just under TYPE_CHECKING:
# LangGraph calls get_type_hints() on router functions passed to
# add_conditional_edges, and an unresolvable "AgentState" forward ref
# raises NameError at build_graph() time. state.py imports langgraph, so
# fall back to a plain dict alias when it isn't installed — that keeps this
# module (and its router functions) unit-testable without langgraph.
try:
    from graph.state import AgentState
except ImportError:
    AgentState = dict
from graph.config import (
    DUPLICATE_AUTO_RESOLVE_MAX_AMOUNT,
    DUPLICATE_DISPUTE_LOOKBACK_YEARS,
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
# Data-reading & audit helpers
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> list:
    """Load a JSON data file from the data/ directory."""
    data_path = Path(__file__).resolve().parent.parent / "data" / filename
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _audit_entry(tool_name: str, args: dict, result: dict) -> dict:
    """Build a single tool audit log entry."""
    from datetime import datetime, timezone
    return {
        "tool": tool_name,
        "args": args,
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_transactions() -> list[dict]:
    return _load_json("transactions.json")


def _found_result(matched: dict, attempts: int, note: str, audit_entries: list) -> dict:
    """State update for a transaction that's been positively identified, however
    it got resolved (direct ID, LLM match, disambiguation, or a post-clarification
    retry). Centralized so these fields can't drift out of sync between the
    different resolution paths again — see the transaction_merchant/
    transaction_location bug this project already hit once from duplicated
    versions of this exact dict.
    """
    return {
        "transaction_id": matched["transaction_id"],
        "transaction_date": matched.get("date"),
        "transaction_location": matched.get("location"),
        "transaction_merchant": matched.get("merchant"),
        "transaction_account": matched.get("account_id"),
        "transaction_amount": matched.get("amount"),
        "lookup_status": "found",
        "lookup_attempts": attempts,
        "candidate_transactions": [],
        "agent_notes": [note],
        "tool_audit_log": audit_entries,
    }


def _interrupt(payload: dict):
    """Invokes LangGraph's interrupt() to pause execution and prompt the user.
    Falls back gracefully if langgraph is not installed in the environment.
    """
    try:
        from langgraph.types import interrupt
        return interrupt(payload)
    except ImportError:
        return None




def transaction_lookup_node(state: "AgentState") -> dict:
    """Finds the transaction the customer is calling about.

    If an exact transaction_id is already provided in state and found in records,
    it returns directly without an LLM call.
    Otherwise, candidate transactions are filtered by the customer's accounts and
    matched via LLM (graph/llm.py:match_transaction) using the customer's
    description and any extracted hints (date, location, amount).
    """
    all_txns = _load_transactions()
    attempts = state.get("lookup_attempts", 0) + 1

    # 1. Direct ID match
    direct_id = state.get("transaction_id")
    if direct_id:
        matched = next((t for t in all_txns if t.get("transaction_id") == direct_id), None)
        if matched:
            audit = _audit_entry(
                "transactions_lookup",
                {"transaction_id": direct_id},
                {"found": True, "method": "direct_id"},
            )
            return _found_result(matched, attempts, f"transaction {direct_id} found via direct ID", [audit])

    # 2. Gather candidates for the customer
    if state.get("candidate_transactions"):
        candidates = state["candidate_transactions"]
    else:
        customer_accounts = state.get("accounts", {})
        if isinstance(customer_accounts, dict) and customer_accounts:
            candidates = [t for t in all_txns if t.get("account_id") in customer_accounts]
        elif isinstance(customer_accounts, (list, set)) and customer_accounts:
            candidates = [t for t in all_txns if t.get("account_id") in customer_accounts]
        elif state.get("transaction_account"):
            candidates = [t for t in all_txns if t.get("account_id") == state.get("transaction_account")]
        else:
            candidates = all_txns

    if not candidates:
        audit = _audit_entry(
            "transactions_lookup",
            {"accounts": list(customer_accounts.keys()) if isinstance(customer_accounts, dict) else customer_accounts},
            {"found": False, "candidates_count": 0},
        )
        return {
            "lookup_status": "not_found",
            "lookup_attempts": attempts,
            "candidate_transactions": [],
            "agent_notes": ["transaction lookup: no candidate transactions available"],
            "tool_audit_log": [audit],
        }

    # 3. LLM-assisted matching
    from graph.llm import match_transaction

    description = state.get("customer_description", "")
    hints = {}
    if state.get("transaction_date"):
        hints["date"] = state["transaction_date"]
    if state.get("transaction_location"):
        hints["location"] = state["transaction_location"]
    if state.get("transaction_amount") is not None:
        hints["amount"] = state["transaction_amount"]

    result = match_transaction(candidates, description, hints)
    status = result.get("match_status", "not_found")
    audit = _audit_entry(
        "transactions_matching",
        {"candidates_count": len(candidates), "hints": hints},
        {"match_status": status, "matched_id": result.get("matched_transaction_id")},
    )

    if status == "exact_match":
        matched_id = result.get("matched_transaction_id")
        matched = next((t for t in candidates if t.get("transaction_id") == matched_id), None)
        if matched:
            return _found_result(
                matched, attempts,
                f"transaction {matched_id} matched via LLM: {result.get('reasoning', '')}",
                [audit],
            )
        else:
            return {
                "lookup_status": "not_found",
                "lookup_attempts": attempts,
                "candidate_transactions": [],
                "agent_notes": [f"LLM matched ID {matched_id} not present in candidate records"],
                "tool_audit_log": [audit],
            }

    elif status == "ambiguous":
        cand_ids = set(result.get("candidate_transaction_ids", []))
        matched_candidates = [t for t in candidates if t.get("transaction_id") in cand_ids]
        if not matched_candidates:
            matched_candidates = candidates[:3]

        # 1. Format the numbered choices
        lines = [
            f"{i+1}. ${c['amount']:.2f} at {c.get('merchant') or c.get('location')} on {c.get('date')} ({c['transaction_id']})"
            for i, c in enumerate(matched_candidates)
        ]
        prompt = (
            "I found multiple transactions that match your description:\n"
            + "\n".join(lines)
            + "\nPlease reply with the option number (e.g. 1) or transaction ID:"
        )

        # 2. Pause and wait for user's selection
        user_selection = _interrupt({
            "action": "transaction_disambiguation",
            "prompt": prompt,
            "candidate_ids": [c["transaction_id"] for c in matched_candidates],
        })

        # 3. Option A: Parse choice by numeric index (1, 2, ...) or exact ID (TXN0001)
        chosen = None
        if user_selection is not None:
            sel_str = str(user_selection).strip()
            if sel_str.isdigit():
                idx = int(sel_str) - 1
                if 0 <= idx < len(matched_candidates):
                    chosen = matched_candidates[idx]
            if not chosen:
                chosen = next(
                    (c for c in matched_candidates if c.get("transaction_id", "").upper() == sel_str.upper()),
                    None,
                )

        if chosen:
            disambig_audit = _audit_entry(
                "transactions_disambiguation",
                {"selection": str(user_selection), "options_count": len(matched_candidates)},
                {"resolved_id": chosen["transaction_id"], "status": "found"},
            )
            return _found_result(
                chosen, attempts,
                f"disambiguation resolved to {chosen['transaction_id']} from selection '{user_selection}'",
                [audit, disambig_audit],
            )
        else:
            return {
                "lookup_status": "ambiguous",
                "lookup_attempts": attempts,
                "candidate_transactions": matched_candidates,
                "agent_notes": [
                    f"ambiguous transaction match unresolved (selection='{user_selection}'): {result.get('reasoning', '')}"
                ],
                "tool_audit_log": [audit],
            }


    else:
        # Gate 4: give the customer one chance to clarify before giving up —
        # route_after_lookup falls straight to out_of_scope after this, since
        # MAX_LOOKUP_ATTEMPTS makes the external retry loop a no-op at 1.
        clarification = _interrupt({
            "action": "lookup_clarification",
            "prompt": (
                "I couldn't find a transaction matching that in your account history. "
                "Could you give me a bit more detail — the exact date, amount, or "
                "merchant name?"
            ),
        })

        if not clarification:
            return {
                "lookup_status": "not_found",
                "lookup_attempts": attempts,
                "candidate_transactions": [],
                "agent_notes": [f"transaction not found: {result.get('reasoning', '')}"],
                "tool_audit_log": [audit],
            }

        enriched_description = f"{description} (clarification: {clarification})"
        retry_result = match_transaction(candidates, enriched_description, hints)
        retry_status = retry_result.get("match_status", "not_found")
        retry_audit = _audit_entry(
            "transactions_matching",
            {"candidates_count": len(candidates), "hints": hints, "clarification": str(clarification)},
            {"match_status": retry_status, "matched_id": retry_result.get("matched_transaction_id")},
        )

        if retry_status == "exact_match":
            matched_id = retry_result.get("matched_transaction_id")
            matched = next((t for t in candidates if t.get("transaction_id") == matched_id), None)
            if matched:
                return _found_result(
                    matched, attempts,
                    f"transaction {matched_id} matched via LLM after clarification: {retry_result.get('reasoning', '')}",
                    [audit, retry_audit],
                )

        # Still ambiguous or not_found after one clarification round — hand off
        # rather than looping clarification requests indefinitely.
        return {
            "lookup_status": "not_found",
            "lookup_attempts": attempts,
            "candidate_transactions": [],
            "agent_notes": [
                f"transaction still not found after clarification "
                f"(clarification='{clarification}'): {retry_result.get('reasoning', '')}"
            ],
            "tool_audit_log": [audit, retry_audit],
        }



# ---------------------------------------------------------------------------
# Data-reading nodes
# ---------------------------------------------------------------------------






def authenticate_node(state: "AgentState") -> dict:
    """Verifies the caller's identity by comparing auth_code_provided against
    the auth_code in customers.json.

    If auth_code_provided is not present in state, calls LangGraph's interrupt()
    to pause execution and prompt the user for their 4-digit code.
    If the code fails, auth_code_provided is reset to None so the retry loop
    re-prompts the user with remaining attempts.
    """
    current_attempts = state.get("auth_attempts", 0)
    provided_code = state.get("auth_code_provided")

    # 1. Prompt user via interrupt() if not provided
    if not provided_code:
        remaining = MAX_AUTH_ATTEMPTS - current_attempts
        if current_attempts == 0:
            prompt = "For your security, please enter the 4-digit verification code sent to your phone."
        else:
            prompt = (
                f"Incorrect code. You have {remaining} "
                f"attempt{'s' if remaining != 1 else ''} remaining. "
                "Please enter your 4-digit verification code:"
            )

        payload = _interrupt({
            "action": "otp_request",
            "prompt": prompt,
            "remaining_attempts": remaining,
        })
        if isinstance(payload, dict):
            provided_code = payload.get("code") or payload.get("auth_code") or str(payload)
        else:
            provided_code = str(payload) if payload is not None else None


    # 2. Check against customers.json
    customers = _load_json("customers.json")
    customer_id = state.get("customer_id", "")
    customer = next((c for c in customers if c.get("customer_id") == customer_id), None)
    expected_code = customer.get("auth_code") if customer else None

    is_match = bool(
        customer
        and provided_code is not None
        and str(provided_code).strip() == str(expected_code).strip()
    )
    new_attempts = current_attempts + 1

    audit = _audit_entry(
        "customers_lookup",
        {"customer_id": customer_id},
        {"found": customer is not None, "auth_match": is_match, "attempt": new_attempts},
    )

    return {
        "auth_attempts": new_attempts,
        "authenticated": is_match,
        "auth_code_provided": str(provided_code).strip() if is_match else None,
        "agent_notes": [
            f"auth attempt {new_attempts}/{MAX_AUTH_ATTEMPTS}: "
            f"code={'****' if provided_code else 'none'}, match={is_match}"
        ],
        "tool_audit_log": [audit],
    }



def account_lookup_node(state: "AgentState") -> dict:
    """Pulls all accounts belonging to the customer and checks standing.
    Sets account_status from the primary account (first active account, or
    first account if none are active).
    """
    accounts_data = _load_json("accounts.json")
    customer_id = state.get("customer_id", "")

    customer_accounts = [a for a in accounts_data if a.get("customer_id") == customer_id]
    accounts_dict = {a["account_id"]: a for a in customer_accounts}

    # Pick the primary account for standing check
    primary = next(
        (a for a in customer_accounts if a.get("active_or_frozen") == "active"),
        customer_accounts[0] if customer_accounts else None,
    )

    if primary:
        account_status = "active" if primary.get("active_or_frozen") == "active" else "frozen"
        account_standing_ok = account_status == "active"
        from datetime import date
        opened = date.fromisoformat(primary.get("date_opened", "2000-01-01"))
        tenure_years = round((date.today() - opened).days / 365.25, 1)
    else:
        account_status = "on_hold"
        account_standing_ok = False
        tenure_years = 0.0

    audit = _audit_entry(
        "accounts_lookup",
        {"customer_id": customer_id},
        {"account_count": len(customer_accounts), "primary_status": account_status},
    )

    return {
        "accounts": accounts_dict,
        "account_status": account_status,
        "account_standing_ok": account_standing_ok,
        "account_tenure_years": tenure_years,
        "agent_notes": [
            f"account lookup: {len(customer_accounts)} accounts found, "
            f"standing_ok={account_standing_ok}"
        ],
        "tool_audit_log": [audit],
    }


def diagnostics_node(state: "AgentState") -> dict:
    """Reads the diagnostics table for the found transaction.
    Populates fraud_flag, duplicate_match_id, transaction_status, and block_reason
    from diagnostics.json. The data uses null (not the string 'none') for absent values.
    """
    diagnostics_data = _load_json("diagnostics.json")
    transaction_id = state.get("transaction_id")

    diag = next(
        (d for d in diagnostics_data if d.get("transaction_id") == transaction_id),
        None,
    )

    if diag:
        fraud_flag = bool(diag.get("fraud_flag", False))
        duplicate_match_id = diag.get("duplicate_match_id")  # None or a TXN ID
        transaction_status = diag.get("transaction_status", "posted")
        block_reason = diag.get("block_reason")  # None or a reason string
        found = True
    else:
        # No diagnostics record — treat as clean posted transaction
        fraud_flag = False
        duplicate_match_id = None
        transaction_status = "posted"
        block_reason = None
        found = False

    audit = _audit_entry(
        "diagnostics_lookup",
        {"transaction_id": transaction_id},
        {
            "found": found,
            "fraud_flag": fraud_flag,
            "has_duplicate": duplicate_match_id is not None,
            "transaction_status": transaction_status,
            "block_reason": block_reason,
        },
    )

    return {
        "fraud_flag": fraud_flag,
        "duplicate_match_id": duplicate_match_id,
        "transaction_status": transaction_status,
        "block_reason": block_reason,
        "agent_notes": [
            f"diagnostics: txn={transaction_id}, fraud={fraud_flag}, "
            f"status={transaction_status}, block_reason={block_reason}"
        ],
        "tool_audit_log": [audit],
    }


# ---------------------------------------------------------------------------

# Resolution nodes (LLM-backed)
#
# Each node follows the same pattern:
#   1. Python makes the policy decision (escalate? auto-resolve? reroute?)
#   2. Python picks the relevant guidance string for the situation
#   3. LLM generates a customer-facing explanation using that guidance
#
# The guidance mappings live here (not in the prompt) so they're auditable,
# unit-testable, and don't bloat the system prompt as cases grow.
# ---------------------------------------------------------------------------

BLOCK_REASON_GUIDANCE = {
    "insufficient_funds": "The account did not have enough funds to cover this transaction. Suggest the customer check their balance and try again later.",
    "card_frozen": "The card is currently frozen for security reasons. Direct the customer to the fraud department to unfreeze it before retrying.",
    "system_error": "A temporary system issue caused the decline. Suggest the customer try again shortly.",
    None: "The specific reason for the decline is unclear. Suggest the customer contact support for further investigation.",
}

DUPLICATE_OUTCOME_GUIDANCE = {
    "auto_resolved": "A refund has been initiated for the duplicate charge. Confirm the refund amount and that it will appear in 3-5 business days.",
    "escalated": "This duplicate charge requires further review by a specialist. Explain that someone will follow up within 24-48 hours.",
}

PENDING_GUIDANCE = {
    "normal": "This is a standard authorization hold. Most pending charges settle or drop off within a few business days.",
    "abnormal": "This pending charge has been held longer than usual. The team is looking into it and will follow up.",
}

FRAUD_FREEZE_GUIDANCE = {
    "frozen": "Confirm that the customer's card has been frozen as a precaution while the fraud team investigates, and that they'll be contacted before it's unfrozen.",
    "declined": "The customer chose not to freeze their card right now. Do not offer to freeze it again — just confirm the case has been escalated.",
}


def resolution_fraud_node(state: "AgentState") -> dict:
    """Always escalates. Asks the customer to confirm a card freeze first.

    Python decides: always escalate, and Python — never the LLM — decides
    card_frozen, based solely on the customer's explicit yes/no from the
    interrupt. LLM explains: tells the customer what's happening and
    confirms whatever the freeze outcome actually was.
    """
    from graph.llm import explain_fraud_escalation

    description = state.get("customer_description", "")
    amount = state.get("transaction_amount", 0.0)
    merchant = state.get("transaction_merchant")
    account_id = state.get("transaction_account") or "your account"
    fraud_flag = state.get("fraud_flag", False)

    freeze_response = _interrupt({
        "action": "card_freeze_confirmation",
        "prompt": (
            f"For your protection, would you like me to temporarily freeze the card "
            f"linked to account {account_id} right now while our fraud team investigates? "
            "(Reply 'Yes' or 'No')"
        ),
    })

    card_frozen = str(freeze_response or "").strip().lower() in ("yes", "y", "freeze")

    freeze_audit = _audit_entry(
        "freeze_card",
        {"account_id": account_id},
        {"card_frozen": card_frozen, "customer_response": str(freeze_response)},
    )

    source = "system-flagged" if fraud_flag else "customer-reported, no system flag"
    case_summary = (
        f"Fraud case — TXN: {state.get('transaction_id', 'unknown')}, "
        f"Amount: ${amount:.2f}, Merchant: {merchant or 'unknown'}, "
        f"Customer: {state.get('customer_id', 'unknown')}, "
        f"Card frozen: {card_frozen}, Source: {source}"
    )

    guidance = FRAUD_FREEZE_GUIDANCE["frozen" if card_frozen else "declined"]
    explanation = explain_fraud_escalation(description, amount, merchant, guidance)

    return {
        "resolution_type": "escalated",
        "resolution_status": "escalated",
        "escalated": True,
        "card_frozen": card_frozen,
        "case_summary": case_summary,
        "resolution_notes": [
            f"fraud path: escalated to human agent (card_frozen={card_frozen})",
            f"customer explanation: {explanation}",
        ],
        "tool_audit_log": [freeze_audit],
    }


def resolution_duplicate_node(state: "AgentState") -> dict:
    """Auto-resolves under the $50 / 1-year-lookback policy, else escalates.

    Python decides: check amount < threshold AND no recent dispute in lookback window.
    LLM explains: communicates the refund or escalation to the customer.
    """
    from graph.llm import explain_duplicate_charge
    from datetime import datetime, timedelta

    description = state.get("customer_description", "")
    amount = state.get("transaction_amount", 0.0)
    merchant = state.get("transaction_merchant")
    dispute_history = state.get("dispute_history", [])

    # Policy decision: auto-resolve only if amount < threshold AND no recent dispute
    cutoff = datetime.now() - timedelta(days=DUPLICATE_DISPUTE_LOOKBACK_YEARS * 365)
    recent_disputes = [
        d for d in dispute_history
        if datetime.fromisoformat(d.get("date", "1970-01-01")) > cutoff
    ]
    can_auto_resolve = amount < DUPLICATE_AUTO_RESOLVE_MAX_AMOUNT and len(recent_disputes) == 0

    if can_auto_resolve:
        outcome = "auto_resolved"
        resolution_status = "resolved"
    else:
        outcome = "escalated"
        resolution_status = "escalated"

    guidance = DUPLICATE_OUTCOME_GUIDANCE[outcome]
    explanation = explain_duplicate_charge(description, amount, merchant, guidance)

    return {
        "resolution_type": outcome,
        "resolution_status": resolution_status,
        "escalated": not can_auto_resolve,
        "resolution_notes": [
            f"duplicate path: {outcome} (amount=${amount:.2f}, recent_disputes={len(recent_disputes)})",
            f"customer explanation: {explanation}",
        ],
    }


def resolution_failed_node(state: "AgentState") -> dict:
    """Explains the block reason; offers retry or escalation.

    Python decides: pick the right guidance for this block_reason,
    increment fix_attempts, escalate if ceiling hit.
    LLM explains: tells the customer why it failed and what to do.
    """
    from graph.llm import explain_failed_transaction

    description = state.get("customer_description", "")
    amount = state.get("transaction_amount", 0.0)
    merchant = state.get("transaction_merchant")
    block_reason = state.get("block_reason")
    fix_attempts = state.get("fix_attempts", 0) + 1

    guidance = BLOCK_REASON_GUIDANCE.get(block_reason, BLOCK_REASON_GUIDANCE[None])

    # If we've hit the fix attempt ceiling, escalate instead of looping
    if fix_attempts >= MAX_FIX_ATTEMPTS:
        escalated = True
        resolution_status = "escalated"
    else:
        escalated = False
        resolution_status = "resolved"

    explanation = explain_failed_transaction(
        description, amount, merchant, block_reason, guidance
    )

    return {
        "resolution_type": "explained" if not escalated else "escalated",
        "resolution_status": resolution_status,
        "fix_attempts": fix_attempts,
        "escalated": escalated,
        "resolution_notes": [
            f"failed-transaction path: block_reason={block_reason}, fix_attempts={fix_attempts}",
            f"customer explanation: {explanation}",
        ],
    }


def resolution_pending_node(state: "AgentState") -> dict:
    """Explain-only, no action tools. May reroute to diagnostics once if the
    pending charge looks abnormally stuck.

    Python decides: normal vs abnormal hold, whether to reroute.
    LLM explains: tells the customer about authorization holds.
    """
    from graph.llm import explain_pending_charge

    description = state.get("customer_description", "")
    amount = state.get("transaction_amount", 0.0)
    merchant = state.get("transaction_merchant")
    pending_reroutes = state.get("pending_reroutes", 0)

    # Simple heuristic: if this is a reroute (we've been here before), it's abnormal
    if pending_reroutes > 0:
        guidance = PENDING_GUIDANCE["abnormal"]
    else:
        guidance = PENDING_GUIDANCE["normal"]

    explanation = explain_pending_charge(description, amount, merchant, guidance)

    return {
        "resolution_type": "explained",
        "resolution_status": "resolved",
        "pending_reroutes": pending_reroutes,
        "resolution_notes": [
            f"pending path: reroutes={pending_reroutes}",
            f"customer explanation: {explanation}",
        ],
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

def route_after_intake(state: "AgentState") -> str:
    """If intake classified the call as out_of_scope, short-circuit immediately
    to out_of_scope rather than putting the caller through auth, account lookup,
    and transaction lookup.
    """
    if state.get("call_reason") == "out_of_scope":
        return "out_of_scope"
    return "authenticate"


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

    One deliberate exception at the bottom of that chain: a customer-stated
    fraud claim still goes to resolution_fraud even when diagnostics come back
    completely clean. "Not flagged" in a mock diagnostics table isn't the same
    as "confirmed safe," and a dismissed fraud claim is uniquely high-liability
    compared to a customer being wrong about a duplicate/failed/pending charge
    — so fraud gets a human-escalation safety net that the other three call
    reasons deliberately don't. resolution_fraud_node distinguishes system-
    flagged vs. customer-reported-only in its case_summary for the human
    reviewer's benefit.
    """
    if state["fraud_flag"]:
        return "resolution_fraud"
    if state["duplicate_match_id"]:
        return "resolution_duplicate"
    if state["transaction_status"] == "failed":
        return "resolution_failed"
    if state["transaction_status"] == "pending":
        return "resolution_pending"
    if state.get("call_reason") == "fraud":
        return "resolution_fraud"
    # posted, clean, no flags, and the customer wasn't even claiming fraud —
    # nothing to resolve
    return "out_of_scope"
