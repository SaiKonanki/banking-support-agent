"""
test_resolution_nodes.py

Unit tests for the four resolution nodes' Python policy logic.
Mocks the LLM explain functions so tests run without any external dependency.
What's tested here: that Python makes the right policy decision (escalate vs
auto-resolve, correct guidance selection, fix_attempts tracking) and passes
the right inputs to the LLM. What's NOT tested: whether the LLM generates
good prose — that's for the eval suite in step 6.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import (
    resolution_fraud_node,
    resolution_duplicate_node,
    resolution_failed_node,
    resolution_pending_node,
    BLOCK_REASON_GUIDANCE,
    DUPLICATE_OUTCOME_GUIDANCE,
    PENDING_GUIDANCE,
)
from graph.config import DUPLICATE_AUTO_RESOLVE_MAX_AMOUNT, MAX_FIX_ATTEMPTS


# ---------------------------------------------------------------------------
# Fraud node
# ---------------------------------------------------------------------------

def test_fraud_always_escalates():
    state = {
        "customer_description": "I see a $500 charge I didn't make",
        "transaction_amount": 500.0,
        "transaction_location": "Best Buy",
        "transaction_id": "TXN0003",
        "transaction_account": "ACC001",
        "customer_id": "CUST001",
    }
    with patch("graph.nodes._interrupt", return_value="no"), \
         patch("graph.llm.explain_fraud_escalation", return_value="We've flagged this.") as mock_explain:
        update = resolution_fraud_node(state)

    mock_explain.assert_called_once()
    assert mock_explain.call_args[0][:3] == ("I see a $500 charge I didn't make", 500.0, "Best Buy")
    assert update["resolution_type"] == "escalated"
    assert update["resolution_status"] == "escalated"
    assert update["escalated"] is True
    assert "TXN0003" in update["case_summary"]
    assert "We've flagged this." in update["resolution_notes"][1]


def test_fraud_freeze_confirmed():
    state = {
        "customer_description": "I see a $500 charge I didn't make",
        "transaction_amount": 500.0,
        "transaction_location": "Best Buy",
        "transaction_id": "TXN0003",
        "transaction_account": "ACC001",
        "customer_id": "CUST001",
    }
    with patch("graph.nodes._interrupt", return_value="yes") as mock_interrupt, \
         patch("graph.llm.explain_fraud_escalation", return_value="Your card is frozen.") as mock_explain:
        update = resolution_fraud_node(state)

    mock_interrupt.assert_called_once()
    assert mock_interrupt.call_args[0][0]["action"] == "card_freeze_confirmation"
    assert "ACC001" in mock_interrupt.call_args[0][0]["prompt"]

    assert update["card_frozen"] is True
    assert "Card frozen: True" in update["case_summary"]
    assert update["tool_audit_log"][0]["tool"] == "freeze_card"
    assert update["tool_audit_log"][0]["result"]["card_frozen"] is True
    # Guidance passed to the LLM should reflect the frozen outcome
    assert "frozen" in mock_explain.call_args[0][3].lower()


def test_fraud_freeze_declined():
    state = {
        "customer_description": "I see a $500 charge I didn't make",
        "transaction_amount": 500.0,
        "transaction_location": "Best Buy",
        "transaction_id": "TXN0003",
        "transaction_account": "ACC001",
        "customer_id": "CUST001",
    }
    with patch("graph.nodes._interrupt", return_value="no"), \
         patch("graph.llm.explain_fraud_escalation", return_value="Understood, no freeze.") as mock_explain:
        update = resolution_fraud_node(state)

    assert update["card_frozen"] is False
    assert "Card frozen: False" in update["case_summary"]
    assert update["tool_audit_log"][0]["result"]["card_frozen"] is False
    assert "not" in mock_explain.call_args[0][3].lower() or "declined" not in mock_explain.call_args[0][3].lower()


def test_fraud_freeze_no_response_defaults_to_not_frozen():
    """Standalone/non-interrupt environments (e.g. langgraph not installed)
    get None back from _interrupt — that must never be treated as consent."""
    state = {
        "customer_description": "I see a $500 charge I didn't make",
        "transaction_amount": 500.0,
        "transaction_location": "Best Buy",
        "transaction_id": "TXN0003",
        "transaction_account": "ACC001",
        "customer_id": "CUST001",
    }
    with patch("graph.nodes._interrupt", return_value=None), \
         patch("graph.llm.explain_fraud_escalation", return_value="Understood."):
        update = resolution_fraud_node(state)

    assert update["card_frozen"] is False


# ---------------------------------------------------------------------------
# Duplicate node — policy logic
# ---------------------------------------------------------------------------

def test_duplicate_auto_resolves_small_amount_no_disputes():
    state = {
        "customer_description": "I got charged twice for coffee",
        "transaction_amount": 12.50,  # under $50 threshold
        "transaction_location": "Starbucks",
        "dispute_history": [],
    }
    with patch("graph.llm.explain_duplicate_charge", return_value="Refund on the way.") as mock_explain:
        update = resolution_duplicate_node(state)

    assert update["resolution_type"] == "auto_resolved"
    assert update["resolution_status"] == "resolved"
    assert update["escalated"] is False
    # Verify the auto_resolved guidance was passed to the LLM
    mock_explain.assert_called_once()
    call_args = mock_explain.call_args
    assert "refund" in call_args[0][3].lower()  # guidance arg


def test_duplicate_escalates_large_amount():
    state = {
        "customer_description": "Duplicate charge for $200",
        "transaction_amount": 200.0,  # over $50 threshold
        "transaction_location": "Target",
        "dispute_history": [],
    }
    with patch("graph.llm.explain_duplicate_charge", return_value="A specialist will review."):
        update = resolution_duplicate_node(state)

    assert update["resolution_type"] == "escalated"
    assert update["resolution_status"] == "escalated"
    assert update["escalated"] is True


def test_duplicate_escalates_recent_dispute():
    state = {
        "customer_description": "Another duplicate charge",
        "transaction_amount": 10.0,  # under threshold, but has recent dispute
        "transaction_location": "Amazon",
        "dispute_history": [{"date": "2026-09-01", "amount": 15.0}],
    }
    with patch("graph.llm.explain_duplicate_charge", return_value="A specialist will review."):
        update = resolution_duplicate_node(state)

    assert update["resolution_type"] == "escalated"
    assert update["escalated"] is True


# ---------------------------------------------------------------------------
# Failed transaction node — guidance selection + fix_attempts
# ---------------------------------------------------------------------------

def test_failed_insufficient_funds_guidance():
    state = {
        "customer_description": "My purchase was declined",
        "transaction_amount": 150.0,
        "transaction_location": "Target",
        "block_reason": "insufficient_funds",
        "fix_attempts": 0,
    }
    with patch("graph.llm.explain_failed_transaction", return_value="Not enough funds.") as mock_explain:
        update = resolution_failed_node(state)

    assert update["resolution_type"] == "explained"
    assert update["resolution_status"] == "resolved"
    assert update["fix_attempts"] == 1
    assert update["escalated"] is False
    # Check the right guidance was passed
    call_args = mock_explain.call_args
    assert "insufficient_funds" == call_args[0][3]  # block_reason arg
    assert "balance" in call_args[0][4].lower()  # guidance arg


def test_failed_card_frozen_guidance():
    state = {
        "customer_description": "Can't use my card",
        "transaction_amount": 75.0,
        "transaction_location": "Shell",
        "block_reason": "card_frozen",
        "fix_attempts": 0,
    }
    with patch("graph.llm.explain_failed_transaction", return_value="Card is frozen.") as mock_explain:
        update = resolution_failed_node(state)

    call_args = mock_explain.call_args
    assert "frozen" in call_args[0][4].lower()  # guidance mentions frozen


def test_failed_escalates_after_max_fix_attempts():
    state = {
        "customer_description": "Still declined",
        "transaction_amount": 100.0,
        "transaction_location": "Amazon",
        "block_reason": "system_error",
        "fix_attempts": MAX_FIX_ATTEMPTS - 1,  # one more will hit ceiling
    }
    with patch("graph.llm.explain_failed_transaction", return_value="Escalating."):
        update = resolution_failed_node(state)

    assert update["resolution_type"] == "escalated"
    assert update["resolution_status"] == "escalated"
    assert update["escalated"] is True
    assert update["fix_attempts"] == MAX_FIX_ATTEMPTS


# ---------------------------------------------------------------------------
# Pending node — normal vs abnormal guidance
# ---------------------------------------------------------------------------

def test_pending_normal_first_visit():
    state = {
        "customer_description": "My payment is stuck pending",
        "transaction_amount": 50.0,
        "transaction_location": "Netflix",
        "pending_reroutes": 0,
    }
    with patch("graph.llm.explain_pending_charge", return_value="Standard hold.") as mock_explain:
        update = resolution_pending_node(state)

    assert update["resolution_type"] == "explained"
    assert update["resolution_status"] == "resolved"
    # Verify normal guidance was passed
    call_args = mock_explain.call_args
    assert "standard" in call_args[0][3].lower()  # normal guidance


def test_pending_abnormal_after_reroute():
    state = {
        "customer_description": "Still pending after a week",
        "transaction_amount": 50.0,
        "transaction_location": "Netflix",
        "pending_reroutes": 1,
    }
    with patch("graph.llm.explain_pending_charge", return_value="Looking into it.") as mock_explain:
        update = resolution_pending_node(state)

    # Verify abnormal guidance was passed
    call_args = mock_explain.call_args
    assert "longer than usual" in call_args[0][3].lower()  # abnormal guidance


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
