"""
test_data_nodes.py

Unit tests for the live data-reading nodes:
- authenticate_node (against data/customers.json)
- account_lookup_node (against data/accounts.json)
- diagnostics_node (against data/diagnostics.json)

Verifies both data reading and audit log creation without any network/LLM dependencies.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch

from graph.nodes import authenticate_node, account_lookup_node, diagnostics_node


def test_authenticate_node_success():
    # CUST001 has auth_code "2824" in customers.json
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": "2824",
        "auth_attempts": 0,
    }
    update = authenticate_node(state)
    assert update["authenticated"] is True
    assert update["auth_attempts"] == 1
    assert len(update["tool_audit_log"]) == 1
    assert update["tool_audit_log"][0]["tool"] == "customers_lookup"
    assert update["tool_audit_log"][0]["result"]["auth_match"] is True


def test_authenticate_node_wrong_code():
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": "0000",
        "auth_attempts": 1,
    }
    update = authenticate_node(state)
    assert update["authenticated"] is False
    assert update["auth_attempts"] == 2
    assert update["auth_code_provided"] is None  # cleared on failure for retry
    assert update["tool_audit_log"][0]["result"]["auth_match"] is False


def test_authenticate_node_missing_code_no_interrupt_input():
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": None,
        "auth_attempts": 0,
    }
    with patch("graph.nodes._interrupt", return_value=None):
        update = authenticate_node(state)
    assert update["authenticated"] is False
    assert update["auth_attempts"] == 1


def test_authenticate_node_interrupt_initial_prompt():
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": None,
        "auth_attempts": 0,
    }
    with patch("graph.nodes._interrupt", return_value="2824") as mock_interrupt:
        update = authenticate_node(state)

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["action"] == "otp_request"
    assert payload["remaining_attempts"] == 3
    assert "security" in payload["prompt"].lower()

    assert update["authenticated"] is True
    assert update["auth_code_provided"] == "2824"
    assert update["auth_attempts"] == 1


def test_authenticate_node_interrupt_retry_prompt():
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": None,
        "auth_attempts": 1,
    }
    with patch("graph.nodes._interrupt", return_value="9999") as mock_interrupt:
        update = authenticate_node(state)

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["remaining_attempts"] == 2
    assert "2 attempts remaining" in payload["prompt"]

    assert update["authenticated"] is False
    assert update["auth_code_provided"] is None  # cleared on failure
    assert update["auth_attempts"] == 2


def test_authenticate_node_interrupt_final_retry_prompt():
    state = {
        "customer_id": "CUST001",
        "auth_code_provided": None,
        "auth_attempts": 2,
    }
    with patch("graph.nodes._interrupt", return_value="9999") as mock_interrupt:
        update = authenticate_node(state)

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["remaining_attempts"] == 1
    assert "1 attempt remaining" in payload["prompt"]

    assert update["authenticated"] is False
    assert update["auth_code_provided"] is None
    assert update["auth_attempts"] == 3



def test_account_lookup_node_found():
    # CUST001 has ACC001 (checking, active, date_opened: 2023-04-12)
    state = {
        "customer_id": "CUST001",
    }
    update = account_lookup_node(state)
    assert "ACC001" in update["accounts"]
    assert update["account_status"] == "active"
    assert update["account_standing_ok"] is True
    assert update["account_tenure_years"] > 0
    assert len(update["tool_audit_log"]) == 1
    assert update["tool_audit_log"][0]["tool"] == "accounts_lookup"


def test_account_lookup_node_nonexistent():
    state = {
        "customer_id": "CUST_DOES_NOT_EXIST",
    }
    update = account_lookup_node(state)
    assert update["accounts"] == {}
    assert update["account_status"] == "on_hold"
    assert update["account_standing_ok"] is False


def test_diagnostics_node_flagged():
    # TXN0007 is failed with insufficient_funds in diagnostics.json
    state = {
        "transaction_id": "TXN0007",
    }
    update = diagnostics_node(state)
    assert update["transaction_status"] == "failed"
    assert update["block_reason"] == "insufficient_funds"
    assert update["fraud_flag"] is False
    assert update["duplicate_match_id"] is None
    assert len(update["tool_audit_log"]) == 1
    assert update["tool_audit_log"][0]["tool"] == "diagnostics_lookup"
    assert update["tool_audit_log"][0]["result"]["found"] is True


def test_diagnostics_node_fraud_flagged():
    # TXN0029 has fraud_flag = True in diagnostics.json
    state = {
        "transaction_id": "TXN0029",
    }
    update = diagnostics_node(state)
    assert update["fraud_flag"] is True
    assert update["block_reason"] is None


def test_diagnostics_node_clean_or_unrecorded():
    state = {
        "transaction_id": "TXN_NOT_IN_DIAGNOSTICS",
    }
    update = diagnostics_node(state)
    assert update["transaction_status"] == "posted"
    assert update["fraud_flag"] is False
    assert update["block_reason"] is None
    assert update["duplicate_match_id"] is None
    assert update["tool_audit_log"][0]["result"]["found"] is False


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
