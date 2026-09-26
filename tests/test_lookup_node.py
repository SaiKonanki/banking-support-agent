"""
test_lookup_node.py

Unit tests for transaction_lookup_node's matching logic.
Mocks graph.llm.match_transaction for LLM-assisted matching, so tests run
instantaneously without any external dependency, network call, or API key.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import transaction_lookup_node


def test_direct_id_match():
    # Direct ID present in transactions.json should resolve immediately without LLM
    state = {
        "transaction_id": "TXN0001",
        "lookup_attempts": 0,
    }
    with patch("graph.llm.match_transaction") as mock_match:
        update = transaction_lookup_node(state)

    mock_match.assert_not_called()
    assert update["lookup_status"] == "found"
    assert update["transaction_id"] == "TXN0001"
    assert update["transaction_account"] == "ACC001"
    assert update["transaction_amount"] == 206.77
    assert update["transaction_merchant"] == "Chipotle"
    assert update["lookup_attempts"] == 1
    assert "TXN0001 found via direct ID" in update["agent_notes"][0]


def test_fuzzy_exact_match():
    state = {
        "customer_description": "I bought something at Best Buy for about $420",
        "transaction_amount": 421.81,
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "exact_match",
        "matched_transaction_id": "TXN0003",
        "candidate_transaction_ids": [],
        "reasoning": "Single Best Buy purchase matching the approximate amount.",
    }
    with patch("graph.llm.match_transaction", return_value=fake_match_result) as mock_match:
        update = transaction_lookup_node(state)

    mock_match.assert_called_once()
    assert update["lookup_status"] == "found"
    assert update["transaction_id"] == "TXN0003"
    assert update["transaction_amount"] == 421.81
    assert update["transaction_location"] == "Austin, TX"
    assert update["transaction_merchant"] == "Best Buy"
    assert update["lookup_attempts"] == 1


def test_fuzzy_ambiguous_match():
    state = {
        "customer_description": "I have two charges around $200-$300",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "ambiguous",
        "matched_transaction_id": None,
        "candidate_transaction_ids": ["TXN0001", "TXN0004"],
        "reasoning": "Both TXN0001 ($206.77) and TXN0004 ($287.57) are in this range.",
    }
    with patch("graph.llm.match_transaction", return_value=fake_match_result) as mock_match:
        update = transaction_lookup_node(state)

    mock_match.assert_called_once()
    assert update["lookup_status"] == "ambiguous"
    assert "transaction_id" not in update
    assert len(update["candidate_transactions"]) == 2
    matched_ids = [t["transaction_id"] for t in update["candidate_transactions"]]
    assert "TXN0001" in matched_ids
    assert "TXN0004" in matched_ids
    assert update["lookup_attempts"] == 1


def test_disambiguation_select_by_index():
    state = {
        "customer_description": "I have two charges around $200-$300",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "ambiguous",
        "matched_transaction_id": None,
        "candidate_transaction_ids": ["TXN0001", "TXN0004"],
        "reasoning": "Both TXN0001 ($206.77) and TXN0004 ($287.57) are in this range.",
    }
    # Mock match_transaction AND mock _interrupt returning "1" (option 1 -> TXN0001)
    with patch("graph.llm.match_transaction", return_value=fake_match_result), \
         patch("graph.nodes._interrupt", return_value="1") as mock_interrupt:
        update = transaction_lookup_node(state)

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["action"] == "transaction_disambiguation"
    assert "TXN0001" in payload["candidate_ids"]
    assert "TXN0004" in payload["candidate_ids"]

    assert update["lookup_status"] == "found"
    assert update["transaction_id"] == "TXN0001"
    assert update["transaction_amount"] == 206.77
    assert update["transaction_merchant"] == "Chipotle"
    assert update["candidate_transactions"] == []
    assert len(update["tool_audit_log"]) == 2  # matching + disambiguation


def test_disambiguation_select_by_id():
    state = {
        "customer_description": "I have two charges around $200-$300",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "ambiguous",
        "matched_transaction_id": None,
        "candidate_transaction_ids": ["TXN0001", "TXN0004"],
        "reasoning": "Both TXN0001 and TXN0004 match.",
    }
    # User types exact ID "TXN0004"
    with patch("graph.llm.match_transaction", return_value=fake_match_result), \
         patch("graph.nodes._interrupt", return_value="TXN0004"):
        update = transaction_lookup_node(state)

    assert update["lookup_status"] == "found"
    assert update["transaction_id"] == "TXN0004"
    assert update["transaction_amount"] == 287.57
    assert update["candidate_transactions"] == []


def test_disambiguation_invalid_choice():
    state = {
        "customer_description": "I have two charges around $200-$300",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "ambiguous",
        "matched_transaction_id": None,
        "candidate_transaction_ids": ["TXN0001", "TXN0004"],
        "reasoning": "Both TXN0001 and TXN0004 match.",
    }
    # User types invalid option number "99"
    with patch("graph.llm.match_transaction", return_value=fake_match_result), \
         patch("graph.nodes._interrupt", return_value="99"):
        update = transaction_lookup_node(state)

    assert update["lookup_status"] == "ambiguous"
    assert len(update["candidate_transactions"]) == 2



def test_fuzzy_not_found():
    state = {
        "customer_description": "A flight to London for $3000",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "not_found",
        "matched_transaction_id": None,
        "candidate_transaction_ids": [],
        "reasoning": "No transactions match this international travel description.",
    }
    with patch("graph.llm.match_transaction", return_value=fake_match_result) as mock_match:
        update = transaction_lookup_node(state)

    mock_match.assert_called_once()
    assert update["lookup_status"] == "not_found"
    assert update["candidate_transactions"] == []
    assert update["lookup_attempts"] == 1


def test_not_found_clarification_finds_transaction():
    state = {
        "customer_description": "A $300 charge at a gas station I don't recognize",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    first_result = {
        "match_status": "not_found",
        "matched_transaction_id": None,
        "candidate_transaction_ids": [],
        "reasoning": "No transaction close to $300 at a gas station.",
    }
    second_result = {
        "match_status": "exact_match",
        "matched_transaction_id": "TXN0001",
        "candidate_transaction_ids": [],
        "reasoning": "Customer clarified it was actually a $206.77 Chipotle charge.",
    }
    with patch("graph.llm.match_transaction", side_effect=[first_result, second_result]) as mock_match, \
         patch("graph.nodes._interrupt", return_value="Sorry, it was actually Chipotle for $206.77") as mock_interrupt:
        update = transaction_lookup_node(state)

    mock_interrupt.assert_called_once()
    assert mock_interrupt.call_args[0][0]["action"] == "lookup_clarification"
    assert mock_match.call_count == 2

    assert update["lookup_status"] == "found"
    assert update["transaction_id"] == "TXN0001"
    assert len(update["tool_audit_log"]) == 2


def test_not_found_clarification_still_not_found():
    state = {
        "customer_description": "A $300 charge at a gas station I don't recognize",
        "accounts": {"ACC001": {}},
        "lookup_attempts": 0,
    }
    fake_match_result = {
        "match_status": "not_found",
        "matched_transaction_id": None,
        "candidate_transaction_ids": [],
        "reasoning": "Still no match.",
    }
    with patch("graph.llm.match_transaction", return_value=fake_match_result) as mock_match, \
         patch("graph.nodes._interrupt", return_value="I really don't remember any more detail"):
        update = transaction_lookup_node(state)

    assert mock_match.call_count == 2  # initial attempt + one clarification retry
    assert update["lookup_status"] == "not_found"
    assert "clarification" in update["agent_notes"][0]


def test_empty_candidates_returns_not_found():
    state = {
        "customer_description": "Some purchase",
        "accounts": {"NON_EXISTENT_ACC": {}},
        "lookup_attempts": 0,
    }
    with patch("graph.llm.match_transaction") as mock_match:
        update = transaction_lookup_node(state)

    mock_match.assert_not_called()
    assert update["lookup_status"] == "not_found"
    assert update["lookup_attempts"] == 1


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
