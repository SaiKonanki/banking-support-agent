"""
test_intake_node.py

Unit tests for intake_node's own logic — how it shapes an update dict from
whatever classify_intake() returns. These mock graph.llm.classify_intake
directly, so no anthropic package and no live API key are needed to run
them. What's NOT being tested here: whether the LLM classifies correctly.
That's a prompt-quality question, for the eval suite in step 6, not a
unit test.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import intake_node


def test_intake_node_full_extraction():
    fake_result = {
        "call_reason": "fraud",
        "transaction_date": "2026-09-20",
        "transaction_location": "Best Buy",
        "transaction_amount": 249.99,
    }
    with patch("graph.llm.classify_intake", return_value=fake_result) as mock_call:
        state = {"customer_description": "There's a $249.99 charge at Best Buy I didn't make"}
        update = intake_node(state)

    mock_call.assert_called_once_with(state["customer_description"])
    assert update["call_reason"] == "fraud"
    assert update["transaction_date"] == "2026-09-20"
    assert update["transaction_location"] == "Best Buy"
    assert update["transaction_amount"] == 249.99
    assert "agent_notes" in update


def test_intake_node_minimal_extraction_omits_unset_fields():
    fake_result = {
        "call_reason": "pending",
        "transaction_date": None,
        "transaction_location": None,
        "transaction_amount": None,
    }
    with patch("graph.llm.classify_intake", return_value=fake_result):
        state = {"customer_description": "My payment has been pending for days"}
        update = intake_node(state)

    assert update["call_reason"] == "pending"
    # nothing extracted -> these keys shouldn't be in the update at all,
    # so they don't overwrite anything already in state with None
    assert "transaction_date" not in update
    assert "transaction_location" not in update
    assert "transaction_amount" not in update


def test_intake_node_preserves_customer_description():
    fake_result = {
        "call_reason": "out_of_scope",
        "transaction_date": None,
        "transaction_location": None,
        "transaction_amount": None,
    }
    with patch("graph.llm.classify_intake", return_value=fake_result):
        state = {"customer_description": "What are your branch hours?"}
        update = intake_node(state)

    assert update["customer_description"] == "What are your branch hours?"


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
