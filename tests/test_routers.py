"""
test_routers.py

Unit tests for the three router functions, run in isolation from LangGraph
itself (they're pure functions: state in, string out). This is deliberate —
these are the auditable policy decisions, so they get tested directly,
without needing a compiled graph or an LLM anywhere nearby.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.nodes import (
    route_after_intake,
    route_after_auth,
    route_after_lookup,
    route_after_diagnostics,
)
from graph.config import MAX_AUTH_ATTEMPTS, MAX_LOOKUP_ATTEMPTS


def test_route_after_intake_valid_reason():
    assert route_after_intake({"call_reason": "failed"}) == "authenticate"
    assert route_after_intake({"call_reason": "fraud"}) == "authenticate"
    assert route_after_intake({"call_reason": "duplicate"}) == "authenticate"
    assert route_after_intake({"call_reason": "pending"}) == "authenticate"


def test_route_after_intake_out_of_scope():
    assert route_after_intake({"call_reason": "out_of_scope"}) == "out_of_scope"



def test_route_after_auth_success():
    state = {"authenticated": True, "auth_attempts": 1}
    assert route_after_auth(state) == "account_lookup"


def test_route_after_auth_retry():
    state = {"authenticated": False, "auth_attempts": 1}
    assert route_after_auth(state) == "authenticate"


def test_route_after_auth_exhausted():
    state = {"authenticated": False, "auth_attempts": MAX_AUTH_ATTEMPTS}
    assert route_after_auth(state) == "verification_failed"


def test_route_after_lookup_found():
    state = {"lookup_status": "found", "lookup_attempts": 1}
    assert route_after_lookup(state) == "diagnostics"


def test_route_after_lookup_retry():
    state = {"lookup_status": "not_found", "lookup_attempts": 0}
    assert route_after_lookup(state) == "transaction_lookup"


def test_route_after_lookup_exhausted():
    state = {"lookup_status": "ambiguous", "lookup_attempts": MAX_LOOKUP_ATTEMPTS}
    assert route_after_lookup(state) == "out_of_scope"


def test_route_after_diagnostics_fraud_wins():
    # fraud should win even if duplicate_match_id and failed status are ALSO set —
    # this is the priority-order test, the most important one here
    state = {
        "fraud_flag": True,
        "duplicate_match_id": "txn_999",
        "transaction_status": "failed",
    }
    assert route_after_diagnostics(state) == "resolution_fraud"


def test_route_after_diagnostics_duplicate_beats_failed():
    state = {
        "fraud_flag": False,
        "duplicate_match_id": "txn_999",
        "transaction_status": "failed",
    }
    assert route_after_diagnostics(state) == "resolution_duplicate"


def test_route_after_diagnostics_failed():
    state = {"fraud_flag": False, "duplicate_match_id": None, "transaction_status": "failed"}
    assert route_after_diagnostics(state) == "resolution_failed"


def test_route_after_diagnostics_pending():
    state = {"fraud_flag": False, "duplicate_match_id": None, "transaction_status": "pending"}
    assert route_after_diagnostics(state) == "resolution_pending"


def test_route_after_diagnostics_posted_clean():
    state = {"fraud_flag": False, "duplicate_match_id": None, "transaction_status": "posted"}
    assert route_after_diagnostics(state) == "out_of_scope"


def test_route_after_diagnostics_fraud_claim_without_flag_still_escalates():
    # Diagnostics found nothing wrong, but the customer's stated reason was
    # fraud — "not flagged" isn't "confirmed safe," so this still escalates
    # rather than closing out like other clean/posted-clean calls do.
    state = {
        "fraud_flag": False,
        "duplicate_match_id": None,
        "transaction_status": "posted",
        "call_reason": "fraud",
    }
    assert route_after_diagnostics(state) == "resolution_fraud"


def test_route_after_diagnostics_non_fraud_claim_stays_out_of_scope():
    # Same clean diagnostics, but the customer wasn't claiming fraud — the
    # fraud-specific safety net shouldn't apply to duplicate/failed/pending.
    state = {
        "fraud_flag": False,
        "duplicate_match_id": None,
        "transaction_status": "posted",
        "call_reason": "duplicate",
    }
    assert route_after_diagnostics(state) == "out_of_scope"


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
