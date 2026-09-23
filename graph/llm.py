"""
llm.py

Thin wrapper around the Anthropic API, used only by the three nodes that
need language understanding or generation (intake, lookup matching,
resolution explanations). Everything else in this project — routing
functions, account/diagnostics lookups — is plain Python and never
imports this module.

Kept as one small wrapper, not a client.messages.create() call scattered
inside each node, so there's a single place to swap models, add retry or
timeout handling, or point at a different provider later.

The `anthropic` import is deliberately lazy (inside _get_client, not at
module load time) so this module — and anything that imports it, like
nodes.py — stays importable even in an environment without the anthropic
package installed. That matters for testing: a node's *own* logic (how it
shapes the update dict from an LLM result) can be unit-tested by mocking
classify_intake() etc., without needing the real package or a live API key.
"""

import os

from graph.config import LLM_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic  # lazy: see module docstring

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your shell "
                "(or load it from a .env file) before running the graph "
                "for real."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


CALL_REASONS = ["failed", "fraud", "duplicate", "pending", "out_of_scope"]

_CLASSIFY_INTAKE_TOOL = {
    "name": "classify_intake",
    "description": (
        "Classify why a banking customer is calling, and pull out any "
        "transaction details they mentioned, so the agent can look the "
        "transaction up."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "call_reason": {
                "type": "string",
                "enum": CALL_REASONS,
                "description": (
                    "failed: a transaction was declined or blocked. "
                    "fraud: customer suspects a charge they didn't make. "
                    "duplicate: the same charge appears to have gone through "
                    "twice. pending: a charge has been stuck in pending "
                    "status too long. out_of_scope: anything else (balance "
                    "questions, lost card, general questions) — not one of "
                    "the four call types this agent handles."
                ),
            },
            "transaction_date": {
                "type": ["string", "null"],
                "description": "Date the customer mentioned (YYYY-MM-DD if inferable), else null.",
            },
            "transaction_location": {
                "type": ["string", "null"],
                "description": "Merchant or location the customer mentioned, else null.",
            },
            "transaction_amount": {
                "type": ["number", "null"],
                "description": "Dollar amount the customer mentioned, else null.",
            },
        },
        "required": ["call_reason"],
    },
}

_INTAKE_SYSTEM_PROMPT = """You are the intake step of a banking customer \
support agent. Your only job is to read what the customer said about why \
they're calling and classify it into exactly one of five categories, and \
pull out any transaction details they happened to mention. You are not \
resolving anything and you are not speaking to the customer directly — a \
later step handles that. Call the classify_intake tool with your answer. \
If the customer's words don't clearly match failed, fraud, duplicate, or \
pending, use out_of_scope rather than guessing."""


def classify_intake(customer_description: str) -> dict:
    """Classifies a customer's free-text description into a structured
    call_reason, plus any transaction hints the LLM can extract.

    Returns a dict shaped like:
        {
            "call_reason": "failed" | "fraud" | "duplicate" | "pending" | "out_of_scope",
            "transaction_date": str | None,
            "transaction_location": str | None,
            "transaction_amount": float | None,
        }
    """
    client = _get_client()

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=500,
        system=_INTAKE_SYSTEM_PROMPT,
        tools=[_CLASSIFY_INTAKE_TOOL],
        tool_choice={"type": "tool", "name": "classify_intake"},
        messages=[{"role": "user", "content": customer_description}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "classify_intake":
            result = dict(block.input)
            result.setdefault("transaction_date", None)
            result.setdefault("transaction_location", None)
            result.setdefault("transaction_amount", None)
            return result

    # Defensive fallback — shouldn't happen with tool_choice forcing the
    # call, but a node should never crash the graph over a malformed response.
    return {
        "call_reason": "out_of_scope",
        "transaction_date": None,
        "transaction_location": None,
        "transaction_amount": None,
    }
