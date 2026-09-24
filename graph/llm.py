"""
llm.py

Thin wrapper around the LLM API, used only by the three nodes that
need language understanding or generation (intake, lookup matching,
resolution explanations). Everything else in this project — routing
functions, account/diagnostics lookups — is plain Python and never
imports this module.

Supports Anthropic (default) and OpenAI-compatible providers (for local or
open-source LLMs like Ollama, vLLM, or LMStudio) via configuration in
graph/config.py.

Dependency imports (anthropic / openai) are deliberately lazy (inside
_get_client, not at module load time) so this module — and anything that
imports it, like nodes.py — stays importable even in an environment without
either package installed. That keeps a node's *own* logic unit-testable
by mocking classify_intake() and match_transaction(), without needing real
packages or live API keys.
"""

import json
from typing import Optional

from graph.config import LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL, LLM_API_KEY

_client = None


def _get_client():
    global _client
    if _client is None:
        if LLM_PROVIDER == "anthropic":
            import anthropic  # lazy import

            if not LLM_API_KEY:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY (or LLM_API_KEY) is not set. Export it in your shell "
                    "(or load it from a .env file) before running the graph for real."
                )
            _client = anthropic.Anthropic(api_key=LLM_API_KEY)

        elif LLM_PROVIDER in ("openai", "openai_compatible"):
            import openai  # lazy import

            kwargs = {"api_key": LLM_API_KEY or "dummy-key"}
            if LLM_BASE_URL:
                kwargs["base_url"] = LLM_BASE_URL
            _client = openai.OpenAI(**kwargs)

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

    return _client


def _call_structured_llm(
    system_prompt: str,
    user_content: str,
    tool_name: str,
    tool_description: str,
    schema: dict,
) -> dict:
    """Invokes the configured LLM provider forcing a structured tool call.

    Translates tool definitions to the respective provider's format (Anthropic vs.
    OpenAI-compatible) and extracts the parsed arguments dict.
    """
    client = _get_client()

    if LLM_PROVIDER == "anthropic":
        tool_def = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": schema,
        }
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=system_prompt,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input)
        return {}

    elif LLM_PROVIDER in ("openai", "openai_compatible"):
        tool_def = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": schema,
            },
        }
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            tools=[tool_def],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        choice = response.choices[0]
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.function.name == tool_name:
                    return json.loads(tc.function.arguments)
        return {}

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def _call_text_llm(system_prompt: str, user_content: str) -> str:
    """Invokes the configured LLM provider for a plain-text response.

    Used by resolution explanation nodes where the output goes directly to the
    customer and no structured parsing is needed.
    """
    client = _get_client()

    if LLM_PROVIDER == "anthropic":
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        # Anthropic returns a list of content blocks; concatenate text blocks
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

    elif LLM_PROVIDER in ("openai", "openai_compatible"):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")



# ---------------------------------------------------------------------------
# 1. Intake Classification
# ---------------------------------------------------------------------------

CALL_REASONS = ["failed", "fraud", "duplicate", "pending", "out_of_scope"]

_CLASSIFY_INTAKE_SCHEMA = {
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
    raw = _call_structured_llm(
        system_prompt=_INTAKE_SYSTEM_PROMPT,
        user_content=customer_description,
        tool_name="classify_intake",
        tool_description=(
            "Classify why a banking customer is calling, and pull out any "
            "transaction details they mentioned, so the agent can look the "
            "transaction up."
        ),
        schema=_CLASSIFY_INTAKE_SCHEMA,
    )

    call_reason = raw.get("call_reason")
    if call_reason not in CALL_REASONS:
        call_reason = "out_of_scope"

    return {
        "call_reason": call_reason,
        "transaction_date": raw.get("transaction_date"),
        "transaction_location": raw.get("transaction_location"),
        "transaction_amount": raw.get("transaction_amount"),
    }


# ---------------------------------------------------------------------------
# 2. Transaction Matching
# ---------------------------------------------------------------------------

_MATCH_TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "match_status": {
            "type": "string",
            "enum": ["exact_match", "ambiguous", "not_found"],
            "description": (
                "exact_match: exactly one candidate transaction clearly matches the customer's description/hints. "
                "ambiguous: multiple candidate transactions could plausibly match, needing customer clarification. "
                "not_found: none of the candidate transactions match what the customer described."
            ),
        },
        "matched_transaction_id": {
            "type": ["string", "null"],
            "description": "The transaction_id (e.g. 'TXN0003') of the matched transaction if exact_match, else null.",
        },
        "candidate_transaction_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of candidate transaction_ids that plausibly match if ambiguous, else empty list.",
        },
        "reasoning": {
            "type": "string",
            "description": "Concise explanation of the matching decision.",
        },
    },
    "required": ["match_status", "reasoning"],
}

_MATCH_TRANSACTION_SYSTEM_PROMPT = """You are the transaction matching step of a banking customer support agent. \
Your job is to match the customer's description and intake hints against a list of candidate transactions from their account.

Rules:
1. If exactly one candidate unambiguously matches the merchant/location, amount, or date, return match_status='exact_match' and its matched_transaction_id.
2. If multiple candidates share similar amounts, dates, or merchants such that you cannot be certain which one the customer refers to, return match_status='ambiguous' and list their IDs in candidate_transaction_ids. Do NOT guess when ambiguous.
3. If no candidate corresponds to what the customer described, return match_status='not_found'.
4. Call the match_transaction tool with your structured decision."""


def match_transaction(
    candidates: list[dict],
    customer_description: str,
    hints: Optional[dict] = None,
) -> dict:
    """Matches a customer's description and extracted hints against candidate transactions.

    Returns a dict shaped like:
        {
            "match_status": "exact_match" | "ambiguous" | "not_found",
            "matched_transaction_id": str | None,
            "candidate_transaction_ids": list[str],
            "reasoning": str,
        }
    """
    if not candidates:
        return {
            "match_status": "not_found",
            "matched_transaction_id": None,
            "candidate_transaction_ids": [],
            "reasoning": "No candidate transactions provided to match against.",
        }

    hints = hints or {}
    prompt_content = (
        f"Customer description: {customer_description}\n"
        f"Extracted hints: {json.dumps(hints)}\n\n"
        f"Candidate transactions:\n{json.dumps(candidates, indent=2)}"
    )

    raw = _call_structured_llm(
        system_prompt=_MATCH_TRANSACTION_SYSTEM_PROMPT,
        user_content=prompt_content,
        tool_name="match_transaction",
        tool_description="Match candidate transactions to the customer's request.",
        schema=_MATCH_TRANSACTION_SCHEMA,
    )

    match_status = raw.get("match_status")
    if match_status not in ("exact_match", "ambiguous", "not_found"):
        match_status = "not_found"

    return {
        "match_status": match_status,
        "matched_transaction_id": raw.get("matched_transaction_id") if match_status == "exact_match" else None,
        "candidate_transaction_ids": raw.get("candidate_transaction_ids", []) if match_status == "ambiguous" else [],
        "reasoning": raw.get("reasoning", ""),
    }


# ---------------------------------------------------------------------------
# 3. Resolution Explanations
#
# Four specialized explain functions, one per resolution path.
# Each uses a narrow, focused system prompt with dynamic guidance injected
# by the calling node's Python logic. The LLM handles language only —
# policy decisions (auto-resolve vs escalate, etc.) are made in nodes.py
# before these functions are called.
# ---------------------------------------------------------------------------

_EXPLAIN_FAILED_SYSTEM_PROMPT = """You are explaining a declined or blocked \
transaction to a banking customer. Given the transaction details and the \
specific guidance for this situation, write a clear, empathetic explanation \
of why the transaction was blocked and what the customer can do next.

Rules:
- Never reveal internal system codes, flag names, or diagnostic IDs.
- Never promise the transaction will succeed on retry.
- Keep your response to 2-3 sentences.
- Use a warm, professional tone with no banking jargon.
- Address the customer directly ("your transaction")."""


def explain_failed_transaction(
    customer_description: str,
    amount: float,
    merchant: Optional[str],
    block_reason: Optional[str],
    guidance: str,
) -> str:
    """Generates a customer-facing explanation for a failed/blocked transaction."""
    context = (
        f"Customer said: {customer_description}\n"
        f"Transaction amount: ${amount:.2f}\n"
        f"Merchant: {merchant or 'unknown'}\n"
        f"Guidance for this situation: {guidance}"
    )
    return _call_text_llm(_EXPLAIN_FAILED_SYSTEM_PROMPT, context)


_EXPLAIN_FRAUD_SYSTEM_PROMPT = """You are informing a banking customer that \
a suspected fraudulent transaction has been escalated to the fraud \
investigation team. Given the transaction details, write a reassuring \
explanation.

Rules:
- Never confirm or deny that fraud actually occurred — that is for the investigator.
- Never promise a specific refund amount or timeline.
- Mention that the case has been escalated to a specialist who will follow up.
- Follow the provided guidance exactly for what to say about the customer's card.
- Keep your response to 2-4 sentences.
- Use a calm, reassuring, professional tone."""


def explain_fraud_escalation(
    customer_description: str,
    amount: float,
    merchant: Optional[str],
    guidance: str,
) -> str:
    """Generates a customer-facing explanation for a fraud escalation."""
    context = (
        f"Customer said: {customer_description}\n"
        f"Transaction amount: ${amount:.2f}\n"
        f"Merchant: {merchant or 'unknown'}\n"
        f"Guidance for this situation: {guidance}"
    )
    return _call_text_llm(_EXPLAIN_FRAUD_SYSTEM_PROMPT, context)


_EXPLAIN_DUPLICATE_SYSTEM_PROMPT = """You are explaining the outcome of a \
duplicate charge investigation to a banking customer. The decision (refund \
or escalation) has already been made — your job is to communicate it clearly.

Rules:
- Follow the provided guidance exactly for what outcome to communicate.
- If a refund was issued, confirm the amount and say 3-5 business days.
- If escalated, explain that a specialist will review and follow up within 24-48 hours.
- Never make up a different outcome than what the guidance says.
- Keep your response to 2-3 sentences.
- Use a warm, professional tone."""


def explain_duplicate_charge(
    customer_description: str,
    amount: float,
    merchant: Optional[str],
    guidance: str,
) -> str:
    """Generates a customer-facing explanation for a duplicate charge resolution."""
    context = (
        f"Customer said: {customer_description}\n"
        f"Transaction amount: ${amount:.2f}\n"
        f"Merchant: {merchant or 'unknown'}\n"
        f"Outcome guidance: {guidance}"
    )
    return _call_text_llm(_EXPLAIN_DUPLICATE_SYSTEM_PROMPT, context)


_EXPLAIN_PENDING_SYSTEM_PROMPT = """You are explaining a stuck pending \
transaction to a banking customer. Pending charges are authorization holds \
that haven't settled yet.

Rules:
- Explain what a pending/authorization hold is in simple terms.
- If guidance mentions the hold looks abnormally long, acknowledge that and \
  say the team is looking into it.
- Never promise the charge will drop off by a specific date.
- Keep your response to 2-3 sentences.
- Use a patient, professional tone."""


def explain_pending_charge(
    customer_description: str,
    amount: float,
    merchant: Optional[str],
    guidance: str,
) -> str:
    """Generates a customer-facing explanation for a stuck pending charge."""
    context = (
        f"Customer said: {customer_description}\n"
        f"Transaction amount: ${amount:.2f}\n"
        f"Merchant: {merchant or 'unknown'}\n"
        f"Guidance for this situation: {guidance}"
    )
    return _call_text_llm(_EXPLAIN_PENDING_SYSTEM_PROMPT, context)

