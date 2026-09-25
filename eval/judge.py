"""
judge.py

LLM-as-judge: grades a generated customer-facing explanation against (a) the
ground-truth facts Python computed for that call, and (b) the node's own
guardrail rules (copied from its _EXPLAIN_*_SYSTEM_PROMPT in graph/llm.py).

This is the check unit tests structurally can't do, since they mock the LLM
call out entirely — it's the only thing in this project that actually reads
what the model said and asks "was that both true and allowed?"

Reuses graph.llm._call_structured_llm (forced tool-use) rather than parsing
free text, for the same reliability reason every other structured call in
this project uses it.
"""

from graph.llm import _call_structured_llm

_JUDGE_SYSTEM_PROMPT = """You are a strict QA reviewer for a banking support \
agent's customer-facing responses. You will be given the explanation the \
agent gave a customer, the ground-truth facts about their situation, and a \
list of rules that explanation must follow.

Judge whether the explanation is factually consistent with the ground truth \
AND follows every rule. Minor stylistic variation is fine; a rule violation \
or a stated fact that contradicts the ground truth is not. Be strict but fair."""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "reasoning": {
            "type": "string",
            "description": "1-2 sentences citing the specific fact or rule that passed/failed.",
        },
    },
    "required": ["verdict", "reasoning"],
}


def judge_explanation(explanation: str, ground_truth_facts: dict, guardrail_rules: list) -> dict:
    facts_str = "\n".join(f"- {k}: {v}" for k, v in ground_truth_facts.items())
    rules_str = "\n".join(f"- {r}" for r in guardrail_rules)
    user_content = (
        f"Explanation given to the customer:\n\"{explanation}\"\n\n"
        f"Ground-truth facts:\n{facts_str}\n\n"
        f"Rules the explanation must follow:\n{rules_str}"
    )
    return _call_structured_llm(
        _JUDGE_SYSTEM_PROMPT,
        user_content,
        tool_name="submit_verdict",
        tool_description="Submit the PASS/FAIL verdict and reasoning.",
        schema=_JUDGE_SCHEMA,
    )
