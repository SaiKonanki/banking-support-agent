"""
simulated_customer.py

Plays the customer side of an interrupt for scenarios that don't use fixed
scripted responses. Given the agent's prompt and a persona (which encodes any
private facts the customer would know, like their own OTP code), generates a
short in-character reply via the same LLM backend the graph itself uses.

This is what actually stresses the interrupt/retry loops the way a real,
imperfectly-scripted caller would — a fixed "yes"/"4657" response can never
be more than one path through the conversation.
"""

from graph.llm import _call_text_llm

_SYSTEM_PROMPT_TEMPLATE = """{persona}

You are on a phone call with your bank's support agent. The agent just said
the following to you — reply as the customer would, and nothing else (no
stage directions, no quotation marks around your reply)."""


def respond_as_customer(persona: str, agent_prompt: str) -> str:
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(persona=persona)
    return _call_text_llm(system_prompt, f"The agent says: \"{agent_prompt}\"")
