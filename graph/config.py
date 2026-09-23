"""
config.py

Policy constants for the Agentic Banking Support Agent.

Kept separate from node logic (per the design review) so these are easy to
find, easy to change, and easy to point to in an audit: "the duplicate
auto-resolve threshold is defined here, not buried in a conditional."
"""

# Duplicate charge auto-resolution (Resolution A)
DUPLICATE_AUTO_RESOLVE_MAX_AMOUNT = 50.00  # USD; strictly less than this
DUPLICATE_DISPUTE_LOOKBACK_YEARS = 1  # no similar dispute in this window

# Identity verification
MAX_AUTH_ATTEMPTS = 3

# Escalation
MAX_FIX_ATTEMPTS = 2  # hard ceiling of distinct failed fix attempts

# Bounded retry loops
MAX_LOOKUP_ATTEMPTS = 1  # for not_found / ambiguous transaction lookup
MAX_PENDING_REROUTES = 1  # for abnormally-stuck pending re-diagnosis loop

# LLM
# Used only by the three nodes that need language understanding/generation
# (intake, lookup matching, resolution explanations). Every routing decision
# stays plain Python and never touches this.
import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai_compatible"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5" if LLM_PROVIDER == "anthropic" else "llama3.1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)  # e.g., "http://localhost:11434/v1" for Ollama/vLLM
LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    os.getenv("ANTHROPIC_API_KEY") if LLM_PROVIDER == "anthropic" else os.getenv("OPENAI_API_KEY", "dummy-key")
)

