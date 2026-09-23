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
