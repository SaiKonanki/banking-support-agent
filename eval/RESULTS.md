# Eval Results

This is a write-up of what the Step 6 evaluation suite (`eval/`) actually
found, across several real runs — not a single cherry-picked pass. See
`README.md`'s "Architecture" section for how the graph itself is built, and
`eval/scenarios.py` for the exact scenario definitions referenced below.

## What this suite checks, and why it exists

The 52 unit tests in `tests/` mock the LLM out entirely — they prove each
node's *policy logic* is correct, but they can never catch a bug in what
actually gets sent to, or comes back from, a real model. `eval/` closes that
gap: it runs 5 scenarios through the **real compiled graph**, with **real
LLM calls** (Groq `openai/gpt-oss-120b`) and **real interrupts**, and checks
each one two ways:

1. **State assertions** — deterministic `==` checks on the final state
   (`resolution_type`, `escalated`, etc.), the same style as the unit tests,
   just run against the live graph instead of a mocked node.
2. **LLM-as-judge** — a *separate* LLM call reads the generated customer
   explanation and grades it against that scenario's `ground_truth_facts`
   and `guardrail_rules` (copied from the relevant `_EXPLAIN_*_SYSTEM_PROMPT`
   in `graph/llm.py`). This exists because the explanation text is
   freshly-generated language every run — there's no fixed string to
   `assert` it against, so grading has to be a "does this hold up against
   the rubric" check, not a text match.

Four scenarios use scripted interrupt answers (fixed, deterministic); one
(`fraud_simulated_customer`) uses a second LLM playing the customer
in-character instead, specifically because the fraud path has the richest
interrupt and guardrail surface.

## Results across four runs

State assertions passed **every single time**, across all four runs below —
the deterministic routing logic held up perfectly, which is exactly what
you'd expect given it's plain Python and already covered by unit tests. All
of the variance below is in the **judge's grading of generated text**, which
is the genuinely interesting part.

| Scenario | Run A | Run B | Run C | Run D |
|---|---|---|---|---|
| `duplicate_auto_resolve` | PASS | PASS | PASS | PASS |
| `duplicate_escalate` | PASS | PASS | PASS | PASS |
| `failed_insufficient_funds` | FAIL | PASS | PASS | PASS |
| `pending_normal` | FAIL | PASS | PASS | FAIL |
| `fraud_simulated_customer` | PASS | PASS | PASS | FAIL (infra) |
| **Total** | **3/5** | **5/5** | **5/5** | **3/5** |

### Finding 1: `pending_normal` has a real, recurring judge false-positive

This is the most consistent failure (2 of 4 runs), and both times it failed
for the same reason. Run A:

> "The response says 'These holds usually settle or drop off within a few
> business days,' which promises a specific timeframe and thus violates the
> rule against promising a drop-off date."

Run D:

> "The response says the hold 'usually settles or drops off within a few
> business days,' which promises a specific timeframe, violating the rule to
> never promise a drop-off date."

Both times the judge is grading language that's copied almost verbatim from
`PENDING_GUIDANCE["normal"]` in `graph/nodes.py` — the guidance Python itself
injects into the prompt. The guardrail rule says "never promise the charge
will drop off by a **specific date**"; a vague window like "a few business
days" isn't a specific date, but the judge reads it as one both times.

**Decision (made together, before this write-up): left as-is.** The
guardrail rule's wording is genuinely ambiguous between "no exact calendar
date" (satisfied) and "no timeframe language at all" (not satisfied) — and
an eval suite surfacing that kind of real ambiguity is it doing its job, not
a defect to tune away. Loosening the judge's instructions until it stops
flagging this would risk it also missing a real violation elsewhere. Revisit
if a *third* distinct rule starts showing the same pattern.

### Finding 2: a one-off, narrower judge nitpick (`failed_insufficient_funds`, Run A only)

> "The response uses 'your payment' instead of directly addressing the
> customer with 'your transaction,' violating the rule that requires the
> phrase 'your transaction.'"

This didn't recur in Runs B–D — worth noting as a data point, not a pattern.
The guardrail rule (`Address the customer directly ("your transaction")`)
is ambiguous between "use second-person address" (satisfied — "your
payment" is still direct address) and "use this literal phrase" (not
satisfied), and the judge read it the strict way exactly once.

### Finding 3: infrastructure failures are a distinct category from quality failures

Run D's `fraud_simulated_customer` "failure" wasn't a quality problem at
all — it's a `RateLimitError` from Groq's free tier (8,000 tokens/minute),
hit after running the suite three times back-to-back in the same session:

```
RateLimitError("... Rate limit reached for model `openai/gpt-oss-120b` ...
on tokens per minute (TPM): Limit 8000, Used 7237, Requested 771 ...")
```

`eval/runner.py` deliberately catches any exception and reports it as a
failed scenario with the error message attached (rather than crashing the
whole suite), which is exactly what surfaced this cleanly instead of an
opaque traceback. Worth remembering when reading a report: a FAIL with an
`Error:` line is an infrastructure/quota problem, not evidence the agent did
anything wrong — this project's free Groq tier is fine for occasional
testing, but running the suite repeatedly in a short window (or wiring it
into CI on every commit) would need a paid tier or a slower cadence.

## What this suite is good at, and what it isn't

- **Good at**: catching wiring bugs between nodes end-to-end (this is how
  the transaction_merchant/location field bug got caught, and how it's
  designed to keep catching regressions like it) and giving the routing
  logic a live-LLM smoke test on top of the mocked unit tests.
- **Not designed for**: guaranteeing the judge itself is perfectly
  calibrated — as shown above, it has real, repeatable blind spots against
  ambiguously-worded rules. It's a second opinion worth reading closely, not
  a rubber stamp.
- **Not yet covered**: Gate 2 (transaction disambiguation) and Gate 4
  (lookup clarification) aren't exercised by any of the 5 scenarios live —
  they're covered by unit tests only. Adding a scenario for each would close
  that gap.

## Reproducing this

```bash
python3 -m eval.run_eval
```

Needs a real `LLM_API_KEY` in `.env` — see `README.md`. Writes
`eval/reports/latest.md` (git-ignored, since LLM output isn't byte-stable
run to run) with the full explanation text and judge reasoning for whichever
run you just did.
