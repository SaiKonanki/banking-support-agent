# Agentic Banking Support Agent

A LangGraph-based agentic customer support agent for banking, built as a portfolio
project to demonstrate production-grade agentic AI design: deterministic routing,
human-in-the-loop confirmation gates, auditable policy logic, and bounded escalation.

## Scope

Handles four call reasons through a shared identity/account/lookup/diagnostics
pipeline, then forks into resolution paths:

1. **Failed transaction** — explain the block reason, offer retry or escalation
2. **Suspected fraud** — always escalates to a human; optional card freeze with confirmation
3. **Duplicate charge** — auto-resolves only if amount < $50 and no similar dispute in the past year
4. **Stuck pending charge** — explain-only, no action tools

Diagnostic evidence overrides the customer's stated reason, in priority order:
fraud > duplicate > failed > pending > posted-with-no-issue.

## Status

Build in progress, following this order:

- [x] 1. Mock data and read/action tools
- [x] 2. State schema in code
- [x] 3. Skeleton graph with stub nodes + router functions (unit-tested before any LLM)
- [~] 4. LLM added to intake, lookup matching, customer-facing explanations
  - [x] intake — done (graph/llm.py:classify_intake, forced tool-use for structured call_reason)
  - [ ] lookup matching
  - [ ] customer-facing resolution explanations
- [ ] 5. Interrupts + checkpointer for OTP, clarifications, action confirmations
- [ ] 6. Evaluation suite (scripted scenarios + simulated customer)
- [ ] 7. Polish: tracing, UI, architecture diagram, eval results

## Project structure

```
banking-support-agent/
  data/
    generate_mock_data.py   # generates the four JSON tables below
    customers.json
    accounts.json
    transactions.json
    diagnostics.json
  graph/
    state.py                # AgentState TypedDict — the full shared state schema
    config.py                # policy constants (thresholds, retry caps)
    nodes.py                 # node functions + router functions (no langgraph dependency)
    graph.py                 # StateGraph assembly — wires nodes.py into an actual graph
    llm.py                   # Anthropic API wrapper, used only by LLM-backed nodes
  tests/
    test_routers.py          # unit tests for the three router functions
    test_intake_node.py      # unit tests for intake_node (mocks the LLM call)
```

## Running it for real

`graph/llm.py` reads your API key from the `ANTHROPIC_API_KEY` environment
variable. Set it before running the graph:

```
export ANTHROPIC_API_KEY=sk-ant-...
```

Router and node-logic tests (`tests/`) don't need this — they mock the LLM
call. You'll only need a real key once you're running the compiled graph
end to end.

## Design notes

Routing is deterministic Python, not LLM judgment — the LLM handles language,
Python handles policy. This keeps decisions auditable, unit-testable, and
resistant to prompt injection. Full design rationale, state schema, and node
design live alongside this repo in the project's design docs.
