"""
app.py

A Streamlit chat interface for talking to the compiled graph directly —
real LLM calls, real LangGraph interrupts — so you can actually try the
agent instead of only reading eval reports.

Run with:
    streamlit run ui/app.py

Needs a real LLM_API_KEY, same as eval/run_eval.py — loads the repo-root
.env the same way (no python-dotenv dependency, to keep this project's
runtime deps minimal).
"""

import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")

import streamlit as st
from langgraph.types import Command

from graph.graph import build_graph
from eval.scenarios import base_state
from eval.runner import _extract_explanation

st.set_page_config(page_title="Banking Support Agent", page_icon="\U0001f3e6")


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_data
def load_customers():
    with open(REPO_ROOT / "data" / "customers.json") as f:
        return json.load(f)


def reset_call(customer_id: str) -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.customer_id = customer_id
    st.session_state.messages = []
    st.session_state.phase = "awaiting_description"
    st.session_state.pending_interrupt = None
    st.session_state.final_state = None


def advance(graph, config, invoke_arg) -> None:
    """Runs the graph until it either hits an interrupt or finishes, updating
    session state either way."""
    result = graph.invoke(invoke_arg, config=config)
    if result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        st.session_state.pending_interrupt = payload
        st.session_state.messages.append({"role": "assistant", "content": payload.get("prompt", "")})
        st.session_state.phase = "awaiting_interrupt"
    else:
        st.session_state.pending_interrupt = None
        st.session_state.final_state = result
        explanation = _extract_explanation(result.get("resolution_notes")) or (
            result.get("resolution_notes") or ["The call has ended."]
        )[-1]
        st.session_state.messages.append({"role": "assistant", "content": explanation})
        st.session_state.phase = "done"


customers = load_customers()
customer_labels = {f"{c['name']} ({c['customer_id']})": c["customer_id"] for c in customers}

with st.sidebar:
    st.header("Banking Support Agent")
    st.caption(
        "Talks to the real compiled LangGraph graph — real LLM calls, real "
        "interrupt/resume gates (OTP, disambiguation, card freeze)."
    )
    chosen_label = st.selectbox("Calling as", list(customer_labels.keys()))
    if st.button("Start new call", type="primary"):
        reset_call(customer_labels[chosen_label])
        st.rerun()

    if st.session_state.get("final_state"):
        with st.expander("Case details (what the agent actually decided)"):
            fs = st.session_state.final_state
            st.json({
                "call_reason": fs.get("call_reason"),
                "transaction_id": fs.get("transaction_id"),
                "resolution_type": fs.get("resolution_type"),
                "resolution_status": fs.get("resolution_status"),
                "escalated": fs.get("escalated"),
                "card_frozen": fs.get("card_frozen"),
                "case_summary": fs.get("case_summary"),
            })
            st.caption("Tool audit log")
            st.json([a["tool"] for a in fs.get("tool_audit_log", [])])

if "phase" not in st.session_state:
    reset_call(customer_labels[chosen_label])

st.title("\U0001f3e6 Banking Support")
st.caption(f"Calling as **{chosen_label}** — thread `{st.session_state.thread_id[:8]}`")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

graph = get_graph()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if not os.getenv("LLM_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "No LLM_API_KEY (or ANTHROPIC_API_KEY) found — set one in a .env file at the "
        "repo root before running this. See HANDOFF.md."
    )
elif st.session_state.phase == "awaiting_description":
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("Hi, thanks for calling! What can I help you with today?")
    user_text = st.chat_input("Describe why you're calling...")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        initial_state = base_state(
            st.session_state.customer_id,
            None,  # no transaction_id — let the real lookup/matching run
            user_text,
            st.session_state.thread_id,
        )
        with st.spinner("Thinking..."):
            advance(graph, config, initial_state)
        st.rerun()

elif st.session_state.phase == "awaiting_interrupt":
    action = (st.session_state.pending_interrupt or {}).get("action")

    if action == "card_freeze_confirmation":
        col1, col2 = st.columns(2)
        response = None
        if col1.button("Yes, freeze my card", type="primary"):
            response = "yes"
        if col2.button("No, don't freeze it"):
            response = "no"
        if response:
            st.session_state.messages.append({"role": "user", "content": response})
            with st.spinner("Thinking..."):
                advance(graph, config, Command(resume=response))
            st.rerun()

    else:
        # otp_request, transaction_disambiguation, and anything else: free-text reply
        user_text = st.chat_input("Your reply...")
        if user_text:
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.spinner("Thinking..."):
                advance(graph, config, Command(resume=user_text))
            st.rerun()

else:  # done
    st.success("Call ended. Start a new call from the sidebar to try another scenario.")
