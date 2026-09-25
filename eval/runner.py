"""
runner.py

Drives one Scenario through the real compiled graph: real LLM calls, real
LangGraph interrupts, resumed either with a scenario's fixed scripted
responses or with a simulated customer's in-character replies. Collects the
final state, the interrupt trajectory, and the customer-facing explanation,
then runs the state assertions and the LLM-judge check against it.
"""

from dataclasses import dataclass, field

from langgraph.types import Command

from graph.graph import build_graph
from eval.simulated_customer import respond_as_customer
from eval.judge import judge_explanation
from eval.scenarios import Scenario


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    state_check_failures: list = field(default_factory=list)
    trajectory: list = field(default_factory=list)
    explanation: str = None
    judge_verdict: dict = None
    final_state: dict = None
    error: str = None


def _extract_explanation(resolution_notes: list) -> str:
    prefix = "customer explanation: "
    for note in reversed(resolution_notes or []):
        if note.startswith(prefix):
            return note[len(prefix):]
    return None


def _next_response(scenario: Scenario, interrupt_payload: dict) -> str:
    if scenario.simulated_customer_persona:
        return respond_as_customer(scenario.simulated_customer_persona, interrupt_payload.get("prompt", ""))
    action = interrupt_payload.get("action")
    response = (scenario.scripted_responses or {}).get(action)
    if response is None:
        raise ValueError(
            f"Scenario '{scenario.id}' hit interrupt action '{action}' with no scripted "
            f"response configured for it."
        )
    return response


def run_scenario(scenario: Scenario) -> ScenarioResult:
    graph = build_graph()
    config = {"configurable": {"thread_id": scenario.id}}
    trajectory = []

    try:
        result = graph.invoke(scenario.initial_state, config=config)
        while result.get("__interrupt__"):
            interrupt_obj = result["__interrupt__"][0]
            payload = interrupt_obj.value
            response = _next_response(scenario, payload)
            trajectory.append({"interrupt": payload, "response": response})
            result = graph.invoke(Command(resume=response), config=config)
    except Exception as exc:  # noqa: BLE001 - eval harness: surface any failure as a scenario result
        return ScenarioResult(scenario_id=scenario.id, passed=False, trajectory=trajectory, error=repr(exc))

    state_check_failures = []
    for key, expected_value in scenario.expected.items():
        actual_value = result.get(key)
        if actual_value != expected_value:
            state_check_failures.append(f"{key}: expected {expected_value!r}, got {actual_value!r}")

    explanation = _extract_explanation(result.get("resolution_notes"))
    judge_verdict = None
    if explanation:
        judge_verdict = judge_explanation(explanation, scenario.ground_truth_facts, scenario.guardrail_rules)

    passed = not state_check_failures and (judge_verdict is None or judge_verdict["verdict"] == "PASS")

    return ScenarioResult(
        scenario_id=scenario.id,
        passed=passed,
        state_check_failures=state_check_failures,
        trajectory=trajectory,
        explanation=explanation,
        judge_verdict=judge_verdict,
        final_state=result,
    )
