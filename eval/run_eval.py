"""
run_eval.py

Step 6 evaluation suite entry point. Runs every scenario in eval/scenarios.py
against the real compiled graph (real LLM calls via whatever LLM_PROVIDER is
configured, real LangGraph interrupts) and prints + writes a report.

Needs a real LLM_API_KEY — unlike tests/test_*.py, this is not mocked and not
free of network calls. Loads a repo-root .env if present (no python-dotenv
dependency; this project keeps runtime deps minimal on purpose).

Usage:
    python3 -m eval.run_eval
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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

from eval.scenarios import SCENARIOS  # noqa: E402 - after dotenv load, before config reads
from eval.runner import run_scenario  # noqa: E402


def _format_scenario_report(result) -> str:
    lines = [f"## {result.scenario_id} — {'PASS' if result.passed else 'FAIL'}"]
    if result.error:
        lines.append(f"- **Error:** {result.error}")
        return "\n".join(lines)

    if result.state_check_failures:
        lines.append("- **State assertion failures:**")
        for failure in result.state_check_failures:
            lines.append(f"  - {failure}")
    else:
        lines.append("- State assertions: all passed")

    lines.append(f"- Interrupts hit: {[t['interrupt'].get('action') for t in result.trajectory]}")

    if result.explanation:
        lines.append(f"- Customer explanation: \"{result.explanation}\"")
    if result.judge_verdict:
        lines.append(
            f"- LLM judge: **{result.judge_verdict['verdict']}** — {result.judge_verdict['reasoning']}"
        )
    return "\n".join(lines)


def main() -> int:
    if not os.getenv("LLM_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "No LLM_API_KEY (or ANTHROPIC_API_KEY) found in the environment or .env — "
            "the eval suite needs a real LLM to call. See HANDOFF.md for how this project "
            "sets up a free Groq key.",
            file=sys.stderr,
        )
        return 2

    results = []
    for scenario in SCENARIOS:
        print(f"Running {scenario.id}...", flush=True)
        result = run_scenario(scenario)
        results.append(result)
        print(f"  -> {'PASS' if result.passed else 'FAIL'}")

    passed_count = sum(1 for r in results if r.passed)
    report_lines = [
        f"# Eval report — {datetime.now(timezone.utc).isoformat()}",
        f"\n**{passed_count}/{len(results)} scenarios passed**\n",
    ]
    report_lines.extend(_format_scenario_report(r) for r in results)
    report = "\n\n".join(report_lines)

    print("\n" + "=" * 60)
    print(report)

    reports_dir = REPO_ROOT / "eval" / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "latest.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")

    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
