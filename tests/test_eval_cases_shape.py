"""
Structural check for tools/eval/eval_cases.py fixtures.

Run: python3 tests/test_eval_cases_shape.py

Offline, no network, no API key. Catches a typo'd or missing-key fixture
before it burns real API calls in tools/eval_agents.py - each case is fed
straight into its agent's real `_build_X_prompt()` (the same function
production code uses), so this also doubles as a smoke test that the
prompt builders don't raise on any of the hand-curated eval inputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stock_agents import (
    _build_algo_prompt,
    _build_fundamentals_prompt,
    _build_prediction_prompt,
    _build_social_prompt,
    _build_technical_prompt,
)
from tools.eval.eval_cases import EVAL_CASES

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


BUILDERS = {
    "fundamentals": _build_fundamentals_prompt,
    "technical":    _build_technical_prompt,
    "social":       _build_social_prompt,
    "algo":         _build_algo_prompt,
    "prediction":   _build_prediction_prompt,
}


def test_every_case_has_notes_and_ticker():
    for agent, cases in EVAL_CASES.items():
        for i, case in enumerate(cases):
            check(f"{agent}[{i}] has a non-empty 'notes' rationale",
                  bool(case.get("notes", "").strip()), f"case={case.get('ticker')}")
            check(f"{agent}[{i}] has a 'ticker'",
                  bool(case.get("ticker")))


def test_every_case_builds_a_valid_prompt():
    for agent, cases in EVAL_CASES.items():
        builder = BUILDERS[agent]
        for i, case in enumerate(cases):
            kwargs = {k: v for k, v in case.items() if k != "notes"}
            try:
                system, user = builder(**kwargs)
                ok = isinstance(system, str) and isinstance(user, str) and bool(system.strip()) and bool(user.strip())
                check(f"{agent}[{i}] ({case.get('ticker')}) builds a non-empty (system, user) prompt",
                      ok)
            except TypeError as e:
                check(f"{agent}[{i}] ({case.get('ticker')}) builds a non-empty (system, user) prompt",
                      False, f"kwarg mismatch: {e}")


def test_every_agent_has_at_least_three_cases():
    for agent, cases in EVAL_CASES.items():
        check(f"{agent} has >= 3 eval cases", len(cases) >= 3, f"has {len(cases)}")


if __name__ == "__main__":
    test_every_case_has_notes_and_ticker()
    test_every_case_builds_a_valid_prompt()
    test_every_agent_has_at_least_three_cases()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — eval_cases.py: well-formed, buildable against real prompt builders")
