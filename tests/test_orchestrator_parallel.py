"""
Verification for parallel analysis-agent dispatch (agents/orchestrator.py
:: _run_analysis_agents).

Run: python3 tests/test_orchestrator_parallel.py

Offline/deterministic: all 4 analysis agent functions and _get_client are
patched at their `agents.orchestrator` binding (not `agents.stock_agents`,
since `from agents.stock_agents import run_fundamentals_agent, ...` creates
a separate name in orchestrator's own namespace - patching the source
module wouldn't affect orchestrator's already-bound reference).

What must hold:
  1. The 4 agents actually run concurrently, not sequentially (wall time
     close to one stub's duration, not the sum of all 4).
  2. Each agent's output lands in the correct output key.
  3. One agent raising doesn't prevent the other 3 from completing, and
     produces an "[Agent error: ...]" string in that one slot - matching
     _call()'s own convention for an API-level failure.
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import _run_analysis_agents

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _noop_progress(stage, msg):
    pass


def _call_with_stubs(fundamentals_fn, technical_fn, social_fn, algo_fn):
    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.run_fundamentals_agent", side_effect=fundamentals_fn), \
         patch("agents.orchestrator.run_technical_agent", side_effect=technical_fn), \
         patch("agents.orchestrator.run_social_agent", side_effect=social_fn), \
         patch("agents.orchestrator.run_algo_agent", side_effect=algo_fn):
        return _run_analysis_agents(
            resolved_key="dummy-key", ticker="TEST",
            fundamentals={}, earnings={}, analyst_ratings={},
            indicators={}, signal_summary={}, price_hist_summary="",
            company_name="Test Co", reddit_data={}, stocktwits_data={},
            web_forum_data={}, current_price=100.0, algo_signals={},
            progress=_noop_progress,
        )


STUB_DELAY = 0.3


def _slow_stub(marker):
    def _fn(*args, **kwargs):
        time.sleep(STUB_DELAY)
        return marker
    return _fn


def test_runs_concurrently_not_sequentially():
    start = time.time()
    _call_with_stubs(_slow_stub("F"), _slow_stub("T"), _slow_stub("S"), _slow_stub("A"))
    elapsed = time.time() - start
    # Sequential would take ~4 * STUB_DELAY; concurrent should stay close to
    # 1 * STUB_DELAY. 2x gives generous headroom for scheduling overhead
    # without letting a real regression to sequential execution slip through.
    check("4 agents run concurrently (elapsed < 2x one stub's delay)",
          elapsed < STUB_DELAY * 2,
          f"elapsed={elapsed:.2f}s (1x={STUB_DELAY}s, 4x={STUB_DELAY * 4}s)")


def test_outputs_land_in_correct_keys():
    result = _call_with_stubs(
        _slow_stub("FUND"), _slow_stub("TECH"), _slow_stub("SOC"), _slow_stub("ALGO"),
    )
    check("fundamentals_analysis has fundamentals agent's output",
          result["fundamentals_analysis"] == "FUND")
    check("technical_analysis has technical agent's output",
          result["technical_analysis"] == "TECH")
    check("social_analysis has social agent's output",
          result["social_analysis"] == "SOC")
    check("algo_analysis has algo agent's output",
          result["algo_analysis"] == "ALGO")


def test_one_failure_does_not_block_the_others():
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated agent failure")

    result = _call_with_stubs(_boom, _slow_stub("TECH"), _slow_stub("SOC"), _slow_stub("ALGO"))
    check("failing agent produces an '[Agent error:' string, not an exception",
          isinstance(result["fundamentals_analysis"], str)
          and result["fundamentals_analysis"].startswith("[Agent error:"),
          repr(result["fundamentals_analysis"]))
    check("technical agent still completed despite fundamentals failing",
          result["technical_analysis"] == "TECH")
    check("social agent still completed despite fundamentals failing",
          result["social_analysis"] == "SOC")
    check("algo agent still completed despite fundamentals failing",
          result["algo_analysis"] == "ALGO")


if __name__ == "__main__":
    test_runs_concurrently_not_sequentially()
    test_outputs_land_in_correct_keys()
    test_one_failure_does_not_block_the_others()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — orchestrator parallel dispatch: concurrent, correct, isolated")
