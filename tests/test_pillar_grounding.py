"""
Verification for pillar-score grounding in the 4 analysis-agent prompts
(agents/stock_agents.py's _pillar_block + each _build_X_prompt's optional
`pillar` param, and agents/orchestrator.py::_run_analysis_agents threading
the right pillar sub-dict to each agent).

Run: python3 tests/test_pillar_grounding.py

Offline/deterministic - no network, no API key.

What must hold:
  1. Passing a pillar dict adds a "DETERMINISTIC ... PILLAR SCORE" block
     naming that pillar's score and formula to the built prompt.
  2. Omitting pillar (or passing None) leaves the prompt BYTE-IDENTICAL
     to what it was before this phase - callers that don't opt in see
     no change at all.
  3. _run_analysis_agents routes pillars["technical"] to the technical
     agent, pillars["fundamentals"] to the fundamentals agent, etc. -
     never mixed up - and a missing/empty pillars dict degrades to
     "no pillar block for anyone" rather than raising.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.orchestrator import _run_analysis_agents
from agents.stock_agents import (
    _build_algo_prompt,
    _build_fundamentals_prompt,
    _build_social_prompt,
    _build_technical_prompt,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


SAMPLE_PILLAR = {"score": 62.5, "formula": "mean(rsi, macd, sma voting)"}

BUILDER_CASES = [
    ("fundamentals", "FUNDAMENTALS", _build_fundamentals_prompt,
     dict(ticker="AAPL", fundamentals={}, earnings={}, analyst_ratings={})),
    ("technical", "TECHNICAL", _build_technical_prompt,
     dict(ticker="AAPL", indicators={}, signal_summary={}, price_history_summary="")),
    ("social", "SOCIAL", _build_social_prompt,
     dict(ticker="AAPL", company_name="Apple", reddit_data={}, stocktwits_data={}, web_forum_data={})),
    ("algo", "ALGO", _build_algo_prompt,
     dict(ticker="AAPL", current_price=100.0, algo_signals={}, indicators={})),
]


def test_pillar_block_appears_when_passed():
    for name, label, builder, kwargs in BUILDER_CASES:
        _system, user = builder(**kwargs, pillar=SAMPLE_PILLAR)
        check(f"{name}: pillar block present when pillar is passed",
              f"DETERMINISTIC {label} PILLAR SCORE" in user)
        check(f"{name}: pillar block cites the actual score",
              "62.5/100" in user)
        check(f"{name}: pillar block cites the actual formula",
              "mean(rsi, macd, sma voting)" in user)


def test_prompt_is_byte_identical_without_pillar():
    for name, _label, builder, kwargs in BUILDER_CASES:
        _sys_omit, user_omit = builder(**kwargs)
        _sys_none, user_none = builder(**kwargs, pillar=None)
        _sys_with, user_with = builder(**kwargs, pillar=SAMPLE_PILLAR)
        check(f"{name}: omitting pillar == passing pillar=None",
              user_omit == user_none)
        check(f"{name}: omitting pillar produces a DIFFERENT (shorter) prompt than passing one",
              user_omit != user_with and len(user_omit) < len(user_with))


def _noop_progress(stage, msg):
    pass


def test_orchestrator_routes_correct_pillar_to_each_agent():
    captured = {}

    def _capture(agent_name):
        def _fn(*args, **kwargs):
            captured[agent_name] = kwargs.get("pillar")
            return f"{agent_name}-output"
        return _fn

    pillars = {
        "fundamentals": {"score": 10, "formula": "f"},
        "technical":    {"score": 20, "formula": "t"},
        "social":       {"score": 30, "formula": "s"},
        "algo":         {"score": 40, "formula": "a"},
    }

    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.run_fundamentals_agent", side_effect=_capture("fundamentals")), \
         patch("agents.orchestrator.run_technical_agent", side_effect=_capture("technical")), \
         patch("agents.orchestrator.run_social_agent", side_effect=_capture("social")), \
         patch("agents.orchestrator.run_algo_agent", side_effect=_capture("algo")):
        _run_analysis_agents(
            resolved_key="dummy-key", ticker="TEST",
            fundamentals={}, earnings={}, analyst_ratings={},
            indicators={}, signal_summary={}, price_hist_summary="",
            company_name="Test Co", reddit_data={}, stocktwits_data={},
            web_forum_data={}, current_price=100.0, algo_signals={},
            pillars=pillars, progress=_noop_progress,
        )

    check("fundamentals agent received the fundamentals pillar (not another one)",
          captured.get("fundamentals") == pillars["fundamentals"])
    check("technical agent received the technical pillar (not another one)",
          captured.get("technical") == pillars["technical"])
    check("social agent received the social pillar (not another one)",
          captured.get("social") == pillars["social"])
    check("algo agent received the algo pillar (not another one)",
          captured.get("algo") == pillars["algo"])


def test_empty_pillars_dict_degrades_gracefully():
    captured = {}

    def _capture(agent_name):
        def _fn(*args, **kwargs):
            captured[agent_name] = kwargs.get("pillar")
            return f"{agent_name}-output"
        return _fn

    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.run_fundamentals_agent", side_effect=_capture("fundamentals")), \
         patch("agents.orchestrator.run_technical_agent", side_effect=_capture("technical")), \
         patch("agents.orchestrator.run_social_agent", side_effect=_capture("social")), \
         patch("agents.orchestrator.run_algo_agent", side_effect=_capture("algo")):
        _run_analysis_agents(
            resolved_key="dummy-key", ticker="TEST",
            fundamentals={}, earnings={}, analyst_ratings={},
            indicators={}, signal_summary={}, price_hist_summary="",
            company_name="Test Co", reddit_data={}, stocktwits_data={},
            web_forum_data={}, current_price=100.0, algo_signals={},
            pillars={}, progress=_noop_progress,
        )

    check("empty pillars dict does not raise, and every agent gets pillar=None",
          all(v is None for v in captured.values()), str(captured))


if __name__ == "__main__":
    test_pillar_block_appears_when_passed()
    test_prompt_is_byte_identical_without_pillar()
    test_orchestrator_routes_correct_pillar_to_each_agent()
    test_empty_pillars_dict_degrades_gracefully()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — pillar-score grounding: correct routing, no change when opted out")
