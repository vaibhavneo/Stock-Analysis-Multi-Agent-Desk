"""
Verification for Phase 6: the optional self-critique pass inside
run_prediction_agent (agents/stock_agents.py::_self_critique).

Run: python3 tests/test_self_critique.py

Offline/deterministic - agents.stock_agents._call is stubbed via
unittest.mock.patch with a sequence of canned responses; no network, no
API key.

What must hold:
  1. self_critique=False (the default) never makes a second call - the
     draft is returned as-is, no latency/cost added.
  2. self_critique=True, critique says CONSISTENT -> original draft kept,
     tagged "consistent".
  3. self_critique=True, critique returns a validating revision -> the
     REVISION is adopted, tagged "revised".
  4. self_critique=True, critique response is unparseable garbage ->
     original draft kept, tagged "flagged_but_kept_original" - a broken
     critique must never replace a good draft.
  5. self_critique=True, critique parses but fails validate_prediction
     (e.g. a bad enum) -> same as #4, original kept and flagged.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stock_agents import run_prediction_agent

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


DRAFT = {
    "action": "BUY", "conviction": "HIGH", "time_horizon": "medium-term (1-3 months)",
    "time_horizon_days": 90, "entry_price": "$100", "target_price": "$120",
    "stop_loss": "$90", "upside_pct": 20.0, "downside_pct": -10.0, "risk_reward": "2:1",
    "summary": "Buy on strength.", "bull_case": "ok", "bear_case": "ok", "key_catalysts": [],
    "watch_levels": {"support": "$95", "resistance": "$115"},
    "scores": {"fundamentals": 7, "technical": 7, "social": 6, "algo": 7, "overall": 7},
}

REVISION = dict(DRAFT, action="HOLD", summary="Revised: mixed signals, not a clean buy.")


def _run(self_critique, second_call_response=None):
    responses = [json.dumps(DRAFT)]
    if second_call_response is not None:
        responses.append(second_call_response)
    with patch("agents.stock_agents._call", side_effect=responses) as mock_call:
        result = run_prediction_agent(
            client=object(), ticker="TEST", company_name="Test Co", current_price=100.0,
            fundamentals_analysis="Bullish fundamentals.", technical_analysis="Bullish trend.",
            signal_summary={"score": 70, "direction": "BULLISH"},
            social_analysis="Neutral.", algo_analysis="Bullish momentum.",
            self_critique=self_critique,
        )
        return result, mock_call.call_count


def test_off_by_default_makes_only_one_call():
    result, call_count = _run(self_critique=False)
    check("self_critique=False returns the draft unmodified", result["action"] == "BUY")
    check("self_critique=False never adds a self_critique key", "self_critique" not in result)
    check("self_critique=False makes exactly 1 call (no critique call)", call_count == 1)


def test_consistent_keeps_original_and_tags_it():
    result, call_count = _run(self_critique=True, second_call_response="CONSISTENT")
    check("CONSISTENT keeps the original action", result["action"] == "BUY")
    check("CONSISTENT keeps the original summary", result["summary"] == DRAFT["summary"])
    check("CONSISTENT is tagged", result.get("self_critique") == "consistent")
    check("exactly 2 calls made (draft + critique)", call_count == 2)


def test_valid_revision_is_adopted_and_tagged():
    result, call_count = _run(self_critique=True, second_call_response=json.dumps(REVISION))
    check("a validating revision replaces the draft's action", result["action"] == "HOLD")
    check("a validating revision replaces the draft's summary", result["summary"] == REVISION["summary"])
    check("revision is tagged", result.get("self_critique") == "revised")
    check("exactly 2 calls made", call_count == 2)


def test_unparseable_critique_keeps_original():
    result, call_count = _run(self_critique=True, second_call_response="I refuse to answer.")
    check("unparseable critique keeps the original action", result["action"] == "BUY")
    check("unparseable critique keeps the original summary", result["summary"] == DRAFT["summary"])
    check("unparseable critique is flagged", result.get("self_critique") == "flagged_but_kept_original")
    check("exactly 2 calls made", call_count == 2)


def test_structurally_invalid_revision_keeps_original():
    bad_revision = json.dumps(dict(DRAFT, action="Maybe"))  # parses, fails the enum check
    result, call_count = _run(self_critique=True, second_call_response=bad_revision)
    check("a non-validating revision does NOT replace the draft's action", result["action"] == "BUY")
    check("a non-validating revision is flagged, not silently adopted",
          result.get("self_critique") == "flagged_but_kept_original")
    check("exactly 2 calls made", call_count == 2)


if __name__ == "__main__":
    test_off_by_default_makes_only_one_call()
    test_consistent_keeps_original_and_tags_it()
    test_valid_revision_is_adopted_and_tagged()
    test_unparseable_critique_keeps_original()
    test_structurally_invalid_revision_keeps_original()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — self-critique pass: opt-in, never replaces a good draft with a broken one")
