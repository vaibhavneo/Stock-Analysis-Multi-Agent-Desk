"""
Verification for agents/prediction_schema.py and the retry/validation
wiring inside agents/stock_agents.py::run_prediction_agent.

Run: python3 tests/test_prediction_schema.py

Offline/deterministic - the LLM-calling parts are stubbed via
unittest.mock.patch on agents.stock_agents._call; no network, no API key.

What must hold:
  1. validate_prediction() never raises, on anything (None, {}, a list,
     wrong types, bad enums, out-of-range numbers).
  2. A structurally valid dict passes; each specific defect is named in
     `errors`.
  3. The hardcoded HOLD/LOW fallback dict in run_prediction_agent is
     itself self-consistent with the validator (the bug this phase fixed:
     it used to be missing time_horizon/time_horizon_days/watch_levels).
  4. run_prediction_agent retries once on a structurally-invalid-but-
     syntactically-valid JSON response, same as it already did for an
     empty/error response - and gives up gracefully (fallback +
     validation_errors) if both attempts fail.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.prediction_schema import validate_prediction
from agents.stock_agents import run_prediction_agent

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


VALID = {
    "action": "BUY",
    "conviction": "HIGH",
    "time_horizon": "medium-term (1-3 months)",
    "time_horizon_days": 90,
    "entry_price": "$150-155",
    "target_price": "$180",
    "stop_loss": "$140",
    "upside_pct": 15.5,
    "downside_pct": -7.2,
    "risk_reward": "2.1:1",
    "summary": "Strong buy on fundamentals and technicals.",
    "bull_case": "Earnings beat continues.",
    "bear_case": "Macro headwinds.",
    "key_catalysts": ["earnings", "product launch"],
    "watch_levels": {"support": "$145", "resistance": "$165"},
    "scores": {"fundamentals": 8, "technical": 7, "social": 6, "algo": 7, "overall": 7},
}


def test_valid_dict_passes():
    result = validate_prediction(VALID)
    check("a fully valid dict passes", result.valid, str(result.errors))
    check("a valid dict has no errors", result.errors == [])


def test_missing_required_key():
    bad = dict(VALID)
    del bad["stop_loss"]
    result = validate_prediction(bad)
    check("missing key fails validation", not result.valid)
    check("missing key is named in errors",
          any("stop_loss" in e for e in result.errors), str(result.errors))


def test_wrong_type():
    bad = dict(VALID)
    bad["upside_pct"] = "15.5%"  # string instead of a number
    result = validate_prediction(bad)
    check("wrong type fails validation", not result.valid)
    check("wrong-type field is named in errors",
          any("upside_pct" in e for e in result.errors), str(result.errors))


def test_bad_enum():
    bad = dict(VALID)
    bad["action"] = "Maybe"
    result = validate_prediction(bad)
    check("bad enum value fails validation", not result.valid)
    check("bad enum field is named in errors",
          any("action" in e for e in result.errors), str(result.errors))


def test_out_of_range_score():
    bad = dict(VALID)
    bad["scores"] = dict(VALID["scores"], overall=15)
    result = validate_prediction(bad)
    check("out-of-range sub-score fails validation", not result.valid)
    check("out-of-range field is named in errors",
          any("overall" in e for e in result.errors), str(result.errors))


def test_never_raises_on_garbage():
    for garbage in (None, {}, [], "a string", 42, {"action": "BUY"}):
        try:
            result = validate_prediction(garbage)
            check(f"never raises on {garbage!r}", True)
            if garbage != {"action": "BUY"}:
                check(f"{garbage!r} is invalid", not result.valid)
        except Exception as e:
            check(f"never raises on {garbage!r}", False, f"raised {e!r}")


def test_stray_bool_rejected_for_numeric_field():
    bad = dict(VALID)
    bad["upside_pct"] = True  # bool is technically an int subclass in Python
    result = validate_prediction(bad)
    check("a stray bool is rejected for a numeric field", not result.valid, str(result.errors))


# ── run_prediction_agent retry/validation wiring ────────────────────────

VALID_JSON = json.dumps(VALID)
INVALID_ACTION_JSON = json.dumps(dict(VALID, action="Maybe"))
UNPARSEABLE = "Sorry, I can't help with that."


def _run(call_side_effect):
    with patch("agents.stock_agents._call", side_effect=call_side_effect) as mock_call:
        result = run_prediction_agent(
            client=object(), ticker="TEST", company_name="Test Co", current_price=100.0,
            fundamentals_analysis="...", technical_analysis="...",
            signal_summary={"score": 50, "direction": "NEUTRAL"},
        )
        return result, mock_call.call_count


def test_valid_first_response_returns_immediately():
    result, call_count = _run([VALID_JSON])
    check("valid first response is returned as-is", result["action"] == "BUY")
    check("only 1 call made when the first response is already valid", call_count == 1)


def test_empty_response_retries_once():
    result, call_count = _run(["", VALID_JSON])
    check("empty-then-valid retries and returns the valid response", result["action"] == "BUY")
    check("exactly 2 calls made", call_count == 2)


def test_structurally_invalid_json_retries_once():
    result, call_count = _run([INVALID_ACTION_JSON, VALID_JSON])
    check("invalid-action-then-valid retries and returns the valid response", result["action"] == "BUY")
    check("exactly 2 calls made (validation failure triggers a retry, same as empty/error)", call_count == 2)


def test_both_attempts_fail_falls_back_with_validation_errors():
    result, call_count = _run([INVALID_ACTION_JSON, INVALID_ACTION_JSON])
    check("falls back to HOLD when both attempts are structurally invalid", result["action"] == "HOLD")
    check("fallback carries validation_errors for diagnosability",
          "validation_errors" in result and len(result["validation_errors"]) > 0,
          str(result.get("validation_errors")))
    check("exactly 2 calls made (no 3rd attempt)", call_count == 2)


def test_both_attempts_unparseable_falls_back_without_validation_errors():
    result, call_count = _run([UNPARSEABLE, UNPARSEABLE])
    check("falls back to HOLD when nothing ever parses", result["action"] == "HOLD")
    check("no validation_errors key when nothing ever parsed as JSON at all",
          "validation_errors" not in result)


def test_fallback_dict_itself_passes_validation():
    result, _ = _run([UNPARSEABLE, UNPARSEABLE])
    validation = validate_prediction(result if "validation_errors" not in result
                                      else {k: v for k, v in result.items() if k != "validation_errors"})
    check("the hardcoded fallback dict is itself self-consistent with the validator",
          validation.valid, str(validation.errors))


if __name__ == "__main__":
    test_valid_dict_passes()
    test_missing_required_key()
    test_wrong_type()
    test_bad_enum()
    test_out_of_range_score()
    test_never_raises_on_garbage()
    test_stray_bool_rejected_for_numeric_field()
    test_valid_first_response_returns_immediately()
    test_empty_response_retries_once()
    test_structurally_invalid_json_retries_once()
    test_both_attempts_fail_falls_back_with_validation_errors()
    test_both_attempts_unparseable_falls_back_without_validation_errors()
    test_fallback_dict_itself_passes_validation()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — prediction_schema validator + retry/validation wiring")
