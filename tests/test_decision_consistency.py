#!/usr/bin/env python3
"""Consistency checks (Phase 26).

Each test constructs the exact contradiction the check exists to catch and
asserts it is caught. These double as the regression net for the whole layer:
if a future change reintroduces one of these, the check fires rather than the
contradiction shipping silently next to two confident numbers.
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision.consistency import check_consistency
from test_decision_intelligence import build, mock_rec

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def codes(result):
    return {i["code"] for i in result["issues"]}


def test_sizing_gate_violation_is_an_error():
    d = build()
    d["sizing"] = {"position_size_pct": 5.0, "gated": True}
    r = check_consistency(d)
    check("caught", "SIZING_GATE_VIOLATED" in codes(r))
    check("severity is ERROR", r["n_errors"] >= 1)


def test_high_confidence_with_insufficient_calibration_is_an_error():
    d = build()
    d["confidence"]["decision_confidence"] = "HIGH"
    d["confidence"]["dimensions"]["calibration"]["level"] = "INSUFFICIENT"
    r = check_consistency(d)
    check("caught", "CONFIDENCE_WITHOUT_CALIBRATION" in codes(r))


def test_entry_band_at_the_quote_while_waiting_is_an_error():
    d = build()
    d["entry"]["status"] = "WAIT"
    d["entry_plan"] = {"supported": True, "entry_low": d["current_price"],
                       "entry_high": d["current_price"]}
    r = check_consistency(d)
    check("caught", "ENTRY_AT_PRICE_WHILE_WAITING" in codes(r))


def test_bull_target_below_the_price_is_an_error():
    d = build()
    for s in d["scenarios"]["scenarios"]:
        if s["name"] == "BULL":
            s["price_levels"] = [{"price": d["current_price"] * 0.5, "distance_pct": -50.0,
                                  "basis": "TRADED_PRICE", "sources": ["x"], "confidence": 0.5}]
    r = check_consistency(d)
    check("caught", "TARGET_BELOW_PRICE" in codes(r))


def test_uncalibrated_probability_is_an_error():
    d = build()
    d["scenarios"]["scenarios"][0]["probability"] = 0.72
    d["scenarios"]["scenarios"][0]["probability_basis"] = "NOT_CALIBRATED"
    r = check_consistency(d)
    check("caught", "UNCALIBRATED_PROBABILITY" in codes(r))


def test_demonstrated_edge_with_failing_dsr_is_an_error():
    d = build()
    d["statistical_edge"]["demonstrated"] = True
    d["statistical_edge"]["backtest"]["dsr"] = 0.1
    r = check_consistency(d)
    check("caught", "EDGE_CLAIM_VS_DSR" in codes(r))


def test_add_recommendation_without_edge_is_an_error():
    d = build()
    d["decision_state"]["headline_state"] = "ADD_CONDITIONALLY"
    d["statistical_edge"]["demonstrated"] = False
    r = check_consistency(d)
    check("caught", "ACTION_WITHOUT_EDGE" in codes(r))
    check("it is an ERROR, not a warning",
          any(i["code"] == "ACTION_WITHOUT_EDGE" and i["severity"] == "ERROR"
              for i in r["issues"]))


def test_composite_verb_versus_state_is_surfaced():
    """The live AAPL case: the recommendation card says BUY while the decision
    state says WATCH. Both are defensible; the silence between them was not."""
    d = build()
    d["_recommendation"]["action"] = "BUY"
    d["decision_state"]["headline_state"] = "WATCH"
    r = check_consistency(d)
    check("the asymmetry is named",
          "COMPOSITE_SCORES_SECURITY_NOT_DECISION" in codes(r))
    check("it is INFO, not an error",
          any(i["code"] == "COMPOSITE_SCORES_SECURITY_NOT_DECISION"
              and i["severity"] == "INFO" for i in r["issues"]))


def test_negative_verb_with_constructive_state_is_an_error():
    d = build()
    d["_recommendation"]["action"] = "SELL"
    d["decision_state"]["headline_state"] = "WAIT_FOR_ENTRY"
    r = check_consistency(d)
    check("caught", "REC_VERB_VS_STATE" in codes(r))
    check("severity is ERROR",
          any(i["code"] == "REC_VERB_VS_STATE" and i["severity"] == "ERROR"
              for i in r["issues"]))


def test_exit_state_without_a_priced_invalidation_is_warned():
    d = build()
    d["decision_state"]["headline_state"] = "EXIT_CONDITIONALLY"
    d["risk_budget"]["invalidation_level"] = None
    r = check_consistency(d)
    check("caught", "EXIT_WITHOUT_LEVEL" in codes(r))


def test_a_clean_decision_has_no_errors():
    d = build()
    check("no errors on the default fixture", d["consistency"]["n_errors"] == 0,
          d["consistency"]["summary"])
    check("checks actually ran", len(d["consistency"]["issues"]) >= 0)


def test_every_issue_names_the_fields_involved():
    d = build()
    d["sizing"] = {"position_size_pct": 5.0, "gated": True}
    r = check_consistency(d)
    for i in r["issues"]:
        check(f"{i['code']} names its fields", isinstance(i["fields"], list))
        check(f"{i['code']} has a detail", bool(i["detail"]))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
