"""
Verification for POST /api/intelligence (web/app.py::intelligence_endpoint) -
the new route wiring intelligence/orchestration.py's adaptive computation
into the same synthesize_decision() Decision Report every other decision
endpoint produces.

Run: python3 tests/test_intelligence_endpoint.py

Offline/deterministic - uses Flask's test client (no real server, no
network). Every expensive call (_build_full_recommendation,
_build_backtest_all, _gather_cheap_enrichments, plan_sections,
run_selected) is mocked at its source, following the same lesson learned
earlier this session with tests/test_app_calibration_bugfix.py:
_gather_cheap_enrichments itself calls a real, unrelated, network-hitting
xsection.ranking.run_ranking() - mocking the whole function sidesteps that
rather than needing to know about it here too.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import web.app as app_module

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


FAKE_REC = {
    "ticker": "TEST", "current_price": 100.0, "sector": "Technology", "regime": "MEDIUM",
    "action": "BUY", "composite": 72,
    "confidence": {
        "thesis": {"level": "MEDIUM", "score": 0.6},
        "data": {"level": "HIGH", "score": 0.8},
        "statistical_edge": {"level": "HIGH", "score": 0.8},
        "allocation": {"level": "HIGH", "score": 0.8},
    },
    "pillars": {k: {"score": 60, "confidence": 0.8, "backtestable": True, "flags": []}
                for k in ("technical", "algo", "risk", "fundamentals", "research", "social")},
    "levels": {"stop_loss": 94.0, "target_price": 112.0, "atr_14": 3.0},
    "position_size_pct": 5.0, "position_size_gated": False, "time_horizon_days": 91,
    "honesty_flags": {}, "claims": {}, "decision_fingerprint": "fp0",
}
FAKE_INTEL = {
    "ticker": "TEST", "sections": ["historical_context"], "current_price": 100.0,
    "regime": {"risk_stance": "RISK_ON", "trend": "BULLISH", "confidence": 0.9},
    "historical_context": {"confidence": 0.8, "flags": []},
    "analog": None, "forecast": None, "risk_profile": None, "evidence": None,
    "flags": [],
}


class _MockSet:
    """Starts every patch, exposes the individual mocks, stops everything on
    exit - a context manager since `with patch(...) as m:` can't be built
    from a list literal."""

    def __init__(self, sections_result=None):
        intel = dict(FAKE_INTEL)
        if sections_result is not None:
            intel["sections"] = sections_result
        self._patchers = [
            patch("intelligence.orchestration.plan_sections", return_value=intel["sections"]),
            patch("intelligence.orchestration.run_selected", return_value=intel),
            patch.object(app_module, "_build_full_recommendation", return_value=(FAKE_REC, pd.DataFrame())),
            patch.object(app_module, "_build_backtest_all", return_value=None),
            patch.object(app_module, "_gather_cheap_enrichments", return_value=(None, None, None)),
        ]

    def __enter__(self):
        mocks = [p.start() for p in self._patchers]
        (self.plan_sections, self.run_selected, self.build_full_recommendation,
         self.build_backtest_all, self.gather_cheap_enrichments) = mocks
        return self

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()


def _client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_missing_ticker_returns_400():
    resp = _client().post("/api/intelligence", json={})
    check("missing ticker returns HTTP 400", resp.status_code == 400)
    check("error body mentions no ticker", "ticker" in (resp.get_json() or {}).get("error", "").lower())


def test_basic_request_returns_a_decision_report_with_intelligence_sections():
    with _MockSet() as m:
        resp = _client().post("/api/intelligence", json={"ticker": "test"})
    check("200 OK on a well-formed request", resp.status_code == 200, str(resp.get_json()))
    body = resp.get_json()
    check("ticker is uppercased", body.get("ticker") == "TEST")
    check("final_action present (same Decision Report shape as /api/decision)", "final_action" in body)
    check("market_regime carries through from run_selected's regime", body.get("market_regime") == FAKE_INTEL["regime"])
    check("sections_computed is present in the response", body.get("sections_computed") == FAKE_INTEL["sections"])


def test_plan_sections_called_with_the_request_body_fields():
    with _MockSet() as m:
        _client().post("/api/intelligence", json={"ticker": "test", "sections": "valuation"})
    check("plan_sections was called with the requested sections value",
          m.plan_sections.call_args.args[0] == "valuation"
          or m.plan_sections.call_args.kwargs.get("requested") == "valuation",
          str(m.plan_sections.call_args))


def test_narrow_preset_skips_backtest_all_and_calibration_by_horizon():
    """The valuation preset ([historical_context] only, no evidence/analog)
    must stay cheap: no backtest_all race, no calibration_by_horizon query.
    This is the fix for the ~80s narrow-preset latency found by browser-
    testing the real UI - regression-guarded here so it can't silently
    regress back."""
    with _MockSet(sections_result=["historical_context"]) as m:
        resp = _client().post("/api/intelligence", json={"ticker": "test", "sections": "valuation"})
        called_backtest_all = m.build_backtest_all.called
        called_enrichments = m.gather_cheap_enrichments.called
    body = resp.get_json()
    check("200 OK on the narrow preset", resp.status_code == 200, str(body))
    check("_build_backtest_all is NOT called for a narrow (non-deep) preset - the ~80s fix",
          not called_backtest_all)
    check("_gather_cheap_enrichments is NOT called for a narrow (non-deep) preset",
          not called_enrichments)
    check("calibration_by_horizon is absent from the response for a narrow preset",
          "calibration_by_horizon" not in body, str(body.get("calibration_by_horizon")))


def test_deep_preset_includes_calibration_by_horizon():
    """The full preset (includes both evidence and analog) is the one place
    item 11's forecast-vs-track-record pairing is worth the extra query -
    confirmed reaching the response body with the real ledger function's
    shape (n / win_rate / avg_return_pct / brier per horizon)."""
    fake_by_horizon = {
        5: {"overall": {"n": 40, "win_rate": 0.55, "avg_raw_return_pct": 1.1, "brier": 0.24}},
        20: {"overall": {"n": 38, "win_rate": 0.61, "avg_raw_return_pct": 2.3, "brier": 0.21}},
    }
    deep_sections = ["regime", "historical_context", "analog", "forecast", "risk", "evidence"]
    with _MockSet(sections_result=deep_sections) as m, \
         patch("data.prediction_ledger.calibration_report_all_horizons", return_value=fake_by_horizon) as mock_cal:
        resp = _client().post("/api/intelligence", json={"ticker": "test", "sections": "full"})
    body = resp.get_json()
    check("200 OK on the full/deep preset", resp.status_code == 200, str(body))
    check("calibration_report_all_horizons was actually called for the deep preset", mock_cal.called)
    cbh = body.get("calibration_by_horizon")
    check("calibration_by_horizon is present in the response", cbh is not None, str(body.keys()))
    if cbh:
        check("horizon 5's win_rate carries through correctly", cbh.get("5", {}).get("win_rate") == 0.55, str(cbh))
        check("horizon 20's n carries through correctly", cbh.get("20", {}).get("n") == 38, str(cbh))


def test_calibration_by_horizon_failure_degrades_to_none_not_500():
    deep_sections = ["regime", "historical_context", "analog", "forecast", "risk", "evidence"]
    with _MockSet(sections_result=deep_sections) as m, \
         patch("data.prediction_ledger.calibration_report_all_horizons", side_effect=RuntimeError("db locked")):
        resp = _client().post("/api/intelligence", json={"ticker": "test", "sections": "full"})
    check("a calibration_by_horizon failure still returns 200, not a 500", resp.status_code == 200)
    check("calibration_by_horizon is explicitly None (not silently dropped, not a crash)",
          resp.get_json().get("calibration_by_horizon") is None)


def test_avg_cost_sets_owns_position_and_reaches_run_selected():
    with _MockSet() as m:
        resp = _client().post("/api/intelligence", json={"ticker": "test", "avg_cost": 120.0, "shares": 10})
    check("200 OK with a position supplied", resp.status_code == 200, str(resp.get_json()))
    check("plan_sections received has_position=True from the supplied avg_cost",
          m.plan_sections.call_args.kwargs.get("has_position") is True
          or (len(m.plan_sections.call_args.args) > 1 and m.plan_sections.call_args.args[1] is True),
          str(m.plan_sections.call_args))
    check("run_selected received the avg_cost and shares",
          m.run_selected.call_args.kwargs.get("avg_cost") == 120.0
          and m.run_selected.call_args.kwargs.get("shares") == 10,
          str(m.run_selected.call_args))


def test_no_avg_cost_means_owns_position_false():
    with _MockSet() as m:
        _client().post("/api/intelligence", json={"ticker": "test"})
    check("plan_sections received has_position=False with no avg_cost supplied",
          m.plan_sections.call_args.kwargs.get("has_position") is False
          or (len(m.plan_sections.call_args.args) > 1 and m.plan_sections.call_args.args[1] is False),
          str(m.plan_sections.call_args))


def test_exception_returns_500_not_a_crash():
    with patch("intelligence.orchestration.plan_sections", side_effect=RuntimeError("boom")):
        resp = _client().post("/api/intelligence", json={"ticker": "TEST"})
    check("an internal exception returns HTTP 500, not an unhandled crash", resp.status_code == 500)
    check("error message surfaced in the response body", "boom" in (resp.get_json() or {}).get("error", ""))


if __name__ == "__main__":
    test_missing_ticker_returns_400()
    test_basic_request_returns_a_decision_report_with_intelligence_sections()
    test_plan_sections_called_with_the_request_body_fields()
    test_narrow_preset_skips_backtest_all_and_calibration_by_horizon()
    test_deep_preset_includes_calibration_by_horizon()
    test_calibration_by_horizon_failure_degrades_to_none_not_500()
    test_avg_cost_sets_owns_position_and_reaches_run_selected()
    test_no_avg_cost_means_owns_position_false()
    test_exception_returns_500_not_a_crash()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — /api/intelligence: wires adaptive sections into the Decision Report")
