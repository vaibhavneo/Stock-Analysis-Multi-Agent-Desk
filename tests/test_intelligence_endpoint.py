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
        self.plan_sections, self.run_selected = mocks[0], mocks[1]
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
    test_avg_cost_sets_owns_position_and_reaches_run_selected()
    test_no_avg_cost_means_owns_position_false()
    test_exception_returns_500_not_a_crash()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — /api/intelligence: wires adaptive sections into the Decision Report")
