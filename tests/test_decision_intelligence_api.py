#!/usr/bin/env python3
"""API surface for Decision Intelligence (Phases 20, 27, 28).

Network and the heavy research path are stubbed: these test the ENDPOINT — its
contract, its caching, its refusal to infer a position, and its refusal to leak
a secret — not the research engine underneath, which has its own tests.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from test_decision_intelligence import build

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


def _client():
    import web.app as app_mod
    app_mod._DI_CACHE.clear()
    return app_mod, app_mod.app.test_client()


def test_endpoint_requires_a_ticker():
    _, c = _client()
    r = c.post("/api/decision-intelligence", json={})
    check("400 without a ticker", r.status_code == 400)


def test_endpoint_returns_the_structured_object():
    app_mod, c = _client()
    with patch.object(app_mod, "_build_decision_intelligence", return_value=build()):
        r = c.post("/api/decision-intelligence", json={"ticker": "TEST"})
    check("200", r.status_code == 200)
    d = r.get_json()
    for key in ("decision_state", "thesis", "statistical_edge", "level_map", "entry",
                "playbook", "scenarios", "catalysts", "monitoring", "confidence",
                "quality", "statistical_honesty", "consistency", "evidence"):
        check(f"response carries {key}", key in d)


def test_position_is_never_inferred():
    app_mod, c = _client()
    captured = {}

    def _fake(ticker, period, deep, position, max_risk, narrative):
        captured["position"] = position
        return build(position=position)

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence", json={"ticker": "TEST"})
    check("no position is passed when none is supplied", captured["position"] is None)

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence",
               json={"ticker": "TEST", "avg_cost": 50, "shares": 10, "portfolio_value": 1000})
    check("position is passed through verbatim",
          captured["position"] == {"avg_cost": 50.0, "shares": 10.0, "portfolio_value": 1000.0})


def test_invalid_position_values_are_dropped_not_guessed():
    app_mod, c = _client()
    captured = {}

    def _fake(ticker, period, deep, position, max_risk, narrative):
        captured["position"] = position
        return build()

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence", json={"ticker": "TEST", "avg_cost": "abc"})
    check("an unparseable cost basis yields no position", captured["position"] is None)

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence", json={"ticker": "TEST", "avg_cost": -5})
    check("a negative cost basis yields no position", captured["position"] is None)


def test_repeat_requests_are_served_from_cache():
    """Phase 28: one evidence snapshot, many views — not one research run per
    panel."""
    app_mod, c = _client()
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return build()

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        first = c.post("/api/decision-intelligence", json={"ticker": "TEST"}).get_json()
        second = c.post("/api/decision-intelligence", json={"ticker": "TEST"}).get_json()
    check("the engine ran once", calls["n"] == 1)
    check("the second response is marked cached", second.get("_cached") is True)
    check("the payload is the same decision",
          first["decision_fingerprint"] == second["decision_fingerprint"])


def test_different_position_context_is_a_different_cache_entry():
    app_mod, c = _client()
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return build()

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence", json={"ticker": "TEST"})
        c.post("/api/decision-intelligence", json={"ticker": "TEST", "avg_cost": 90})
    check("a different position recomputes", calls["n"] == 2)


def test_narrative_requests_are_never_cached():
    """A cached narrative could outlive the object it was validated against."""
    app_mod, c = _client()
    calls = {"n": 0}

    def _fake(*a, **k):
        calls["n"] += 1
        return build()

    with patch.object(app_mod, "_build_decision_intelligence", side_effect=_fake):
        c.post("/api/decision-intelligence", json={"ticker": "TEST", "narrative": True})
        c.post("/api/decision-intelligence", json={"ticker": "TEST", "narrative": True})
    check("both narrative requests recomputed", calls["n"] == 2)


def test_no_secret_reaches_the_response():
    app_mod, c = _client()
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-do-not-leak-0000")
    with patch.object(app_mod, "_build_decision_intelligence", return_value=build()):
        body = c.post("/api/decision-intelligence", json={"ticker": "TEST"}).data.decode()
    check("no sk- token in the payload", "sk-" not in body)
    check("no key name in the payload", "DEEPSEEK_API_KEY" not in body)


def test_no_order_placement_route_exists():
    """Paper/research only — this must remain true by inspection, not by
    intention."""
    app_mod, _ = _client()
    rules = [r.rule.lower() for r in app_mod.app.url_map.iter_rules()]
    banned = ("order", "buy", "sell", "trade", "execute", "place")
    offenders = [r for r in rules if any(b in r for b in banned)]
    check("no order/trade/execute route exists", not offenders, offenders)


def test_journal_endpoint_is_read_only():
    app_mod, c = _client()
    r = c.get("/api/decision-journal?limit=5")
    check("200", r.status_code == 200)
    d = r.get_json()
    check("returns a summary", "summary" in d)
    check("returns decisions", isinstance(d.get("decisions"), list))
    check("no write verb is routed",
          c.post("/api/decision-journal", json={}).status_code in (404, 405))


def test_forward_endpoint_reports_without_claiming_an_edge():
    _, c = _client()
    r = c.get("/api/decision-journal/forward?horizon=20")
    check("200", r.status_code == 200)
    v = r.get_json()["validation"]
    check("a verdict is reported", v["verdict"] in ("SUFFICIENT_SAMPLE", "NO_DEMONSTRATED_EDGE"))
    check("the safeguard is restated", "authoritative" in v["safeguard"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
