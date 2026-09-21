#!/usr/bin/env python3
"""The desk half of the OptionsPilot loop.

OptionsPilot weights this desk at 45% of its composite. A name with no frozen
view scores zero on that component and can never clear its bar, so "the desk is
reachable" and "the desk is useful" are different facts — and the whole point of
this module is to report and fix the second.

Every test writes to a throwaway ledger. A suite that can reach the canonical
one is a suite that will eventually poison the calibration record.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision import desk_feed

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


def _tmp_ledger():
    from data import prediction_ledger as pl
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    pl.set_db_path(path)
    return path


def _snapshot(ticker, action, days_ago=0):
    """Freeze one view straight into the throwaway ledger."""
    from data import prediction_ledger as pl
    when = (dt.datetime.now() - dt.timedelta(days=days_ago)).strftime("%Y-%m-%dT00:00:00")
    return pl.freeze_prediction({
        "ticker": ticker, "generated_at": when, "current_price": 100.0,
        "action": action, "time_horizon_days": 91, "expected_return_pct": 5.0,
        "confidence": {"thesis": {"level": "MEDIUM", "score": 0.6},
                       "data": {"level": "MEDIUM", "score": 0.6},
                       "statistical_edge": {"level": "LOW", "score": 0.4, "checks": {}},
                       "allocation": {"level": "NONE", "score": 0.0}},
        "pillars": {"technical": {"score": 60, "confidence": 0.8}},
        "composite": 60.0, "sector": "Technology", "regime": "MEDIUM",
        "position_size_pct": 0.0, "claims": {}, "data_asof": when[:10],
        "experiment_manifest_hash": "test", "decision_fingerprint": "fp-" + ticker,
    })


# ══════════════════════════════════════════════════════════════════════════
# Coverage — reachable is not useful
# ══════════════════════════════════════════════════════════════════════════

def test_the_four_states_are_distinguished():
    path = _tmp_ledger()
    try:
        _snapshot("FRESHBULL", "BUY", days_ago=0)
        _snapshot("OLDBULL", "BUY", days_ago=desk_feed.MAX_AGE_DAYS + 5)
        _snapshot("NODIR", "HOLD", days_ago=0)
        c = desk_feed.coverage(["FRESHBULL", "OLDBULL", "NODIR", "NEVERSEEN"])
        check("directional", c["directional"] == ["FRESHBULL"], c["directional"])
        check("stale", c["stale"] == ["OLDBULL"], c["stale"])
        check("neutral (a HOLD is not a direction)", c["neutral"] == ["NODIR"], c["neutral"])
        check("absent", c["absent"] == ["NEVERSEEN"], c["absent"])
        check("usable fraction counts only directional", c["usable_fraction"] == 0.25,
              c["usable_fraction"])
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_a_hold_is_as_useless_to_the_consumer_as_silence():
    """OptionsPilot scores a neutral view zero on its directional component,
    exactly as it scores an absent one. Counting HOLD as coverage would report
    a desk that is speaking while saying nothing actionable."""
    path = _tmp_ledger()
    try:
        for t in ("H1", "H2", "H3"):
            _snapshot(t, "HOLD", days_ago=0)
        c = desk_feed.coverage(["H1", "H2", "H3"])
        check("nothing is directional", c["directional"] == [])
        check("usable fraction is zero", c["usable_fraction"] == 0.0)
        check("the message says nothing is usable", "NO name" in c["means"], c["means"])
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_the_staleness_bar_matches_the_consumer():
    """A desk that calls a view fresh while its consumer calls it stale is
    worse than either rule alone."""
    check("ten days, same as OptionsPilot", desk_feed.MAX_AGE_DAYS == 10)


def test_stance_vocabulary_matches_the_consumer():
    for action in ("BUY", "accumulate", "STRONG_BUY", "long"):
        check(f"{action} is bullish", desk_feed.stance_for(action) == "bullish")
    for action in ("SELL", "reduce", "underweight"):
        check(f"{action} is bearish", desk_feed.stance_for(action) == "bearish")
    for action in ("HOLD", "", None, "WATCH"):
        check(f"{action!r} is neutral", desk_feed.stance_for(action) == "neutral")


def test_coverage_of_nothing_is_not_a_crash():
    c = desk_feed.coverage([])
    check("universe is zero", c["universe"] == 0)
    check("usable fraction is None, not zero", c["usable_fraction"] is None)
    check("it says so", "No symbols" in c["means"])


# ══════════════════════════════════════════════════════════════════════════
# Refresh
# ══════════════════════════════════════════════════════════════════════════

def test_refresh_only_works_on_names_that_need_it():
    path = _tmp_ledger()
    try:
        _snapshot("HAVEIT", "BUY", days_ago=0)
        called = []

        def fake_freeze(ticker, **kw):
            called.append(ticker)
            _snapshot(ticker, "BUY", days_ago=0)
            return {"ticker": ticker, "status": "done"}

        with patch("agents.heartbeat.forecast_and_freeze", side_effect=fake_freeze):
            r = desk_feed.refresh(["HAVEIT", "NEEDIT"], limit=10)
        check("only the name lacking a view was worked", called == ["NEEDIT"], called)
        check("attempted counts only real work", r["attempted"] == 1)
        check("the gain is reported", r["newly_directional"] == ["NEEDIT"],
              r["newly_directional"])
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_absent_names_are_worked_before_stale_and_neutral():
    """A bounded budget should buy the most coverage it can."""
    path = _tmp_ledger()
    try:
        _snapshot("STALEONE", "BUY", days_ago=desk_feed.MAX_AGE_DAYS + 3)
        _snapshot("NEUTRALONE", "HOLD", days_ago=0)
        called = []

        def fake_freeze(ticker, **kw):
            called.append(ticker)
            return {"ticker": ticker, "status": "done"}

        with patch("agents.heartbeat.forecast_and_freeze", side_effect=fake_freeze):
            desk_feed.refresh(["STALEONE", "NEUTRALONE", "ABSENTONE"], limit=1)
        check("the absent name went first", called == ["ABSENTONE"], called)
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_the_budget_is_respected_and_the_remainder_reported():
    path = _tmp_ledger()
    try:
        called = []

        def fake_freeze(ticker, **kw):
            called.append(ticker)
            return {"ticker": ticker, "status": "done"}

        with patch("agents.heartbeat.forecast_and_freeze", side_effect=fake_freeze):
            r = desk_feed.refresh(["A1", "A2", "A3", "A4", "A5"], limit=2)
        check("only the budget was spent", len(called) == 2, called)
        check("the remainder is reported", r["remaining"] == 3, r["remaining"])
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_one_bad_symbol_does_not_end_the_run():
    path = _tmp_ledger()
    try:
        def fake_freeze(ticker, **kw):
            if ticker == "BOOM":
                raise RuntimeError("provider exploded")
            return {"ticker": ticker, "status": "done"}

        with patch("agents.heartbeat.forecast_and_freeze", side_effect=fake_freeze):
            r = desk_feed.refresh(["BOOM", "FINE"], limit=5)
        states = {x["ticker"]: x["status"] for x in r["results"]}
        check("the bad one is recorded as an error", states.get("BOOM") == "error", states)
        check("the good one still ran", states.get("FINE") == "done", states)
    finally:
        from data import prediction_ledger as pl
        pl.set_db_path(None)
        os.unlink(path)


def test_refresh_reuses_the_path_the_ledger_grades():
    """Not a new scoring path. A view served to a consumer that nothing later
    grades would be worse than no view."""
    import inspect
    src = inspect.getsource(desk_feed.refresh)
    check("it calls the heartbeat's own per-ticker path",
          "forecast_and_freeze" in src)
    check("it does not build its own recommendation",
          "build_recommendation" not in src)


# ══════════════════════════════════════════════════════════════════════════
# The endpoints
# ══════════════════════════════════════════════════════════════════════════

def _client():
    import web.app as app_mod
    return app_mod, app_mod.app.test_client()


def test_coverage_endpoint_requires_symbols():
    _, c = _client()
    check("400 without symbols", c.get("/api/desk/coverage").status_code == 400)


def test_refresh_endpoint_validates_its_input():
    _, c = _client()
    check("400 on a missing list", c.post("/api/desk/refresh", json={}).status_code == 400)
    check("400 on a non-list",
          c.post("/api/desk/refresh", json={"symbols": "AAPL"}).status_code == 400)


def test_refresh_endpoint_caps_the_budget():
    """An unbounded limit would let one request build hundreds of
    recommendations and hold a worker for the whole time."""
    _, c = _client()
    with patch.object(desk_feed, "refresh", return_value={"ok": True}) as ref:
        c.post("/api/desk/refresh", json={"symbols": ["A"], "limit": 9999})
        check("the limit is capped", ref.call_args.kwargs["limit"] <= 40,
              ref.call_args.kwargs)
        c.post("/api/desk/refresh", json={"symbols": ["A"], "limit": 0})
        check("and floored", ref.call_args.kwargs["limit"] >= 1, ref.call_args.kwargs)


def test_the_feed_adds_no_execution_path():
    app_mod, _ = _client()
    rules = [r.rule.lower() for r in app_mod.app.url_map.iter_rules()]
    banned = ("order", "buy", "sell", "trade", "execute", "place", "exercise")
    check("no execution route exists", not [r for r in rules if any(b in r for b in banned)])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
