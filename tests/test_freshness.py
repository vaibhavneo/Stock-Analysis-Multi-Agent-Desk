#!/usr/bin/env python3
"""How old is the number, and has the price left it behind?

The brief showed "PRICE NOW $227.38" while the last trade was $228.87. The
daily bar series had not published that day's session yet — 390 one-minute
bars had already printed and closed — so the headline price, and every
distance-from-here derived from it, were a day behind and labelled current.

The error was 0.65%. Labelling stale data "now" is the real defect, because
nothing on the page let the reader discover it.

These tests hold the two properties that matter: the session calendar must not
silently expire, and a level crossed since the close must be reported — that
is the one piece of real-time information that changes what to do.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas import freshness as F
from mas.freshness import (market_session, market_holidays, build_freshness,
                           OPEN, CLOSED, PRE, POST, ALWAYS_OPEN)

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


def _et(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=F._ET)


# ── The session calendar ──────────────────────────────────────────────────

def test_holidays_are_derived_per_year_not_hard_coded():
    """A hard-coded list expires, and an expired holiday calendar reports
    'the market is open' on Christmas."""
    for year in (2026, 2027, 2028, 2030, 2035):
        hols = market_holidays(year)
        check(f"{year} has ten closures", len(hols) == 10, len(hols))
        check(f"{year} includes Christmas-observed",
              any(h.month == 12 and h.day in (24, 25, 26) for h in hols), hols)
        check(f"{year} has no weekend closures",
              all(h.weekday() < 5 for h in hols),
              [str(h) for h in hols if h.weekday() >= 5])


def test_good_friday_is_found_each_year():
    for year, expected in ((2026, dt.date(2026, 4, 3)),
                           (2027, dt.date(2027, 3, 26)),
                           (2028, dt.date(2028, 4, 14))):
        check(f"{year} Good Friday", expected in market_holidays(year),
              [str(h) for h in market_holidays(year)])


def test_regular_hours_are_recognised():
    check("11am is open", market_session("EQUITY", _et(2026, 9, 23, 11, 0))["state"] == OPEN)
    check("8am is pre", market_session("EQUITY", _et(2026, 9, 23, 8, 0))["state"] == PRE)
    check("5pm is after hours",
          market_session("EQUITY", _et(2026, 9, 23, 17, 0))["state"] == POST)
    check("2am is closed", market_session("EQUITY", _et(2026, 9, 23, 2, 0))["state"] == CLOSED)


def test_weekends_and_holidays_are_closed():
    check("Saturday", market_session("EQUITY", _et(2026, 9, 26, 11, 0))["state"] == CLOSED)
    check("Christmas", market_session("EQUITY", _et(2026, 12, 25, 11, 0))["state"] == CLOSED)
    check("and it says which",
          "holiday" in market_session("EQUITY", _et(2026, 12, 25, 11, 0))["statement"])


def test_asset_classes_keep_their_own_calendars():
    """A coin has no close to be behind; an equity does."""
    check("crypto is always open",
          market_session("CRYPTO", _et(2026, 12, 25, 3, 0))["state"] == ALWAYS_OPEN)
    check("and says there is no close",
          "no close" in market_session("CRYPTO")["statement"])
    check("fx is shut at the weekend",
          market_session("FX", _et(2026, 9, 26, 11, 0))["state"] == CLOSED)


# ── Drift and crossings ───────────────────────────────────────────────────

def _q(price):
    return {"available": True, "price": price, "source": "last trade",
            "as_of": "now", "age_sec": 0, "reason": ""}


def test_drift_is_reported_against_the_price_actually_computed_on():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=227.38,
                          computed_on_date="2026-09-21", quote=_q(228.87))
    check("drift computed", abs(out["drift_pct"] - 0.655) < 0.01, out["drift_pct"])
    check("both prices are in the sentence",
          "227.38" in out["statement"] and "228.87" in out["statement"],
          out["statement"])


def test_a_level_crossed_since_the_close_is_reported():
    """The whole point of the layer."""
    out = build_freshness("NVDA", "EQUITY", computed_on_price=227.38,
                          computed_on_date="2026-09-21", quote=_q(228.87),
                          levels=[{"label": "the nearest resistance", "price": 228.0},
                                  {"label": "the invalidation", "price": 219.36}])
    check("one crossing", len(out["crossed"]) == 1, out["crossed"])
    check("the right one", out["crossed"][0]["price"] == 228.0, out["crossed"])
    check("direction stated", "upward" in out["crossed"][0]["direction"])
    check("marked stale", out["stale"] is True)
    check("and the sentence says the level is behind the analysis",
          "behind the analysis" in out["statement"], out["statement"])


def test_a_downward_crossing_is_caught_too():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=227.38,
                          quote=_q(218.00),
                          levels=[{"label": "the invalidation", "price": 219.36}])
    check("crossed downward", len(out["crossed"]) == 1, out["crossed"])
    check("direction stated", "downward" in out["crossed"][0]["direction"])


def test_levels_not_between_the_two_prices_are_not_reported():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=227.38,
                          quote=_q(228.87),
                          levels=[{"label": "the entry trigger", "price": 233.43},
                                  {"label": "the invalidation", "price": 219.36}])
    check("nothing crossed", out["crossed"] == [], out["crossed"])
    check("and it is not marked stale", out["stale"] is False)


def test_a_large_move_without_a_crossing_still_flags_a_re_run():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=200.0,
                          quote=_q(210.0), levels=[])
    check("stale", out["stale"] is True, out)
    check("and says why", "re-running" in out["statement"], out["statement"])


def test_no_quote_degrades_to_the_computed_price_with_a_reason():
    out = build_freshness("ZZZZ", "EQUITY", computed_on_price=10.0,
                          quote={"available": False, "price": None,
                                 "reason": "the provider returned no last trade"})
    check("no drift invented", out["drift_pct"] is None, out)
    check("the reason is carried",
          "no last trade" in out["statement"], out["statement"])
    check("not marked stale on an absent quote", out["stale"] is False)


def test_a_missing_computed_price_does_not_produce_a_fake_drift():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=None, quote=_q(100.0))
    check("no drift", out["drift_pct"] is None, out)


def test_live_quote_never_raises_and_never_returns_a_bare_none_price():
    out = F.live_quote("DEFINITELY_NOT_A_TICKER_XYZ")
    check("returns a dict", isinstance(out, dict))
    check("available is a bool", isinstance(out["available"], bool))
    if not out["available"]:
        check("price is None and a reason is given",
              out["price"] is None and bool(out["reason"]), out)


def test_the_analysis_basis_is_named_so_the_two_prices_are_not_conflated():
    out = build_freshness("NVDA", "EQUITY", computed_on_price=227.38,
                          quote=_q(228.87))
    check("the basis is stated",
          "settled bar series" in out["computed_on"]["basis"],
          out["computed_on"])


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
