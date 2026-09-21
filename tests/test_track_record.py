#!/usr/bin/env python3
"""The "what this engine has actually done" panel.

The panel this replaces listed eight chips and then restated four of them as
five warning lines — all true, none of it useful. These tests hold the
replacement to being useful: it must lead with the buy-and-hold comparison, it
must treat completed trades rather than bars as the sample, and it must state
plainly that none of it forecasts the next session.

That last one has a test because it is the expectation a reader naturally
brings to a panel labelled "backtest and calibration", and the honest answer is
no.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision.plain import has_jargon
from decision.track_record import (LIVE_SAMPLE_READABLE, TRADES_THIN,
                                   TRADES_TOO_FEW, build_live_record,
                                   build_strategy_comparison, build_track_record)
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


def mock_backtest_all(core_annual=4.6, hold_annual=36.5, core_beats=False):
    return {
        "ticker": "TEST",
        "rows": [
            {"strategy": "sma_crossover", "sharpe": 0.9, "dsr": 0.0,
             "annualized_return_pct": 50.8, "max_dd_pct": 48.9, "n_trades": 4,
             "beats_hold": True},
            {"strategy": "momentum", "sharpe": 0.64, "dsr": 0.0,
             "annualized_return_pct": 37.6, "max_dd_pct": 54.8, "n_trades": 111,
             "beats_hold": True},
            {"strategy": "seven_pillar_core", "sharpe": 0.018, "dsr": 0.0,
             "annualized_return_pct": core_annual, "max_dd_pct": 38.7, "n_trades": 27,
             "beats_hold": core_beats},
            {"strategy": "stat_arb", "sharpe": -0.67, "dsr": 0.0,
             "annualized_return_pct": -7.1, "max_dd_pct": 36.1, "n_trades": 22,
             "beats_hold": False},
        ],
        "buy_hold": {"sharpe": 0.59, "annualized_return_pct": hold_annual, "max_dd_pct": 58.5},
    }


def mock_horizons(excess_by_horizon=None):
    excess_by_horizon = excess_by_horizon or {1: -0.14, 5: -0.34, 20: -1.09,
                                              60: -0.94, 126: -0.51, 252: 8.28}
    return {h: {"overall": {"n": 300, "win_rate": 0.51, "avg_excess_return_pct": e,
                            "brier": 0.30}}
            for h, e in excess_by_horizon.items()}


# ══════════════════════════════════════════════════════════════════════════
# The comparison that should lead the panel
# ══════════════════════════════════════════════════════════════════════════

def test_losing_to_buy_and_hold_is_the_headline():
    d = build()
    tr = build_track_record(mock_rec(), d["statistical_edge"], mock_backtest_all(),
                            mock_horizons())
    check("headline names the buy-and-hold loss",
          "did not beat simply holding" in tr["headline"], tr["headline"])
    check("both annual returns are stated",
          "+4.6%" in tr["comparison"]["statement"] and "+36.5%" in tr["comparison"]["statement"],
          tr["comparison"]["statement"])
    check("it says this is the important number",
          "most important number" in tr["comparison"]["statement"])


def test_beating_buy_and_hold_changes_the_headline():
    """A mutation: the panel must actually respond to the comparison."""
    d = build()
    losing = build_track_record(mock_rec(), d["statistical_edge"],
                                mock_backtest_all(core_annual=4.6, core_beats=False),
                                mock_horizons())
    winning = build_track_record(mock_rec(), d["statistical_edge"],
                                 mock_backtest_all(core_annual=60.0, core_beats=True),
                                 mock_horizons())
    check("headline moves", losing["headline"] != winning["headline"])
    check("winning case does not claim a loss",
          "did not beat" not in winning["headline"], winning["headline"])
    check("winning comparison reads positively",
          "did better than" in winning["comparison"]["statement"],
          winning["comparison"]["statement"])


def test_the_engines_own_strategy_is_marked_in_the_table():
    cmp = build_strategy_comparison(mock_rec(), mock_backtest_all())
    core = [r for r in cmp["rows"] if r["is_core"]]
    check("exactly one row is the core strategy", len(core) == 1)
    check("it is named", core[0]["strategy"] == "seven_pillar_core")
    check("rank is reported", cmp["core_rank"] == 3, cmp["core_rank"])
    check("how many beat holding is reported", cmp["n_beating_hold"] == 2)
    check("the one-stock caveat is stated", "one comparison" in cmp["caveat"])


def test_missing_library_degrades_without_inventing():
    cmp = build_strategy_comparison(mock_rec(), None)
    check("status is UNAVAILABLE", cmp["status"] == "UNAVAILABLE")
    check("no rows are fabricated", cmp["rows"] == [])
    check("it says why", "not run" in cmp["statement"])


# ══════════════════════════════════════════════════════════════════════════
# Trades, not bars
# ══════════════════════════════════════════════════════════════════════════

def test_trade_count_is_treated_as_the_sample():
    d = build()
    for n, expected in ((6, "TOO_FEW"), (27, "TOO_FEW"),
                        (TRADES_THIN - 1, "THIN"), (TRADES_THIN + 50, "ADEQUATE")):
        rec = mock_rec()
        rec["backtest"]["n_trades"] = n
        tr = build_track_record(rec, d["statistical_edge"], mock_backtest_all(), mock_horizons())
        check(f"{n} trades reads {expected}", tr["sample"]["verdict"] == expected,
              tr["sample"]["verdict"])
        check(f"{n} trades is quoted in the sentence", str(n) in tr["sample"]["statement"])


def test_the_panel_explains_why_the_bar_count_misleads():
    d = build()
    tr = build_track_record(mock_rec(), d["statistical_edge"], mock_backtest_all(),
                            mock_horizons())
    check("bar/trade distinction is stated",
          "bars" in tr["sample"]["why_bars_mislead"]
          and "Trades are the sample" in tr["sample"]["why_bars_mislead"])
    check("the bar count is still reported", "bars_tested" in tr["sample"])


def test_too_few_trades_can_become_the_headline():
    """When the strategy DID beat holding but on six trades, the sample is the
    story."""
    d = build()
    rec = mock_rec()
    rec["backtest"]["n_trades"] = 6
    tr = build_track_record(rec, d["statistical_edge"],
                            mock_backtest_all(core_annual=60.0, core_beats=True),
                            mock_horizons())
    check("headline is about the sample", "too few completed trades" in tr["headline"].lower(),
          tr["headline"])


# ══════════════════════════════════════════════════════════════════════════
# The live record
# ══════════════════════════════════════════════════════════════════════════

def test_live_record_reports_every_horizon():
    live = build_live_record(mock_horizons())
    check("six horizons", len(live["rows"]) == 6)
    check("sorted by horizon", [r["horizon_days"] for r in live["rows"]]
          == sorted(r["horizon_days"] for r in live["rows"]))
    check("total is summed", live["total_matured"] == 1800)


def test_live_record_leads_with_excess_return_not_win_rate():
    """Win rate 51% looks fine; a negative average excess return does not. The
    summary must be driven by the second."""
    live = build_live_record(mock_horizons())
    check("names how many horizons trail the benchmark",
          "trailed the benchmark at 5 of 6" in live["statement"], live["statement"])
    check("names the exception", "252" in live["statement"])
    check("the disagreement is explained",
          "counts money" in live["caveat"])


def test_a_uniformly_losing_record_says_so_plainly():
    live = build_live_record(mock_horizons({1: -0.2, 20: -1.0, 252: -3.0}))
    check("all horizons trailing is stated bluntly",
          "not yet a good one" in live["statement"], live["statement"])


def test_thin_horizons_are_flagged():
    thin = {20: {"overall": {"n": LIVE_SAMPLE_READABLE - 1, "win_rate": 0.9,
                             "avg_excess_return_pct": 5.0, "brier": 0.2}}}
    live = build_live_record(thin)
    check("a thin sample is marked unreadable", live["rows"][0]["readable"] is False)


def test_no_live_record_is_a_gap_not_a_pass():
    live = build_live_record(None)
    check("status is UNAVAILABLE", live["status"] == "UNAVAILABLE")
    check("it is framed as a gap in the evidence",
          "gap in the evidence" in live["statement"], live["statement"])


# ══════════════════════════════════════════════════════════════════════════
# The limit a reader will bump into
# ══════════════════════════════════════════════════════════════════════════

def test_the_panel_states_it_does_not_forecast_the_next_session():
    d = build()
    tr = build_track_record(mock_rec(), d["statistical_edge"], mock_backtest_all(),
                            mock_horizons())
    text = tr["what_this_cannot_tell_you"].lower()
    check("next session is named", "next session" in text)
    check("it says none of it predicts", "none of this predicts" in text)
    check("it says what it IS for", bool(tr["what_this_is_good_for"]))
    check("what it is for mentions sizing", "size" in tr["what_this_is_good_for"].lower())


def test_the_rendered_text_is_readable():
    d = build()
    tr = build_track_record(mock_rec(), d["statistical_edge"], mock_backtest_all(),
                            mock_horizons())
    for key, value in (("headline", tr["headline"]),
                       ("comparison", tr["comparison"]["statement"]),
                       ("comparison caveat", tr["comparison"]["caveat"]),
                       ("sample", tr["sample"]["statement"]),
                       ("why bars mislead", tr["sample"]["why_bars_mislead"]),
                       ("live record", tr["live_record"]["statement"]),
                       ("live caveat", tr["live_record"]["caveat"]),
                       ("cannot tell you", tr["what_this_cannot_tell_you"]),
                       ("good for", tr["what_this_is_good_for"])):
        leaks = has_jargon(value)
        check(f"{key} has no jargon", not leaks, leaks)


def test_track_record_is_on_the_default_decision():
    """It was gated behind `deep` alongside a 70-second ranking while costing
    0.58 seconds, which is why the panel was empty."""
    d = build()
    check("present on every decision", "track_record" in d)
    check("it always has a headline", bool(d["track_record"]["headline"]))
    check("it always states the limit", bool(d["track_record"]["what_this_cannot_tell_you"]))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
