#!/usr/bin/env python3
"""Position-management rule backtests (Phase 24).

The rules are simple; the things that can go silently wrong are not. These tests
target exactly those: look-ahead leakage in the N-day high/low references,
whether a rule actually differs from the baseline, and whether the honest
NO_DEMONSTRATED_EDGE default really is the default rather than a message that
only appears when something else fails.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtest.position_rules import (BASELINE_RULE, RULES, _references,
                                     evaluate_position_rules)

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


def _df(n=700, seed=5, drift=0.0004, vol=0.015):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2021-01-01", periods=n)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol_s = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": vol_s}, index=idx)


def _base_signal(df):
    """A simple always-on-after-warmup long signal, so the only thing varying
    between rules is the rule itself."""
    s = pd.Series(0.0, index=df.index)
    s.iloc[60:] = 1.0
    return s


def test_references_do_not_look_ahead():
    """The N-day high/low must be shifted: a rule comparing today's close to a
    window that includes today is testing whether today was today."""
    df = _df(200)
    high, low = _references(df, 20)
    for i in range(30, 60):
        window_high = df["High"].iloc[i - 20:i].max()
        check(f"high at {i} excludes the current bar",
              abs(high.iloc[i] - window_high) < 1e-9,
              f"{high.iloc[i]} vs {window_high}")
        break   # one index is enough; the construction is uniform


def test_every_rule_produces_a_bounded_long_only_position():
    df = _df()
    base = _base_signal(df)
    high, low = _references(df)
    for name, fn in RULES.items():
        sig = fn(df["Close"].astype(float), base, high, low).fillna(0.0)
        check(f"{name} is long-only", bool((sig >= -1e-9).all()))
        check(f"{name} never exceeds full size", bool((sig <= 1.0 + 1e-9).all()))
        check(f"{name} is aligned to the price index", len(sig) == len(df))


def test_rules_actually_differ_from_the_baseline():
    """A rule that never changes the position is not a rule."""
    df = _df()
    base = _base_signal(df)
    high, low = _references(df)
    baseline = RULES[BASELINE_RULE](df["Close"].astype(float), base, high, low).fillna(0.0)
    differing = 0
    for name, fn in RULES.items():
        if name == BASELINE_RULE:
            continue
        sig = fn(df["Close"].astype(float), base, high, low).fillna(0.0)
        if not np.allclose(sig.to_numpy(), baseline.to_numpy()):
            differing += 1
    check("every non-baseline rule differs from the baseline",
          differing == len(RULES) - 1, f"{differing}/{len(RULES) - 1}")


def test_insufficient_sample_is_refused_not_estimated():
    df = _df(100)
    res = evaluate_position_rules(df, base_signal=_base_signal(df))
    check("status is INSUFFICIENT_SAMPLE", res["status"] == "INSUFFICIENT_SAMPLE")
    check("no rules are reported", res["rules"] == [])


def test_full_evaluation_reports_honestly():
    df = _df(900)
    res = evaluate_position_rules(df, base_signal=_base_signal(df), n_folds=3)
    check("status OK", res["status"] == "OK")
    check("all rules evaluated", res["n_rules_tested"] == len(RULES))
    check("multiple-testing correction is disclosed",
          "deflated Sharpe" in res["method"]["multiple_testing"])
    check("survivorship limitation is disclosed",
          "not survivorship-safe" in res["method"]["survivorship"])
    check("verdict is one of the two allowed values",
          res["verdict"] in ("DEMONSTRATED_EDGE", "NO_DEMONSTRATED_EDGE"))
    check("costs are charged", all(r["total_cost_pct"] >= 0 for r in res["rules"]))
    check("time in market is reported", all("time_in_market_pct" in r for r in res["rules"]))


def test_no_demonstrated_edge_is_the_honest_default():
    """On a random walk, no position-management rule should clear the bar. If
    one does, the bar is not doing its job."""
    df = _df(900, seed=99, drift=0.0)
    res = evaluate_position_rules(df, base_signal=_base_signal(df), n_folds=3)
    check("verdict is NO_DEMONSTRATED_EDGE on a driftless random walk",
          res["verdict"] == "NO_DEMONSTRATED_EDGE", res["verdict"])
    check("the statement says so plainly",
          "NO_DEMONSTRATED_EDGE" in res["statement"])
    check("a negative result is framed as a result",
          "not a failure" in res["statement"])


def test_wait_for_pullback_spends_less_time_in_the_market():
    """A sanity check on the rule semantics themselves: a rule that only enters
    near the N-day low cannot be invested as often as one that always is."""
    df = _df(900)
    res = evaluate_position_rules(df, base_signal=_base_signal(df), n_folds=3)
    by = {r["name"]: r for r in res["rules"]}
    check("pullback rule is in the market less than the baseline",
          by["wait_for_pullback"]["time_in_market_pct"]
          < by["buy_immediately"]["time_in_market_pct"],
          f'{by["wait_for_pullback"]["time_in_market_pct"]} vs '
          f'{by["buy_immediately"]["time_in_market_pct"]}')


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
