#!/usr/bin/env python3
"""Tests for the Phase-3 edge analysis instrument.

The danger of a tool like this is not that it crashes - it is that it produces
a confident-looking ordering across buckets from pure noise. These tests guard
the defences that make a thin sample obvious rather than tempting.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.edge_analysis import (ACTIONS, COMPOSITE_BUCKETS, MIN_INDEPENDENT,
                                  _independent_n, _stats, by_action,
                                  format_report, full_report)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _rows(n, excess, day="2025-01-02", days=None):
    if days:
        return [{"excess": excess, "correct": 1, "day": days[i % len(days)],
                 "ticker": f"T{i}", "action": "BUY", "composite": 70, "pillars": {}}
                for i in range(n)]
    return [{"excess": excess, "correct": 1, "day": day, "ticker": f"T{i}",
             "action": "BUY", "composite": 70, "pillars": {}} for i in range(n)]


def test_same_day_predictions_are_one_observation():
    """Fifty tickers forecast on one morning share a market. Counting them as
    fifty independent observations is how a tool like this manufactures
    confidence that does not exist."""
    rows = _rows(50, 1.0, day="2025-03-04")
    check("50 same-day rows collapse to 1 independent observation",
          _independent_n(rows, 5) == 1, str(_independent_n(rows, 5)))


def test_overlapping_windows_reduce_independence():
    """Daily calls graded over a year overlap almost entirely."""
    days = [f"2025-01-{d:02d}" for d in range(1, 29)]
    rows = _rows(280, 1.0, days=days)
    short = _independent_n(rows, 1)
    long = _independent_n(rows, 252)
    check("short horizon keeps more independence", short > long, f"{short} vs {long}")
    check("a 1-year horizon over one month is ~1 window", long <= 2, str(long))


def test_standard_error_uses_independent_n_not_row_count():
    """Dividing by sqrt(row count) would understate the error several-fold."""
    days = [f"2025-01-{d:02d}" for d in range(1, 11)]
    rows = [{"excess": e, "correct": 1, "day": days[i % 10], "ticker": f"T{i}",
             "action": "BUY", "composite": 70, "pillars": {}}
            for i, e in enumerate([1.0, -1.0] * 50)]
    s = _stats(rows, 5)
    check("row count is reported", s["n"] == 100)
    check("independent n is much smaller", s["independent_n"] < 20, str(s["independent_n"]))
    import math
    naive = (s["sd_pct"] / math.sqrt(s["n"]))
    check("stderr is larger than the naive row-count version",
          s["stderr_pct"] > naive, f"{s['stderr_pct']} vs {naive:.3f}")


def test_thin_bucket_gets_no_direction():
    """Below the floor a bucket must report INSUFFICIENT, never 'positive'."""
    rows = _rows(200, 5.0, day="2025-06-02")     # huge mean, 1 independent obs
    s = _stats(rows, 5)
    check("verdict is INSUFFICIENT despite a large mean",
          s["verdict"] == "INSUFFICIENT", s["verdict"])
    check("mean is still reported for transparency", s["mean_excess_pct"] == 5.0)


def test_weak_signal_is_called_inconclusive_not_positive():
    days = [f"2025-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)]
    rows = [{"excess": (0.05 if i % 2 else -0.04), "correct": 1,
             "day": days[i % len(days)], "ticker": f"T{i}", "action": "BUY",
             "composite": 70, "pillars": {}} for i in range(200)]
    s = _stats(rows, 1)
    check("enough independent observations for a verdict",
          s["independent_n"] >= MIN_INDEPENDENT, str(s["independent_n"]))
    check("a near-zero mean is inconclusive, not positive",
          s["verdict"] == "inconclusive", f"{s['verdict']} t={s['t_stat']}")


def test_empty_bucket_reports_no_data_not_zero():
    s = _stats([], 5)
    check("empty bucket says NO DATA", s["verdict"] == "NO DATA")
    check("no fabricated mean", "mean_excess_pct" not in s)


def test_report_runs_against_the_real_ledger_and_labels_itself():
    rep = full_report()
    check("covers every horizon", len(rep["by_horizon"]) >= 6)
    text = format_report(rep, horizons=[5])
    check("states it is measurement only", "measurement only" in text)
    check("states it does not influence scoring",
          "never influences scoring" in text)
    check("explains the independence caveat", "INDEPENDENT observations" in text)
    a = by_action(5)
    check("action buckets present", set(a["actions"]) == set(ACTIONS))


def test_composite_buckets_cover_the_requested_bands():
    labels = [b[2] for b in COMPOSITE_BUCKETS]
    for want in ("50-55", "55-60", "60-65", "65-70", "70-75", "75-80", "80+"):
        check(f"bucket {want} exists", want in labels)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
