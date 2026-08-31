#!/usr/bin/env python3
"""Tests for the daily heartbeat — the accumulation loop.

The properties that matter are not "it runs". They are:
  · the RAW probability is what gets frozen (or calibration trains on its own
    corrections and compounds them),
  · grading happens BEFORE fitting (or today's fit is blind to what matured),
  · one bad ticker cannot end a universe-wide run,
  · re-running the same day does not duplicate history.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# Point the ledger at a THROWAWAY database before anything imports it.
# Earlier runs left 44 synthetic "TEST" snapshots in the production ledger.
# They were inert (a fake ticker has no prices, so nothing ever matured), but
# a test suite must not be able to write into the record the system learns
# from at all. set_db_path is the ledger's own hook for exactly this.
import tempfile

from data import prediction_ledger as _pl

_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="heartbeat-test-"), "ledger.db")

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _tmpdb(fn):
    """Point the ledger at this module's throwaway DB for ONE test, then restore.

    Function-scoped on purpose. Setting the path at module scope (as this file
    originally did) mutates a process-wide global the moment the module is
    IMPORTED, so under pytest every test collected afterwards silently inherits
    the temp database. Production is unaffected - it runs in its own process -
    but a later test reading the ledger without setting its own path would read
    the wrong database, which is the kind of bug that hides for months.
    """
    def wrapper(*a, **k):
        prev = _pl._db_override
        _pl.set_db_path(_TMP_DB)
        try:
            return fn(*a, **k)
        finally:
            _pl.set_db_path(prev)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper



def _fake_rec(ticker="TEST"):
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df = pd.DataFrame({"Close": [50.0 + i * 0.1 for i in range(300)]}, index=idx)
    rec = {
        "ticker": ticker,
        "current_price": float(df["Close"].iloc[-1]),
        "action": "HOLD",
        "composite": 61.0,
        "pillars": {"technical": {"score": 70, "confidence": 1.0},
                    "algo": {"score": 66, "confidence": 1.0},
                    "fundamentals": {"score": 58, "confidence": 0.8}},
        "levels": {"atr_14": 2.0},
        "algo_signals": {"algo_score": 66},
    }
    return rec, df


def _recommend_ok(ticker):
    return _fake_rec(ticker)


def _recommend_boom(ticker):
    raise RuntimeError("data provider down")


def _isolate(hb):
    """Stub the ledger-touching dedup guard. Tests that are not ABOUT the guard
    must not depend on whatever happens to be in the real ledger today."""
    orig = hb.already_frozen
    hb.already_frozen = lambda ticker, day: False
    return orig


@_tmpdb
def test_forecast_and_freeze_stores_the_raw_probability():
    """The frozen number must be the UNCALIBRATED one. If a correction were
    stored, tomorrow's fit would train on today's correction and compound."""
    import agents.heartbeat as hb

    frozen = {}
    orig_af = _isolate(hb)
    orig = hb.pl.freeze_prediction
    hb.pl.freeze_prediction = lambda rec: (frozen.update(rec=rec) or "snap-1")
    try:
        # A calibrator that would visibly move any probability.
        cals = {h: {"applied": True, "map": {"x": [0.0, 1.0], "y": [0.2, 0.2]}}
                for h in hb.HORIZONS}
        out = hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok, calibrators=cals)
        stored = (frozen.get("rec") or {}).get("horizon_probabilities") or {}
        check("freeze succeeded", out["status"] == "done", str(out))
        check("a probability was stored for every horizon",
              set(stored.keys()) == set(hb.HORIZONS))
        check("stored probability is RAW, not the 0.2 the calibrator would give",
              all(abs(v - 0.2) > 1e-9 for v in stored.values()), str(stored))
        check("calibrated view is still returned for the reader",
              out.get("calibrated_p_up") is not None)
        check("and the calibrated view DID apply the map",
              all(abs(v - 0.2) < 1e-6 for v in out["calibrated_p_up"].values()),
              str(out.get("calibrated_p_up")))
    finally:
        hb.pl.freeze_prediction = orig
        hb.already_frozen = orig_af


@_tmpdb
def test_rerunning_the_same_day_does_not_duplicate():
    """Regression: predictions are stamped to the DAY, not the moment.

    freeze_prediction content-hashes created_at, so a wall-clock timestamp made
    every re-run a distinct snapshot - a cron retry would silently double-count
    the same call and bias calibration toward duplicated observations. Caught
    against the real ledger (392 -> 394 rows on a re-run), fixed here.
    """
    import agents.heartbeat as hb

    stamps = []
    orig_af0 = _isolate(hb)
    orig = hb.pl.freeze_prediction
    hb.pl.freeze_prediction = lambda rec: (stamps.append(rec.get("generated_at"))
                                           or "snap")
    try:
        hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok)
        hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok)
        check("two runs on the same day produce the SAME stamp",
              len(stamps) == 2 and stamps[0] == stamps[1], str(stamps))
        check("stamp is pinned to midnight of the day",
              str(stamps[0]).endswith("T00:00:00"), str(stamps[0]))
        # An explicit as_of must win, so a backfill can date its own rows.
        stamps.clear()
        hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok, as_of="2024-05-06")
        check("explicit as_of is honoured", stamps[0] == "2024-05-06T00:00:00",
              str(stamps[0]))
    finally:
        hb.pl.freeze_prediction = orig
        hb.already_frozen = orig_af0

    # The stamp alone is NOT enough - price_at_call moves intraday, so the
    # content hash differs anyway. The already_frozen() guard is what actually
    # prevents the duplicate.
    orig_af = hb.already_frozen
    orig_fp = hb.pl.freeze_prediction
    froze = []
    hb.already_frozen = lambda ticker, day: True
    hb.pl.freeze_prediction = lambda rec: (froze.append(rec) or "snap")
    try:
        out = hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok)
        check("already-predicted ticker is skipped", out["status"] == "skipped", str(out))
        check("and nothing was written", froze == [])
        forced = hb.forecast_and_freeze("TEST", recommend_fn=_recommend_ok, force=True)
        check("force overrides the guard", forced["status"] == "done", str(forced))
    finally:
        hb.already_frozen = orig_af
        hb.pl.freeze_prediction = orig_fp


@_tmpdb
def test_one_bad_ticker_does_not_end_the_run():
    import agents.heartbeat as hb

    def mixed(ticker):
        if ticker == "BAD":
            raise RuntimeError("boom")
        return _fake_rec(ticker)

    orig_af = _isolate(hb)
    orig = hb.pl.freeze_prediction
    hb.pl.freeze_prediction = lambda rec: "snap"
    try:
        res = hb.run_daily(["GOOD", "BAD", "ALSOGOOD"], recommend_fn=mixed,
                           grade=False, refit=False)
        check("run completed", res["tickers"] == 3)
        check("good tickers still frozen", res["frozen"] == 2, str(res["frozen"]))
        check("the bad one is reported, not swallowed", res["errors"] == 1)
        bad = next(r for r in res["results"] if r["ticker"] == "BAD")
        check("failure carries a reason", "recommend" in (bad.get("reason") or ""))
    finally:
        hb.pl.freeze_prediction = orig
        hb.already_frozen = orig_af


@_tmpdb
def test_grading_happens_before_calibration_fit():
    """Order is load-bearing: fitting first would ignore everything that
    matured overnight, which is the whole point of running daily."""
    import agents.heartbeat as hb
    order = []

    orig_af = _isolate(hb)
    orig_refresh = hb.pl.refresh_outcomes
    orig_freeze = hb.pl.freeze_prediction
    hb.pl.refresh_outcomes = lambda ticker=None: (order.append("grade") or {"matured": 3})
    hb.pl.freeze_prediction = lambda rec: "snap"

    import intelligence.calibration as cal
    orig_load = cal.load_calibrators
    cal.load_calibrators = lambda hs, source="all": (order.append("fit") or {})
    try:
        hb.run_daily(["AAA"], recommend_fn=_recommend_ok)
        check("both steps ran", "grade" in order and "fit" in order, str(order))
        check("grading precedes fitting", order.index("grade") < order.index("fit"),
              str(order))
    finally:
        hb.pl.refresh_outcomes = orig_refresh
        hb.pl.freeze_prediction = orig_freeze
        cal.load_calibrators = orig_load
        hb.already_frozen = orig_af


@_tmpdb
def test_no_llm_call_in_the_daily_loop():
    """The loop must stay deterministic and keyless - that is what makes a
    wide daily universe affordable and reproducible.

    Checked against the AST rather than raw text: the docstring legitimately
    NAMES the explainer while explaining that it is a separate opt-in concern,
    and a substring scan would flag that mention as a call."""
    import ast
    import inspect

    import agents.heartbeat as hb
    tree = ast.parse(inspect.getsource(hb))

    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                referenced.add(a.asname or a.name.split(".")[-1])

    for banned in ("run_prediction_agent", "run_fundamentals_agent",
                   "run_technical_agent", "run_social_agent", "run_algo_agent",
                   "run_decision_explainer", "OpenAI", "_get_client",
                   "analyze_stock"):
        check(f"no LLM entry point referenced in code: {banned}",
              banned not in referenced)


@_tmpdb
def test_calibration_status_is_reported_even_when_inactive():
    """A silent no-op is indistinguishable from a broken loop. The run must say
    which horizons are active - including 'none'."""
    import agents.heartbeat as hb
    import intelligence.calibration as cal

    orig_af = _isolate(hb)
    orig_freeze = hb.pl.freeze_prediction
    orig_refresh = hb.pl.refresh_outcomes
    orig_load = cal.load_calibrators
    hb.pl.freeze_prediction = lambda rec: "snap"
    hb.pl.refresh_outcomes = lambda ticker=None: {"matured": 0}
    cal.load_calibrators = lambda hs, source="all": {
        h: {"applied": False, "n": 5, "effective_n": 1,
            "reason": "insufficient_independent_windows"} for h in hs}
    try:
        res = hb.run_daily(["AAA"], recommend_fn=_recommend_ok)
        check("calibration block present", "calibration" in res)
        check("active horizons reported as empty, not omitted",
              res.get("calibration_active_horizons") == [])
        check("the reason survives into the run record",
              "insufficient" in str(res["calibration"]))
    finally:
        hb.pl.freeze_prediction = orig_freeze
        hb.pl.refresh_outcomes = orig_refresh
        cal.load_calibrators = orig_load
        hb.already_frozen = orig_af


@_tmpdb
def test_independence_report_tracks_distance_to_the_gate():
    """The number to watch while the flywheel spins up."""
    from agents.heartbeat import independence_report
    rep = independence_report()
    check("every horizon reported", len(rep) > 0)
    for h, r in rep.items():
        if r.get("error"):
            continue
        check(f"h={h} reports effective_n", "effective_n" in r)
        check(f"h={h} reports the shortfall to the gate", "shortfall" in r)
        check(f"h={h} gate_met agrees with the shortfall",
              r["gate_met"] == (r["shortfall"] == 0))


@_tmpdb
def test_runner_status_only_changes_nothing():
    """--status-only must be safe to run any time, including from a cron
    healthcheck, without touching the ledger."""
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run([sys.executable, "run_heartbeat.py", "--status-only"],
                         cwd=repo, capture_output=True, text=True, timeout=300)
    check("status-only exits cleanly", out.returncode == 0, out.stderr[-300:])
    check("prints the independence report",
          "effective_n" in out.stdout, out.stdout[:300])


@_tmpdb
def test_runner_exits_nonzero_with_no_tickers():
    """A silent cron failure is worse than a loud one."""
    import subprocess
    repo = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run([sys.executable, "run_heartbeat.py"],
                         cwd=repo, capture_output=True, text=True, timeout=300)
    check("no tickers -> exit code 2", out.returncode == 2, str(out.returncode))


@_tmpdb
def test_watchlist_parser_ignores_commas_inside_comments():
    """Regression: the parser split on commas BEFORE stripping comments, so a
    comment containing a comma - "(p50 3.4s, p90 19.8s, max 42.0s)" - fragmented
    into pieces where only the first still began with '#'. The rest were sent to
    the data provider as ticker symbols and produced five bogus errors on the
    first real cron run."""
    import argparse
    import tempfile

    import run_heartbeat as rh

    content = (
        "# universe notes\n"
        "#   cost is ~9.0s/ticker (p50 3.4s, p90 19.8s, max 42.0s) - cold caches\n"
        "AAPL\n"
        "MSFT, NVDA\n"
        "\n"
        "  # indented comment, with a comma\n"
        "BRK.B\n"
        "aapl\n"                      # duplicate, different case
    )
    path = os.path.join(tempfile.mkdtemp(prefix="wl-"), "watchlist.txt")
    open(path, "w").write(content)

    got = rh._load_tickers(argparse.Namespace(file=path, tickers=None))
    check("no comment fragments leaked in as symbols",
          got == ["AAPL", "MSFT", "NVDA", "BRK.B"], str(got))
    check("commas on a data line still separate symbols", "NVDA" in got)
    check("dotted symbols survive", "BRK.B" in got)
    check("case-insensitive de-duplication", got.count("AAPL") == 1)

    inline = rh._load_tickers(argparse.Namespace(file=None, tickers="aapl, msft,NVDA"))
    check("--tickers parses the same way", inline == ["AAPL", "MSFT", "NVDA"], str(inline))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
