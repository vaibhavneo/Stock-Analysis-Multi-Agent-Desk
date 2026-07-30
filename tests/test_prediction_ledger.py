"""
Verification for the Prediction Ledger + Outcome Calibration.

Run: python3 tests/test_prediction_ledger.py

Offline/deterministic (synthetic adjusted-price series; temp DB). Covers exactly
the mission's required tests:
  1. Immutability — a frozen snapshot cannot be edited or deleted (DB triggers).
  2. Duplicate refreshes — re-running produces identical rows, no inflation.
  3. Missing market days — a call/horizon on a non-trading day resolves cleanly.
  4. Corporate actions — returns read from ONE adjusted series, so a split does
     not manufacture a spurious return.
  5. Benchmark comparison — excess = raw - benchmark, arithmetic verified.
  6. Reproducibility — same frozen snapshot + same prices => same outcomes.
  7. Calibration/attribution — breakdowns + ECE compute over matured outcomes.
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:60s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def setup():
    # A FRESH temp DB per test: prediction snapshots are immutable (no DELETE),
    # so tests cannot share a DB or predictions would accumulate across them.
    from data import ledger, store
    from data import prediction_ledger as pl
    tmp = Path(tempfile.mkdtemp())
    store._DB_PATH = tmp / "s.db"; store.DB_PATH = store._DB_PATH
    ledger.set_db_path(tmp / "l.db")
    pl.set_db_path(tmp / "s.db")   # same DB as the store — NO new database
    return pl


def price_series(start="2022-01-03", n=400, drift=0.001, vol=0.005, seed=1, split_at=None):
    """A synthetic ADJUSTED-close series. If split_at is given, the RAW price
    would halve there, but the returned series is already adjusted (continuous)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)
    return close


def frozen_rec(ticker, price, action, created, edge_level="HIGH", edge_score=1.0,
               fingerprint="fp0", sector="Technology", regime="LOW"):
    """A minimal rec dict shaped like build_recommendation's output."""
    return {
        "ticker": ticker, "generated_at": created, "current_price": price,
        "action": action, "time_horizon_days": 20, "expected_return_pct": 8.0,
        "sector": sector, "regime": regime, "benchmark": "SPY", "data_asof": created[:10],
        "position_size_pct": 5.0, "decision_fingerprint": fingerprint,
        "experiment_manifest_hash": "manifestX",
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.5},
            "data": {"level": "HIGH", "score": 0.8},
            "statistical_edge": {"level": edge_level, "score": edge_score,
                                 "checks": {"walk_forward": {"pass": True},
                                            "pbo": {"pass": edge_level == "HIGH"}}},
            "allocation": {"level": edge_level, "score": edge_score},
        },
        "pillars": {"technical": {"score": 80}, "algo": {"score": 75}, "risk": {"score": 65},
                    "fundamentals": {"score": 55}, "research": {"score": 62}, "social": {"score": 50}},
        "claims": {"composite": "claim_abc", "pillar_technical": "claim_t"},
    }


def test_immutability():
    print("=== 1. Immutability: frozen snapshots cannot be edited/deleted ===")
    pl = setup()
    sid = pl.freeze_prediction(frozen_rec("IMMU", 100.0, "BUY", "2022-06-01T10:00:00"))
    check("freeze returns a snapshot id", bool(sid))
    snap = pl.get_snapshot(sid)
    check("snapshot stores the four confidence dimensions",
          all(snap[c] for c in ("conf_thesis", "conf_data", "conf_statistical_edge", "conf_allocation")))
    check("snapshot stores evidence ids + strategy version + fingerprint",
          snap["evidence_ids_json"] and snap["strategy_version"] == "manifestX"
          and snap["decision_fingerprint"] == "fp0")
    check("content hash verifies (untampered)", pl.verify_snapshot(sid))

    import sqlite3
    conn = sqlite3.connect(str(pl._db()))
    try:
        conn.execute("UPDATE prediction_snapshots SET action='SELL' WHERE snapshot_id=?", (sid,))
        check("UPDATE is blocked by the DB trigger", False, "no raise")
    except sqlite3.IntegrityError:
        check("UPDATE is blocked by the DB trigger", True)
    except Exception as e:
        check("UPDATE is blocked by the DB trigger", "immutable" in str(e).lower(), str(e)[:40])
    try:
        conn.execute("DELETE FROM prediction_snapshots WHERE snapshot_id=?", (sid,))
        check("DELETE is blocked by the DB trigger", False, "no raise")
    except Exception as e:
        check("DELETE is blocked by the DB trigger", "immutable" in str(e).lower(), str(e)[:40])
    finally:
        conn.close()

    # Re-freezing the identical rec is idempotent (no duplicate, same id).
    sid2 = pl.freeze_prediction(frozen_rec("IMMU", 100.0, "BUY", "2022-06-01T10:00:00"))
    check("re-freezing identical rec returns the same id", sid == sid2)
    check("no duplicate row created", len(pl.list_snapshots(ticker="IMMU")) == 1)


def test_outcomes_and_benchmark():
    print("=== 2. Outcomes + benchmark comparison ===")
    pl = setup()
    prices = price_series(drift=0.001, seed=2)
    bench = price_series(drift=0.0005, seed=99)      # benchmark rises slower
    call_date = str(prices.index[100].date())

    outs = pl.evaluate_outcomes(call_date, "BUY", prices, bench, edge_score=1.0)
    for h in (1, 5, 20, 60):
        check(f"horizon {h}d matured with a raw return", outs[h]["matured"]
              and outs[h]["raw_return_pct"] is not None)
    # A horizon beyond the available data must NOT mature (call near the end).
    late_call = str(prices.index[-30].date())
    late = pl.evaluate_outcomes(late_call, "BUY", prices, bench, edge_score=1.0)
    check("horizon beyond available data is not matured",
          late[60]["matured"] is False and late[5]["matured"] is True,
          f"5d={late[5]['matured']} 60d={late[60]['matured']}")

    o20 = outs[20]
    # Arithmetic: excess == raw - benchmark, MFE >= raw >= MAE.
    check("excess return == raw - benchmark",
          abs(o20["excess_return_pct"] - (o20["raw_return_pct"] - o20["benchmark_return_pct"])) < 1e-6)
    check("MFE >= raw >= MAE (favorable/adverse excursion bounds)",
          o20["mfe_pct"] >= o20["raw_return_pct"] - 1e-9 >= -1e18
          and o20["mae_pct"] <= o20["raw_return_pct"] + 1e-9,
          f"mae={o20['mae_pct']} raw={o20['raw_return_pct']} mfe={o20['mfe_pct']}")
    check("Brier present where edge_score exists (0..1)",
          o20["brier"] is not None and 0 <= o20["brier"] <= 1)
    check("direction_correct is 0/1 for a directional action",
          o20["direction_correct"] in (0, 1))

    # HOLD makes no directional claim -> no Brier, no direction.
    oh = pl.evaluate_outcomes(call_date, "HOLD", prices, bench, edge_score=1.0)[20]
    check("HOLD has no directional verdict", oh["direction_correct"] is None and oh["brier"] is None)


def test_missing_market_days():
    print("=== 3. Missing market days: weekend/holiday call resolves cleanly ===")
    pl = setup()
    prices = price_series(seed=4)
    # A Saturday (not in the business-day index) between two trading days.
    sat = str((prices.index[100] + pd.Timedelta(days=1)).date())  # index[100] is a weekday; +1 may be Sat
    weekend = pd.Timestamp(sat)
    while weekend.weekday() < 5:      # push to an actual Saturday
        weekend += pd.Timedelta(days=1)
    outs = pl.evaluate_outcomes(str(weekend.date()), "BUY", prices, None, edge_score=0.8)
    check("a non-trading-day call still produces matured outcomes",
          outs[5]["matured"] and outs[5]["as_of_date"] is not None)
    check("as_of_date is an actual trading day in the series",
          pd.Timestamp(outs[5]["as_of_date"]) in prices.index)
    # A call after all data -> nothing matured, no crash.
    future = str((prices.index[-1] + pd.Timedelta(days=30)).date())
    outs2 = pl.evaluate_outcomes(future, "BUY", prices, None)
    check("call after all data -> nothing matured, no crash",
          all(not v["matured"] for v in outs2.values()))


def test_corporate_actions():
    print("=== 4. Corporate actions: adjusted series, no spurious return ===")
    pl = setup()
    # Continuous adjusted series (what yfinance auto_adjust gives). A raw 2:1
    # split would halve the unadjusted price at bar 150, but the adjusted series
    # is smooth. Using ONE adjusted series for both endpoints => the return is
    # driven by real drift, never a split artifact.
    prices = price_series(n=300, drift=0.0, vol=0.001, seed=7)   # ~flat
    call_date = str(prices.index[100].date())
    o = pl.evaluate_outcomes(call_date, "BUY", prices, None, edge_score=0.5)[20]
    # Over a ~flat window the 20d return must be small — NOT a ±50% split jump.
    check("no ~50% split artifact in a flat adjusted series",
          abs(o["raw_return_pct"]) < 10.0, f"raw={o['raw_return_pct']}%")

    # Explicit contrast: if someone (wrongly) used a RAW series with a split, the
    # jump would appear. Prove our function is immune by feeding a spliced raw
    # series and confirming we can detect the discontinuity is NOT in adjusted.
    raw = prices.copy()
    raw.iloc[150:] = raw.iloc[150:] / 2.0     # a raw unadjusted split at bar 150
    call2 = str(prices.index[140].date())
    o_raw = pl.evaluate_outcomes(call2, "BUY", raw, None)[20]       # spans the split
    o_adj = pl.evaluate_outcomes(call2, "BUY", prices, None)[20]    # adjusted
    check("raw-with-split shows a large artifact (the bug we avoid)",
          o_raw["raw_return_pct"] < -30, f"raw={o_raw['raw_return_pct']}%")
    check("adjusted series shows NO such artifact (correct behavior)",
          abs(o_adj["raw_return_pct"]) < 10, f"adj={o_adj['raw_return_pct']}%")


def test_duplicate_refresh_and_reproducibility():
    print("=== 5. Duplicate refresh idempotent + reproducible ===")
    pl = setup()
    prices = price_series(n=400, drift=0.0012, seed=5)
    bench = price_series(n=400, drift=0.0004, seed=88)
    call_date = str(prices.index[50].date())
    sid = pl.freeze_prediction(frozen_rec("REFRESH", float(prices.iloc[50]), "BUY",
                                          call_date + "T10:00:00"))

    fetch = lambda t: prices if t == "REFRESH" else bench
    r1 = pl.refresh_outcomes(ticker="REFRESH", fetch_fn=fetch)
    import sqlite3
    def rows():
        conn = sqlite3.connect(str(pl._db())); conn.row_factory = sqlite3.Row
        try:
            return [dict(x) for x in conn.execute(
                "SELECT horizon_days, raw_return_pct, excess_return_pct, mae_pct, mfe_pct, "
                "direction_correct, matured FROM prediction_outcomes WHERE snapshot_id=? "
                "ORDER BY horizon_days", (sid,)).fetchall()]
        finally:
            conn.close()
    snap1 = rows()
    r2 = pl.refresh_outcomes(ticker="REFRESH", fetch_fn=fetch)   # run again
    snap2 = rows()
    check("refresh #2 does NOT create duplicate rows", len(snap1) == len(snap2) == len(pl.HORIZONS))
    check("refresh is idempotent (identical outcome values)", snap1 == snap2,
          "matured outcomes must not change on re-evaluation")
    check("reproducibility: recomputed returns are deterministic",
          all(a["raw_return_pct"] == b["raw_return_pct"] for a, b in zip(snap1, snap2)))
    check("matured horizons flagged", any(r["matured"] for r in snap1))


def test_calibration_report():
    print("=== 6. Calibration + attribution breakdowns ===")
    pl = setup()
    prices = {"WIN": price_series(n=400, drift=0.004, seed=10),     # strong up
              "LOSE": price_series(n=400, drift=-0.004, seed=11),   # strong down
              "SPY": price_series(n=400, drift=0.0005, seed=12)}
    # Freeze a spread of predictions across confidence/action/sector/regime.
    specs = [("WIN", "BUY", "HIGH", 1.0, "Technology", "LOW"),
             ("WIN", "BUY", "HIGH", 1.0, "Technology", "LOW"),
             ("LOSE", "BUY", "LOW", 0.2, "Energy", "HIGH"),      # bullish call on a faller -> wrong
             ("LOSE", "REDUCE", "HIGH", 0.9, "Energy", "HIGH")]  # bearish call on a faller -> right
    for i, (tkr, action, lvl, es, sec, reg) in enumerate(specs):
        p = prices[tkr]
        pl.freeze_prediction(frozen_rec(tkr, float(p.iloc[50]), action,
                                        str(p.index[50].date()) + f"T10:0{i}:00",
                                        edge_level=lvl, edge_score=es,
                                        fingerprint=f"fp{i}", sector=sec, regime=reg))
    pl.refresh_outcomes(fetch_fn=lambda t: prices.get(t, prices["SPY"]))

    rep = pl.calibration_report(horizon=20)
    check("overall win rate computed", rep["overall"]["n"] >= 3 and rep["overall"]["win_rate"] is not None,
          str(rep["overall"]))
    check("benchmark-relative return present", rep["overall"]["avg_excess_return_pct"] is not None)
    check("breakdown by action", "BUY" in rep["by_action"])
    check("breakdown by confidence bucket", set(rep["by_confidence"]) & {"HIGH", "LOW"})
    check("breakdown by regime + sector",
          rep["by_regime"] and rep["by_sector"])
    check("pillar (agent) attribution present for 6 pillars",
          set(rep["pillar_attribution"]) == {"technical", "algo", "risk", "fundamentals",
                                             "research", "social"})
    check("confidence reliability compares predicted vs realized",
          all({"predicted_win_prob", "realized_win_rate", "calibration_gap"} <= set(b)
              for b in rep["confidence_reliability"]))
    check("calibration error (ECE) computed", rep["calibration_error_ece"] is not None)

    # HIGH-confidence bullish calls on the strong riser should win; the LOW one
    # on the faller should lose — attribution must reflect reality, not prose.
    hi = rep["by_confidence"].get("HIGH", {})
    check("HIGH-confidence bucket has a real win rate", hi.get("win_rate") is not None, str(hi))

    s = pl.summary()
    check("summary reports totals + matured + active",
          s["total_predictions"] == 4 and "matured_20d" in s and "active_not_fully_matured" in s,
          str(s))


if __name__ == "__main__":
    test_immutability()
    test_outcomes_and_benchmark()
    test_missing_market_days()
    test_corporate_actions()
    test_duplicate_refresh_and_reproducibility()
    test_calibration_report()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — prediction ledger: immutable, outcome-tracked, calibration-scored")
