"""
Verification for the additive data/prediction_ledger.py extensions this
session: the 126-day (6mo) horizon, per-horizon probabilities
(horizon_probabilities_json), calibration_report_all_horizons(), and
pillar_correlation. tests/test_prediction_ledger.py (unmodified) still
covers everything it did before - this file covers only what's new.

Run: python3 tests/test_prediction_ledger_v2.py

Offline/deterministic, fresh temp DB per test (predictions are immutable -
tests cannot share a DB or rows would accumulate across runs).
"""
import json
import sqlite3
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
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def setup():
    from data import ledger, store
    from data import prediction_ledger as pl
    tmp = Path(tempfile.mkdtemp())
    store._DB_PATH = tmp / "s.db"
    store.DB_PATH = store._DB_PATH
    ledger.set_db_path(tmp / "l.db")
    pl.set_db_path(tmp / "s.db")
    return pl


def price_series(n=400, drift=0.001, vol=0.005, seed=1, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


def frozen_rec(ticker, price, action, created, pillars=None, horizon_probabilities=None,
               edge_level="HIGH", edge_score=1.0, fingerprint="fp0"):
    return {
        "ticker": ticker, "generated_at": created, "current_price": price,
        "action": action, "time_horizon_days": 20, "expected_return_pct": 8.0,
        "sector": "Technology", "regime": "LOW", "benchmark": "SPY", "data_asof": created[:10],
        "position_size_pct": 5.0, "decision_fingerprint": fingerprint,
        "experiment_manifest_hash": "manifestX",
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.5},
            "data": {"level": "HIGH", "score": 0.8},
            "statistical_edge": {"level": edge_level, "score": edge_score, "checks": {}},
            "allocation": {"level": edge_level, "score": edge_score},
        },
        "pillars": pillars or {"technical": {"score": 80}, "algo": {"score": 75}, "risk": {"score": 65},
                                "fundamentals": {"score": 55}, "research": {"score": 62}, "social": {"score": 50}},
        "claims": {},
        "horizon_probabilities": horizon_probabilities,
    }


def test_horizon_126_is_tracked():
    pl = setup()
    check("126 (6mo) is in HORIZONS", 126 in pl.HORIZONS)
    check("existing horizons 1/5/20/60/252 are untouched (not renamed)",
          set(pl.HORIZONS) >= {1, 5, 20, 60, 252})
    outs = pl.evaluate_outcomes("2022-01-03", "BUY", price_series(n=400), None)
    check("evaluate_outcomes returns a 126-day entry", 126 in outs)


def test_horizon_probabilities_round_trip():
    pl = setup()
    hp = {5: 0.9, 21: 0.8, 126: 0.55}
    sid = pl.freeze_prediction(frozen_rec("HPTEST", 100.0, "BUY", "2022-06-01T10:00:00",
                                          horizon_probabilities=hp))
    snap = pl.get_snapshot(sid)
    check("snapshot stores horizon_probabilities_json", bool(snap.get("horizon_probabilities_json")))
    stored = json.loads(snap["horizon_probabilities_json"])
    check("round-trips the exact values",
          {int(k): v for k, v in stored.items()} == hp, str(stored))


def test_horizon_probabilities_absent_is_byte_identical_to_before():
    pl = setup()
    sid = pl.freeze_prediction(frozen_rec("NOHPT", 100.0, "BUY", "2022-06-01T10:00:00"))
    snap = pl.get_snapshot(sid)
    check("horizon_probabilities_json is None when not supplied", snap.get("horizon_probabilities_json") is None)

    prices = price_series(n=400)
    outs_without_param = pl.evaluate_outcomes("2022-01-03", "BUY", prices, None, edge_score=0.6)
    outs_with_none = pl.evaluate_outcomes("2022-01-03", "BUY", prices, None, edge_score=0.6,
                                          horizon_probabilities=None)
    check("evaluate_outcomes is byte-identical whether horizon_probabilities is omitted or None",
          outs_without_param == outs_with_none)


def test_horizon_probabilities_change_brier_at_that_horizon_only():
    pl = setup()
    prices = price_series(n=400, drift=0.001, seed=5)
    # A per-horizon probability that DIFFERS sharply from the blended edge_score
    # default should change that horizon's Brier score specifically, and leave
    # every OTHER horizon's Brier exactly as the edge_score-only path computes.
    outs_default = pl.evaluate_outcomes("2022-01-03", "BUY", prices, None, edge_score=0.5)
    outs_override = pl.evaluate_outcomes("2022-01-03", "BUY", prices, None, edge_score=0.5,
                                         horizon_probabilities={20: 0.99})
    check("the overridden horizon (20) gets a different Brier score",
          outs_default[20]["brier"] != outs_override[20]["brier"],
          f"{outs_default[20]['brier']} vs {outs_override[20]['brier']}")
    for h in (1, 5, 60, 126, 252):
        check(f"horizon {h} (not overridden) has an unchanged Brier score",
              outs_default[h]["brier"] == outs_override[h]["brier"])


def test_calibration_report_all_horizons_shape():
    pl = setup()
    prices = price_series(n=400, seed=2)
    sid = pl.freeze_prediction(frozen_rec("ALLH", 100.0, "BUY", "2022-01-03T10:00:00"))
    outs = pl.evaluate_outcomes("2022-01-03", "BUY", prices, None, edge_score=0.7)
    conn = sqlite3.connect(str(pl._db()))
    for h, o in outs.items():
        pl._upsert_outcome(conn, sid, h, o)
    conn.commit()
    conn.close()

    reports = pl.calibration_report_all_horizons()
    check("returns a report for every tracked horizon", set(reports.keys()) == set(pl.HORIZONS))
    for h in pl.HORIZONS:
        check(f"horizon {h}'s report has the standard calibration_report shape",
              reports[h]["horizon_days"] == h and "overall" in reports[h]
              and "pillar_correlation" in reports[h])


def _insert_synthetic_snapshot_and_outcome(pl, conn, sid, ticker, pillars, raw_return_pct, action="BUY"):
    conn.execute(
        """INSERT INTO prediction_snapshots
           (snapshot_id, created_at, ticker, price_at_call, action, horizon_days,
            pillars_json, frozen_json, content_hash)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (sid, "2022-01-03T10:00:00", ticker, 100.0, action, 20,
         json.dumps(pillars), "{}", "hash_" + sid))
    conn.execute(
        """INSERT INTO prediction_outcomes
           (snapshot_id, horizon_days, evaluated_at, matured, raw_return_pct, direction_correct, brier)
           VALUES (?,?,?,?,?,?,?)""",
        (sid, 20, "2022-02-03T10:00:00", 1, raw_return_pct, int(raw_return_pct > 0), 0.1))


def test_pillar_correlation_distinguishes_signal_from_noise():
    pl = setup()
    conn = pl._conn()  # ensures the schema actually exists before raw INSERTs
    rng = np.random.default_rng(42)

    # "technical" pillar score is a near-perfect linear function of the
    # eventual return - a real, strong signal.
    # "social" pillar score is pure noise, independent of the return.
    for i in range(20):
        ret = rng.normal(0, 10)
        technical_score = max(0, min(100, 50 + ret * 3))   # strongly correlated
        social_score = rng.uniform(0, 100)                  # pure noise
        pillars = {"technical": {"score": technical_score}, "social": {"score": social_score}}
        # calibration_report reads pillars_json as {pillar: score}, not
        # {pillar: {"score": ...}} - match freeze_prediction's own flattening
        # (frozen["pillars"] = {k: v.get("score") ...}).
        flat_pillars = {"technical": technical_score, "social": social_score}
        _insert_synthetic_snapshot_and_outcome(pl, conn, f"sig{i}", "CORR", flat_pillars, ret)
    conn.commit()
    conn.close()

    report = pl.calibration_report(horizon=20)
    tech_corr = report["pillar_correlation"]["technical"]["correlation_with_return"]
    social_corr = report["pillar_correlation"]["social"]["correlation_with_return"]
    check("a genuinely predictive pillar shows a strong positive correlation",
          tech_corr is not None and tech_corr > 0.8, f"technical corr={tech_corr}")
    check("a pure-noise pillar shows a weak correlation, clearly distinguishable from the real signal",
          social_corr is not None and abs(social_corr) < 0.5, f"social corr={social_corr}")
    check("the noise pillar's correlation is clearly smaller in magnitude than the real signal's",
          abs(social_corr) < abs(tech_corr))


def test_pillar_correlation_none_with_too_little_data():
    pl = setup()
    conn = pl._conn()
    _insert_synthetic_snapshot_and_outcome(pl, conn, "s1", "TINY", {"technical": 70}, 5.0)
    conn.commit()
    conn.close()
    report = pl.calibration_report(horizon=20)
    check("correlation is None (not fabricated) with only 1 data point",
          report["pillar_correlation"]["technical"]["correlation_with_return"] is None)


def test_immutability_triggers_survive_the_new_column():
    """Explicit regression check: horizon_probabilities_json is a real schema
    change - confirm it didn't accidentally weaken the immutability guarantee
    tests/test_prediction_ledger.py already covers for the pre-existing columns."""
    pl = setup()
    sid = pl.freeze_prediction(frozen_rec("IMMU2", 100.0, "BUY", "2022-06-01T10:00:00",
                                          horizon_probabilities={21: 0.7}))
    conn = sqlite3.connect(str(pl._db()))
    try:
        conn.execute("UPDATE prediction_snapshots SET action='SELL' WHERE snapshot_id=?", (sid,))
        check("UPDATE still blocked after the new column exists", False, "no raise")
    except sqlite3.IntegrityError:
        check("UPDATE still blocked after the new column exists", True)
    try:
        conn.execute("DELETE FROM prediction_snapshots WHERE snapshot_id=?", (sid,))
        check("DELETE still blocked after the new column exists", False, "no raise")
    except sqlite3.IntegrityError:
        check("DELETE still blocked after the new column exists", True)
    conn.close()


def test_migration_from_old_shape_db_is_safe():
    """Simulate a pre-existing DB created before horizon_probabilities_json
    existed (mirrors how replay_run_id's own migration is exercised) - the
    module must add the column and keep existing data intact, never drop it."""
    pl = setup()
    tmp_db = Path(tempfile.mkdtemp()) / "old.db"
    conn = sqlite3.connect(str(tmp_db))
    old_schema = pl._SCHEMA.replace(
        "    horizon_probabilities_json TEXT, -- {horizon_days: p_up}; NULL for pre-existing rows\n", "")
    conn.executescript(old_schema)
    conn.execute(
        """INSERT INTO prediction_snapshots
           (snapshot_id, created_at, ticker, price_at_call, action, frozen_json, content_hash)
           VALUES (?,?,?,?,?,?,?)""",
        ("pre_existing", "2021-01-01T00:00:00", "OLD", 50.0, "HOLD", "{}", "h1"))
    conn.commit()
    conn.close()

    pl.set_db_path(tmp_db)
    snap = pl.get_snapshot("pre_existing")
    check("pre-existing row survives the migration", snap is not None and snap["ticker"] == "OLD")
    check("pre-existing row's new column reads as None, not an error", snap.get("horizon_probabilities_json") is None)

    new_sid = pl.freeze_prediction(frozen_rec("NEWROW", 100.0, "BUY", "2022-01-01T00:00:00",
                                              horizon_probabilities={21: 0.6}))
    check("a NEW row after migration can use the new column", bool(new_sid))
    new_snap = pl.get_snapshot(new_sid)
    check("the new row's horizon_probabilities_json is actually populated",
          bool(new_snap.get("horizon_probabilities_json")))


if __name__ == "__main__":
    test_horizon_126_is_tracked()
    test_horizon_probabilities_round_trip()
    test_horizon_probabilities_absent_is_byte_identical_to_before()
    test_horizon_probabilities_change_brier_at_that_horizon_only()
    test_calibration_report_all_horizons_shape()
    test_pillar_correlation_distinguishes_signal_from_noise()
    test_pillar_correlation_none_with_too_little_data()
    test_immutability_triggers_survive_the_new_column()
    test_migration_from_old_shape_db_is_safe()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — prediction ledger v2: horizons, probabilities, correlation, migration-safe")
