"""
Verification for the recommendation engine (agents/recommendation.py).

Run: python3 tests/test_recommendation.py

Offline/deterministic (synthetic OHLCV; temp DBs for both the EvidenceLedger
and the recommendations store, so re-runs are idempotent and no fixture ever
pollutes the real registries — the lesson from M2a, applied from the start).

What must hold for the engine to be honest:
  1. Fully keyless — a complete recommendation with llm_prose=None.
  2. Every number traceable — pillars + composite have explainable claims.
  3. Deterministic conviction — the gates fire as documented.
  4. time_horizon_days round-trips to the store (the always-NULL column, fixed).
  5. Honesty flags present and truthful (core-only backtest, tracked-forward).
  6. Levels geometry — investing-wide stops, 2:1 R:R, entry zone brackets price.
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
_TMP = Path(tempfile.mkdtemp())


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def setup_dbs():
    """Isolate BOTH stores. The ledger has a public override; the store's path
    is module-level, so the test points it at a temp file directly."""
    from data import ledger, store
    ledger.set_db_path(_TMP / "rec_ledger.db")
    store._DB_PATH = _TMP / "rec_store.db"
    store.DB_PATH = store._DB_PATH


def make_inputs(kind="up", n=500, seed=7):
    rng = np.random.default_rng(seed)
    drift = {"up": 0.002, "down": -0.002, "flat": 0.0}[kind]
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n))), index=idx)
    df = pd.DataFrame({"Open": close.shift(1).fillna(close.iloc[0]),
                       "High": close * 1.006, "Low": close * 0.994, "Close": close,
                       "Volume": pd.Series(rng.uniform(1e6, 2e6, n), index=idx)})
    from tools.market_data import (compute_algo_signals, compute_indicators,
                                   compute_signal_summary)
    ind = compute_indicators(df)
    return df, ind, compute_signal_summary(ind), compute_algo_signals(df, ind)


FUND = {"trailingPE": 18, "priceToBook": 3, "profitMargins": 0.22, "returnOnEquity": 0.35,
        "revenueGrowth": 0.12, "marketCap": 1e12, "freeCashflow": 5e10, "totalDebt": 1e11,
        "totalCash": 6e10, "beta": 1.1, "recommendationMean": 2.0, "targetMeanPrice": None,
        "numberOfAnalystOpinions": 30, "shortRatio": 2.0,
        "fiftyTwoWeekHigh": None, "fiftyTwoWeekLow": None}


def test_keyless_and_structure():
    print("=== 1. Keyless: a full recommendation with no LLM anywhere ===")
    setup_dbs()
    from agents.recommendation import build_recommendation

    df, ind, ss, algo = make_inputs("up")
    fund = dict(FUND); fund["targetMeanPrice"] = ind["current_price"] * 1.15
    fund["fiftyTwoWeekHigh"], fund["fiftyTwoWeekLow"] = ind["52w_high"], ind["52w_low"]
    rec = build_recommendation("TESTUP", df, ind, ss, algo, fund, run_id="t1")

    for f in ("action", "conviction", "composite", "pillars", "levels",
              "position_size_pct", "time_horizon_days", "backtest",
              "honesty_flags", "claims", "disclaimer"):
        check(f"field present: {f}", f in rec)
    check("thesis is None without llm_prose (no fabricated narrative)",
          rec["thesis"] is None)
    check("action is an investing verb",
          rec["action"] in ("BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL"), rec["action"])
    check("uptrend composite > 55", rec["composite"] > 55, f"{rec['composite']}")
    check("all 6 pillars reported", len(rec["pillars"]) == 6, str(list(rec["pillars"])))
    check("backtest block carries honest dSR denominator",
          rec["backtest"]["n_trials"] >= 1, f"n_trials={rec['backtest']['n_trials']}")

    # thesis appears ONLY as labelled narrative when prose is supplied
    rec2 = build_recommendation("TESTUP", df, ind, ss, algo, fund,
                                llm_prose={"summary": "words", "bull_case": "b"}, run_id="t1b")
    check("thesis labelled as narrative-only",
          "narrative only" in rec2["thesis"]["source"])


def test_claims_traceable():
    print("=== 2. Every number traceable to an explainable claim ===")
    setup_dbs()
    from agents.recommendation import build_recommendation
    from data import ledger

    df, ind, ss, algo = make_inputs("up")
    rec = build_recommendation("TRACE", df, ind, ss, algo, dict(FUND), run_id="t2")

    for name, p in rec["pillars"].items():
        cid = p["claim_id"]
        check(f"pillar {name} has a claim", bool(cid))
        if cid:
            e = ledger.explain(cid)
            check(f"pillar {name} claim explains (formula + evidence)",
                  bool(e.get("formula")) and len(e.get("evidence", [])) >= 1)
    comp_id = rec["claims"].get("composite")
    check("composite has a claim", bool(comp_id))
    if comp_id:
        e = ledger.explain(comp_id)
        check("composite claim cites the pillar datums", len(e["evidence"]) == 6,
              f"{len(e['evidence'])} datums")
        check("snapshot claims honestly carry as_of_honored=False",
              e["as_of_honored"] is False)


def test_conviction_gates():
    print("=== 3. Conviction tracks PROVEN EDGE (statistical), not narrative ===")
    setup_dbs()
    from agents.recommendation import build_recommendation

    # Downtrend: no proven edge -> conviction is LOW or NONE, never HIGH, and
    # size is gated to 0. (conviction == statistical-edge level now.)
    df, ind, ss, algo = make_inputs("down")
    rec = build_recommendation("TESTDN", df, ind, ss, algo, dict(FUND), run_id="t3")
    check("downtrend never yields HIGH conviction",
          rec["conviction"] in ("LOW", "NONE", "MEDIUM"), rec["conviction"])
    check("downtrend action is not a buy",
          rec["action"] in ("HOLD", "REDUCE", "SELL"), rec["action"])
    check("no proven edge -> position gated to 0",
          rec["position_size_pct"] == 0.0 and rec["position_size_gated"] is True,
          f"size={rec['position_size_pct']} gated={rec['position_size_gated']}")

    # The four dimensions are all present and independent.
    for dim in ("thesis", "data", "statistical_edge", "allocation"):
        check(f"confidence dimension present: {dim}", dim in rec["confidence"])
    check("statistical_edge reports its gate checks",
          set(rec["confidence"]["statistical_edge"]["checks"]) >=
          {"min_sample", "net_cost_positive", "dsr", "walk_forward", "pbo", "embargo"})
    check("thesis confidence is labelled narrative-only (gates nothing)",
          "narrative only" in rec["confidence"]["thesis"]["note"])

    # Veto: even a proven edge is constrained by an active risk veto; and a veto
    # can never produce a HIGH allocation.
    df, ind, ss, algo = make_inputs("up")
    algo_v = dict(algo); algo_v["vol_regime"], algo_v["vol_expanding"] = "HIGH", True
    rec_v = build_recommendation("TESTVETO", df, ind, ss, algo_v, dict(FUND), run_id="t3b")
    check("risk veto keeps allocation confidence below HIGH",
          rec_v["confidence"]["allocation"]["level"] != "HIGH",
          f"alloc={rec_v['confidence']['allocation']['level']} veto={rec_v['risk_veto']}")

    # THE gate rule: non-zero Kelly requires statistical_edge == HIGH.
    check("non-zero size IMPLIES statistical edge HIGH",
          (rec_v["position_size_pct"] == 0.0)
          or (rec_v["confidence"]["statistical_edge"]["level"] == "HIGH"),
          f"size={rec_v['position_size_pct']} edge={rec_v['confidence']['statistical_edge']['level']}")


def test_persistence_roundtrip():
    print("=== 4. Persistence: time_horizon_days round-trips, hit-rate filter works ===")
    setup_dbs()
    from agents.recommendation import build_recommendation, log_composite_recommendation
    from data import store

    df, ind, ss, algo = make_inputs("up")
    rec = build_recommendation("PERSIST", df, ind, ss, algo, dict(FUND), run_id="t4")
    rid = log_composite_recommendation(rec)
    check("recommendation persisted (row id returned)", isinstance(rid, int), str(rid))

    rows = store.get_history(ticker="PERSIST")
    check("row visible in history", len(rows) == 1)
    row = rows[0]
    check("grounding_strategy tags the composite engine",
          row["grounding_strategy"] == "seven_pillar_composite")
    check("action/conviction persisted",
          row["action"] == rec["action"] and row["conviction"] == rec["conviction"])

    # THE FIX: the column that was always NULL now carries a real horizon.
    import sqlite3
    conn = sqlite3.connect(str(store._DB_PATH)); conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT time_horizon_days, raw_json FROM recommendations WHERE id=?",
                     (rid,)).fetchone()
    conn.close()
    check("time_horizon_days is non-NULL (the always-NULL column, fixed)",
          r["time_horizon_days"] == rec["time_horizon_days"],
          f"stored={r['time_horizon_days']}")
    check("raw_json preserves pillar scores (audit trail not amputated)",
          '"pillars"' in r["raw_json"])

    # Outcome tracking + engine-filtered hit rate work through existing plumbing.
    out = store.check_outcome(rid, current_price=rec["current_price"] * 1.05)
    check("outcome tracker accepts the rec", out.get("direction_correct") is not None)
    hr = store.get_historical_hit_rate(strategy_source="seven_pillar_composite")
    check("hit rate filterable by composite engine", hr["total"] >= 1, str(hr))


def test_levels_geometry():
    print("=== 5. Levels: investing geometry ===")
    setup_dbs()
    from agents.recommendation import build_recommendation

    df, ind, ss, algo = make_inputs("flat")
    rec = build_recommendation("LVL", df, ind, ss, algo, dict(FUND), run_id="t5")
    lv, p = rec["levels"], rec["current_price"]
    check("entry zone brackets current price",
          lv["entry_zone_low"] < p <= lv["entry_zone_high"] + 0.01,
          f"[{lv['entry_zone_low']}, {lv['entry_zone_high']}] vs {p}")
    check("stop at least 2% below price", lv["stop_loss"] <= p * 0.98 + 1e-6)
    check("2:1 reward:risk", abs((lv["target_price"] - p) - 2 * (p - lv["stop_loss"])) < 0.05,
          f"target={lv['target_price']} stop={lv['stop_loss']}")
    check("horizon maps from vol regime",
          rec["time_horizon_days"] in (45, 91, 126), str(rec["time_horizon_days"]))
    check("honesty flags name the core-only backtest",
          rec["honesty_flags"]["backtest_covers_core_only"] is True
          and rec["honesty_flags"]["social_research_tracked_forward_only"] is True)


def test_fundamentals_via_gateway_not_yfinance():
    print("=== 6. Recommendation fundamentals: SEC/gateway, never legacy yfinance ===")
    setup_dbs()
    from agents.recommendation import build_recommendation

    df, ind, ss, algo = make_inputs("up")

    # A stub EDGAR PIT result (the shape analyze_fundamentals_pit returns).
    pit = {"available": True, "as_of_honored": True, "provider": "sec-edgar",
           "ratios": {"net_margin": {"value": 0.25}, "return_on_equity": {"value": 0.4},
                      "debt_to_equity": {"value": 0.5}, "operating_margin": {"value": 0.3}}}

    # Deliberately hand the engine RICH yfinance accounting numbers. In strict
    # mode they must be IGNORED — the fundamentals pillar must be SEC-sourced.
    yf_fund = {"profitMargins": 0.99, "returnOnEquity": 9.9, "totalDebt": 1, "marketCap": 1e12,
               "trailingPE": 5, "priceToBook": 1, "revenueGrowth": 0.9,
               "beta": 1.0, "recommendationMean": 1.0, "targetMeanPrice": ind["current_price"] * 2,
               "numberOfAnalystOpinions": 40, "fiftyTwoWeekHigh": ind["52w_high"],
               "fiftyTwoWeekLow": ind["52w_low"]}

    with_pit = build_recommendation("SECOK", df, ind, ss, algo, yf_fund, pit=pit, run_id="t6")
    fp = with_pit["pillars"]["fundamentals"]
    check("fundamentals pillar is SEC-sourced",
          with_pit["honesty_flags"]["fundamentals_source"] == "sec-edgar")
    check("fundamentals flags say PIT used, NOT restated snapshot",
          "pit_fundamentals_used" in fp["flags"]
          and "restated_snapshot_fundamentals" not in fp["flags"])

    # Prove yfinance is not the source: swap the yfinance accounting numbers to
    # garbage while keeping the SAME PIT. The fundamentals pillar must not move.
    yf_garbage = dict(yf_fund); yf_garbage.update(
        {"profitMargins": -9, "returnOnEquity": -9, "trailingPE": 999, "revenueGrowth": -9})
    with_garbage = build_recommendation("SECOK", df, ind, ss, algo, yf_garbage, pit=pit, run_id="t6b")
    check("fundamentals score independent of yfinance accounting fields (SEC only)",
          with_pit["pillars"]["fundamentals"]["score"]
          == with_garbage["pillars"]["fundamentals"]["score"],
          f"{with_pit['pillars']['fundamentals']['score']} vs {with_garbage['pillars']['fundamentals']['score']}")

    # Non-filer (no PIT): pillar is neutral + flagged, NEVER a yfinance fallback.
    no_pit = build_recommendation("NOFILER", df, ind, ss, algo, yf_fund, pit=None, run_id="t6c")
    fp2 = no_pit["pillars"]["fundamentals"]
    check("no EDGAR -> fundamentals_source unavailable",
          no_pit["honesty_flags"]["fundamentals_source"] == "unavailable")
    check("no EDGAR -> flagged no_sec_source, NOT restated yfinance",
          "fundamentals_no_sec_source" in fp2["flags"]
          and "restated_snapshot_fundamentals" not in fp2["flags"])

    # The live rec paths must actually route through the gateway (code check).
    root = Path(__file__).parent.parent
    app_txt = (root / "web" / "app.py").read_text()
    orch_txt = (root / "agents" / "orchestrator.py").read_text()
    check("/api/recommendation fetches PIT via analyze_fundamentals_pit",
          "analyze_fundamentals_pit" in app_txt)
    check("orchestrator rec path fetches PIT via analyze_fundamentals_pit",
          "analyze_fundamentals_pit" in orch_txt)


if __name__ == "__main__":
    test_keyless_and_structure()
    test_claims_traceable()
    test_conviction_gates()
    test_persistence_roundtrip()
    test_levels_geometry()
    test_fundamentals_via_gateway_not_yfinance()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — recommendation engine: keyless, traceable, deterministic")
