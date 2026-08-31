"""
Point-in-Time Historical Replay.

Populate the Prediction Ledger with historically VALID recommendations, so
calibration / agent attribution / confidence reliability can be measured now,
without waiting for live predictions to mature.

The whole thing rests on two facts that already hold:
  1. build_recommendation is a pure function of its inputs (reproducible —
     decision_fingerprint), and
  2. EDGAR fundamentals already support `as_of` (filings filed after the date
     are excluded; restatements collapse to the latest surviving filing).

So a replay at date D is just: feed the UNCHANGED recommendation gates inputs
that were knowable at D, freeze the result dated D, and let the existing outcome
engine evaluate it forward. No look-ahead:
  - prices: the full series TRUNCATED at D (a bar dated <=D was knowable at D);
  - technical / algo / volatility: recomputed from the truncated series;
  - fundamentals: analyze_fundamentals_pit(ticker, as_of=D) — genuine PIT;
  - 52-week range: from the truncated series (not the current 52w).

What has NO point-in-time history in free data is OMITTED, never synthesized
(the mission forbids fabricating it):
  - social sentiment (reddit/stocktwits are now-only scrapes) -> pillar flagged;
  - analyst consensus / targets (current values would be look-ahead) -> omitted;
  - beta, short interest (current) -> omitted;
  - macro (FRED is current-vintage, not PIT) -> not a recommendation input anyway.
Every omission is recorded as a `missing input`, and each replay item records
its data coverage, exclusions, strategy version, and replay run id.

Reuses: FinancialDataGateway (bars + EDGAR as_of), EvidenceLedger, Experiment
Registry, the recommendation gates (unchanged), and the immutable Prediction
Ledger. No new agents/signals, no trading. Run state lives in the EXISTING
recommendations.db (resumable) — no new database.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from data import prediction_ledger as pl
from data import store

# Inputs that free data cannot supply point-in-time; omitted, never faked.
ALWAYS_MISSING = ["social_sentiment", "analyst_consensus", "beta", "short_interest"]
# Macro is current-vintage (FRED), not PIT, and is not a recommendation input.
STANDING_EXCLUSIONS = ["macro_current_vintage_not_pit"]
MIN_BARS = 60          # minimum truncated history to attempt a recommendation

_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT,
    status      TEXT NOT NULL,       -- running | complete | interrupted
    config_json TEXT NOT NULL,
    total_items INTEGER DEFAULT 0,
    done_items  INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS replay_items (
    run_id          TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    as_of           TEXT NOT NULL,
    status          TEXT NOT NULL,   -- done | skipped | error
    snapshot_id     TEXT,
    coverage_json   TEXT,
    missing_json    TEXT,
    exclusions_json TEXT,
    reason          TEXT,
    evaluated_at    TEXT,
    PRIMARY KEY (run_id, ticker, as_of)
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(pl._db()), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_RUN_SCHEMA)
    conn.commit()
    return conn


# ── Point-in-time input construction (the no-look-ahead core) ───────────────

def pit_inputs(ticker: str, as_of: str, full_prices,
               pit_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None):
    """Build the recommendation inputs KNOWABLE at `as_of`.

    Returns (df_asof, indicators, signal_summary, algo_signals, pit_fund,
             minimal_fundamentals, coverage, missing, exclusions) or raises
    ValueError('insufficient_history') when there aren't enough bars.
    """
    import pandas as pd
    from tools.market_data import (compute_algo_signals, compute_indicators,
                                   compute_signal_summary)

    prices = full_prices.dropna()
    cutoff = pd.Timestamp(as_of)
    df = prices[prices.index <= cutoff]                 # TRUNCATION = no look-ahead
    if len(df) < MIN_BARS:
        raise ValueError("insufficient_history")

    # market_data expects an OHLCV frame; reconstruct from the close series
    # (replay operates on adjusted close — OHLC approximated, volume neutral).
    ohlcv = pd.DataFrame({"Open": df, "High": df, "Low": df, "Close": df,
                          "Volume": pd.Series(1.0, index=df.index)})
    indicators = compute_indicators(ohlcv)
    signal_summary = compute_signal_summary(indicators)
    algo_signals = compute_algo_signals(ohlcv, indicators)

    # EDGAR point-in-time fundamentals (genuine as_of; restatements excluded).
    pit_fn = pit_fn or _default_pit_fn
    try:
        pit_fund = pit_fn(ticker, as_of)
    except Exception as e:
        pit_fund = {"available": False, "reason": f"pit_error:{str(e)[:40]}"}

    # Only as-of-SAFE market fields. 52-week range comes from the TRUNCATED
    # series (indicators["52w_high"/"52w_low"]), never the current 52w. beta,
    # short interest, analyst consensus, margins are omitted (look-ahead).
    minimal_fundamentals = {
        "fiftyTwoWeekHigh": indicators.get("52w_high"),
        "fiftyTwoWeekLow": indicators.get("52w_low"),
    }

    coverage = {
        "price_bars_as_of": len(df),
        "price_last_bar": str(df.index[-1].date()),
        "fundamentals_pit": bool(pit_fund.get("available")),
        "fundamentals_as_of_honored": bool(pit_fund.get("as_of_honored")),
        "sentiment": "excluded_no_pit_history",
        "analyst_consensus": "excluded_no_pit_history",
        "macro": "excluded_current_vintage_only",
    }
    missing = list(ALWAYS_MISSING)
    if not pit_fund.get("available"):
        missing.append("sec_fundamentals")
    exclusions = list(STANDING_EXCLUSIONS)
    return df, indicators, signal_summary, algo_signals, pit_fund, minimal_fundamentals, \
        coverage, missing, exclusions


def _default_pit_fn(ticker: str, as_of: str) -> Dict[str, Any]:
    from agents.fundamentals_pit import analyze_fundamentals_pit
    return analyze_fundamentals_pit(ticker, as_of=as_of, run_id=f"replay:{as_of}")


def _pit_horizon_probabilities(ticker: str, rec: Dict[str, Any],
                               algo_signals: Dict[str, Any], df) -> Optional[Dict[int, float]]:
    """Per-horizon p_up for a REPLAYED prediction, computed point-in-time.

    Without this a replayed snapshot freezes an action but no probability, and
    the calibration layer has nothing to learn from — which is exactly why a
    ledger full of matured replay outcomes was still unusable for calibration.

    Two inputs are deliberately withheld, and both would be lookahead:

      regime      compute_market_regime() reads TODAY's SPY/VIX, not the market
                  as it stood at as_of. Passing it would let a 2024 replay see
                  2026 conditions. There is no PIT regime source yet, so the
                  forecast is made without one and flags itself accordingly.

      calibrators A calibration map is fitted on outcomes that, relative to
                  as_of, live in the future. Feeding it back into a replayed
                  forecast would both leak and be circular — the engine would
                  be graded on a probability it could only state with hindsight.

    Everything else (pillars, algo signals, ATR, price) already comes from
    pit_inputs(), so it carries the same as_of guarantee the recommendation does.

    Returns None on any failure: a replay that freezes an action without a
    probability is degraded but still honest, which is better than aborting the
    row or storing a guess.
    """
    try:
        from data.prediction_ledger import HORIZONS
        from intelligence.prediction_engine import forecast_horizons

        fc = forecast_horizons(
            ticker,
            float(df.iloc[-1]),
            rec.get("pillars", {}),
            algo_signals,
            regime=None,             # see docstring - would be lookahead
            atr_14=(rec.get("levels") or {}).get("atr_14"),
            horizons=tuple(HORIZONS),
            calibrators=None,        # see docstring - would leak and be circular
        )
        probs = {h["horizon_days"]: h["p_up"] for h in (fc.get("horizons") or {}).values()}
        return probs or None
    except Exception:
        return None


def replay_one(ticker: str, as_of: str, full_prices, run_id: str,
               benchmark: str = "SPY",
               pit_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Replay a single (ticker, as_of): build a PIT recommendation dated `as_of`
    and freeze it. Returns {status, snapshot_id, coverage, missing, exclusions}.
    Never raises for ordinary problems — they become status='skipped'/'error'."""
    import pandas as pd
    from agents.recommendation import build_recommendation

    try:
        (df, ind, ss, algo, pit_fund, min_fund, coverage, missing, exclusions) = \
            pit_inputs(ticker, as_of, full_prices, pit_fn=pit_fn)
    except ValueError as e:
        return {"status": "skipped", "reason": str(e), "snapshot_id": None,
                "coverage": {"price_bars_as_of": 0}, "missing": ALWAYS_MISSING,
                "exclusions": STANDING_EXCLUSIONS + [str(e)]}
    except Exception as e:
        return {"status": "error", "reason": f"pit_inputs:{str(e)[:60]}", "snapshot_id": None,
                "coverage": {}, "missing": [], "exclusions": []}

    try:
        rec = build_recommendation(ticker, df.to_frame("Close").assign(
            Open=df, High=df, Low=df, Volume=1.0), ind, ss, algo, min_fund,
            reddit=None, stocktwits=None, pit=pit_fund, run_id=f"replay:{run_id}:{as_of}")
        # Date the prediction AS OF the replay date so the outcome engine
        # evaluates it forward from D (fingerprint excludes timestamps, unchanged).
        as_of_dt = f"{as_of}T00:00:00"
        rec["generated_at"] = as_of_dt
        rec["data_asof"] = str(df.index[-1].date())
        rec["benchmark"] = benchmark
        rec["replay_run_id"] = run_id
        rec["replay_meta"] = {"coverage": coverage, "missing": missing,
                              "exclusions": exclusions, "as_of": as_of}
        rec["horizon_probabilities"] = _pit_horizon_probabilities(
            ticker, rec, algo, df)
        sid = pl.freeze_prediction(rec)
        return {"status": "done", "snapshot_id": sid, "coverage": coverage,
                "missing": missing, "exclusions": exclusions,
                "action": rec.get("action"), "fingerprint": rec.get("decision_fingerprint")}
    except Exception as e:
        return {"status": "error", "reason": f"build:{str(e)[:60]}", "snapshot_id": None,
                "coverage": coverage, "missing": missing, "exclusions": exclusions}


# ── Sampling cadence ─────────────────────────────────────────────────────────

def sample_dates(start: str, end: str, cadence: str = "M") -> List[str]:
    """Calendar sample dates between start and end. cadence: 'W' weekly, 'M'
    monthly, or an int string = every N calendar days. replay_one snaps each to
    the actual trading day <= the sample date."""
    import pandas as pd
    freq = {"W": "W-FRI", "M": "MS"}.get(cadence)
    if freq is None:
        n = int(cadence)
        dr = pd.date_range(start, end, freq=f"{n}D")
    else:
        dr = pd.date_range(start, end, freq=freq)
    return [str(d.date()) for d in dr]


# ── Run orchestration (resumable) ────────────────────────────────────────────

def start_run(config: Dict[str, Any]) -> str:
    run_id = config.get("run_id") or f"replay-{uuid.uuid4().hex[:10]}"
    tickers = [t.upper() for t in config["tickers"]]
    dates = sample_dates(config["start"], config["end"], config.get("cadence", "M"))
    total = len(tickers) * len(dates)
    conn = _conn()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO replay_runs
               (run_id, created_at, updated_at, status, config_json, total_items, done_items)
               VALUES (?,?,?,?,?,?,0)""",
            (run_id, datetime.now().isoformat(timespec="seconds"),
             datetime.now().isoformat(timespec="seconds"), "running",
             json.dumps({**config, "tickers": tickers, "dates_sampled": len(dates)}), total))
        conn.commit()
    finally:
        conn.close()
    return run_id


def _completed_cells(run_id: str) -> set:
    conn = _conn()
    try:
        return {(r["ticker"], r["as_of"]) for r in conn.execute(
            "SELECT ticker, as_of FROM replay_items WHERE run_id=? AND status IN ('done','skipped')",
            (run_id,))}
    finally:
        conn.close()


def _record_item(run_id, ticker, as_of, res):
    conn = _conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO replay_items
               (run_id, ticker, as_of, status, snapshot_id, coverage_json, missing_json,
                exclusions_json, reason, evaluated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (run_id, ticker, as_of, res["status"], res.get("snapshot_id"),
             json.dumps(res.get("coverage")), json.dumps(res.get("missing")),
             json.dumps(res.get("exclusions")), res.get("reason"),
             datetime.now().isoformat(timespec="seconds")))
        done = conn.execute(
            "SELECT COUNT(*) AS n FROM replay_items WHERE run_id=? AND status IN ('done','skipped','error')",
            (run_id,)).fetchone()["n"]
        conn.execute("UPDATE replay_runs SET done_items=?, updated_at=? WHERE run_id=?",
                     (done, datetime.now().isoformat(timespec="seconds"), run_id))
        conn.commit()
    finally:
        conn.close()


def run_replay(config: Dict[str, Any],
               fetch_fn: Optional[Callable[[str], Any]] = None,
               benchmark_fetch_fn: Optional[Callable[[str], Any]] = None,
               pit_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
               progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
               evaluate: bool = True) -> Dict[str, Any]:
    """Run (or RESUME) a replay. Idempotent + resumable: cells already done or
    skipped are not redone, and freezing is content-addressed so re-runs never
    duplicate a snapshot. On completion, evaluates outcomes through the existing
    engine (evaluate=False for tests that want to check separately)."""
    run_id = start_run(config)
    tickers = [t.upper() for t in config["tickers"]]
    dates = sample_dates(config["start"], config["end"], config.get("cadence", "M"))
    benchmark = config.get("benchmark", "SPY")

    if fetch_fn is None:
        def fetch_fn(t):
            from tools.market_data import fetch_price_history
            return fetch_price_history(t, period="max")["Close"]

    done_cells = _completed_cells(run_id)
    n_done = n_skipped = n_error = 0
    price_cache: Dict[str, Any] = {}

    for ticker in tickers:
        # Fetch full history once per ticker; a fetch failure = delisted/missing.
        if ticker not in price_cache:
            try:
                price_cache[ticker] = fetch_fn(ticker)
            except Exception as e:
                price_cache[ticker] = None
                for as_of in dates:
                    if (ticker, as_of) in done_cells:
                        continue
                    res = {"status": "skipped", "reason": f"symbol_unavailable:{str(e)[:40]}",
                           "snapshot_id": None, "coverage": {}, "missing": [],
                           "exclusions": ["delisted_or_missing_symbol"]}
                    _record_item(run_id, ticker, as_of, res); n_skipped += 1
                continue
        prices = price_cache[ticker]
        if prices is None:
            continue
        for as_of in dates:
            if (ticker, as_of) in done_cells:
                continue
            res = replay_one(ticker, as_of, prices, run_id, benchmark=benchmark, pit_fn=pit_fn)
            _record_item(run_id, ticker, as_of, res)
            n_done += res["status"] == "done"
            n_skipped += res["status"] == "skipped"
            n_error += res["status"] == "error"
            if progress_cb:
                progress_cb(get_run(run_id))

    # Mark the run complete and evaluate outcomes for the tickers touched.
    _finalize(run_id)
    evaluated = None
    if evaluate:
        evaluated = {}
        for ticker in tickers:
            try:
                evaluated[ticker] = pl.refresh_outcomes(
                    ticker=ticker, fetch_fn=fetch_fn, benchmark_fetch_fn=benchmark_fetch_fn or fetch_fn)
            except Exception as e:
                evaluated[ticker] = {"error": str(e)[:60]}
    return {"run_id": run_id, "done": n_done, "skipped": n_skipped, "errors": n_error,
            "evaluated": evaluated, "run": get_run(run_id)}


def _finalize(run_id: str) -> None:
    conn = _conn()
    try:
        run = conn.execute("SELECT total_items, done_items FROM replay_runs WHERE run_id=?",
                          (run_id,)).fetchone()
        status = "complete" if run and run["done_items"] >= run["total_items"] else "interrupted"
        conn.execute("UPDATE replay_runs SET status=?, updated_at=? WHERE run_id=?",
                     (status, datetime.now().isoformat(timespec="seconds"), run_id))
        conn.commit()
    finally:
        conn.close()


def get_run(run_id: str) -> Dict[str, Any]:
    conn = _conn()
    try:
        run = conn.execute("SELECT * FROM replay_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run:
            return {"error": f"no run {run_id}"}
        d = dict(run)
        d["config"] = json.loads(d.pop("config_json"))
        d["pct"] = round(100.0 * d["done_items"] / d["total_items"], 1) if d["total_items"] else 0.0
        by_status = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM replay_items WHERE run_id=? GROUP BY status", (run_id,))}
        d["items_by_status"] = by_status
        d["recent"] = [dict(r) for r in conn.execute(
            "SELECT ticker, as_of, status, snapshot_id, reason FROM replay_items "
            "WHERE run_id=? ORDER BY evaluated_at DESC LIMIT 8", (run_id,))]
        return d
    finally:
        conn.close()


def list_runs(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT run_id, created_at, status, total_items, done_items FROM replay_runs "
            "ORDER BY created_at DESC LIMIT ?", (limit,))]
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    cfg = {"tickers": (sys.argv[1].split(",") if len(sys.argv) > 1 else ["AAPL", "MSFT"]),
           "start": sys.argv[2] if len(sys.argv) > 2 else "2022-01-01",
           "end": sys.argv[3] if len(sys.argv) > 3 else "2024-06-01",
           "cadence": sys.argv[4] if len(sys.argv) > 4 else "M"}
    print(json.dumps(run_replay(cfg), indent=2, default=str))
