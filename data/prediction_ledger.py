"""
Prediction Ledger + Outcome Calibration.

Freeze every recommendation at creation time, evaluate it later against real
prices, and measure whether the confidence dimensions and pillar (agent) signals
have historically been reliable. This is a MEASUREMENT layer — it observes
outcomes, it never feeds them back to tune strategies or change the
recommendation gates (that would be look-ahead of the worst kind: fitting on the
future you were supposed to predict).

Two tables, added to the EXISTING recommendations.db (no new database):

  prediction_snapshots — IMMUTABLE. Written once per recommendation; SQLite
    BEFORE UPDATE / BEFORE DELETE triggers ABORT any attempt to edit or remove a
    frozen prediction. Stores the full frozen recommendation plus queryable
    columns (confidence dimensions, pillar scores, gates, evidence ids, strategy
    version, data timestamps, decision fingerprint). A content_hash makes tamper
    evident.

  prediction_outcomes — the MUTABLE evaluation layer, keyed by
    (snapshot_id, horizon_days) for horizons {1, 5, 20, 60, 252} trading days.
    Deterministic from historical adjusted prices, so a refresh is idempotent;
    matured horizons don't move. Corporate actions are handled by reading both
    endpoints from the SAME adjusted series at evaluation time.

Nothing here uses LLM prose — outcomes are pure price arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from data import store

HORIZONS = (1, 5, 20, 60, 252)          # trading days
DEFAULT_BENCHMARK = "SPY"

_db_override: Optional[str] = None


def set_db_path(path: Optional[Any]) -> None:
    """Tests point this at a temp file. None -> the real store DB."""
    global _db_override
    _db_override = str(path) if path else None


def _db() -> str:
    return _db_override or str(store.DB_PATH)


_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    snapshot_id        TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    price_at_call      REAL NOT NULL,
    action             TEXT NOT NULL,
    horizon_days       INTEGER,
    expected_return_pct REAL,
    conf_thesis        TEXT,
    conf_data          TEXT,
    conf_statistical_edge TEXT,
    conf_allocation    TEXT,
    edge_score         REAL,        -- probability-like field for Brier/calibration
    position_size_pct  REAL,
    pillars_json       TEXT,
    gates_json         TEXT,
    evidence_ids_json  TEXT,
    strategy_version   TEXT,        -- experiment manifest hash
    data_timestamps_json TEXT,
    decision_fingerprint TEXT,
    sector             TEXT,
    regime             TEXT,
    benchmark          TEXT,
    replay_run_id      TEXT,        -- NULL = live prediction; set = historical replay
    frozen_json        TEXT NOT NULL,
    content_hash       TEXT NOT NULL
);

-- IMMUTABILITY: a frozen prediction can never be edited or deleted. Enforced in
-- the database itself, so no code path — present or future — can rewrite history.
CREATE TRIGGER IF NOT EXISTS prediction_snapshots_no_update
BEFORE UPDATE ON prediction_snapshots
BEGIN SELECT RAISE(ABORT, 'prediction snapshots are immutable (no UPDATE)'); END;

CREATE TRIGGER IF NOT EXISTS prediction_snapshots_no_delete
BEFORE DELETE ON prediction_snapshots
BEGIN SELECT RAISE(ABORT, 'prediction snapshots are immutable (no DELETE)'); END;

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    snapshot_id        TEXT NOT NULL,
    horizon_days       INTEGER NOT NULL,
    evaluated_at       TEXT NOT NULL,
    as_of_date         TEXT,
    matured            INTEGER NOT NULL DEFAULT 0,
    price_at_horizon   REAL,
    raw_return_pct     REAL,
    benchmark_return_pct REAL,
    excess_return_pct  REAL,
    mae_pct            REAL,        -- maximum ADVERSE excursion over the window
    mfe_pct            REAL,        -- maximum FAVORABLE excursion
    direction_correct  INTEGER,
    brier              REAL,
    PRIMARY KEY (snapshot_id, horizon_days),
    FOREIGN KEY (snapshot_id) REFERENCES prediction_snapshots(snapshot_id)
);
"""

_BULLISH = {"BUY", "ACCUMULATE", "LONG"}
_BEARISH = {"SELL", "REDUCE", "SHORT", "AVOID"}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db(), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Migration for DBs created before replay_run_id existed (additive column;
    # does not touch existing rows, so immutability is preserved).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(prediction_snapshots)")}
    if "replay_run_id" not in cols:
        conn.execute("ALTER TABLE prediction_snapshots ADD COLUMN replay_run_id TEXT")
    conn.commit()
    return conn


def _source_where(source: str, alias: str = "s") -> str:
    """SQL fragment to filter live vs historical(replay) snapshots."""
    if source == "live":
        return f"AND {alias}.replay_run_id IS NULL"
    if source in ("replay", "historical"):
        return f"AND {alias}.replay_run_id IS NOT NULL"
    return ""      # "all"


# ── Freeze (immutable snapshot) ─────────────────────────────────────────────

def _content_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze_prediction(rec: Dict[str, Any]) -> Optional[str]:
    """Freeze a recommendation as an immutable snapshot. Returns snapshot_id.

    Idempotent: the id is a content hash of the frozen payload (including
    created_at), so re-freezing the identical rec returns the existing id and
    writes nothing. A logging failure returns None and never breaks the caller.
    """
    try:
        conf = rec.get("confidence") or {}
        frozen = {
            "ticker": rec["ticker"], "created_at": rec.get("generated_at"),
            "price_at_call": rec.get("current_price"), "action": rec.get("action"),
            "horizon_days": rec.get("time_horizon_days"),
            "expected_return_pct": rec.get("expected_return_pct"),
            "confidence": {k: v.get("level") for k, v in conf.items()},
            "confidence_scores": {k: v.get("score") for k, v in conf.items()},
            "pillars": {k: v.get("score") for k, v in (rec.get("pillars") or {}).items()},
            "gates": {g: c.get("pass") for g, c in
                      (conf.get("statistical_edge", {}).get("checks") or {}).items()},
            "evidence_ids": rec.get("claims") or {},
            "strategy_version": rec.get("experiment_manifest_hash"),
            "data_timestamps": {"data_asof": rec.get("data_asof"),
                                "generated_at": rec.get("generated_at")},
            "decision_fingerprint": rec.get("decision_fingerprint"),
            "sector": rec.get("sector"), "regime": rec.get("regime"),
            "benchmark": rec.get("benchmark", DEFAULT_BENCHMARK),
            "position_size_pct": rec.get("position_size_pct"),
            "replay_run_id": rec.get("replay_run_id"),
            "replay_meta": rec.get("replay_meta"),   # coverage/missing/exclusions (frozen, immutable)
        }
        sid = _content_hash(frozen)
        content_hash = _content_hash({k: v for k, v in frozen.items() if k != "created_at"})
        edge_score = (conf.get("statistical_edge") or {}).get("score")

        conn = _conn()
        try:
            # INSERT OR IGNORE only — never REPLACE (REPLACE would fire the
            # no-DELETE trigger, and rewriting a frozen prediction is forbidden).
            conn.execute(
                """INSERT OR IGNORE INTO prediction_snapshots
                   (snapshot_id, created_at, ticker, price_at_call, action, horizon_days,
                    expected_return_pct, conf_thesis, conf_data, conf_statistical_edge,
                    conf_allocation, edge_score, position_size_pct, pillars_json, gates_json,
                    evidence_ids_json, strategy_version, data_timestamps_json,
                    decision_fingerprint, sector, regime, benchmark, replay_run_id,
                    frozen_json, content_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, frozen["created_at"], frozen["ticker"], frozen["price_at_call"],
                 frozen["action"], frozen["horizon_days"], frozen["expected_return_pct"],
                 frozen["confidence"].get("thesis"), frozen["confidence"].get("data"),
                 frozen["confidence"].get("statistical_edge"),
                 frozen["confidence"].get("allocation"), edge_score,
                 frozen["position_size_pct"], json.dumps(frozen["pillars"]),
                 json.dumps(frozen["gates"]), json.dumps(frozen["evidence_ids"]),
                 frozen["strategy_version"], json.dumps(frozen["data_timestamps"]),
                 frozen["decision_fingerprint"], frozen["sector"], frozen["regime"],
                 frozen["benchmark"], frozen["replay_run_id"],
                 json.dumps(frozen, default=str), content_hash))
            conn.commit()
            return sid
        finally:
            conn.close()
    except Exception:
        return None


def get_snapshot(snapshot_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM prediction_snapshots WHERE snapshot_id=?",
                           (snapshot_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def verify_snapshot(snapshot_id: str) -> bool:
    """Recompute the content hash from the frozen payload; True iff untampered."""
    snap = get_snapshot(snapshot_id)
    if not snap:
        return False
    frozen = json.loads(snap["frozen_json"])
    recomputed = _content_hash({k: v for k, v in frozen.items() if k != "created_at"})
    return recomputed == snap["content_hash"]


def list_snapshots(ticker: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM prediction_snapshots WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM prediction_snapshots ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Outcome evaluation (pure price arithmetic) ──────────────────────────────

def evaluate_outcomes(price_at_call_date: str, action: str, prices, benchmark,
                      edge_score: Optional[float] = None,
                      horizons=HORIZONS) -> Dict[int, Dict[str, Any]]:
    """Compute outcomes at each horizon from ADJUSTED price series.

    Args:
      price_at_call_date: ISO date of the recommendation.
      prices/benchmark:   adjusted-close pd.Series indexed by date (the SAME
                          current series for both endpoints, so a split between
                          call and horizon is handled consistently — corporate
                          actions never manufacture a spurious return).
      edge_score:         probability-like field; enables Brier where present.

    Missing market days: the call date is snapped to the nearest available
    trading day at/after it, and a horizon of N is N POSITIONS along the
    trading-day index (so holidays/weekends are skipped naturally).
    """
    import numpy as np
    import pandas as pd

    prices = prices.dropna()
    idx = prices.index
    call_ts = pd.Timestamp(price_at_call_date)
    pos = int(idx.searchsorted(call_ts))
    out: Dict[int, Dict[str, Any]] = {}
    if pos >= len(idx):
        # The call date is after all available data — nothing matured yet.
        for h in horizons:
            out[h] = {"matured": False}
        return out

    p_call = float(prices.iloc[pos])
    bullish = action.upper() in _BULLISH
    bearish = action.upper() in _BEARISH
    p_dir = None if edge_score is None else 0.5 + 0.5 * float(edge_score)   # P(direction correct)

    # Benchmark aligned to the same trading-day grid.
    bench = benchmark.dropna() if benchmark is not None else None

    for h in horizons:
        hpos = pos + h
        if hpos >= len(idx):
            out[h] = {"matured": False}
            continue
        p_h = float(prices.iloc[hpos])
        raw = (p_h / p_call - 1.0) * 100.0
        window = prices.iloc[pos:hpos + 1]
        mfe = (float(window.max()) / p_call - 1.0) * 100.0
        mae = (float(window.min()) / p_call - 1.0) * 100.0

        bench_ret = None
        excess = None
        if bench is not None and len(bench):
            bpos = int(bench.index.searchsorted(idx[pos]))
            bhpos = int(bench.index.searchsorted(idx[hpos]))
            if bpos < len(bench) and bhpos < len(bench):
                b0, b1 = float(bench.iloc[bpos]), float(bench.iloc[bhpos])
                if b0:
                    bench_ret = (b1 / b0 - 1.0) * 100.0
                    excess = raw - bench_ret

        if bullish:
            direction_correct = int(raw > 0)
        elif bearish:
            direction_correct = int(raw < 0)
        else:
            direction_correct = None      # HOLD makes no directional claim

        brier = None
        if p_dir is not None and direction_correct is not None:
            brier = (p_dir - direction_correct) ** 2

        out[h] = {
            "matured": True, "as_of_date": str(idx[hpos].date()),
            "price_at_horizon": round(p_h, 4),
            "raw_return_pct": round(raw, 4),
            "benchmark_return_pct": round(bench_ret, 4) if bench_ret is not None else None,
            "excess_return_pct": round(excess, 4) if excess is not None else None,
            "mae_pct": round(mae, 4), "mfe_pct": round(mfe, 4),
            "direction_correct": direction_correct,
            "brier": round(brier, 6) if brier is not None else None,
        }
    return out


def _upsert_outcome(conn, sid: int, h: int, o: Dict[str, Any]) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO prediction_outcomes
           (snapshot_id, horizon_days, evaluated_at, as_of_date, matured,
            price_at_horizon, raw_return_pct, benchmark_return_pct, excess_return_pct,
            mae_pct, mfe_pct, direction_correct, brier)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (sid, h, datetime.now().isoformat(timespec="seconds"), o.get("as_of_date"),
         int(bool(o.get("matured"))), o.get("price_at_horizon"), o.get("raw_return_pct"),
         o.get("benchmark_return_pct"), o.get("excess_return_pct"), o.get("mae_pct"),
         o.get("mfe_pct"), o.get("direction_correct"), o.get("brier")))


def refresh_outcomes(ticker: Optional[str] = None,
                     fetch_fn: Optional[Callable[[str], Any]] = None,
                     benchmark_fetch_fn: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Re-evaluate outcomes for snapshots (optionally one ticker). Idempotent:
    re-running yields the same matured values (deterministic from history).

    fetch_fn(ticker) -> adjusted-close pd.Series (injectable for tests; live uses
    tools.market_data.fetch_price_history's Close). Prices are fetched ONCE per
    ticker and reused for every snapshot + horizon on that ticker.
    """
    import pandas as pd

    if fetch_fn is None:
        def fetch_fn(t):
            from tools.market_data import fetch_price_history
            return fetch_price_history(t, period="2y")["Close"]
    if benchmark_fetch_fn is None:
        benchmark_fetch_fn = fetch_fn

    snaps = list_snapshots(ticker=ticker)
    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for s in snaps:
        by_ticker.setdefault(s["ticker"], []).append(s)

    n_snap = 0
    n_outcome = 0
    n_matured = 0
    errors: List[str] = []
    bench_cache: Dict[str, Any] = {}

    conn = _conn()
    try:
        for tkr, group in by_ticker.items():
            try:
                prices = fetch_fn(tkr)
            except Exception as e:
                errors.append(f"{tkr}: price fetch failed: {str(e)[:60]}")
                continue
            for s in group:
                bmark = s.get("benchmark") or DEFAULT_BENCHMARK
                if bmark not in bench_cache:
                    try:
                        bench_cache[bmark] = benchmark_fetch_fn(bmark)
                    except Exception:
                        bench_cache[bmark] = None
                outs = evaluate_outcomes(
                    s["created_at"][:10], s["action"], prices, bench_cache[bmark],
                    edge_score=s.get("edge_score"))
                n_snap += 1
                for h, o in outs.items():
                    _upsert_outcome(conn, s["snapshot_id"], h, o)
                    n_outcome += 1
                    if o.get("matured"):
                        n_matured += 1
        conn.commit()
    finally:
        conn.close()
    return {"snapshots_evaluated": n_snap, "outcome_rows": n_outcome,
            "matured": n_matured, "errors": errors,
            "refreshed_at": datetime.now().isoformat(timespec="seconds")}


# ── Calibration + attribution (read-only measurement) ───────────────────────

def _bucket_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    dirs = [r["direction_correct"] for r in rows if r["direction_correct"] is not None]
    excess = [r["excess_return_pct"] for r in rows if r["excess_return_pct"] is not None]
    raw = [r["raw_return_pct"] for r in rows if r["raw_return_pct"] is not None]
    briers = [r["brier"] for r in rows if r["brier"] is not None]
    return {
        "n": n,
        "win_rate": round(sum(dirs) / len(dirs), 3) if dirs else None,
        "avg_raw_return_pct": round(sum(raw) / len(raw), 2) if raw else None,
        "avg_excess_return_pct": round(sum(excess) / len(excess), 2) if excess else None,
        "brier": round(sum(briers) / len(briers), 4) if briers else None,
    }


def calibration_report(horizon: int = 20, source: str = "all") -> Dict[str, Any]:
    """Everything the dashboard shows: overall win rate + calibration, plus
    breakdowns by action, confidence, regime, sector, and pillar attribution.
    `source`: 'all' | 'live' | 'replay' (historical). Read-only — this NEVER
    writes back into the recommendation path."""
    conn = _conn()
    try:
        rows = [dict(r) for r in conn.execute(
            f"""SELECT o.*, s.action AS s_action, s.conf_statistical_edge, s.conf_thesis,
                      s.conf_data, s.conf_allocation, s.edge_score, s.regime, s.sector,
                      s.pillars_json, s.replay_run_id
               FROM prediction_outcomes o
               JOIN prediction_snapshots s ON s.snapshot_id = o.snapshot_id
               WHERE o.horizon_days=? AND o.matured=1 {_source_where(source)}""",
            (horizon,)).fetchall()]
    finally:
        conn.close()

    overall = _bucket_stats(rows)

    def group(key_fn):
        g: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            g.setdefault(str(key_fn(r)), []).append(r)
        return {k: _bucket_stats(v) for k, v in sorted(g.items())}

    by_action = group(lambda r: r["s_action"])
    by_confidence = group(lambda r: r["conf_statistical_edge"] or "—")
    by_regime = group(lambda r: r["regime"] or "unknown")
    by_sector = group(lambda r: r["sector"] or "unknown")

    # Pillar (agent) attribution: for each pillar, outcomes when it was LEANING
    # (score >= 60 at call time). Shows which agent signals actually preceded wins.
    pillar_attr: Dict[str, Any] = {}
    for pillar in ("technical", "algo", "risk", "fundamentals", "research", "social"):
        leaning = []
        for r in rows:
            try:
                pj = json.loads(r["pillars_json"] or "{}")
            except Exception:
                pj = {}
            if pj.get(pillar) is not None and pj.get(pillar) >= 60:
                leaning.append(r)
        pillar_attr[pillar] = _bucket_stats(leaning)

    # Confidence reliability / calibration error: per statistical-edge bucket,
    # compare the predicted probability (0.5 + 0.5*edge_score) with the realized
    # win rate. ECE = sample-weighted mean gap. This is the honest answer to
    # "does HIGH confidence actually win more?".
    reliability = []
    ece_num = 0.0
    ece_den = 0
    for level in ("HIGH", "MEDIUM", "LOW", "NONE"):
        bucket = [r for r in rows if (r["conf_statistical_edge"] or "") == level
                  and r["direction_correct"] is not None]
        if not bucket:
            continue
        realized = sum(r["direction_correct"] for r in bucket) / len(bucket)
        preds = [0.5 + 0.5 * (r["edge_score"] or 0.0) for r in bucket]
        predicted = sum(preds) / len(preds)
        gap = abs(predicted - realized)
        reliability.append({"level": level, "n": len(bucket),
                            "predicted_win_prob": round(predicted, 3),
                            "realized_win_rate": round(realized, 3),
                            "calibration_gap": round(gap, 3)})
        ece_num += gap * len(bucket)
        ece_den += len(bucket)

    return {
        "horizon_days": horizon,
        "overall": overall,
        "calibration_error_ece": round(ece_num / ece_den, 4) if ece_den else None,
        "by_action": by_action,
        "by_confidence": by_confidence,
        "by_regime": by_regime,
        "by_sector": by_sector,
        "pillar_attribution": pillar_attr,
        "confidence_reliability": reliability,
    }


def summary(source: str = "all") -> Dict[str, Any]:
    """Active vs matured counts for the dashboard header (optionally filtered to
    live vs historical/replay predictions)."""
    conn = _conn()
    try:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM prediction_snapshots s WHERE 1=1 {_source_where(source)}"
        ).fetchone()["n"]
        matured_20 = conn.execute(
            f"""SELECT COUNT(*) AS n FROM prediction_outcomes o
                JOIN prediction_snapshots s ON s.snapshot_id=o.snapshot_id
                WHERE o.horizon_days=20 AND o.matured=1 {_source_where(source)}""").fetchone()["n"]
        pending = conn.execute(
            f"""SELECT COUNT(DISTINCT s.snapshot_id) AS n FROM prediction_snapshots s
                LEFT JOIN prediction_outcomes o
                  ON o.snapshot_id=s.snapshot_id AND o.horizon_days=252 AND o.matured=1
                WHERE o.snapshot_id IS NULL {_source_where(source)}""").fetchone()["n"]
        by_source = {r["src"]: r["n"] for r in conn.execute(
            "SELECT CASE WHEN replay_run_id IS NULL THEN 'live' ELSE 'historical' END AS src, "
            "COUNT(*) AS n FROM prediction_snapshots GROUP BY src")}
        return {"total_predictions": total, "matured_20d": matured_20,
                "active_not_fully_matured": pending, "by_source": by_source}
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        tkr = sys.argv[2] if len(sys.argv) > 2 else None
        print(json.dumps(refresh_outcomes(ticker=tkr), indent=2))
    else:
        print(json.dumps(summary(), indent=2))
        print(json.dumps(calibration_report(), indent=2, default=str))
