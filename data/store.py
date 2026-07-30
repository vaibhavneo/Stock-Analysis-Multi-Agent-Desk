"""
Phase 4 — Recommendation Persistence & Tracking
SQLite-backed store (stdlib sqlite3, no new deps).

Tables:
  recommendations — one row per prediction call
  outcomes        — one row per outcome check (appended, not upserted)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

_DB_PATH = Path(__file__).parent / "recommendations.db"

# Public alias for sibling modules (ledger.py) that add their own tables to the
# SAME database. Exported rather than re-derived so the path can never drift
# into two files pointing at two different DBs.
DB_PATH = _DB_PATH

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    price_at_call       REAL    NOT NULL,
    action              TEXT,
    entry_price         REAL,
    target_price        REAL,
    stop_loss           REAL,
    conviction          TEXT,
    grounding_strategy  TEXT,
    kelly_fraction      REAL,
    time_horizon_days   INTEGER,
    raw_json            TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id   INTEGER NOT NULL,
    checked_at          TEXT    NOT NULL,
    price_at_check      REAL    NOT NULL,
    realized_pct_change REAL    NOT NULL,
    hit_target          INTEGER,
    hit_stop            INTEGER,
    direction_correct   INTEGER,
    FOREIGN KEY (recommendation_id) REFERENCES recommendations(id)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def log_recommendation(rec: dict) -> int:
    """Persist a prediction. Returns the new row id."""
    conn = _connect()
    try:
        pred   = rec.get("prediction") or {}
        grnd   = rec.get("grounding") or {}
        action = pred.get("action") or pred.get("recommendation")

        cur = conn.execute(
            """
            INSERT INTO recommendations
              (ticker, created_at, price_at_call, action,
               entry_price, target_price, stop_loss, conviction,
               grounding_strategy, kelly_fraction, time_horizon_days, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.get("ticker", ""),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                float(rec.get("current_price") or 0),
                str(action or ""),
                _safe_float(pred.get("entry_price")),
                _safe_float(pred.get("target_price")),
                _safe_float(pred.get("stop_loss")),
                str(pred.get("conviction") or ""),
                str(grnd.get("grounding_strategy") or ""),
                _safe_float(grnd.get("position_size")),
                _safe_int(pred.get("time_horizon_days")),
                # 16000 (was 8000): composite recommendations carry pillar
                # scores + honesty flags + claim ids in the payload, and a
                # truncated raw_json would silently amputate the audit trail.
                json.dumps({k: v for k, v in rec.items() if k not in ("fundamentals",)},
                           default=str)[:16000],
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def check_outcome(recommendation_id: int, current_price: float | None = None) -> dict:
    """
    Fetch a recommendation, compute realized P&L against the logged price_at_call,
    persist to outcomes, and return a summary dict.

    current_price: pass in the live price if you already have it (avoids a yfinance call);
    otherwise the function fetches it via yfinance.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM recommendations WHERE id=?", (recommendation_id,)
        ).fetchone()
        if not row:
            return {"error": f"No recommendation with id={recommendation_id}"}

        rec = dict(row)

        if current_price is None:
            try:
                import yfinance as yf
                hist = yf.Ticker(rec["ticker"]).history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
            except Exception:
                current_price = None

        if current_price is None:
            return {"error": "Could not fetch current price"}

        price_at_call = rec["price_at_call"] or 1.0
        realized_pct  = (current_price - price_at_call) / price_at_call * 100

        # Include the 7-pillar composite's investing verbs: ACCUMULATE is a
        # bullish call (like BUY), REDUCE a bearish one (like SELL). Without
        # these, composite recommendations would never get a direction and their
        # forward hit-rate would be permanently empty.
        action = (rec.get("action") or "").upper()
        if action in ("BUY", "LONG", "ACCUMULATE"):
            direction_correct = int(current_price > price_at_call)
        elif action in ("SELL", "SHORT", "AVOID", "REDUCE"):
            direction_correct = int(current_price < price_at_call)
        else:
            direction_correct = None

        target = rec.get("target_price")
        stop   = rec.get("stop_loss")
        hit_target = int(current_price >= target) if target else None
        hit_stop   = int(current_price <= stop)   if stop   else None

        conn.execute(
            """
            INSERT INTO outcomes
              (recommendation_id, checked_at, price_at_check,
               realized_pct_change, hit_target, hit_stop, direction_correct)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                recommendation_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                current_price,
                round(realized_pct, 4),
                hit_target,
                hit_stop,
                direction_correct,
            ),
        )
        conn.commit()
        return {
            "recommendation_id": recommendation_id,
            "ticker":            rec["ticker"],
            "action":            rec["action"],
            "price_at_call":     price_at_call,
            "current_price":     current_price,
            "realized_pct":      round(realized_pct, 2),
            "hit_target":        hit_target,
            "hit_stop":          hit_stop,
            "direction_correct": direction_correct,
        }
    finally:
        conn.close()


def get_historical_hit_rate(
    strategy_source: str | None = None,
    ticker: str | None = None,
) -> dict:
    """
    Aggregate win-rate across logged+checked recommendations.
    Returns a dict with keys: total, correct, hit_rate (0-1), "no_data" flag.
    Optionally filtered by grounding_strategy or ticker.
    """
    conn = _connect()
    try:
        conditions = ["o.direction_correct IS NOT NULL"]
        params: list = []
        if strategy_source:
            conditions.append("r.grounding_strategy = ?")
            params.append(strategy_source)
        if ticker:
            conditions.append("r.ticker = ?")
            params.append(ticker.upper())

        where = " AND ".join(conditions)
        sql = f"""
            SELECT COUNT(*) AS total,
                   SUM(o.direction_correct) AS correct
            FROM outcomes o
            JOIN recommendations r ON r.id = o.recommendation_id
            WHERE {where}
        """
        row = conn.execute(sql, params).fetchone()
        total   = row["total"]   or 0
        correct = row["correct"] or 0

        if total == 0:
            return {"total": 0, "correct": 0, "hit_rate": None, "no_data": True}

        return {
            "total":    total,
            "correct":  correct,
            "hit_rate": round(correct / total, 3),
            "no_data":  False,
        }
    finally:
        conn.close()


def get_history(ticker: str | None = None, limit: int = 50) -> list[dict]:
    """Return recent recommendations + their latest outcome."""
    conn = _connect()
    try:
        conditions = []
        params: list = []
        if ticker:
            conditions.append("r.ticker = ?")
            params.append(ticker.upper())
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        sql = f"""
            SELECT r.id, r.ticker, r.created_at, r.price_at_call, r.action,
                   r.entry_price, r.target_price, r.stop_loss, r.conviction,
                   r.grounding_strategy,
                   o.checked_at, o.price_at_check, o.realized_pct_change,
                   o.hit_target, o.hit_stop, o.direction_correct
            FROM recommendations r
            LEFT JOIN outcomes o ON o.recommendation_id = r.id
              AND o.id = (SELECT MAX(id) FROM outcomes WHERE recommendation_id = r.id)
            {where}
            ORDER BY r.id DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
