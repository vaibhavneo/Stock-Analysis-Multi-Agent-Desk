"""Keep the last research result per symbol, so "what changed" has a prior.

Append-only and small. Only the fields change detection reads are stored —
the evidence keys, their directions, values and freshness, plus the consensus
— because storing the whole result would grow without bound for no gain, and
because a snapshot is a comparison basis rather than a second copy of the
answer.

Snapshots are never rewritten. A later model producing a different reading
must not retroactively change what the earlier one said, or "what changed"
becomes a comparison of the present against itself.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol     TEXT NOT NULL,
    intent     TEXT,
    as_of      TEXT NOT NULL,
    epoch      REAL NOT NULL,
    consensus  TEXT,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_snapshots_symbol
    ON research_snapshots(symbol, epoch DESC);
"""

# Rewriting history would make "what changed" a comparison of the present
# against itself.
_IMMUTABLE = """
CREATE TRIGGER IF NOT EXISTS research_snapshots_no_update
BEFORE UPDATE ON research_snapshots
BEGIN SELECT RAISE(ABORT, 'research snapshots are append-only'); END;
"""

_applied: set = set()


def _conn():
    from data import prediction_ledger as pl
    conn = pl._conn()
    key = pl._db()
    if key not in _applied:
        conn.executescript(_SCHEMA)
        try:
            conn.executescript(_IMMUTABLE)
        except sqlite3.OperationalError:
            pass
        conn.commit()
        _applied.add(key)
    return conn


def _slim(result: Dict[str, Any]) -> Dict[str, Any]:
    """Only what change detection reads."""
    items = []
    for i in ((result.get("evidence") or {}).get("items") or []):
        items.append({k: i.get(k) for k in
                      ("key", "source", "direction", "value", "freshness",
                       "usable_for_decision", "statement", "tier")})
    return {"evidence": {"items": items},
            "synthesis": {"consensus": (result.get("synthesis") or {}).get("consensus")},
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%S")}


def save(symbol: str, result: Dict[str, Any]) -> Optional[int]:
    """Store one snapshot. Never raises — losing a comparison basis must not
    take down the answer it was comparing."""
    if not symbol:
        return None
    try:
        slim = _slim(result)
        conn = _conn()
        try:
            cur = conn.execute(
                "INSERT INTO research_snapshots "
                "(symbol, intent, as_of, epoch, consensus, payload) "
                "VALUES (?,?,?,?,?,?)",
                (symbol.upper(), (result.get("plan") or {}).get("intent"),
                 slim["as_of"], time.time(),
                 slim["synthesis"]["consensus"], json.dumps(slim)))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except Exception:
        return None


def latest(symbol: str, before_epoch: Optional[float] = None
           ) -> Optional[Dict[str, Any]]:
    """The most recent snapshot for this symbol, or None."""
    if not symbol:
        return None
    try:
        conn = _conn()
        try:
            if before_epoch is None:
                row = conn.execute(
                    "SELECT payload FROM research_snapshots WHERE symbol=? "
                    "ORDER BY epoch DESC LIMIT 1", (symbol.upper(),)).fetchone()
            else:
                row = conn.execute(
                    "SELECT payload FROM research_snapshots WHERE symbol=? "
                    "AND epoch < ? ORDER BY epoch DESC LIMIT 1",
                    (symbol.upper(), float(before_epoch))).fetchone()
            return json.loads(row["payload"]) if row else None
        finally:
            conn.close()
    except Exception:
        return None


def history(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        conn = _conn()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT id, symbol, intent, as_of, consensus FROM "
                "research_snapshots WHERE symbol=? ORDER BY epoch DESC LIMIT ?",
                (symbol.upper(), int(limit)))]
        finally:
            conn.close()
    except Exception:
        return []
