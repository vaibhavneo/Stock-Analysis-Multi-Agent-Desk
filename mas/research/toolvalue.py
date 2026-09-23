"""Does a tool actually change anything?

A capability called a hundred times that never alters a decision is not
obviously worth its latency — but that is a claim about a distribution, not
about one request, and it cannot be settled from a single trace. So outcomes
are accumulated per capability over time and reported as a record.

NOTHING IS REMOVED ON THE STRENGTH OF THIS
------------------------------------------
Measuring first is the point. A tool that changed no decision this month may
be the one that catches the month something breaks, and a low rate is a
prompt to look rather than a verdict. The report says what happened; it does
not recommend deletion.

FOUR OUTCOMES, BECAUSE THEY ARE FOUR DIFFERENT FACTS
----------------------------------------------------
    invoked            it ran
    produced_evidence  it returned something normalisable
    read               synthesis categorised it
    weighed            it entered the directional reading
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_value (
    capability        TEXT PRIMARY KEY,
    invoked           INTEGER NOT NULL DEFAULT 0,
    produced          INTEGER NOT NULL DEFAULT 0,
    read_count        INTEGER NOT NULL DEFAULT 0,
    weighed           INTEGER NOT NULL DEFAULT 0,
    failed            INTEGER NOT NULL DEFAULT 0,
    total_ms          INTEGER NOT NULL DEFAULT 0,
    last_seen         TEXT
);
"""

_applied: set = set()


def _conn():
    from data import prediction_ledger as pl
    conn = pl._conn()
    key = pl._db()
    if key not in _applied:
        conn.executescript(_SCHEMA)
        conn.commit()
        _applied.add(key)
    return conn


def record(result: Dict[str, Any]) -> bool:
    """Accumulate one request's outcomes. Never raises."""
    try:
        steps = (result.get("trace") or {}).get("steps") or []
        if not steps:
            return False
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn = _conn()
        try:
            for st in steps:
                cap = st.get("capability")
                if not cap:
                    continue
                ok = st.get("outcome") == "SUCCESS"
                conn.execute(
                    "INSERT INTO tool_value (capability, invoked, produced, "
                    "read_count, weighed, failed, total_ms, last_seen) "
                    "VALUES (?,1,?,?,?,?,?,?) "
                    "ON CONFLICT(capability) DO UPDATE SET "
                    " invoked=invoked+1, produced=produced+?, "
                    " read_count=read_count+?, weighed=weighed+?, "
                    " failed=failed+?, total_ms=total_ms+?, last_seen=?",
                    (cap, 1 if ok else 0, 1 if st.get("consumed") else 0,
                     1 if st.get("changed_synthesis") else 0,
                     0 if ok else 1, int(st.get("elapsed_ms") or 0), now,
                     1 if ok else 0, 1 if st.get("consumed") else 0,
                     1 if st.get("changed_synthesis") else 0,
                     0 if ok else 1, int(st.get("elapsed_ms") or 0), now))
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def report() -> Dict[str, Any]:
    """What each capability has actually contributed. Never raises."""
    try:
        conn = _conn()
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM tool_value ORDER BY invoked DESC")]
        finally:
            conn.close()
    except Exception as e:
        return {"available": False,
                "reason": f"{type(e).__name__}: {e}", "capabilities": []}

    out = []
    for r in rows:
        inv = r["invoked"] or 1
        out.append({
            "capability": r["capability"],
            "invoked": r["invoked"],
            "produced_evidence": r["produced"],
            "read_by_synthesis": r["read_count"],
            "entered_weighing": r["weighed"],
            "failed": r["failed"],
            "produce_rate": round(r["produced"] / inv, 3),
            "weigh_rate": round(r["weighed"] / inv, 3),
            "avg_ms": round((r["total_ms"] or 0) / inv),
            "last_seen": r["last_seen"],
            "note": (
                "never entered a directional reading — worth looking at, "
                "though a tool that changes nothing this month may be the one "
                "that catches the month something breaks"
                if r["weighed"] == 0 and r["invoked"] >= 5 else
                "contributes to the directional reading"
                if r["weighed"] else "too few runs to say"),
        })
    total = sum(r["invoked"] for r in rows)
    never = [o["capability"] for o in out
             if o["entered_weighing"] == 0 and o["invoked"] >= 5]
    return {
        "available": True, "capabilities": out, "total_invocations": total,
        "never_weighed": never,
        "statement": (
            f"{len(out)} capabilities over {total} invocation(s). "
            + (f"{len(never)} produced evidence that never entered a "
               f"directional reading: {', '.join(never)}. Measured, not acted "
               f"on — nothing is removed on the strength of this."
               if never else
               "Every capability with enough runs has contributed to a "
               "directional reading at least once.")),
        "caveat": ("Counts accumulate across requests and model versions. A "
                   "low rate is a prompt to look, never a verdict."),
    }
