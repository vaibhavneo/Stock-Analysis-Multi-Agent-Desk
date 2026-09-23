"""Every research result, frozen as it was produced.

The decision journal already records the BRIEF. It does not record the
research that produced it — which plan was made, which tools were reachable,
which specialists ran, what evidence they returned, or what was missing. So a
decision could be re-read later without any way to reconstruct why it had the
inputs it had.

APPEND-ONLY, ENFORCED BY THE DATABASE
-------------------------------------
A later model producing a different reading must never retroactively change
what an earlier one recorded. If it could, forward evaluation would be
grading the present against a version of the past the present wrote — which
is not evaluation, it is confirmation. UPDATE and DELETE are refused by
trigger, not by convention.

MODEL AND DATA VERSIONS ARE STORED WITH THE ROW
-----------------------------------------------
Because the same question, asked twice across a model change, SHOULD produce
different answers, and without the version stamps that difference is
indistinguishable from the market having moved.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_journal (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint    TEXT NOT NULL UNIQUE,
    symbol         TEXT NOT NULL,
    intent         TEXT,
    question       TEXT,
    as_of          TEXT NOT NULL,
    epoch          REAL NOT NULL,
    price          REAL,
    horizon_days   INTEGER,
    consensus      TEXT,
    decision_state TEXT,
    depth          TEXT,
    model_version  TEXT,
    data_version   TEXT,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_research_journal_symbol
    ON research_journal(symbol, epoch DESC);
"""

_IMMUTABLE = """
CREATE TRIGGER IF NOT EXISTS research_journal_no_update
BEFORE UPDATE ON research_journal
BEGIN SELECT RAISE(ABORT, 'the research journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS research_journal_no_delete
BEFORE DELETE ON research_journal
BEGIN SELECT RAISE(ABORT, 'the research journal is append-only'); END;
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


def _model_version() -> str:
    """Identifies the engine that produced a row, so a later difference can be
    attributed to a model change rather than to the market."""
    try:
        from mas import registry
        reg = registry.load()
        return f"mas.v{reg.get('version')}+journal.v{SCHEMA_VERSION}"
    except Exception:
        return f"journal.v{SCHEMA_VERSION}"


def _data_version(result: Dict[str, Any]) -> str:
    tools = [t.get("tool") for t in ((result.get("plan") or {}).get("tools") or [])
             if t.get("tool")]
    disc = result.get("tool_discovery") or {}
    return f"tools:{len(set(tools))}/{disc.get('n_available')}of{disc.get('n_total')}"


def record(result: Dict[str, Any]) -> Dict[str, Any]:
    """The frozen record, exactly the fields Phase 29 names."""
    plan = result.get("plan") or {}
    dec = result.get("decision") or {}
    sections = {s.get("key"): s for s in (dec.get("sections") or [])}

    def sec(name):
        s = sections.get(name) or {}
        return {"status": s.get("status"), "headline": s.get("headline"),
                "detail": s.get("detail")} if s else None

    return {
        "question": result.get("question"),
        "symbol": (plan.get("symbols") or [None])[0],
        "symbols": plan.get("symbols"),
        "intent": plan.get("intent"),
        "horizon": plan.get("horizon"),
        "position": plan.get("position"),
        "research_plan": {
            "objective": plan.get("objective"),
            "required_evidence": plan.get("required_evidence"),
            "optional_evidence": plan.get("optional_evidence"),
            "missing": plan.get("missing"),
            "answerable": plan.get("answerable"),
        },
        "tools_used": [{"evidence": t.get("evidence"), "tool": t.get("tool"),
                        "necessity": t.get("necessity")}
                       for t in (plan.get("tools") or [])],
        "agents_used": [{"capability": c.get("capability"),
                         "necessity": c.get("necessity"),
                         "question": (c.get("brief") or {}).get("question")}
                        for c in (plan.get("capabilities") or [])],
        "evidence": [{k: i.get(k) for k in
                      ("key", "tier", "source", "direction", "horizon",
                       "confidence", "weight", "freshness", "statement",
                       "usable_for_decision", "uncertainty", "flags")}
                     for i in ((result.get("evidence") or {}).get("items") or [])],
        "synthesis": {k: (result.get("synthesis") or {}).get(k) for k in
                      ("consensus", "agreement_ratio", "n_directional_sources",
                       "bullish_weight", "bearish_weight", "n_conflicts",
                       "n_decision_changing", "statement")},
        "decision_state": sec("state"),
        "entry_conditions": sec("entry"),
        "add_conditions": sec("add"),
        "reduce_conditions": sec("reduce"),
        "exit_conditions": sec("exit"),
        "invalidation": sec("invalidation"),
        "monitoring": sec("monitoring"),
        "scenarios": sec("scenarios"),
        "risk": sec("risk"),
        "catalysts": [i for i in
                      ((result.get("evidence") or {}).get("items") or [])
                      if str(i.get("key", "")).startswith("catalyst")],
        "escalation": result.get("escalation"),
        "adversarial": result.get("adversarial"),
        "validation": result.get("validation"),
        "accounting": result.get("accounting"),
        "trace": (result.get("trace") or {}).get("summary"),
        "model_version": _model_version(),
        "data_version": _data_version(result),
        "schema_version": SCHEMA_VERSION,
    }


def _fingerprint(rec: Dict[str, Any]) -> str:
    """Content-addressed, so re-asking the same question against unchanged
    evidence does not create a second row."""
    basis = json.dumps({
        "symbol": rec.get("symbol"), "intent": rec.get("intent"),
        "evidence": [(e.get("key"), e.get("value") if "value" in e else
                      e.get("statement")) for e in (rec.get("evidence") or [])],
        "synthesis": (rec.get("synthesis") or {}).get("consensus"),
        "model_version": rec.get("model_version"),
    }, sort_keys=True, default=str)
    return hashlib.sha256(basis.encode()).hexdigest()[:24]


def journal(result: Dict[str, Any]) -> Optional[str]:
    """Freeze one research result. Never raises — losing a journal row must
    not take down the answer it was recording."""
    try:
        rec = record(result)
        if not rec.get("symbol"):
            return None
        fp = _fingerprint(rec)
        price = None
        core = ((result.get("decision") or {}).get("sections") or [])
        for i in ((result.get("evidence") or {}).get("items") or []):
            if i.get("key") == "research:composite":
                price = None
        conn = _conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO research_journal "
                "(fingerprint, symbol, intent, question, as_of, epoch, price, "
                " horizon_days, consensus, decision_state, depth, "
                " model_version, data_version, payload) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fp, rec["symbol"], rec.get("intent"), rec.get("question"),
                 time.strftime("%Y-%m-%dT%H:%M:%S"), time.time(), price,
                 (rec.get("horizon") or {}).get("days"),
                 (rec.get("synthesis") or {}).get("consensus"),
                 (rec.get("decision_state") or {}).get("status"),
                 (result.get("escalation") or {}).get("to"),
                 rec["model_version"], rec["data_version"],
                 json.dumps(rec, default=str)))
            conn.commit()
            return fp
        finally:
            conn.close()
    except Exception:
        return None


def history(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        conn = _conn()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT id, fingerprint, symbol, intent, question, as_of, "
                "consensus, decision_state, depth, model_version "
                "FROM research_journal WHERE symbol=? "
                "ORDER BY epoch DESC LIMIT ?", (symbol.upper(), int(limit)))]
        finally:
            conn.close()
    except Exception:
        return []


def get(fingerprint: str) -> Optional[Dict[str, Any]]:
    try:
        conn = _conn()
        try:
            row = conn.execute(
                "SELECT payload FROM research_journal WHERE fingerprint=?",
                (fingerprint,)).fetchone()
            return json.loads(row["payload"]) if row else None
        finally:
            conn.close()
    except Exception:
        return None
