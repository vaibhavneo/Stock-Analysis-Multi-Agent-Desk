"""
Decision journal (Phase 22) — an append-only record of what was decided, and why.

The prediction ledger already freezes the RECOMMENDATION (action verb, pillar
scores, gates). It does not record the decision: the state, the entry
conditions, the add/reduce/exit conditions, the invalidation. Those are exactly
what needs grading later — "was HOLD the right call?" is answerable, "did the
composite predict returns?" is a different and narrower question.

Design follows the prediction ledger's own, because that design has already
proven itself here:

  - **Immutable.** BEFORE UPDATE and BEFORE DELETE triggers in the database
    itself, so no code path — present or future — can rewrite a past decision.
    When the model changes, new rows are written; old rows keep saying what the
    old model said. A journal you can edit is a story, not evidence.
  - **Content-addressed.** The row id is a hash of the frozen payload, so
    re-journaling an identical decision is a no-op rather than a duplicate.
  - **Same database, separate table.** It joins to `prediction_snapshots` by
    ticker and date for analysis, and shares the ledger's role/quarantine
    discipline so test rows never become evidence.
  - **Failure is silent to the caller.** Journaling must never break a decision.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_journal (
    decision_id          TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    ticker               TEXT NOT NULL,
    market_price         REAL,
    data_asof            TEXT,

    headline_state       TEXT NOT NULL,
    state_if_owned       TEXT,
    state_if_not_owned   TEXT,
    ownership            TEXT,

    thesis_direction     TEXT,
    thesis_strength      TEXT,
    thesis_net_weight    REAL,

    edge_level           TEXT,
    edge_demonstrated    INTEGER,
    entry_status         TEXT,
    decision_confidence  TEXT,
    position_size_pct    REAL,
    sizing_gated         INTEGER,

    horizon_days         INTEGER,
    review_date          TEXT,

    entry_conditions_json      TEXT,
    add_conditions_json        TEXT,
    reduce_conditions_json     TEXT,
    exit_conditions_json       TEXT,
    invalidation_json          TEXT,
    confirmation_json          TEXT,
    catalysts_json             TEXT,
    levels_json                TEXT,
    evidence_snapshot_json     TEXT,
    quality_json               TEXT,
    consistency_json           TEXT,

    model_version        TEXT,
    data_version         TEXT,
    calibration_state    TEXT,
    recommendation_fingerprint TEXT,
    decision_fingerprint TEXT,
    frozen_json          TEXT NOT NULL,
    content_hash         TEXT NOT NULL
);

-- IMMUTABILITY: a journaled decision can never be edited or deleted. This is
-- what makes "what did the system say at the time" answerable rather than
-- reconstructed.
CREATE TRIGGER IF NOT EXISTS decision_journal_no_update
BEFORE UPDATE ON decision_journal
BEGIN SELECT RAISE(ABORT, 'decision journal entries are immutable (no UPDATE)'); END;

CREATE TRIGGER IF NOT EXISTS decision_journal_no_delete
BEFORE DELETE ON decision_journal
BEGIN SELECT RAISE(ABORT, 'decision journal entries are immutable (no DELETE)'); END;

CREATE INDEX IF NOT EXISTS idx_decision_journal_ticker
    ON decision_journal(ticker, created_at);

-- Forward outcomes for a journaled decision, one row per horizon. Separate
-- table so the decision row stays immutable while outcomes accumulate.
CREATE TABLE IF NOT EXISTS decision_outcomes (
    decision_id        TEXT NOT NULL,
    horizon_days       INTEGER NOT NULL,
    evaluated_at       TEXT NOT NULL,
    as_of_date         TEXT,
    matured            INTEGER NOT NULL DEFAULT 0,
    price_at_horizon   REAL,
    raw_return_pct     REAL,
    benchmark_return_pct REAL,
    excess_return_pct  REAL,
    mae_pct            REAL,
    mfe_pct            REAL,
    hit_invalidation   INTEGER,
    days_to_invalidation INTEGER,
    hit_confirmation   INTEGER,
    days_to_confirmation INTEGER,
    state_was_correct  INTEGER,
    PRIMARY KEY (decision_id, horizon_days),
    FOREIGN KEY (decision_id) REFERENCES decision_journal(decision_id)
);
"""

_db_override: Optional[str] = None


def set_db_path(path: Optional[Any]) -> None:
    """Point the journal at a specific database — used by tests so a suite can
    never write into the canonical evidence ledger."""
    global _db_override
    _db_override = str(path) if path else None


def _db() -> str:
    if _db_override:
        return _db_override
    from data import prediction_ledger as pl
    return pl._db()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _conditions(playbook: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    return [{"condition": c.get("condition"), "measurable_as": c.get("measurable_as"),
             "basis": c.get("basis"), "deterministic": c.get("deterministic")}
            for c in (playbook.get(key) or [])]


def journal_decision(decision: Dict[str, Any],
                     model_version: Optional[str] = None) -> Optional[str]:
    """Freeze one decision. Returns the decision_id, or None on any failure.

    Idempotent: the id is a content hash including `created_at`, so re-journaling
    the identical object returns the existing id and writes nothing new.
    """
    try:
        if decision.get("status") != "OK":
            return None

        ds = decision.get("decision_state") or {}
        thesis = decision.get("thesis") or {}
        edge = decision.get("statistical_edge") or {}
        playbook = decision.get("playbook") or {}
        mc = decision.get("mind_changers") or {}
        lm = decision.get("level_map") or {}

        frozen = {
            "ticker": decision["ticker"],
            "created_at": decision.get("generated_at") or datetime.now().isoformat(timespec="seconds"),
            "market_price": decision.get("current_price"),
            "data_asof": decision.get("data_asof"),
            "headline_state": ds.get("headline_state"),
            "state_if_owned": (ds.get("if_owned") or {}).get("state"),
            "state_if_not_owned": (ds.get("if_not_owned") or {}).get("state"),
            "ownership": ds.get("ownership"),
            "thesis": {"direction": thesis.get("direction"),
                       "strength": thesis.get("strength"),
                       "net_weight": thesis.get("net_weight"),
                       "statement": thesis.get("statement"),
                       "supporting": thesis.get("supporting_evidence"),
                       "opposing": thesis.get("opposing_evidence")},
            "edge": {"level": edge.get("level"), "demonstrated": edge.get("demonstrated"),
                     "verdict": edge.get("verdict"),
                     "gates": {g["gate"]: g["passed"] for g in (edge.get("gates") or [])}},
            "entry_status": (decision.get("entry") or {}).get("status"),
            "entry_conditions": (decision.get("entry") or {}).get("conditions"),
            "entry_plan": {k: (decision.get("entry_plan") or {}).get(k)
                           for k in ("supported", "entry_low", "entry_high", "status")},
            "add_conditions": _conditions(playbook, "add_conditions"),
            "reduce_conditions": _conditions(playbook, "reduce_conditions"),
            "exit_conditions": _conditions(playbook, "exit_conditions"),
            "hold_conditions": _conditions(playbook, "hold_conditions"),
            "invalidation": mc.get("invalidation"),
            "confirmation": mc.get("confirmation"),
            "catalysts": [{"date": e.get("date"), "event": e.get("event"),
                           "relevance": e.get("relevance")}
                          for e in ((decision.get("catalysts") or {}).get("upcoming") or [])],
            "levels": {"ladder": [{"price": l.get("price"), "kind": l.get("kind"),
                                   "basis": l.get("basis"), "sources": l.get("sources")}
                                  for l in (lm.get("ladder") or [])],
                       "nearest_support": lm.get("nearest_support"),
                       "nearest_resistance": lm.get("nearest_resistance")},
            "evidence_snapshot": [{"source": e.get("source"), "direction": e.get("direction"),
                                   "horizon": e.get("horizon"), "weight": e.get("weight"),
                                   "validation_status": e.get("validation_status"),
                                   "raw_value": e.get("raw_value")}
                                  for e in (decision.get("evidence") or [])],
            "confidence": decision.get("confidence", {}).get("decision_confidence"),
            "confidence_dimensions": {k: v.get("level") for k, v in
                                      (decision.get("confidence", {}).get("dimensions") or {}).items()},
            "quality": (decision.get("quality") or {}).get("components"),
            "consistency": {"n_errors": (decision.get("consistency") or {}).get("n_errors"),
                            "n_warnings": (decision.get("consistency") or {}).get("n_warnings"),
                            "codes": [i["code"] for i in
                                      (decision.get("consistency") or {}).get("issues", [])]},
            "sizing": decision.get("sizing"),
            "horizon_days": decision.get("horizon_days"),
            "review_date": playbook.get("review_date"),
            "model_version": model_version or "decision_intelligence_v1",
            "data_version": (decision.get("_recommendation") or {}).get("decision_fingerprint"),
            "calibration_state": (edge.get("calibration") or {}).get("status"),
            "decision_fingerprint": decision.get("decision_fingerprint"),
        }

        did = _hash(frozen)
        content_hash = _hash({k: v for k, v in frozen.items() if k != "created_at"})

        conn = _conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO decision_journal
                   (decision_id, created_at, ticker, market_price, data_asof,
                    headline_state, state_if_owned, state_if_not_owned, ownership,
                    thesis_direction, thesis_strength, thesis_net_weight,
                    edge_level, edge_demonstrated, entry_status, decision_confidence,
                    position_size_pct, sizing_gated, horizon_days, review_date,
                    entry_conditions_json, add_conditions_json, reduce_conditions_json,
                    exit_conditions_json, invalidation_json, confirmation_json,
                    catalysts_json, levels_json, evidence_snapshot_json, quality_json,
                    consistency_json, model_version, data_version, calibration_state,
                    recommendation_fingerprint, decision_fingerprint, frozen_json, content_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (did, frozen["created_at"], frozen["ticker"], frozen["market_price"],
                 frozen["data_asof"], frozen["headline_state"], frozen["state_if_owned"],
                 frozen["state_if_not_owned"], frozen["ownership"],
                 frozen["thesis"]["direction"], frozen["thesis"]["strength"],
                 frozen["thesis"]["net_weight"], frozen["edge"]["level"],
                 1 if frozen["edge"]["demonstrated"] else 0, frozen["entry_status"],
                 frozen["confidence"], (frozen["sizing"] or {}).get("position_size_pct"),
                 1 if (frozen["sizing"] or {}).get("gated") else 0,
                 frozen["horizon_days"], frozen["review_date"],
                 json.dumps(frozen["entry_conditions"], default=str),
                 json.dumps(frozen["add_conditions"], default=str),
                 json.dumps(frozen["reduce_conditions"], default=str),
                 json.dumps(frozen["exit_conditions"], default=str),
                 json.dumps(frozen["invalidation"], default=str),
                 json.dumps(frozen["confirmation"], default=str),
                 json.dumps(frozen["catalysts"], default=str),
                 json.dumps(frozen["levels"], default=str),
                 json.dumps(frozen["evidence_snapshot"], default=str),
                 json.dumps(frozen["quality"], default=str),
                 json.dumps(frozen["consistency"], default=str),
                 frozen["model_version"], frozen["data_version"],
                 frozen["calibration_state"],
                 (decision.get("_recommendation") or {}).get("decision_fingerprint"),
                 frozen["decision_fingerprint"],
                 json.dumps(frozen, default=str), content_hash))
            conn.commit()
        finally:
            conn.close()

        # Same quarantine discipline as the prediction ledger: a synthetic
        # ticker or a non-canonical deployment records what it did, but its rows
        # must never count as evidence.
        try:
            from data import prediction_ledger as pl
            if pl.is_synthetic_ticker(frozen["ticker"]) or not pl.is_canonical_ledger():
                pl.quarantine_snapshot(
                    did,
                    "synthetic_ticker" if pl.is_synthetic_ticker(frozen["ticker"])
                    else f"non_canonical_origin:{pl.ledger_role()}",
                    frozen["ticker"])
        except Exception:
            pass

        return did
    except Exception:
        return None


def get_decision(decision_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM decision_journal WHERE decision_id=?",
                           (decision_id,)).fetchone()
        return json.loads(row["frozen_json"]) if row else None
    finally:
        conn.close()


def latest_decision(ticker: str) -> Optional[Dict[str, Any]]:
    """The most recent journaled decision for a ticker — the input the add
    analysis needs to answer "has the thesis strengthened since last time?"."""
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM decision_journal WHERE ticker=? ORDER BY created_at DESC LIMIT 1",
            (ticker.upper(),)).fetchone()
        return json.loads(row["frozen_json"]) if row else None
    finally:
        conn.close()


def list_decisions(ticker: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM decision_journal WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decision_journal ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def summary() -> Dict[str, Any]:
    conn = _conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM decision_journal").fetchone()[0]
        by_state = dict(conn.execute(
            "SELECT headline_state, COUNT(*) FROM decision_journal GROUP BY headline_state"
        ).fetchall())
        graded = conn.execute(
            "SELECT COUNT(*) FROM decision_outcomes WHERE matured=1").fetchone()[0]
        tickers = conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM decision_journal").fetchone()[0]
        return {"total_decisions": total, "by_state": by_state,
                "matured_outcomes": graded, "distinct_tickers": tickers}
    finally:
        conn.close()
