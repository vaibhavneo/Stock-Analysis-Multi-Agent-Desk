"""
EvidenceLedger + TrialRegistry.

Two append-only records that together answer the two questions a trading desk
must never guess at:

  EvidenceLedger — "where did this number come from?"
      Every numeric claim (a P/E, a net margin, a target price) links to the
      exact Datums it was computed from and the formula that combined them.
      record_claim() REFUSES to store a claim with no evidence: "no unsupported
      numeric claims" is enforced by the schema, not by author discipline. An
      LLM cannot talk its way past a NOT NULL constraint.

  TrialRegistry — "how many things did we try before this one looked good?"
      Every hypothesis ever backtested gets a permanent row, alive or dead.
      Deflated Sharpe needs the TRUE trial count; taking it from a per-run
      constant (the old n_trials=7) understates it by orders of magnitude and
      turns dSR into a comfortable lie. Dead ideas are the point — they are the
      denominator. Nothing here is ever deleted (P7).

Lives in the same SQLite file as recommendations/outcomes (additive: those
tables have a live coach reader and are untouched — see memory/log.md C25.1).
The DB is rebuildable and gitignored; the files and providers are the truth (P2).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import DB_PATH

_SCHEMA = """
PRAGMA journal_mode=WAL;

-- Datums: the evidence. Content-addressed, so re-fetching the same fact does
-- not duplicate it and old claims keep pointing at stable ids.
CREATE TABLE IF NOT EXISTS data_points (
    datum_id     TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    symbol       TEXT,
    concept      TEXT,
    value        REAL,
    value_json   TEXT,
    unit         TEXT,
    period_start TEXT,
    period_end   TEXT,
    available_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    provider     TEXT NOT NULL,
    document     TEXT,
    source_ref   TEXT,
    source_url   TEXT,
    confidence   REAL,
    status       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dp_symbol_concept ON data_points(symbol, concept);

-- Claims: the numbers a report shows. as_of_honored travels WITH the claim so a
-- reader can never see the number without seeing whether it was PIT-clean.
CREATE TABLE IF NOT EXISTS claims (
    claim_id      TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    symbol        TEXT,
    name          TEXT NOT NULL,
    value         REAL,
    unit          TEXT,
    formula       TEXT NOT NULL,
    as_of         TEXT,
    as_of_honored INTEGER NOT NULL,
    run_id        TEXT,
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_claims_run ON claims(run_id);

-- The link table. A claim with zero rows here is a claim with no evidence,
-- which record_claim() refuses to create.
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL,
    datum_id TEXT NOT NULL,
    role     TEXT NOT NULL,
    PRIMARY KEY (claim_id, datum_id, role),
    FOREIGN KEY (claim_id) REFERENCES claims(claim_id),
    FOREIGN KEY (datum_id) REFERENCES data_points(datum_id)
);

-- TrialRegistry: append-only count of every hypothesis ever tested.
CREATE TABLE IF NOT EXISTS trials (
    trial_id     TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    hypothesis   TEXT NOT NULL,
    family       TEXT,
    strategy     TEXT,
    params_json  TEXT NOT NULL,
    universe     TEXT,
    period_start TEXT,
    period_end   TEXT,
    sharpe       REAL,
    outcome      TEXT NOT NULL,
    flags_json   TEXT,
    notes        TEXT
);
CREATE INDEX IF NOT EXISTS idx_trials_family ON trials(family);
"""

OUTCOMES = ("alive", "dead", "inconclusive")

# Tests point this at a temp file. Without it, test fixtures would accumulate in
# the real TrialRegistry and inflate n_trials forever — the exact pollution that
# already made the platform's own metrics.jsonl success rate meaningless
# (PLATFORM_IMPROVEMENT_REPORT M-Q1b). The trial count is the denominator of
# dSR's honesty; letting test rows into it would corrupt the one number this
# whole milestone exists to make trustworthy.
_db_override: Optional[Path] = None


def set_db_path(path: Optional[Any]) -> None:
    """Redirect the ledger to another SQLite file (tests only). None restores."""
    global _db_override
    _db_override = Path(path) if path else None


def _db() -> Path:
    return _db_override or DB_PATH


class LedgerError(ValueError):
    pass


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db()), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Evidence ───────────────────────────────────────────────────────────────

def record_datum(d: Dict[str, Any]) -> str:
    """Persist one Datum. Idempotent by content id — same fact, same row."""
    for f in ("datum_id", "kind", "available_at", "retrieved_at", "source", "status"):
        if not d.get(f):
            raise LedgerError(f"datum missing required field {f!r}")
    src = d["source"]
    val = d.get("value")
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO data_points
               (datum_id, kind, symbol, concept, value, value_json, unit,
                period_start, period_end, available_at, retrieved_at, provider,
                document, source_ref, source_url, confidence, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (d["datum_id"], d["kind"], d.get("symbol"), d.get("concept"),
             float(val) if isinstance(val, (int, float)) else None,
             json.dumps(val, default=str) if not isinstance(val, (int, float)) else None,
             d.get("unit"), d.get("period_start"), d.get("period_end"),
             d["available_at"], d["retrieved_at"], src.get("provider"),
             src.get("document"), src.get("ref"), src.get("url"),
             d.get("confidence"), d["status"]),
        )
        conn.commit()
        return d["datum_id"]
    finally:
        conn.close()


def record_claim(name: str, value: Optional[float], formula: str,
                 evidence: Dict[str, Dict[str, Any]], *,
                 symbol: Optional[str] = None, unit: Optional[str] = None,
                 as_of: Optional[str] = None, as_of_honored: bool = False,
                 run_id: Optional[str] = None, notes: Optional[str] = None) -> str:
    """Record a numeric claim and bind it to the evidence it came from.

    Args:
      formula:  human-readable, e.g. "net_income / revenue". Its variable names
                should match `evidence` keys so explain() reads as arithmetic.
      evidence: {role -> Datum}, e.g. {"net_income": <datum>, "revenue": <datum>}.
                Datums are auto-recorded; callers cannot cite a fact they never
                fetched.

    Raises LedgerError on empty evidence. That refusal IS the feature: it is
    what makes "no unsupported numeric claims" a property of the system rather
    than an instruction a model may ignore under pressure.
    """
    if not name or not formula:
        raise LedgerError("a claim requires both a name and a formula")
    if not evidence:
        raise LedgerError(
            f"claim {name!r} has no evidence — every numeric claim must cite the "
            f"datums it was computed from (pass evidence={{role: datum}})")

    for role, d in evidence.items():
        if not isinstance(d, dict) or "datum_id" not in d:
            raise LedgerError(f"evidence[{role!r}] is not a Datum (missing datum_id)")
        record_datum(d)

    # Deterministic id from content + evidence: recomputing the same claim from
    # the same facts is the same claim, which keeps replays idempotent.
    blob = json.dumps({"name": name, "symbol": symbol, "value": value,
                       "formula": formula, "as_of": as_of,
                       "ev": sorted((r, d["datum_id"]) for r, d in evidence.items())},
                      sort_keys=True, default=str)
    claim_id = hashlib.sha1(blob.encode()).hexdigest()[:16]

    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO claims
               (claim_id, created_at, symbol, name, value, unit, formula,
                as_of, as_of_honored, run_id, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (claim_id, _now(), symbol, name,
             float(value) if value is not None else None, unit, formula,
             as_of, int(bool(as_of_honored)), run_id, notes),
        )
        for role, d in evidence.items():
            conn.execute(
                "INSERT OR IGNORE INTO claim_evidence (claim_id, datum_id, role) VALUES (?,?,?)",
                (claim_id, d["datum_id"], role))
        conn.commit()
        return claim_id
    finally:
        conn.close()


def explain(claim_id: str) -> Dict[str, Any]:
    """Full audit trail for one claim: the number, the formula, and every datum
    behind it with its filing document — the end-to-end reproduction path a
    human uses to check the math by hand against the original filing."""
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
        if not row:
            return {"error": f"no claim {claim_id!r}"}
        ev = conn.execute(
            """SELECT ce.role, dp.* FROM claim_evidence ce
               JOIN data_points dp ON dp.datum_id = ce.datum_id
               WHERE ce.claim_id=? ORDER BY ce.role""", (claim_id,)).fetchall()
        claim = dict(row)
        claim["as_of_honored"] = bool(claim["as_of_honored"])
        claim["evidence"] = [dict(e) for e in ev]
        # Surfaced so a reader never has to infer freshness: the newest input is
        # the effective age of the claim.
        claim["evidence_available_at_max"] = max(
            (e["available_at"] for e in claim["evidence"]), default=None)
        return claim
    finally:
        conn.close()


def claims_for_run(run_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM claims WHERE run_id=? ORDER BY created_at", (run_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── Trials ─────────────────────────────────────────────────────────────────

def record_trial(hypothesis: str, params: Dict[str, Any], outcome: str, *,
                 family: Optional[str] = None, strategy: Optional[str] = None,
                 universe: Optional[str] = None, period_start: Optional[str] = None,
                 period_end: Optional[str] = None, sharpe: Optional[float] = None,
                 flags: Optional[Dict[str, Any]] = None,
                 notes: Optional[str] = None) -> str:
    """Register one tested hypothesis, forever.

    Call this for EVERY backtest — especially the failures. A registry of only
    the winners would inflate nothing and deflate nothing; it is the losers that
    make dSR honest.
    """
    if outcome not in OUTCOMES:
        raise LedgerError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    if not hypothesis:
        raise LedgerError("a trial requires a hypothesis statement")

    params_json = json.dumps(params, sort_keys=True, default=str)
    # Identity = the experiment, not the moment: re-running an identical test
    # must not inflate the trial count, or dSR would decay with every re-run.
    trial_id = hashlib.sha1(
        f"{family}|{strategy}|{hypothesis}|{params_json}|{universe}|"
        f"{period_start}|{period_end}".encode()).hexdigest()[:16]

    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO trials
               (trial_id, created_at, hypothesis, family, strategy, params_json,
                universe, period_start, period_end, sharpe, outcome, flags_json, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (trial_id, _now(), hypothesis, family, strategy, params_json, universe,
             period_start, period_end,
             float(sharpe) if sharpe is not None else None, outcome,
             json.dumps(flags or {}, sort_keys=True), notes),
        )
        conn.commit()
        return trial_id
    finally:
        conn.close()


def n_trials(family: Optional[str] = None) -> int:
    """The TRUE trial count for Deflated Sharpe.

    Pass `family` to count only comparable experiments (dSR's N is "variants
    tried on this idea", not "every test in history"); omit it for the total.
    """
    conn = _connect()
    try:
        if family:
            row = conn.execute("SELECT COUNT(*) AS n FROM trials WHERE family=?",
                               (family,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()
        return int(row["n"] or 0)
    finally:
        conn.close()


def strategy_family(ticker: str) -> str:
    """The dSR selection problem is "which strategy for THIS ticker?", so trials
    are counted per ticker — matching what deflated_sharpe_ratio() documents as
    its n. Trials on AAPL do not deflate a conclusion about MSFT."""
    return f"strategy_selection:{ticker.upper().strip()}"


def effective_n_trials(ticker: str, tested_this_run: int = 0) -> int:
    """The n_trials to hand deflated_sharpe_ratio(). THE fix for the
    strategy-shopping loophole.

    Before this, n came from the current process: run all 7 strategies and you
    got n=7; note the winner, re-run it alone, and you got n=1 — no deflation,
    dSR=1.0. Same data, same strategy, better score, purely from re-running it
    by itself. At the Sharpe range real strategies actually live in (0.8-1.2)
    that flips the verdict from "fails the 0.5 bar" to "certain". The registry
    closes it by remembering: the second run still knows about the first seven.

    max() of three terms, each load-bearing:
      registered      — every variant ever tried on this ticker, across sessions
      tested_this_run — trials from this run that may not be recorded yet, so a
                        caller can never under-count by forgetting to record
      1               — dSR's floor; 0 would disable deflation (and now raises)

    Monotone by construction: it can only ever rise as more is tried, which is
    the direction honesty runs in.
    """
    return max(n_trials(family=strategy_family(ticker)), int(tested_this_run), 1)


def list_trials(family: Optional[str] = None, outcome: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        where, params = [], []
        if family:
            where.append("family=?"); params.append(family)
        if outcome:
            where.append("outcome=?"); params.append(outcome)
        sql = "SELECT * FROM trials"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def trial_stats() -> Dict[str, Any]:
    """Survival summary. A healthy research process kills most of what it tries;
    a registry that is 90% `alive` is evidence of a rubber stamp, not of skill."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT outcome, COUNT(*) AS n FROM trials GROUP BY outcome").fetchall()
        by = {r["outcome"]: r["n"] for r in rows}
        total = sum(by.values())
        return {
            "total": total,
            "by_outcome": by,
            "kill_rate": round(by.get("dead", 0) / total, 3) if total else None,
        }
    finally:
        conn.close()
