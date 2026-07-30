"""
Cross-sectional ranking engine + immutable Ranking Ledger.

Ties the pieces together for one `as_of` date:
  universe.members(as_of)  (survivorship-safe, PIT)
    -> features.compute_features per security (point-in-time)
    -> factors.score_cross_section (pre-registered weights)
    -> ranks + screens
    -> freeze into the immutable Ranking Ledger (content-hash, DB triggers).

Determinism: no randomness, no LLM in the numeric path. Identical (universe,
as_of, config) => identical decision fingerprint. Re-running is idempotent
(content-addressed run id, INSERT OR IGNORE). Historical ranks are never
rewritten (BEFORE UPDATE/DELETE triggers ABORT).

Stored in the EXISTING recommendations.db — no new database.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from backtest import experiments
from data import store
from xsection import factors as fac
from xsection import features as ft
from xsection.universe import UNIVERSE_INCOMPLETE, UniverseIncomplete, get_provider

_db_override: Optional[str] = None


def set_db_path(path: Optional[Any]) -> None:
    global _db_override
    _db_override = str(path) if path else None


def _db() -> str:
    return _db_override or str(store.DB_PATH)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ranking_runs (
    ranking_run_id   TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    as_of            TEXT NOT NULL,
    universe_id      TEXT NOT NULL,
    survivorship_safe INTEGER NOT NULL,
    config_version   TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    feature_version  TEXT NOT NULL,
    weights_json     TEXT NOT NULL,
    n_members        INTEGER,
    n_ranked         INTEGER,
    n_excluded       INTEGER,
    decision_fingerprint TEXT,
    universe_snapshot_hash TEXT,
    content_hash     TEXT NOT NULL,
    result_json      TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS ranking_runs_no_update
BEFORE UPDATE ON ranking_runs
BEGIN SELECT RAISE(ABORT, 'ranking runs are immutable (no UPDATE)'); END;
CREATE TRIGGER IF NOT EXISTS ranking_runs_no_delete
BEFORE DELETE ON ranking_runs
BEGIN SELECT RAISE(ABORT, 'ranking runs are immutable (no DELETE)'); END;
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db(), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _hash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def run_ranking(as_of: str, universe_id: str = "reference-smallcap-demo",
                config_version: str = "xsec-v1", persist: bool = True,
                provider: Optional["UniverseProvider"] = None) -> Dict[str, Any]:
    """Produce a full survivorship-safe, point-in-time ranking for `as_of`.
    Returns the ranking result (and freezes it immutably when persist=True).
    Pass `provider` directly to reuse one instance across multiple dates."""
    cfg = experiments.ranking_config(config_version)
    cfg_hash = experiments.ranking_config_hash(config_version)

    if provider is None:
        provider = get_provider(universe_id)
    try:
        members = provider.members(as_of)
        universe_incomplete = None
    except UniverseIncomplete as e:
        return {"status": UNIVERSE_INCOMPLETE, "reason": str(e), "as_of": as_of,
                "universe_id": universe_id, "survivorship_safe": getattr(provider, "survivorship_safe", False)}

    sm = provider.security_master()
    bench_id = provider.benchmark_id()
    cov = provider.coverage()
    bench_px = provider.prices(bench_id, cov["start"], as_of) if bench_id else None
    fund_fn = provider.fundamentals_fn() if hasattr(provider, "fundamentals_fn") else None

    # Point-in-time features per constituent.
    feature_rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for m in members:
        raw_sec = sm.get(m["security_id"]) or {}
        px = provider.prices(m["security_id"], cov["start"], as_of)
        row = ft.compute_features(m, as_of, px, bench_px, raw_sec, fundamentals_fn=fund_fn)
        if row["status"] in (ft.INSUFFICIENT_HISTORY, ft.EXCLUDED) or row["coverage"] < cfg["min_coverage"]:
            excluded.append({"security_id": row["security_id"], "ticker_as_of": m.get("ticker_as_of"),
                             "status": row["status"], "coverage": row["coverage"],
                             "reason": row.get("exclusion_reason") or f"coverage<{cfg['min_coverage']}"})
        else:
            feature_rows.append(row)

    if not feature_rows:
        return {"status": "NO_ELIGIBLE_SECURITIES", "as_of": as_of, "universe_id": universe_id,
                "excluded": excluded, "n_members": len(members)}

    # Factor + composite scoring (deterministic, pre-registered weights).
    scored = fac.score_cross_section(feature_rows, cfg)
    rows = scored["rows"]

    # Assemble ranked table with sector/industry ranks.
    ranked = []
    for i, r in enumerate(feature_rows):
        s = rows[i]
        ranked.append({
            "security_id": r["security_id"], "ticker_as_of": r["ticker_as_of"],
            "name": r.get("name"), "sector": r.get("sector"), "industry": r.get("industry"),
            "listing_status": r.get("listing_status"), "status": r["status"],
            "coverage": r["coverage"], "data_confidence": r["confidence"],
            "factor_scores": s["factor_scores"], "risk_score": s["risk_score"],
            "alpha_score": s["alpha_score"], "risk_penalty_applied": s["risk_penalty_applied"],
            "composite_raw": s["composite_raw"], "composite_percentile": s["composite_percentile"],
            "risk_veto": s["risk_veto"], "factors_present": s["factors_present"],
            "flags": r["flags"], "evidence_ids": sorted({f["evidence_id"] for f in r["features"]
                                                          if f.get("evidence_id")}),
            "features": r["features"],
        })
    ranked.sort(key=lambda x: (x["composite_raw"] if x["composite_raw"] is not None else -9), reverse=True)
    for rk, x in enumerate(ranked, 1):
        x["rank"] = rk
    # sector/industry ranks
    for key in ("sector", "industry"):
        groups: Dict[str, List[Dict]] = {}
        for x in ranked:
            groups.setdefault(x.get(key) or "unknown", []).append(x)
        for _, g in groups.items():
            g.sort(key=lambda x: x["composite_raw"], reverse=True)
            for rk, x in enumerate(g, 1):
                x[f"rank_in_{key}"] = rk

    n = len(ranked)
    decile = max(1, n // 10)
    result = {
        "status": "OK", "as_of": as_of, "universe_id": universe_id,
        "survivorship_safe": provider.survivorship_safe,
        "config_version": config_version, "config_hash": cfg_hash,
        "feature_version": ft.FEATURE_VERSION, "weights": cfg["weights"],
        "risk_penalty": cfg["risk_penalty"], "coverage": cov,
        "n_members": len(members), "n_ranked": n, "n_excluded": len(excluded),
        "top": ranked[:decile], "bottom": ranked[-decile:], "ranked": ranked,
        "excluded": excluded, "factors_present": scored["factors_present"],
        "screens": _screens(ranked),
        "labels": _labels(provider, scored),
        "universe_snapshot_hash": _hash([m["security_id"] for m in members] + [as_of]),
        "experiment_manifest_hash": experiments.manifest_hash(),
    }
    # Decision fingerprint: the deterministic ranking (excludes narrative).
    result["decision_fingerprint"] = _hash({
        "as_of": as_of, "universe": universe_id, "cfg": cfg_hash,
        "ranks": [(x["security_id"], x["rank"], x["composite_raw"]) for x in ranked],
        "excluded": sorted(e["security_id"] for e in excluded)})

    if persist:
        result["ranking_run_id"] = _freeze(result)
    return result


def _labels(provider, scored) -> Dict[str, Any]:
    """The honesty labels the dashboard MUST show."""
    return {
        "PIT_SAFE": True,
        "SURVIVORSHIP_SAFE": bool(getattr(provider, "survivorship_safe", False)),
        "CURRENT_CONSTITUENTS_ONLY": not bool(getattr(provider, "survivorship_safe", False)),
        "DELISTING_HANDLING": "included; conservative delisting return (DELISTING_POLICY.md)",
        "ESTIMATE_HISTORY": "UNAVAILABLE (no PIT estimate history in free data)",
        "DATA_SOURCE": "reference_fixture (synthetic) — production needs paid constituent data",
    }


def _screens(ranked: List[Dict]) -> Dict[str, List[str]]:
    """Named screens (mission §7). Values are ticker_as_of lists (top few)."""
    def by(keyfn, reverse=True, k=5, filt=None):
        pool = [x for x in ranked if (filt is None or filt(x))]
        pool = sorted(pool, key=keyfn, reverse=reverse)
        return [x["ticker_as_of"] for x in pool[:k]]
    fs = lambda x, f: (x["factor_scores"].get(f) or 0)
    return {
        "top_opportunities": [x["ticker_as_of"] for x in ranked[:5]],
        "strongest_momentum": by(lambda x: fs(x, "momentum")),
        "best_quality": by(lambda x: fs(x, "quality")),
        "best_value": by(lambda x: fs(x, "valuation")),
        "best_value_rel_growth": by(lambda x: fs(x, "valuation") + fs(x, "growth")),
        "largest_deterioration": [x["ticker_as_of"] for x in ranked[-5:]],
        "highest_risk": by(lambda x: x["risk_score"]),
        "low_confidence": by(lambda x: -x["data_confidence"], reverse=True),
    }


def _freeze(result: Dict[str, Any]) -> Optional[str]:
    frozen = {k: result[k] for k in ("as_of", "universe_id", "config_hash",
                                     "decision_fingerprint", "universe_snapshot_hash",
                                     "feature_version", "weights")}
    content_hash = _hash(frozen)
    run_id = _hash({**frozen, "created": datetime.now().isoformat(timespec="seconds")})
    try:
        conn = _conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO ranking_runs
                   (ranking_run_id, created_at, as_of, universe_id, survivorship_safe,
                    config_version, config_hash, feature_version, weights_json, n_members,
                    n_ranked, n_excluded, decision_fingerprint, universe_snapshot_hash,
                    content_hash, result_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, datetime.now().isoformat(timespec="seconds"), result["as_of"],
                 result["universe_id"], int(result["survivorship_safe"]), result["config_version"],
                 result["config_hash"], result["feature_version"], json.dumps(result["weights"]),
                 result["n_members"], result["n_ranked"], result["n_excluded"],
                 result["decision_fingerprint"], result["universe_snapshot_hash"], content_hash,
                 json.dumps(result, default=str)))
            conn.commit()
            return run_id
        finally:
            conn.close()
    except Exception:
        return None


def get_ranking(ranking_run_id: str) -> Optional[Dict[str, Any]]:
    conn = _conn()
    try:
        row = conn.execute("SELECT result_json FROM ranking_runs WHERE ranking_run_id=?",
                          (ranking_run_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None
    finally:
        conn.close()


def list_rankings(limit: int = 50, universe_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        if universe_id:
            rows = conn.execute(
                "SELECT ranking_run_id, created_at, as_of, universe_id, n_ranked, n_excluded, "
                "survivorship_safe, config_version, decision_fingerprint FROM ranking_runs "
                "WHERE universe_id=? ORDER BY created_at DESC LIMIT ?", (universe_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT ranking_run_id, created_at, as_of, universe_id, n_ranked, n_excluded, "
                "survivorship_safe, config_version, decision_fingerprint FROM ranking_runs "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def latest_ranking(universe_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    rows = list_rankings(limit=1, universe_id=universe_id)
    return get_ranking(rows[0]["ranking_run_id"]) if rows else None
