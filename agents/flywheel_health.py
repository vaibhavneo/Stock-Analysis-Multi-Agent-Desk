"""
Daily flywheel health report — monitoring, NOT intelligence.

This module answers one question: is the evidence flywheel accumulating clean,
trustworthy data? It reads the ledger and reports. It does not score, rank,
recommend, calibrate, or influence any decision, and nothing downstream
consumes its output. Keep it that way — the moment a health metric feeds back
into the strategy, it stops being a measurement and becomes a tuning knob.

The distinction that matters most here is EVIDENCE vs ROWS. A row in
prediction_snapshots is not automatically evidence: synthetic tickers and
snapshots frozen by a secondary (non-canonical) deployment are quarantined and
must be excluded. Every integrity percentage below is computed over evidence
only, because that is the population calibration will actually read.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from data import prediction_ledger as pl
from data.prediction_ledger import HORIZONS

CORE_PILLARS = ("technical", "algo", "fundamentals")


def _evidence_rows(conn, day: Optional[str]) -> List[Dict[str, Any]]:
    where = "WHERE s.snapshot_id NOT IN (SELECT snapshot_id FROM snapshot_quarantine)"
    args: tuple = ()
    if day:
        where += " AND substr(s.created_at,1,10)=?"
        args = (day,)
    return [dict(r) for r in conn.execute(
        f"""SELECT s.snapshot_id, s.ticker, s.price_at_call, s.action,
                   s.pillars_json, s.horizon_probabilities_json, s.created_at
              FROM prediction_snapshots s {where}""", args).fetchall()]


def _pct(n: int, d: int) -> Optional[float]:
    return round(100.0 * n / d, 1) if d else None


def health_report(day: Optional[str] = None) -> Dict[str, Any]:
    """Ledger health for `day` (default today), plus cumulative maturation.

    Never raises: a monitoring report that crashes tells you nothing on the day
    you most need it.
    """
    day = day or datetime.now().strftime("%Y-%m-%d")
    out: Dict[str, Any] = {"date": day, "ledger_role": "unknown"}
    try:
        out["ledger_role"] = pl.ledger_role()
        conn = pl._conn()
    except Exception as e:
        out["error"] = f"ledger_unavailable: {e}"
        return out

    try:
        q = lambda s, *a: conn.execute(s, a).fetchone()[0]
        rows = _evidence_rows(conn, day)
        n = len(rows)

        valid_price = sum(
            1 for r in rows
            if isinstance(r["price_at_call"], (int, float))
            and r["price_at_call"] is not None
            and r["price_at_call"] > 0
            and r["price_at_call"] == r["price_at_call"])          # NaN check
        complete_pillars = 0
        for r in rows:
            try:
                pj = json.loads(r["pillars_json"] or "{}")
            except Exception:
                pj = {}
            if all(pj.get(k) is not None for k in CORE_PILLARS):
                complete_pillars += 1
        with_probs = sum(1 for r in rows if r["horizon_probabilities_json"])
        with_action = sum(1 for r in rows if r["action"])

        distinct = len({r["ticker"] for r in rows})
        dup_rows = n - distinct        # >1 evidence row for a ticker on one day

        out["today"] = {
            "evidence_rows": n,
            "distinct_tickers": distinct,
            "duplicate_rows": dup_rows,
            "quarantined_rows": q(
                """SELECT COUNT(*) FROM prediction_snapshots s
                    JOIN snapshot_quarantine qn ON qn.snapshot_id = s.snapshot_id
                   WHERE substr(s.created_at,1,10)=?""", day),
            "action_distribution": {},
        }
        actions: Dict[str, int] = {}
        for r in rows:
            actions[r["action"] or "(none)"] = actions.get(r["action"] or "(none)", 0) + 1
        out["today"]["action_distribution"] = dict(sorted(actions.items(), key=lambda kv: -kv[1]))

        out["integrity"] = {
            "prices_valid_pct": _pct(valid_price, n),
            "pillars_complete_pct": _pct(complete_pillars, n),
            "probabilities_present_pct": _pct(with_probs, n),
            "actions_present_pct": _pct(with_action, n),
            "duplicate_rate_pct": _pct(dup_rows, n),
        }

        out["cumulative"] = {
            "evidence_rows": q(
                """SELECT COUNT(*) FROM prediction_snapshots s
                    WHERE s.snapshot_id NOT IN (SELECT snapshot_id FROM snapshot_quarantine)"""),
            "quarantined_rows": q("SELECT COUNT(*) FROM snapshot_quarantine"),
            "total_rows": q("SELECT COUNT(*) FROM prediction_snapshots"),
        }

        matured = {}
        for h in HORIZONS:
            matured[int(h)] = q(
                f"""SELECT COUNT(*) FROM prediction_outcomes o
                      JOIN prediction_snapshots s ON s.snapshot_id = o.snapshot_id
                     WHERE o.horizon_days=? AND o.matured=1
                       AND o.direction_correct IS NOT NULL
                       {pl._source_where('all', 's')}""", h)
        out["matured_outcomes"] = matured
    except Exception as e:
        out["error"] = f"report_failed: {e}"
        return out
    finally:
        conn.close()

    # Calibration readiness. Reports the gate; never changes it.
    try:
        from intelligence.calibration import (MIN_EFFECTIVE_OBS, _pairs_for_horizon,
                                              effective_sample_size)
        cal = {}
        for h in HORIZONS:
            pairs = _pairs_for_horizon(int(h))
            eff = effective_sample_size([d for _, _, d in pairs], int(h)) if pairs else 0
            cal[int(h)] = {"rows": len(pairs), "effective_n": eff,
                           "needed": MIN_EFFECTIVE_OBS,
                           "sufficient": eff >= MIN_EFFECTIVE_OBS,
                           "shortfall": max(0, MIN_EFFECTIVE_OBS - eff)}
        out["calibration_readiness"] = cal
    except Exception as e:
        out["calibration_readiness"] = {"error": str(e)[:120]}

    return out


def format_report(rep: Dict[str, Any]) -> str:
    """Plain-text rendering, for a log file or a terminal."""
    if rep.get("error"):
        return f"STOCK AGENT — DAILY FLYWHEEL HEALTH\n\nDate: {rep['date']}\n  ERROR: {rep['error']}\n"

    t, i, c = rep["today"], rep["integrity"], rep["cumulative"]
    L = ["STOCK AGENT — DAILY FLYWHEEL HEALTH", "",
         f"Date: {rep['date']}    ledger role: {rep['ledger_role']}", ""]

    L.append(f"  Evidence rows today:   {t['evidence_rows']:>6}")
    L.append(f"  Distinct tickers:      {t['distinct_tickers']:>6}")
    L.append(f"  Duplicate rows:        {t['duplicate_rows']:>6}")
    L.append(f"  Quarantined today:     {t['quarantined_rows']:>6}")
    L.append(f"  Evidence rows total:   {c['evidence_rows']:>6}"
             f"   (of {c['total_rows']} rows; {c['quarantined_rows']} quarantined)")
    L.append("")

    def pc(v):
        return "   n/a" if v is None else f"{v:>5.1f}%"

    L.append("Data integrity (evidence only):")
    L.append(f"  Prices valid:        {pc(i['prices_valid_pct'])}")
    L.append(f"  Pillars complete:    {pc(i['pillars_complete_pct'])}")
    L.append(f"  Probabilities:       {pc(i['probabilities_present_pct'])}")
    L.append(f"  Actions present:     {pc(i['actions_present_pct'])}")
    L.append(f"  Duplicate rate:      {pc(i['duplicate_rate_pct'])}")
    L.append("")

    if t["action_distribution"]:
        L.append("Action distribution today:")
        for a, n in t["action_distribution"].items():
            L.append(f"  {a:<12} {n:>4}")
        L.append("")

    L.append("Matured outcomes (cumulative):")
    for h, n in sorted(rep["matured_outcomes"].items()):
        L.append(f"  {h:>4}d: {n:>6}")
    L.append("")

    cal = rep.get("calibration_readiness") or {}
    if "error" in cal:
        L.append(f"Calibration readiness: unavailable ({cal['error']})")
    else:
        L.append("Calibration readiness (independent windows, not row count):")
        for h, r in sorted(cal.items()):
            status = "sufficient" if r["sufficient"] else f"insufficient (short {r['shortfall']})"
            L.append(f"  {h:>4}d: effective_n {r['effective_n']:>4} / {r['needed']}   {status}")
    L.append("")
    L.append("Reporting only — this report never influences scoring or calibration.")
    return "\n".join(L)
