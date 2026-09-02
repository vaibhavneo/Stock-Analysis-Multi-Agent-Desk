"""
Phase 3 edge analysis — does the agent's information predict future EXCESS return?

This is a measurement instrument. It reads frozen forecasts and matured
outcomes and reports relationships. It never changes a score, a weight, a
threshold, or a recommendation, and nothing downstream consumes it. That
separation is the point: the moment this feeds back into the strategy, every
number it produces becomes a number the strategy was fitted to.

WHY EXCESS RETURN, NOT RAW RETURN
    In a rising market every bucket looks profitable. Excess return over the
    benchmark is the only version of the question that can distinguish signal
    from beta.

WHY EVERY ROW CARRIES ITS OWN UNCERTAINTY
    The honest failure mode of a tool like this is producing a seductive
    ordering across buckets that is entirely noise. Three defences:

      1. EFFECTIVE n, not row count. Predictions on 50 tickers on one day
         share a market; overlapping evaluation windows share a price path.
         Both are reported, and they differ by an order of magnitude.
      2. A standard error and t-statistic on every mean, so "bucket 80+ earned
         +2.1%" can be read against how noisy that estimate is.
      3. An explicit verdict of INSUFFICIENT until a bucket clears minimum
         independent observations. A bucket below the floor reports no
         direction at all rather than a tempting one.

    None of these make a small sample trustworthy. They make it obvious.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from data import prediction_ledger as pl
from data.prediction_ledger import HORIZONS

# Composite bands the user asked for. Open-ended at the top.
COMPOSITE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (0.0, 50.0, "<50"), (50.0, 55.0, "50-55"), (55.0, 60.0, "55-60"),
    (60.0, 65.0, "60-65"), (65.0, 70.0, "65-70"), (70.0, 75.0, "70-75"),
    (75.0, 80.0, "75-80"), (80.0, 1e9, "80+"),
)
ACTIONS = ("BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL")
PILLARS = ("technical", "algo", "fundamentals", "social", "research", "risk")

# Minimum INDEPENDENT observations before a bucket is allowed a verdict.
MIN_INDEPENDENT = 20


def _rows(horizon: int, source: str = "all") -> List[Dict[str, Any]]:
    """Matured outcomes joined to their frozen forecast, evidence only."""
    conn = pl._conn()
    try:
        raw = [dict(r) for r in conn.execute(
            f"""SELECT o.excess_return_pct, o.raw_return_pct, o.direction_correct,
                       o.as_of_date, s.ticker, s.action, s.created_at,
                       s.pillars_json, s.frozen_json
                  FROM prediction_outcomes o
                  JOIN prediction_snapshots s ON s.snapshot_id = o.snapshot_id
                 WHERE o.horizon_days=? AND o.matured=1
                       AND o.excess_return_pct IS NOT NULL
                       {pl._source_where(source, 's')}""", (horizon,)).fetchall()]
    finally:
        conn.close()

    out = []
    for r in raw:
        try:
            fj = json.loads(r["frozen_json"] or "{}")
        except Exception:
            fj = {}
        try:
            pj = json.loads(r["pillars_json"] or "{}")
        except Exception:
            pj = {}
        out.append({
            "ticker": r["ticker"], "action": r["action"],
            "day": str(r["created_at"] or "")[:10],
            "excess": float(r["excess_return_pct"]),
            "raw": float(r["raw_return_pct"]) if r["raw_return_pct"] is not None else None,
            "correct": r["direction_correct"],
            "composite": fj.get("composite"),
            "pillars": pj,
        })
    return out


def _independent_n(rows: Sequence[Dict[str, Any]], horizon: int) -> int:
    """Distinct NON-OVERLAPPING call days.

    Two ceilings, smaller wins: the number of distinct call dates (you cannot
    observe more often than you sample) and the calendar span divided by the
    horizon (overlapping windows share a price path). Fifty tickers forecast on
    one morning are one observation of one market, not fifty.
    """
    days = sorted({r["day"] for r in rows if r["day"]})
    if len(days) < 2:
        return len(days)
    try:
        from datetime import date
        span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days
    except Exception:
        return len(days)
    if horizon <= 0:
        return len(days)
    windows = max(1, int((span * 252.0 / 365.0) // horizon))
    return min(windows, len(days))


def _stats(rows: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    vals = [r["excess"] for r in rows]
    n = len(vals)
    if n == 0:
        return {"n": 0, "independent_n": 0, "verdict": "NO DATA"}
    mean = sum(vals) / n
    ind = _independent_n(rows, horizon)
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        sd = math.sqrt(var)
        # SE uses INDEPENDENT n, not row count: dividing by sqrt(460) when
        # there are 12 independent windows would understate the error ~6x.
        se = sd / math.sqrt(max(1, ind))
    else:
        sd = se = float("nan")
    t = (mean / se) if se and se == se and se > 0 else float("nan")
    hit = [r for r in rows if r["correct"] is not None]
    return {
        "n": n, "independent_n": ind,
        "mean_excess_pct": round(mean, 3),
        "median_excess_pct": round(sorted(vals)[n // 2], 3),
        "sd_pct": round(sd, 3) if sd == sd else None,
        "stderr_pct": round(se, 3) if se == se else None,
        "t_stat": round(t, 2) if t == t else None,
        "hit_rate": round(sum(r["correct"] for r in hit) / len(hit), 3) if hit else None,
        "verdict": ("INSUFFICIENT" if ind < MIN_INDEPENDENT else
                    ("inconclusive" if (t != t or abs(t) < 2.0) else
                     ("positive" if t > 0 else "negative"))),
    }


def by_composite(horizon: int, source: str = "all") -> Dict[str, Any]:
    rows = [r for r in _rows(horizon, source) if r["composite"] is not None]
    out = {"horizon_days": horizon, "rows_with_composite": len(rows), "buckets": {}}
    for lo, hi, label in COMPOSITE_BUCKETS:
        out["buckets"][label] = _stats(
            [r for r in rows if lo <= float(r["composite"]) < hi], horizon)
    return out


def by_action(horizon: int, source: str = "all") -> Dict[str, Any]:
    rows = _rows(horizon, source)
    return {"horizon_days": horizon, "rows": len(rows),
            "actions": {a: _stats([r for r in rows if r["action"] == a], horizon)
                        for a in ACTIONS}}


def by_pillar(horizon: int, threshold: float = 60.0, source: str = "all") -> Dict[str, Any]:
    """Outcomes when a pillar was LEANING (score >= threshold) vs not.

    A crude first cut, deliberately: it answers 'did outcomes differ when this
    pillar was positive', not 'what is this pillar's marginal contribution
    holding the others fixed'. The latter needs far more independent data than
    exists, and a regression on 12 independent windows would be theatre.
    """
    rows = _rows(horizon, source)
    out = {"horizon_days": horizon, "threshold": threshold, "pillars": {}}
    for p in PILLARS:
        lean = [r for r in rows if (r["pillars"] or {}).get(p) is not None
                and float(r["pillars"][p]) >= threshold]
        rest = [r for r in rows if (r["pillars"] or {}).get(p) is not None
                and float(r["pillars"][p]) < threshold]
        s_lean, s_rest = _stats(lean, horizon), _stats(rest, horizon)
        spread = (None if s_lean.get("mean_excess_pct") is None
                  or s_rest.get("mean_excess_pct") is None
                  else round(s_lean["mean_excess_pct"] - s_rest["mean_excess_pct"], 3))
        out["pillars"][p] = {"leaning": s_lean, "not_leaning": s_rest,
                             "spread_pct": spread}
    return out


def full_report(source: str = "all") -> Dict[str, Any]:
    return {"generated_for": "phase-3 measurement only",
            "min_independent_for_verdict": MIN_INDEPENDENT,
            "by_horizon": {int(h): {"composite": by_composite(int(h), source),
                                    "action": by_action(int(h), source),
                                    "pillar": by_pillar(int(h), source=source)}
                           for h in HORIZONS}}


def _fmt_stats(s: Dict[str, Any]) -> str:
    if s.get("n", 0) == 0:
        return f"{'—':>9} {'':>7} {'':>7}  no data"
    mean = f"{s['mean_excess_pct']:+.2f}%"
    t = "n/a" if s.get("t_stat") is None else f"{s['t_stat']:+.2f}"
    hit = "n/a" if s.get("hit_rate") is None else f"{s['hit_rate']:.0%}"
    return (f"{mean:>9} {t:>7} {hit:>7}   n={s['n']:<5} ind={s['independent_n']:<4} {s['verdict']}")


def format_report(rep: Dict[str, Any], horizons: Optional[Sequence[int]] = None) -> str:
    L = ["STOCK AGENT — PHASE 3 EDGE ANALYSIS (measurement only)", "",
         "Mean EXCESS return vs benchmark. t uses INDEPENDENT observations, not row",
         f"count. A bucket needs >= {rep['min_independent_for_verdict']} independent observations before it gets a",
         "verdict at all; below that it reports INSUFFICIENT and no direction.", ""]
    for h, blk in sorted(rep["by_horizon"].items()):
        if horizons and h not in horizons:
            continue
        L.append(f"── horizon {h}d " + "─" * 52)
        L.append(f"  {'BY ACTION':<14}{'mean':>9} {'t':>7} {'hit':>7}")
        for a, s in blk["action"]["actions"].items():
            L.append(f"  {a:<14}{_fmt_stats(s)}")
        L.append("")
        comp = blk["composite"]
        L.append(f"  {'BY COMPOSITE':<14}{'mean':>9} {'t':>7} {'hit':>7}"
                 f"    (rows with composite: {comp['rows_with_composite']})")
        for b, s in comp["buckets"].items():
            L.append(f"  {b:<14}{_fmt_stats(s)}")
        L.append("")
    L.append("This report never influences scoring, weights, thresholds or calibration.")
    return "\n".join(L)
