"""
Horizon-aware synthesis (Phase 13).

The composite this app has always produced blends a technical read measured in
days with a fundamental read measured in quarters, using fixed weights. That
number is useful, and it stays — but it cannot answer "which horizon is driving
this?", and that question is the difference between "the stock is weak" and
"the stock is weak *this month* while the multi-year trend is intact". Those
call for opposite actions.

So this module does NOT re-score anything. It partitions the already-normalized
DecisionEvidence by the horizon each item was assigned in decision/evidence.py
and reports each band separately, plus which band is currently dominant and
whether the bands disagree.

The weighted lean inside a band is reliability × magnitude, signed by direction
— the same weighting intelligence/evidence_synthesis.py already uses, applied
within a band instead of across all of them. Items that make no directional
claim (the statistical gate, the track record, the risk pillar, the user's cost
basis) are excluded entirely rather than counted as neutral votes: a NEUTRAL
vote drags a lean toward zero, and "we have no opinion about direction" must not
be able to do that.
"""
from __future__ import annotations

from typing import Any, Dict, List

from decision.evidence import DecisionEvidence, HORIZON_DESCRIPTION, directional

# A band needs at least this much signed weight before it is called directional
# at all. Below it, the band reads MIXED/NEUTRAL rather than committing.
LEAN_THRESHOLD = 0.15

# How far apart two bands' leans must be before they are reported as a horizon
# conflict. 0.6 on a −1..+1 scale means one band is meaningfully bullish while
# another is meaningfully bearish, not merely two shades of the same view.
HORIZON_CONFLICT_GAP = 0.6

BAND_ORDER = ("SHORT", "MEDIUM", "LONG")


def _band_lean(items: List[DecisionEvidence]) -> Dict[str, Any]:
    """Signed, weight-normalized lean in −1..+1 for one horizon band."""
    dir_items = directional(items)
    if not dir_items:
        return {"lean": None, "direction": "NO_SIGNAL", "n_items": 0,
                "weight_total": 0.0, "bullish": [], "bearish": []}

    signed = 0.0
    weight_total = 0.0
    bullish, bearish = [], []
    for e in dir_items:
        w = e.weight
        weight_total += w
        if e.direction == "BULLISH":
            signed += w
            bullish.append(e.source)
        else:
            signed -= w
            bearish.append(e.source)

    lean = round(signed / weight_total, 3) if weight_total > 0 else 0.0
    if weight_total < LEAN_THRESHOLD:
        direction = "NEUTRAL"
    elif lean >= 0.3:
        direction = "BULLISH"
    elif lean <= -0.3:
        direction = "BEARISH"
    elif bullish and bearish:
        direction = "MIXED"
    else:
        direction = "NEUTRAL"

    return {"lean": lean, "direction": direction, "n_items": len(dir_items),
            "weight_total": round(weight_total, 3),
            "bullish": bullish, "bearish": bearish}


def _summarize_band(name: str, band: Dict[str, Any],
                    items: List[DecisionEvidence]) -> str:
    if band["direction"] == "NO_SIGNAL":
        return f"{name.title()} term ({HORIZON_DESCRIPTION[name]}): no directional evidence."
    top = sorted(directional(items), key=lambda e: e.weight, reverse=True)[:2]
    detail = "; ".join(e.observation for e in top)
    return (f"{name.title()} term ({HORIZON_DESCRIPTION[name]}): "
            f"{band['direction'].lower()} (lean {band['lean']:+.2f}). {detail}")


def synthesize_by_horizon(items: List[DecisionEvidence]) -> Dict[str, Any]:
    """Partition evidence into SHORT / MEDIUM / LONG and report each band.

    `ALL`-horizon items are deliberately NOT distributed into the three bands.
    They are horizon-independent by definition, and copying them into every band
    would let one observation be counted three times — the exact double-count
    that makes a blended composite unreadable.
    """
    by_band: Dict[str, List[DecisionEvidence]] = {b: [] for b in BAND_ORDER}
    horizon_independent: List[DecisionEvidence] = []
    for e in items:
        if e.horizon == "ALL":
            horizon_independent.append(e)
        else:
            by_band[e.horizon].append(e)

    bands = {b: _band_lean(by_band[b]) for b in BAND_ORDER}
    summaries = {b: _summarize_band(b, bands[b], by_band[b]) for b in BAND_ORDER}

    # Dominant band = the one carrying the most directional weight. "Dominant"
    # means "most of what we actually know is about this timeframe", not "most
    # important" — a user with a 5-year horizon should read a SHORT-dominant
    # verdict as a warning that the evidence is not about their timeframe.
    scored = [(b, bands[b]["weight_total"]) for b in BAND_ORDER
              if bands[b]["direction"] not in ("NO_SIGNAL",)]
    dominant = max(scored, key=lambda t: t[1])[0] if scored else None

    conflicts: List[Dict[str, Any]] = []
    for i, a in enumerate(BAND_ORDER):
        for b in BAND_ORDER[i + 1:]:
            la, lb = bands[a]["lean"], bands[b]["lean"]
            if la is None or lb is None:
                continue
            gap = abs(la - lb)
            if gap >= HORIZON_CONFLICT_GAP and (la > 0) != (lb > 0):
                conflicts.append({
                    "horizons": [a, b],
                    "leans": {a: la, b: lb},
                    "gap": round(gap, 3),
                    "description": (
                        f"{a.title()}-term evidence leans {bands[a]['direction'].lower()} "
                        f"({la:+.2f}) while {b.lower()}-term leans "
                        f"{bands[b]['direction'].lower()} ({lb:+.2f}). These are claims "
                        f"about different futures, not a contradiction to average away — "
                        f"which one matters depends on the holding period."),
                })

    return {
        "bands": bands,
        "summaries": summaries,
        "dominant_horizon": dominant,
        "dominant_reason": (
            f"Most directional evidence ({bands[dominant]['weight_total']:.2f} of "
            f"{sum(bands[b]['weight_total'] for b in BAND_ORDER):.2f} total weight) is "
            f"{dominant.lower()}-term." if dominant else
            "No horizon carries directional evidence."),
        "horizon_conflicts": conflicts,
        "horizon_independent": [e.source for e in horizon_independent],
    }
