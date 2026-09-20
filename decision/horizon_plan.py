"""
Per-horizon action plans — "what do I do over the next few weeks, versus over
the next year?"

The engine already knew this and could not say it. Evidence carries a horizon
(decision/evidence.py), and every price level carries a timeframe
(decision/levels.py: 1M / 3M / 6M / 1Y). What was missing was the join: a
reader planning to hold for a year was being handed a support level derived
from a 20-day window, and a single `horizon_days` picked from the volatility
regime.

This module groups both by horizon and produces one plan per band. It computes
no new number: the stance comes from that band's own lean, the levels come from
the ladder filtered to that band's timeframes.

Two honesty rules hold in every plan:

  1. **Sizing is horizon-independent.** The statistical gate is a statement
     about the method, not about a holding period, so a gated 0% is 0% in all
     three plans. A per-horizon plan must not become three different ways to
     imply that sizing is available.
  2. **Thin evidence says so.** Measured across six tickers, the long band is
     frequently carried by a single item — the fundamentals pillar, which has
     never been backtested. A long-term plan built on one reading must not look
     as solid as a short-term plan built on three, so `evidence_depth` and a
     plain warning ride along with it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from decision.evidence import DecisionEvidence, directional

# Which level timeframes belong to which planning horizon, and how long the
# horizon is in trading days. The timeframe strings come from
# decision/levels.py::SOURCE_TIMEFRAME and are matched, not re-derived.
HORIZON_SPECS: Dict[str, Dict[str, Any]] = {
    "SHORT": {
        "label": "Next few weeks",
        "days": 21,
        "timeframes": ("1M", "current"),
        "horizon_description": "days to about a month",
    },
    "MEDIUM": {
        "label": "Three to six months",
        "days": 126,
        "timeframes": ("3M", "6M"),
        "horizon_description": "one to six months",
    },
    "LONG": {
        "label": "A year or more",
        "days": 252,
        "timeframes": ("1Y",),
        "horizon_description": "six months and beyond",
    },
}
HORIZON_ORDER = ("SHORT", "MEDIUM", "LONG")

# A band carried by this many directional items or fewer is reported as thin.
# One item is not a consensus; it is one reading wearing a horizon label.
THIN_EVIDENCE_MAX = 1

# Plain stance vocabulary. Deliberately small and written the way a person
# would say it, because these strings are read by the user, not by code.
STANCE_TEXT = {
    "BUY_THE_DIPS":        "Buy weakness, not strength",
    "WAIT_FOR_LOWER":      "Wait for a better price",
    "HOLD":                "Hold what you have",
    "TRIM_INTO_STRENGTH":  "Trim into strength",
    "STAY_OUT":            "Stay out",
    "NOT_ENOUGH_EVIDENCE": "Not enough evidence to plan this far out",
}


def _levels_for(level_map: Optional[Dict[str, Any]], timeframes, kind: str
                ) -> List[Dict[str, Any]]:
    """Levels of one kind that ANY of this horizon's timeframes contributed to.

    Matched on the cluster's individual SOURCES rather than on its headline
    timeframe. A cluster carries the timeframe of its highest-confidence member,
    so a 20-day mean clustered with a 50-day mean gets filed as 3M — which sent
    the nearest short-term level to the medium-term plan and left the
    few-weeks plan pointing at a level 22% away. A level is relevant to every
    horizon that contributed to it, and clusters may legitimately appear in more
    than one plan.
    """
    if not level_map or level_map.get("status") != "OK":
        return []
    from decision.levels import SOURCE_TIMEFRAME
    key = "supports" if kind == "SUPPORT" else "resistances"
    wanted = set(timeframes)
    return [l for l in (level_map.get(key) or [])
            if wanted & {SOURCE_TIMEFRAME.get(src) for src in (l.get("sources") or [])}
            or l.get("timeframe") in wanted]


def _fallback(level_map: Optional[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    if not level_map or level_map.get("status") != "OK":
        return []
    return list(level_map.get("supports" if kind == "SUPPORT" else "resistances") or [])


def _zone(level: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not level:
        return None
    return {
        "price": level.get("price"),
        "low": level.get("zone_low", level.get("price")),
        "high": level.get("zone_high", level.get("price")),
        "distance_pct": level.get("distance_pct"),
        "basis": level.get("basis"),
        "sources": level.get("sources"),
        "confidence": level.get("confidence"),
        "traded_here_before": level.get("observed", False),
    }


def _stance(band: Dict[str, Any], edge_demonstrated: bool, entry_status: str,
            owns: Optional[bool], thin: bool, risk_veto: bool) -> str:
    """The stance for one horizon.

    Reads only that band's own direction — which is the entire point. A
    short-term bullish read and a long-term bearish read are not a contradiction
    to resolve; they are two answers to two different questions.
    """
    direction = band.get("direction")

    if direction in ("NO_SIGNAL", None):
        return "NOT_ENOUGH_EVIDENCE"
    if risk_veto:
        return "HOLD" if owns else "STAY_OUT"
    if direction == "BEARISH":
        return "TRIM_INTO_STRENGTH" if owns else "STAY_OUT"
    if direction in ("NEUTRAL", "MIXED"):
        return "HOLD" if owns else "WAIT_FOR_LOWER"

    # Bullish from here. Whether to act NOW is an entry-location question; the
    # gate decides how much, and it is not horizon-specific.
    if entry_status in ("ATTRACTIVE",):
        return "BUY_THE_DIPS"
    if entry_status in ("ACCEPTABLE",):
        return "BUY_THE_DIPS" if not thin else "WAIT_FOR_LOWER"
    return "HOLD" if owns else "WAIT_FOR_LOWER"


def _why(name: str, band: Dict[str, Any], items: List[DecisionEvidence]) -> str:
    """One plain sentence naming what is actually driving this horizon."""
    from decision.plain import plain_source

    spec = HORIZON_SPECS[name]
    if band.get("direction") in ("NO_SIGNAL", None) or not items:
        return (f"Nothing in the current evidence speaks to {spec['horizon_description']}, "
                f"so there is no read to act on at this range.")

    top = sorted(items, key=lambda e: e.weight, reverse=True)[:2]
    drivers = " and ".join(plain_source(e.source) for e in top)
    # Verbless: the driver phrases mix singular and plural ("analyst consensus"
    # vs "the momentum models"), so any verb gets the agreement wrong half the
    # time.
    lean = {"BULLISH": "pointing upward", "BEARISH": "pointing downward",
            "MIXED": "split", "NEUTRAL": "flat"}.get(band["direction"], "unclear")
    return f"Over {spec['horizon_description']}: {drivers}, {lean}."


def build_horizon_plans(
    horizon_read: Dict[str, Any],
    items: List[DecisionEvidence],
    level_map: Optional[Dict[str, Any]],
    entry: Dict[str, Any],
    edge: Dict[str, Any],
    position_context: Dict[str, Any],
    sizing: Dict[str, Any],
    rec: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One plan per horizon: stance, where to buy, where to take profit, where
    to get out, and what is driving it."""
    rec = rec or {}
    bands = (horizon_read or {}).get("bands") or {}
    owns = position_context.get("status") == "PROVIDED"
    edge_demonstrated = bool(edge.get("demonstrated"))
    entry_status = entry.get("status", "INSUFFICIENT_DATA")
    risk_veto = bool(rec.get("risk_veto"))

    by_band: Dict[str, List[DecisionEvidence]] = {b: [] for b in HORIZON_ORDER}
    for e in directional(items):
        if e.horizon in by_band:
            by_band[e.horizon].append(e)

    plans: List[Dict[str, Any]] = []
    for name in HORIZON_ORDER:
        spec = HORIZON_SPECS[name]
        band = bands.get(name) or {"direction": "NO_SIGNAL", "lean": None,
                                   "n_items": 0, "weight_total": 0.0}
        band_items = by_band[name]
        depth = len(band_items)
        thin = depth <= THIN_EVIDENCE_MAX and band.get("direction") not in ("NO_SIGNAL", None)

        supports = _levels_for(level_map, spec["timeframes"], "SUPPORT")
        resistances = _levels_for(level_map, spec["timeframes"], "RESISTANCE")
        borrowed = False
        if not supports:
            supports, borrowed = _fallback(level_map, "SUPPORT"), True
        if not resistances:
            resistances, borrowed = _fallback(level_map, "RESISTANCE"), True

        buy_level = supports[0] if supports else None
        buy_zone = _zone(buy_level)
        take_profit = _zone(resistances[0] if resistances else None)

        # The exit is the next sourced level BELOW where you would have bought:
        # if the level you bought at fails, the reason for buying there has
        # failed with it. Taking the deepest level in range instead produced a
        # "get out below" 63% down, which is not an exit — it is a shrug.
        exit_level = None
        if buy_level is not None:
            all_supports = _fallback(level_map, "SUPPORT")   # nearest-first
            below = [l for l in all_supports if l["price"] < buy_level["price"]]
            exit_level = below[0] if below else None
        exit_below = _zone(exit_level)

        stance = _stance(band, edge_demonstrated, entry_status, owns, thin, risk_veto)

        # ── Fix 3: no view, no plan. Showing a buy area under a stance that
        # says "not enough evidence" reads as a recommendation. The levels are
        # still worth seeing, so they move to `reference_levels` and say what
        # they are.
        reference_levels = None
        if stance == "NOT_ENOUGH_EVIDENCE":
            reference_levels = {"support": buy_zone, "resistance": take_profit,
                                "note": ("Shown for orientation only. There is no directional "
                                         "read at this range, so there is nothing to act on.")}
            buy_zone = take_profit = exit_below = None

        warnings: List[str] = []
        if thin:
            from decision.plain import plain_source
            named = ", ".join(plain_source(e.source) for e in band_items)
            warnings.append(
                f"This rests on a single reading — {named} — so treat it as weaker than the "
                f"other horizons even though it is stated the same way.")
        if borrowed:
            warnings.append(
                "No price level was available at this timeframe, so levels from a different "
                "timeframe are shown — they are less well matched to this holding period.")
        if (buy_zone and not buy_zone["traded_here_before"]
                and stance not in ("TRIM_INTO_STRENGTH", "STAY_OUT", "NOT_ENOUGH_EVIDENCE")):
            warnings.append(
                "The buy area is a calculated line, not a price this stock has actually "
                "traded at.")

        # What the levels MEAN depends on the stance. A bearish plan showing a
        # "Buy near" row is the same defect as a plan with no view showing one:
        # the number is right and the label turns it into advice the evidence
        # does not support.
        if stance == "NOT_ENOUGH_EVIDENCE":
            action_rows: List[Dict[str, Any]] = []
        elif stance in ("TRIM_INTO_STRENGTH", "STAY_OUT"):
            action_rows = [
                {"label": ("Trim into any rally toward" if stance == "TRIM_INTO_STRENGTH"
                           else "Would only reconsider above"),
                 "zone": take_profit, "empty": "no level in range"},
                {"label": "Get out below", "zone": buy_zone, "empty": "not priced"},
            ]
        else:
            action_rows = [
                {"label": "Add near" if owns else "Buy near",
                 "zone": buy_zone, "empty": "no level to buy at"},
                {"label": "Take profit near", "zone": take_profit, "empty": "no level in range"},
                {"label": "Get out below", "zone": exit_below, "empty": "not priced"},
            ]

        plans.append({
            "horizon": name,
            "action_rows": action_rows,
            "label": spec["label"],
            "horizon_days": spec["days"],
            "horizon_description": spec["horizon_description"],
            "stance": stance,
            "stance_text": STANCE_TEXT[stance],
            "direction": band.get("direction"),
            "lean": band.get("lean"),
            "evidence_depth": depth,
            "evidence_thin": thin,
            "drivers": [e.source for e in sorted(band_items, key=lambda x: x.weight,
                                                 reverse=True)[:3]],
            "why": _why(name, band, band_items),
            "buy_zone": buy_zone,
            "reference_levels": reference_levels,
            "take_profit_near": take_profit,
            "exit_below": exit_below,
            "how_much": sizing.get("statement"),
            "sizing_gated": bool(sizing.get("gated")),
            "warnings": warnings,
        })

    distinct = len({p["stance"] for p in plans})
    return {
        "plans": plans,
        "n_distinct_stances": distinct,
        "owns": owns,
        "note": ("Levels are places the price has been, not places it is going. How much to "
                 "commit is decided by the statistical gate, which is the same for every "
                 "horizon — a longer holding period does not unlock size."),
        "sizing_statement": sizing.get("statement"),
    }
