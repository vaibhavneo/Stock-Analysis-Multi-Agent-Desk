"""
Entry engine (Phase 7) — "is here a good place to start, and if not, what would
make it one?"

The app's previous answer was structurally incapable of saying no. Its entry
zone was `[price − 0.5·ATR, price + 0.25·ATR]`, which always contains the
current price, so whenever an entry zone was displayed the answer was yes. That
is not an entry analysis; it is a restatement of the quote.

This module answers the question against the level ladder instead:

  - WHERE is price sitting between the nearest sourced support and resistance?
  - What is the reward:risk of entering HERE, measured to levels the market has
    actually traded at rather than to an ATR multiple?
  - Is the setup stretched (statistically extended, overbought, far above
    support) or is it near a level that has been defended?

and then, crucially, states the CONDITIONS under which a currently-unattractive
entry would become attractive. Those conditions are the deliverable: a "WAIT"
with no condition attached is useless.

Entry states: ATTRACTIVE / ACCEPTABLE / WAIT / OVEREXTENDED / HIGH_RISK /
INVALIDATED / INSUFFICIENT_DATA.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Reward:risk measured to sourced levels. Stated priors, documented, not fitted.
RR_ATTRACTIVE = 2.0
RR_ACCEPTABLE = 1.5

# "Near" a level, as a fraction of price. Wide enough that daily noise does not
# flip the classification, tight enough that it means something.
NEAR_LEVEL_PCT = 3.0

# Position in the support→resistance range, 0 = at support, 1 = at resistance.
RANGE_LOW = 0.35
RANGE_HIGH = 0.75

RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0


def _range_position(price: float, support: Optional[float],
                    resistance: Optional[float]) -> Optional[float]:
    if support is None or resistance is None or resistance <= support:
        return None
    return round(max(0.0, min(1.0, (price - support) / (resistance - support))), 3)


def assess_entry(
    current_price: Optional[float],
    thesis: Dict[str, Any],
    edge: Dict[str, Any],
    level_map: Optional[Dict[str, Any]] = None,
    indicators: Optional[Dict[str, Any]] = None,
    algo_signals: Optional[Dict[str, Any]] = None,
    rec: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Current entry attractiveness, plus the conditions that would change it.

    Returns a dict whose `status` is one of the seven states above and whose
    `conditions` list is always populated when the status is not ATTRACTIVE —
    "wait" without "wait for what" is not an answer.
    """
    ind = indicators or {}
    algo = algo_signals or {}
    rec = rec or {}

    if not current_price or current_price <= 0:
        return {"status": "INSUFFICIENT_DATA",
                "reason": "No current price.",
                "geometry": {}, "conditions": [], "avoid_conditions": []}
    if not level_map or level_map.get("status") != "OK":
        return {"status": "INSUFFICIENT_DATA",
                "reason": (level_map or {}).get("reason") or
                          ("No price level could be sourced from history or indicators, so "
                           "entry location cannot be assessed. An entry band derived only "
                           "from today's price would always contain today's price and "
                           "would answer nothing."),
                "level_map_status": (level_map or {}).get("status", "MISSING"),
                "geometry": {}, "conditions": [], "avoid_conditions": []}

    support = (level_map.get("nearest_support") or {}).get("price")
    resistance = (level_map.get("nearest_resistance") or {}).get("price")
    major_support = (level_map.get("major_support") or {}).get("price")
    major_resistance = (level_map.get("major_resistance") or {}).get("price")

    risk_price = support if support is not None else major_support
    reward_price = major_resistance if major_resistance is not None else resistance

    risk = (current_price - risk_price) if risk_price else None
    reward = (reward_price - current_price) if reward_price else None
    rr = round(reward / risk, 2) if (risk and reward and risk > 0 and reward > 0) else None

    geometry = {
        "current_price": round(current_price, 2),
        "nearest_support": support,
        "nearest_resistance": resistance,
        "major_support": major_support,
        "major_resistance": major_resistance,
        "downside_to_support_pct": round((risk_price / current_price - 1) * 100, 2) if risk_price else None,
        "upside_to_resistance_pct": round((reward_price / current_price - 1) * 100, 2) if reward_price else None,
        "reward_risk_to_levels": rr,
        "range_position": _range_position(current_price, support, resistance),
        "reward_risk_basis": ("reward to the strongest nearby resistance, risk to the nearest "
                              "support — both sourced levels, not ATR multiples"),
    }

    rsi = ind.get("rsi_14")
    bb_pct = ind.get("bb_pct")
    vol_regime = algo.get("vol_regime")
    vol_expanding = algo.get("vol_expanding")
    risk_veto = bool(rec.get("risk_veto"))
    range_pos = geometry["range_position"]

    flags: List[str] = []
    if rsi is not None and rsi >= RSI_OVERBOUGHT:
        flags.append(f"RSI {rsi:.0f} — overbought")
    if rsi is not None and rsi <= RSI_OVERSOLD:
        flags.append(f"RSI {rsi:.0f} — oversold")
    if bb_pct is not None and bb_pct >= 1.0:
        flags.append("price above the upper Bollinger band")
    if bb_pct is not None and bb_pct <= 0.0:
        flags.append("price below the lower Bollinger band")
    if risk_veto:
        flags.append("risk veto active (high and expanding volatility)")
    if vol_regime == "HIGH":
        flags.append(f"volatility regime HIGH{' and expanding' if vol_expanding else ''}")

    # ── State machine. Order matters: hard blocks first, then location. ──
    reasons: List[str] = []

    if thesis["direction"] == "BEARISH":
        status = "INVALIDATED"
        reasons.append("The directional thesis is bearish — there is no long entry to locate. "
                       "This is a statement about a new long position only; it is not a "
                       "recommendation to short, which this system does not evaluate.")
    elif risk_veto:
        status = "HIGH_RISK"
        reasons.append("A risk veto is active: volatility is high and expanding. Entry "
                       "location is secondary to the fact that position sizing is capped "
                       "regardless of where price sits.")
    elif thesis["direction"] == "NEUTRAL":
        status = "WAIT"
        reasons.append("No directional thesis. Entering without one means taking volatility "
                       "with no stated reason to expect compensation.")
    elif rsi is not None and rsi >= RSI_OVERBOUGHT and (range_pos is not None and range_pos >= RANGE_HIGH):
        status = "OVEREXTENDED"
        reasons.append(f"Price sits {range_pos:.0%} of the way up the support-resistance range "
                       f"with RSI at {rsi:.0f}. Entering here pays the full distance from "
                       f"support while most of the move to resistance is behind it.")
    elif rr is not None and rr >= RR_ATTRACTIVE and (range_pos is None or range_pos <= RANGE_HIGH):
        status = "ATTRACTIVE"
        reasons.append(f"Reward:risk to sourced levels is {rr:.1f}:1 "
                       f"({geometry['upside_to_resistance_pct']:+.1f}% to resistance against "
                       f"{geometry['downside_to_support_pct']:+.1f}% to support).")
    elif rr is not None and rr >= RR_ACCEPTABLE:
        status = "ACCEPTABLE"
        if rr >= RR_ATTRACTIVE and range_pos is not None and range_pos > RANGE_HIGH:
            # The ratio alone would qualify as attractive; location is what held
            # it back, and saying so is the difference between a grade and a
            # reason. Being high in the range means most of the distance to
            # invalidation is already being paid for.
            reasons.append(
                f"Reward:risk to sourced levels is {rr:.1f}:1, which alone would read as "
                f"attractive — but price sits {range_pos:.0%} of the way up the "
                f"support-resistance range, so the entry is paying most of the distance "
                f"to its own invalidation. Acceptable, not attractive.")
        else:
            reasons.append(f"Reward:risk to sourced levels is {rr:.1f}:1 — workable but not "
                           f"compelling; a better location may appear.")
    elif rr is not None and rr < RR_ACCEPTABLE:
        status = "WAIT"
        reasons.append(f"Reward:risk to sourced levels is only {rr:.1f}:1. The distance already "
                       f"travelled from support is large relative to what remains to "
                       f"resistance.")
    else:
        status = "WAIT"
        reasons.append("Reward:risk could not be computed from the sourced levels "
                       "(price may sit outside the sourced range).")

    if status in ("ATTRACTIVE", "ACCEPTABLE") and not edge.get("demonstrated"):
        reasons.append("Note: entry LOCATION is favourable, but the statistical gate has not "
                       "cleared. A good location for an unproven signal is still an unproven "
                       "signal — this affects size, not location.")

    # ── Conditions that would make an entry attractive ───────────────────
    conditions: List[Dict[str, Any]] = []
    if status in ("WAIT", "OVEREXTENDED", "HIGH_RISK", "ACCEPTABLE"):
        if support is not None:
            conditions.append({
                "label": "A — pullback to sourced support",
                "all_of": [
                    f"Price retraces to {support} ({geometry['downside_to_support_pct']:+.1f}%), "
                    f"a level with basis {(level_map.get('nearest_support') or {}).get('basis')}",
                    "Price stabilizes there rather than closing decisively through it",
                    f"The {thesis['direction'].lower()} thesis is still intact at that point "
                    f"(evidence has not flipped)",
                ],
                "why": "Entering at a defended level shortens the distance to invalidation, "
                       "which improves reward:risk without requiring a better forecast.",
            })
        if resistance is not None:
            vol_ratio = ind.get("volume_ratio")
            conditions.append({
                "label": "B — confirmation above sourced resistance",
                "all_of": [
                    f"A close above {resistance} ({geometry['upside_to_resistance_pct']:+.1f}%)",
                    (f"Volume on the breakout above its 20-day average "
                     f"(currently {vol_ratio:.2f}× average)" if vol_ratio is not None
                     else "Volume confirmation on the breakout"),
                    "The thesis is still intact at that point",
                ],
                "why": "A breakout entry accepts a worse price in exchange for the level "
                       "having been resolved rather than assumed.",
            })
        if risk_veto or vol_regime == "HIGH":
            conditions.append({
                "label": "C — volatility normalizes",
                "all_of": ["Volatility regime leaves HIGH, or stops expanding",
                           "The risk veto clears"],
                "why": "Sizing is capped while the veto is active regardless of entry price, "
                       "so location alone cannot make the entry workable.",
            })

    # ── Conditions that should stop an entry outright ────────────────────
    avoid: List[Dict[str, Any]] = [{
        "label": "Thesis invalidation",
        "any_of": [f"Price closes below {support}" if support else
                   "The leading evidence flips direction",
                   "The leading bullish evidence turns bearish"],
        "why": "The reason for entering has stopped being true.",
    }]
    if vol_regime == "HIGH" or risk_veto:
        avoid.append({
            "label": "Unacceptable volatility regime",
            "any_of": ["Volatility regime is HIGH and expanding",
                       "The risk veto is active"],
            "why": "The position's dispersion is larger than the analysis can justify.",
        })
    nxt = (catalysts or {}).get("next_event")
    if nxt and nxt.get("days_away", 999) <= 14:
        avoid.append({
            "label": "Imminent binary event",
            "any_of": [f"{nxt['event']} is {nxt['days_away']} days away"],
            "why": ("Entering immediately before a scheduled repricing means taking the "
                    "event's variance without any demonstrated ability to predict it. "
                    "Waiting until after it is a legitimate choice."),
        })

    return {
        "status": status,
        "reason": " ".join(reasons),
        "reasons": reasons,
        "geometry": geometry,
        "flags": flags,
        "conditions": conditions,
        "avoid_conditions": avoid,
        "edge_demonstrated": bool(edge.get("demonstrated")),
        "note": ("Entry location and position size are separate questions. A favourable "
                 "location does not unlock size; only the statistical gate does."),
    }
