"""
The options overlay — OptionsPilot's structures, normalized for this view.

This is a RENDERING of another service's answer, not a second opinion about it.
Nothing here re-prices, re-ranks or re-judges an option; if the numbers are
wrong they are wrong in OptionsPilot and belong fixed there.

Two rules shape the whole module.

**Its honesty is carried, never stripped.** OptionsPilot labels its own output
hard: candidates are ORDERED BY PAYOFF SHAPE, not by merit; its research
baseline reads NO_DEMONSTRATED_EDGE; the expected-payoff spread across
structures is the volatility smile rather than an edge. A ranked list of option
trades rendered without those labels is strictly more dangerous than no list at
all, because ranking implies a recommendation that the source explicitly
disclaims. So `honesty` is assembled first and every consumer gets it.

**The direction is ours, and that is stated.** The link hands this engine's
thesis direction to OptionsPilot rather than letting it read a stored desk view
— its own status endpoint reports that most names carry no fresh view at all.
That makes the two agents agree by construction, and `direction_source` records
which read produced the structures so a surprising set can be traced back.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# This engine's thesis direction as OptionsPilot names outcomes. NEUTRAL is
# passed through as None rather than guessed at: a neutral read has no
# directional structure, and inventing one would be the link asserting a view
# neither service holds.
DIRECTION_TO_VIEW = {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "NEUTRAL": None}

# Plain names for the structures OptionsPilot builds. Its own `label` is
# already readable; this covers the machine `structure` key where it surfaces.
STRUCTURE_TEXT = {
    "bull_call_spread": "buy a call spread",
    "bear_put_spread": "buy a put spread",
    "long_call": "buy a call",
    "long_put": "buy a put",
    "short_put_spread": "sell a put spread",
    "short_call_spread": "sell a call spread",
}


def view_for(thesis: Optional[Dict[str, Any]]) -> Optional[str]:
    """The view this engine would hand over, or None when it has no direction."""
    if not thesis:
        return None
    return DIRECTION_TO_VIEW.get(thesis.get("direction"))


def _legs_text(legs: List[Dict[str, Any]]) -> str:
    """"Buy 1 call at 335, sell 1 call at 355" — the position in words."""
    parts = []
    for leg in legs or []:
        action = str(leg.get("action", "")).lower()
        qty = abs(leg.get("quantity") or 0)
        right = str(leg.get("right", "")).lower()
        strike = leg.get("strike")
        parts.append(f"{action} {qty} {right}{'s' if qty != 1 else ''} at "
                     f"${strike:,.2f}" if strike is not None else f"{action} {qty} {right}")
    if not parts:
        return ""
    return "; ".join(parts).capitalize() + "."


def _candidate(raw: Dict[str, Any], spot: Optional[float]) -> Dict[str, Any]:
    profile = raw.get("profile") or {}
    rr = raw.get("risk_reward") or {}
    breakevens = profile.get("breakevens") or []
    pop = raw.get("probability_of_profit")

    be_text = None
    if breakevens and spot:
        first = breakevens[0]
        move = (first / spot - 1) * 100
        be_text = (f"Breaks even at ${first:,.2f}, {move:+.1f}% from here"
                   + (f"; second break-even at ${breakevens[1]:,.2f}." if len(breakevens) > 1
                      else "."))

    return {
        "rank": raw.get("rank"),
        "label": raw.get("label"),
        "structure": raw.get("structure"),
        "structure_text": STRUCTURE_TEXT.get(raw.get("structure"), raw.get("label")),
        "legs": raw.get("legs") or [],
        "legs_text": _legs_text(raw.get("legs") or []),
        "note": raw.get("note"),
        "breakevens": breakevens,
        "breakeven_text": be_text,
        # A probability OptionsPilot computed from the chain's own implied
        # volatility. It is model-derived, not a measured frequency, and the
        # label says so wherever it is shown.
        "probability_of_profit": pop,
        "probability_basis": ("implied by the option chain's own volatility — a model "
                              "number, not a measured frequency"),
        "risk_reward_ratio": rr.get("ratio"),
        "risk_reward_note": rr.get("note"),
        "max_loss": profile.get("max_loss"),
        "max_gain": profile.get("max_gain"),
        "expected_payoff": raw.get("expected_payoff"),
        "criterion_value": raw.get("criterion_value"),
        "diagram": raw.get("diagram"),
        "move_profile": raw.get("move_profile"),
        # Carried verbatim. This is the sentence that stops a ranked list from
        # reading as a recommendation.
        "edge_note": raw.get("edge"),
    }


def build_options_overlay(raw: Optional[Dict[str, Any]],
                          thesis: Optional[Dict[str, Any]] = None,
                          requested_view: Optional[str] = None) -> Dict[str, Any]:
    """Normalize one OptionsPilot instruments payload for this view."""
    if not raw or not raw.get("available"):
        reason = (raw or {}).get("reason") or "OptionsPilot was not consulted."
        return {
            "status": "UNAVAILABLE",
            "reason": reason,
            "authenticated": (raw or {}).get("authenticated"),
            "candidates": [],
            "headline": "No options structures are available for this symbol right now.",
            "honesty": [],
        }

    sets = raw.get("sets") or []
    first = sets[0] if sets else {}
    candidates_raw = first.get("candidates") or []
    spot = raw.get("spot")

    candidates = [_candidate(c, spot) for c in candidates_raw]
    research = (candidates_raw[0].get("research") if candidates_raw else None) or {}

    # ── Honesty, assembled before anything else is read ──────────────────
    honesty: List[str] = []
    if first.get("ranked_not_best"):
        honesty.append(first["ranked_not_best"])
    if raw.get("edge"):
        honesty.append(raw["edge"])
    if research.get("verdict") and not research.get("demonstrated_edge", False):
        honesty.append(
            f"OptionsPilot's own research verdict is {research['verdict'].replace('_', ' ').lower()} "
            f"(baseline {research.get('baseline')}). It has not shown that choosing structures "
            f"this way makes money.")
    carry = first.get("carry_identity") or {}
    if carry.get("deviation_means"):
        honesty.append(carry["deviation_means"])
    if raw.get("conflicted_note"):
        honesty.append(raw["conflicted_note"])
    if raw.get("freshness_note"):
        honesty.append(raw["freshness_note"])

    # ── Where the direction came from ────────────────────────────────────
    outcome = raw.get("outcome")
    if requested_view:
        direction_source = {
            "view": outcome,
            "source": "THIS_ENGINE",
            "statement": (f"These structures were built for a {str(outcome).lower()} view, which "
                          f"is the direction this engine's own evidence produced — not a stored "
                          f"view read from elsewhere."),
        }
    else:
        direction_source = {
            "view": outcome,
            "source": "OPTIONSPILOT",
            "statement": (f"These structures were built for a {str(outcome).lower()} view that "
                          f"OptionsPilot derived on its own. This engine did not supply the "
                          f"direction."),
        }

    liquidity = first.get("liquidity") or {}
    quoting = first.get("quoting") or {}
    unpriceable = first.get("unpriceable") or []

    warnings: List[str] = []
    if unpriceable:
        warnings.append(f"{len(unpriceable)} structure(s) could not be priced from the chain.")
    failed = quoting.get("failed") or []
    if failed:
        warnings.append(f"{len(failed)} option leg(s) had no usable quote.")
    for row in (liquidity.get("by_structure") or []):
        if row.get("all_tradeable") is False:
            warnings.append(f"{row.get('structure')} has at least one illiquid leg.")
        elif (row.get("worst_spread_pct") or 0) > 25:
            warnings.append(f"{row.get('structure')} has a wide bid-ask spread "
                            f"({row['worst_spread_pct']:.0f}% at its worst leg).")

    if candidates:
        headline = (f"{len(candidates)} option structures for a {str(outcome).lower()} view, "
                    f"expiring {raw.get('expiry')} ({raw.get('days')} days out).")
    else:
        headline = "OptionsPilot found no structure it could price for this view."

    return {
        "status": "OK",
        "ticker": raw.get("ticker") or raw.get("symbol"),
        "spot": spot,
        "expiry": raw.get("expiry"),
        "days_to_expiry": raw.get("days"),
        "priced_at": raw.get("priced_at"),
        "headline": headline,
        "direction_source": direction_source,
        "criterion": first.get("criterion"),
        "candidates": candidates,
        "n_candidates": len(candidates),
        "liquidity": liquidity,
        "warnings": warnings,
        "research": research,
        "honesty": honesty,
        "rules_version": first.get("rules_version"),
        "source": {
            "service": "OptionsPilot",
            "relationship": ("A separate service with its own release cycle and its own "
                             "research freeze. This view renders its answer; it does not "
                             "re-price, re-rank or second-guess it."),
        },
        "no_execution": ("Neither service can place an order. These are structures to study, "
                         "not orders to send."),
    }
