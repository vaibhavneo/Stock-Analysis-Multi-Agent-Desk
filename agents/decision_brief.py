"""
Decision Brief v2 — one 20-second-readable investment decision from all Stock Agent outputs.

This is a PURE CONSUMER layer on top of `decision_synthesis.synthesize_decision()`. It creates
no new agents, strategies, factors, scores, or databases — it calls the existing synthesis
twice (once per ownership branch) to get the same deterministic numeric backbone, then reshapes
that into a compact brief: one action, ownership-aware guidance, exactly three decisive
insights, an action plan, upgrade/downgrade triggers, an evidence-quality strip, better-ranked
alternatives, and links to the underlying evidence.

Deterministic by construction: `final_action`, confidence levels, sizing, and price levels all
come from `recommendation.py` / `decision_synthesis.py` arithmetic. Any LLM-authored prose in
the inputs (thesis summaries, catalysts) may be quoted verbatim in insights/scenarios but never
changes the action, confidence, sizing, or price levels.

Action vocabulary: BUY / ACCUMULATE / HOLD / WATCH / REDUCE / AVOID / INSUFFICIENT_EVIDENCE.
  - BUY / ACCUMULATE are ownership-independent (thesis positive; ACCUMULATE = edge not yet HIGH
    or allocation gated, so build gradually rather than sizing fully).
  - HOLD (owned) / WATCH (not owned) — same "neutral" tier, worded for the reader's position.
    This is exactly the "strong company, weak edge/timing" case: thesis positive but the
    statistical-edge gate hasn't cleared HIGH.
  - REDUCE (owned) / AVOID (not owned) — same "negative" tier (thesis negative, or a risk veto
    forced a downgrade). AVOID is used instead of SELL whenever ownership is False or unknown.
"""
from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.decision_synthesis import synthesize_decision

REQUIRED_KEYS = ("current_price", "action", "composite", "confidence", "levels")

_INSTRUCTION = {
    "BUY":         "Add within the entry range — this is the highest-conviction tier.",
    "ACCUMULATE":  "Build the position gradually (partial size); do not size fully yet.",
    "HOLD":        "Keep the current position; do not add until conviction improves.",
    "WATCH":       "Do not initiate yet — wait for a stronger statistical edge or catalyst.",
    "REDUCE":      "Trim exposure toward the review point below.",
    "AVOID":       "Do not initiate a position at this time.",
}


def build_decision_brief(
    ticker: str,
    recommendation: Dict[str, Any],
    backtest_all: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    xsec_ranking: Optional[Dict[str, Any]] = None,
    prediction_summary: Optional[Dict[str, Any]] = None,
    orchestrator_result: Optional[Dict[str, Any]] = None,
    owns_position: Optional[bool] = None,
    position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce the compact Decision Brief v2.

    Architecture: seven-agent analysis -> objective stock verdict (tier-based,
    position-agnostic) -> combined with the user's own position context
    (`position={"avg_cost": ..., "shares": ...}`) -> a position-aware decision.
    Supplying `position` implies `owns_position=True` (an average cost only makes
    sense if you hold the position) unless the caller explicitly overrides it.

    `owns_position=None` with no `position` given means unknown — treated the
    same as False for the headline action (AVOID, never SELL)."""
    ticker = (ticker or "").upper().strip()
    now = datetime.now().isoformat(timespec="seconds")
    rec = recommendation or {}

    missing = _missing_required(rec)
    if missing:
        return {
            "ticker": ticker or None,
            "generated_at": now,
            "final_action": "INSUFFICIENT_EVIDENCE",
            "one_line_verdict": "Required inputs are missing — no decision can be produced.",
            "missing_inputs": missing,
        }

    position_context = _build_position_context(rec, position)
    if position_context and owns_position is None:
        owns_position = True

    # Two ownership branches of the SAME deterministic engine — no new logic, just reused
    # twice. Their numeric backbone (composite/confidence/backtest/calibration/xsec/scenarios)
    # is identical; only `final_action`/`action_rationale` differ, which we don't use directly
    # (we derive our own 6-word vocabulary from the shared backbone below).
    shared = synthesize_decision(ticker, rec, backtest_all, calibration, xsec_ranking,
                                  prediction_summary, orchestrator_result, owns_position=False)

    tier = _tier(rec)
    objective_action_owned = _refine_action(tier, "owned", rec, shared)
    action_if_not_owned = _refine_action(tier, "not_owned", rec, shared)

    # ── Position-aware layer: the ONLY place a user's specific entry price can
    #    change the decision, and only via measurable, deterministic thresholds
    #    on unrealized P&L — never LLM judgment. Can even override the objective
    #    tier (e.g. HOLD -> REDUCE for profit-taking on a large unrealized gain).
    action_if_owned, position_note = _apply_position_awareness(
        objective_action_owned, position_context)

    top_action = action_if_owned if owns_position is True else action_if_not_owned
    use_position_note = bool(position_context) and owns_position is True

    rationale = _one_line_verdict(tier, rec, shared, top_action,
                                   position_note if use_position_note else None)
    insights = _select_three_insights(rec, shared)
    action_plan = _build_action_plan(rec, shared, top_action)
    triggers = _build_triggers(rec, shared)
    evidence_status = _build_evidence_status(shared)
    alternatives = _find_alternatives(ticker, xsec_ranking)

    brief: Dict[str, Any] = {
        "ticker": ticker,
        "generated_at": now,
        "current_price": rec.get("current_price"),
        "composite_score": rec.get("composite"),

        "final_action": top_action,
        "one_line_verdict": rationale,
        "objective_action": objective_action_owned,  # position-agnostic, for transparency

        "position_context": position_context,

        "guidance": {
            "if_owned": {"action": action_if_owned,
                         "instruction": _INSTRUCTION[action_if_owned],
                         "position_note": position_note},
            "if_not_owned": {"action": action_if_not_owned,
                              "instruction": _INSTRUCTION[action_if_not_owned]},
        },

        "decisive_insights": insights,
        "action_plan": action_plan,
        "triggers": triggers,
        "evidence_status": evidence_status,
        "alternatives": alternatives,

        "evidence_links": shared["evidence"],
        "warnings": shared["warnings"],
        "confidence": shared["confidence"],

        "disclaimer": ("Decision Brief synthesized from computed outputs — "
                        "not financial advice. Numbers are historical, not predictive."),
    }
    brief["brief_fingerprint"] = _fingerprint(brief)
    brief["recommendation_fingerprint"] = rec.get("decision_fingerprint")
    return brief


# ── Position-aware layer (user's cost basis + current price) ──────────────

def _build_position_context(rec: Dict[str, Any],
                            position: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not position:
        return None
    avg_cost = position.get("avg_cost")
    current_price = rec.get("current_price")
    if not avg_cost or avg_cost <= 0 or not current_price:
        return None
    pl_pct = (current_price - avg_cost) / avg_cost * 100
    shares = position.get("shares")
    pl_dollar = (current_price - avg_cost) * shares if shares else None
    return {
        "avg_cost": round(float(avg_cost), 4),
        "current_price": current_price,
        "shares": shares,
        "unrealized_pl_pct": round(pl_pct, 2),
        "unrealized_pl_dollar": round(pl_dollar, 2) if pl_dollar is not None else None,
    }


def _apply_position_awareness(objective_action: str,
                              position_context: Optional[Dict[str, Any]]) -> tuple:
    """Refines the objective (position-agnostic) owned-branch action using the
    user's own unrealized P&L. Returns (refined_action, position_note).

    Thresholds are fixed and documented here — not tuned, not LLM-chosen:
      +20% unrealized on a neutral (HOLD) signal -> profit-taking override to REDUCE.
      -20% unrealized on a neutral (HOLD) signal -> flagged, no override (patience).
      -15% unrealized on a positive (BUY/ACCUMULATE) signal -> averaging-down note.
    """
    if not position_context:
        return objective_action, None
    pl = position_context["unrealized_pl_pct"]

    if objective_action in ("BUY", "ACCUMULATE"):
        if pl <= -15:
            return objective_action, (
                f"Thesis still supports adding — this pulls your average cost down "
                f"while you're underwater ({pl:+.1f}% unrealized).")
        return objective_action, (
            f"Signals support adding to the position ({pl:+.1f}% unrealized).")

    if objective_action == "HOLD":
        if pl >= 20:
            return "REDUCE", (
                f"Objective signal is neutral, but with a {pl:+.1f}% unrealized gain, "
                "trimming to lock in profit is prudent even though the company itself "
                "isn't flashing a sell signal.")
        if pl <= -20:
            return objective_action, (
                f"You're down {pl:.1f}% on a neutral signal — patience may be warranted, "
                "but this is a good time to revisit whether the original thesis still holds.")
        return objective_action, f"Signal is neutral; unrealized P&L is {pl:+.1f}%."

    if objective_action == "REDUCE":
        if pl >= 0:
            return objective_action, (
                f"Signals are negative and you're still in profit ({pl:+.1f}%) — trimming "
                "now protects gains before further downside.")
        return objective_action, (
            f"Signals are negative and you're at a loss ({pl:.1f}%) — cutting losses may be "
            "prudent rather than waiting for a recovery the data doesn't support.")

    return objective_action, None


# ── Required-input guard ───────────────────────────────────────────────────

def _missing_required(rec: Dict[str, Any]) -> List[str]:
    missing = []
    if rec.get("current_price") is None:
        missing.append("current_price")
    if not rec.get("action"):
        missing.append("action")
    if rec.get("composite") is None:
        missing.append("composite")
    if not rec.get("confidence"):
        missing.append("confidence")
    if not rec.get("levels"):
        missing.append("levels")
    return missing


# ── Tier + action refinement (deterministic) ───────────────────────────────

def _tier(rec: Dict[str, Any]) -> str:
    """One of 'positive' / 'neutral' / 'negative' — independent of ownership.

    A risk veto can only push the tier DOWN (never up): it can turn 'positive' into
    'neutral', but a thesis that's already negative stays negative under a veto too."""
    action = rec.get("action", "HOLD")
    composite = rec.get("composite", 50)
    risk_veto = rec.get("risk_veto", False)

    if action in ("SELL", "REDUCE") or composite < 35:
        return "negative"

    if risk_veto:
        return "neutral"

    stat_level = (rec.get("confidence") or {}).get("statistical_edge", {}).get("level")
    if action in ("BUY", "ACCUMULATE") and stat_level in ("MEDIUM", "HIGH"):
        return "positive"
    return "neutral"


def _refine_action(tier: str, branch: str, rec: Dict[str, Any], shared: Dict[str, Any]) -> str:
    """branch is 'owned' or 'not_owned' — only matters for the neutral/negative tiers
    (BUY/ACCUMULATE read the same whether or not you already hold the position)."""
    if tier == "positive":
        stat_level = shared["confidence"]["statistical"].get("level")
        gated = rec.get("position_size_gated", True)
        risk_veto = rec.get("risk_veto", False)
        if stat_level == "HIGH" and not gated and not risk_veto:
            return "BUY"
        return "ACCUMULATE"
    if tier == "neutral":
        return "HOLD" if branch == "owned" else "WATCH"
    return "REDUCE" if branch == "owned" else "AVOID"


def _one_line_verdict(tier: str, rec: Dict[str, Any], shared: Dict[str, Any], top_action: str,
                      position_note: Optional[str] = None) -> str:
    if position_note:
        return f"{top_action}: {position_note}"
    composite = rec.get("composite")
    stat_level = shared["confidence"]["statistical"].get("level", "NONE")
    if tier == "positive":
        if top_action == "BUY":
            return (f"{top_action}: thesis is positive, statistical edge is HIGH, and sizing "
                     f"is supported — composite {composite}.")
        return (f"{top_action}: thesis is positive (composite {composite}) but statistical "
                 f"edge is only {stat_level}, so build gradually rather than sizing fully.")
    if tier == "neutral":
        if rec.get("risk_veto"):
            return (f"{top_action}: a risk veto is active (elevated/expanding volatility) — "
                     "conviction and sizing are capped regardless of the underlying score.")
        return (f"{top_action}: the company screens acceptably (composite {composite}) but "
                 f"statistical edge is {stat_level} and/or allocation is gated — not enough "
                 "proof yet to size a position.")
    if rec.get("risk_veto"):
        return (f"{top_action}: a risk veto is active and the underlying thesis is not "
                 "strong enough to override it.")
    return (f"{top_action}: thesis is negative (composite {composite}) — pillar consensus "
             "and/or backtest evidence argue against holding exposure here.")


# ── Three decisive insights (deduplicated by category, priority-ordered) ──

def _select_three_insights(rec: Dict[str, Any], shared: Dict[str, Any]) -> List[str]:
    candidates: List[tuple] = []

    if rec.get("risk_veto"):
        candidates.append(("risk_veto",
            "A risk veto is active — elevated/expanding volatility overrides pillar "
            "scores and caps conviction."))

    stat = shared["confidence"]["statistical"]
    stat_level = stat.get("level", "NONE")
    bt = shared["backtest"]
    dsr = bt.get("dsr")
    if stat_level == "HIGH":
        candidates.append(("statistical_edge",
            "Statistical edge is HIGH — the core strategy clears walk-forward, PBO, "
            "and cost-adjusted dSR bars."))
    else:
        dsr_txt = f" (dSR {dsr:.2f} vs 0.5 bar)" if dsr is not None else ""
        candidates.append(("statistical_edge",
            f"Statistical edge is {stat_level}{dsr_txt} — not yet proven beyond luck "
            "after multiple-testing correction."))

    xsec = shared["cross_sectional_rank"]
    if xsec.get("status") == "OK":
        candidates.append(("cross_sectional", xsec["interpretation"]))

    cal = shared["calibration"]
    if cal.get("status") == "INSUFFICIENT_HISTORY":
        candidates.append(("calibration",
            "No mature live track record yet — prediction accuracy hasn't been measured."))
    elif cal.get("status") == "OK":
        candidates.append(("calibration", cal["interpretation"]))

    if rec.get("position_size_gated"):
        candidates.append(("allocation",
            "Position sizing is gated to 0% until the statistical edge is established."))

    cons = shared["consensus"]
    if cons.get("n_conflicting", 0) >= 2 or cons.get("n_agreeing", 0) >= 3:
        candidates.append(("consensus", cons["summary"]))

    candidates.append(("backtest", bt["interpretation"]))

    scenarios = shared["scenarios"]
    if scenarios.get("thesis_breakers"):
        candidates.append(("thesis_breaker", scenarios["thesis_breakers"][0]))

    seen = set()
    picked: List[str] = []
    for category, text in candidates:
        if category in seen:
            continue
        seen.add(category)
        picked.append(text)
        if len(picked) == 3:
            break
    while len(picked) < 3:
        picked.append("No additional decisive evidence available.")
    return picked[:3]


# ── Action plan ─────────────────────────────────────────────────────────

def _build_action_plan(rec: Dict[str, Any], shared: Dict[str, Any], top_action: str) -> Dict[str, Any]:
    levels = rec.get("levels") or {}
    horizon_days = rec.get("time_horizon_days")
    gated = rec.get("position_size_gated", True)
    size_pct = rec.get("position_size_pct")
    stat_level = shared["confidence"]["statistical"].get("level")

    # Only show a price entry range when the action is actually to acquire/add exposure —
    # showing a "target entry" next to WATCH/REDUCE/AVOID would read as a buy signal it isn't.
    show_entry = top_action in ("BUY", "ACCUMULATE")
    entry_low = levels.get("entry_zone_low") if show_entry else None
    entry_high = levels.get("entry_zone_high") if show_entry else None

    if gated or stat_level not in ("MEDIUM", "HIGH") or top_action not in ("BUY", "ACCUMULATE"):
        max_exposure_pct = None
        max_exposure_note = "Unsupported — sizing withheld until the statistical edge is proven."
    else:
        max_exposure_pct = size_pct
        max_exposure_note = f"{size_pct}% (half-Kelly, capped)" if size_pct is not None else "—"

    review_note = (f"in ~{horizon_days} trading days or at the next earnings/catalyst, "
                    "whichever comes first") if horizon_days else "at the next earnings/catalyst"

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "instruction": _INSTRUCTION.get(top_action, "Monitor."),
        "time_horizon_days": horizon_days,
        "review_note": review_note,
        "max_exposure_pct": max_exposure_pct,
        "max_exposure_note": max_exposure_note,
    }


# ── Upgrade / downgrade triggers (measurable, deterministic) ──────────────

def _build_triggers(rec: Dict[str, Any], shared: Dict[str, Any]) -> Dict[str, List[str]]:
    stat = shared["confidence"]["statistical"]
    bt = shared["backtest"]
    dsr = bt.get("dsr")
    max_dd = bt.get("max_drawdown")
    xsec = shared["cross_sectional_rank"]

    upgrade: List[str] = []
    downgrade: List[str] = []

    if stat.get("level") != "HIGH":
        dsr_txt = f"; currently {dsr:.2f}" if dsr is not None else ""
        upgrade.append(f"Statistical edge reaches HIGH (dSR ≥ 0.5 across walk-forward "
                        f"+ PBO checks{dsr_txt})")
    if xsec.get("status") == "OK" and (xsec.get("percentile") or 0) < 0.8:
        upgrade.append("Cross-sectional rank improves into the top quintile "
                        "(≥80th percentile) of its universe")
    if rec.get("risk_veto"):
        upgrade.append("Risk veto clears (volatility regime normalizes, no longer expanding)")
    if not upgrade:
        upgrade.append("No further upgrade condition tracked — already at the highest "
                        "supported tier.")

    if dsr is not None:
        downgrade.append(f"dSR falls further below 0.5 or turns negative (currently {dsr:.2f})")
    if max_dd is not None:
        dd_bar = max(max_dd, 0.30)
        downgrade.append(f"Max drawdown exceeds {dd_bar*100:.0f}% (current: {max_dd:.0%})")
    if rec.get("risk_veto"):
        downgrade.append("Risk veto remains active past the review point")
    else:
        downgrade.append("Volatility regime shifts to HIGH and expanding "
                          "(would trigger a risk veto)")
    if not downgrade:
        downgrade.append("No further downgrade condition tracked.")

    return {"upgrade": upgrade[:3], "downgrade": downgrade[:3]}


# ── Evidence-quality strip ─────────────────────────────────────────────────

def _build_evidence_status(shared: Dict[str, Any]) -> Dict[str, str]:
    conf = shared["confidence"]
    xsec = shared["cross_sectional_rank"]
    cal = shared["calibration"]
    if xsec.get("status") == "OK":
        xsec_txt = f"#{xsec['rank']}/{xsec['n_securities']} ({xsec['percentile']*100:.0f}th pct)"
    else:
        xsec_txt = "not available"
    if cal.get("status") == "OK":
        cal_txt = f"{cal.get('n_predictions', 0)} matured predictions"
    else:
        cal_txt = "insufficient history"
    return {
        "thesis_confidence": conf["thesis"].get("level", "—"),
        "data_confidence": conf["data"].get("level", "—"),
        "statistical_edge": conf["statistical"].get("level", "—"),
        "cross_sectional_rank": xsec_txt,
        "calibration_maturity": cal_txt,
    }


# ── Better-ranked alternatives ─────────────────────────────────────────────

def _find_alternatives(ticker: str, xsec_ranking: Optional[Dict[str, Any]],
                        top_n: int = 3) -> Dict[str, Any]:
    if not xsec_ranking or xsec_ranking.get("status") != "OK":
        return {"status": "UNAVAILABLE", "items": []}

    ranked = xsec_ranking.get("ranked", [])
    ticker_entry = next(
        (r for r in ranked if (r.get("ticker_as_of") or "").upper() == ticker.upper()), None)
    if not ticker_entry:
        return {"status": "UNAVAILABLE", "items": []}

    my_rank = ticker_entry.get("rank", len(ranked) + 1)
    better = sorted((r for r in ranked if r.get("rank", 1 << 30) < my_rank),
                     key=lambda r: r.get("rank", 1 << 30))
    items = [{"ticker": r.get("ticker_as_of"), "rank": r.get("rank"),
              "percentile": r.get("composite_percentile")} for r in better[:top_n]]
    return {"status": "OK" if items else "NONE_BETTER", "items": items}


# ── Fingerprint (excludes the timestamp so repeated calls on the same
#    inputs are reproducible, matching decision_synthesis's convention) ────

def _fingerprint(brief: Dict[str, Any]) -> str:
    d = {
        "ticker": brief["ticker"],
        "final_action": brief["final_action"],
        "composite_score": brief["composite_score"],
        "confidence": brief["confidence"],
        "guidance": brief["guidance"],
        "recommendation_fingerprint": brief.get("recommendation_fingerprint"),
    }
    return hashlib.sha1(_json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
