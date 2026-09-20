"""
Position-management playbook (Phase 9) and monitoring plan (Phase 15).

The playbook is the answer to "what do I actually do, and when do I stop doing
it?" — expressed as CONDITIONS rather than instructions, because a condition
remains checkable after the fact and an instruction does not. Every condition
here is phrased against something this system already computes, which is what
makes the forward-validation framework (decision/forward.py) able to grade them
later.

Deterministic wherever possible: a condition is generated from a measured level,
a gate result, a budget comparison, or a named conflict. Where a condition can
only be stated qualitatively, it says so.

The monitoring plan turns those conditions into a schedule: what to check, how
often, and what a check would have to show to change the state. Review cadence
is derived from the volatility regime the recommendation engine already uses to
set its horizon, so the plan and the horizon can never drift apart.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

# Review cadence in calendar days, keyed to the same volatility regime that
# drives the recommendation's own time horizon (HIGH→45d, MEDIUM→91d, LOW→126d).
# A fast-moving name needs checking more often; this keeps the two consistent.
REVIEW_CADENCE_DAYS = {"HIGH": 7, "MEDIUM": 14, "LOW": 30}
DEFAULT_CADENCE_DAYS = 14


def _level_condition(level: Optional[Dict[str, Any]], verb: str) -> Optional[Dict[str, Any]]:
    if not level:
        return None
    return {
        "condition": f"Price {verb} {level['price']} ({level['distance_pct']:+.1f}%)",
        "measurable_as": f"daily close {'<' if 'below' in verb else '>'} {level['price']}",
        "basis": level["basis"],
        "level_sources": level.get("sources"),
        "confidence": level.get("confidence"),
        "deterministic": True,
    }


def build_playbook(
    decision_state: Dict[str, Any],
    thesis: Dict[str, Any],
    edge: Dict[str, Any],
    entry: Dict[str, Any],
    conflict: Dict[str, Any],
    position_context: Dict[str, Any],
    risk_budget: Dict[str, Any],
    add_analysis: Dict[str, Any],
    level_map: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
    mind_changers: Optional[Dict[str, Any]] = None,
    algo_signals: Optional[Dict[str, Any]] = None,
    horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """HOLD / ADD / REDUCE / EXIT conditions plus a DO-NOT-ADD list."""
    lm = level_map or {}
    support = lm.get("nearest_support") if lm.get("status") == "OK" else None
    major_support = lm.get("major_support") if lm.get("status") == "OK" else None
    resistance = lm.get("nearest_resistance") if lm.get("status") == "OK" else None

    owned_state = (decision_state.get("if_owned") or {}).get("state")

    # ── HOLD ─────────────────────────────────────────────────────────────
    hold: List[Dict[str, Any]] = []
    c = _level_condition(support, "holds above")
    if c:
        hold.append({**c, "currently_true": True,
                     "why": "The nearest level the market has defended is still intact."})
    if thesis["direction"] == "BULLISH":
        hold.append({"condition": f"The {thesis['strength'].lower()} bullish thesis stays intact "
                                  f"(net directional weight stays positive)",
                     "measurable_as": "thesis.net_weight > 0",
                     "basis": "EVIDENCE", "deterministic": True,
                     "currently_true": (thesis.get("net_weight") or 0) > 0,
                     "why": "The reason for owning it has not reversed."})
    n_dc = conflict.get("n_decision_changing", 0)
    hold.append({"condition": "No decision-changing conflict appears",
                 "measurable_as": "conflict.n_decision_changing == 0",
                 "basis": "EVIDENCE", "deterministic": True,
                 # A hold condition that is ALREADY false is the most important
                 # thing on the list, and printing it identically to the ones
                 # that hold would bury it.
                 "currently_true": n_dc == 0,
                 "why": (f"Currently {n_dc} decision-changing conflict(s) — this condition is "
                         f"already violated; holding is a judgement to sit through a named "
                         f"conflict, not an absence of one."
                         if n_dc else "No decision-changing conflict is currently named.")})
    if (risk_budget.get("position_risk") or {}).get("risk_budget_pct") is not None:
        hold.append({"condition": f"Portfolio risk at invalidation stays within "
                                  f"{risk_budget['position_risk']['risk_budget_pct']}%",
                     "measurable_as": "position_risk.portfolio_risk_if_invalidated_pct <= budget",
                     "basis": "RISK_BUDGET", "deterministic": True,
                     "currently_true": (risk_budget.get("position_risk") or {}).get(
                         "within_risk_budget") is not False,
                     "why": "Exposure stays inside what was agreed in advance."})

    # ── ADD ──────────────────────────────────────────────────────────────
    add: List[Dict[str, Any]] = []
    if add_analysis.get("status") == "OK":
        for cond in (add_analysis.get("conditions") or []):
            for a in cond.get("all_of", []):
                add.append({"condition": a, "measurable_as": "see add_analysis.checks",
                            "basis": "COMPOSITE", "deterministic": True,
                            "why": cond.get("why")})
    else:
        add.append({"condition": "POSITION CONTEXT NOT PROVIDED — add conditions cannot be "
                                 "evaluated without an existing position",
                    "measurable_as": None, "basis": "MISSING_INPUT", "deterministic": False,
                    "why": "Adding is a question about an existing holding."})
    if not edge.get("demonstrated"):
        add.insert(0, {"condition": "The statistical gate clears (sizing stops being 0%)",
                       "measurable_as": "confidence.statistical_edge.level == 'HIGH'",
                       "basis": "STATISTICAL_GATE", "deterministic": True,
                       "why": f"Currently {edge.get('n_gates_passed')}/{edge.get('n_gates')} "
                              f"gates pass; sizing is gated at 0% and that applies to an "
                              f"addition exactly as to a new position."})

    # ── REDUCE ───────────────────────────────────────────────────────────
    reduce: List[Dict[str, Any]] = []
    if (risk_budget.get("position_risk") or {}).get("within_risk_budget") is False:
        reduce.append({"condition": "Position risk already exceeds the stated budget",
                       "measurable_as": "position_risk.within_risk_budget == False",
                       "basis": "RISK_BUDGET", "deterministic": True,
                       "why": (risk_budget.get("position_risk") or {}).get("budget_note"),
                       "currently_true": True})
    if position_context.get("concentration") == "CONCENTRATED":
        reduce.append({"condition": f"Position weight stays above "
                                    f"{position_context.get('weight_pct')}% of the portfolio",
                       "measurable_as": "position weight > 25%",
                       "basis": "RISK_BUDGET", "deterministic": True,
                       "why": "Single-name concentration, independent of the thesis.",
                       "currently_true": True})
    reduce.append({"condition": "The thesis weakens materially (net directional weight falls "
                                "toward or through zero)",
                   "measurable_as": "thesis.net_weight declines below 0.30",
                   "basis": "EVIDENCE", "deterministic": True,
                   "why": f"Currently {thesis.get('net_weight'):+.2f}."})
    reduce.append({"condition": "Volatility regime moves to HIGH and expanding (risk veto fires)",
                   "measurable_as": "recommendation.risk_veto == True",
                   "basis": "RISK", "deterministic": True,
                   "why": f"Currently regime "
                          f"{(algo_signals or {}).get('vol_regime')}, expanding="
                          f"{(algo_signals or {}).get('vol_expanding')}."})

    # ── EXIT ─────────────────────────────────────────────────────────────
    exit_conds: List[Dict[str, Any]] = []
    c = _level_condition(support, "closes below")
    if c:
        exit_conds.append({**c, "why": "Thesis invalidation — the nearest defended level failed.",
                           "category": "THESIS_INVALIDATION"})
    c = _level_condition(major_support, "closes below")
    if c and (not support or major_support["price"] != support["price"]):
        exit_conds.append({**c, "why": "Structural break — the strongest nearby level the "
                                       "market has traded at failed.",
                           "category": "STRUCTURAL_BREAK"})
    exit_conds.append({"condition": "The statistical picture deteriorates further "
                                    "(walk-forward mean OOS Sharpe turns negative)",
                       "measurable_as": "statistical_edge.checks.walk_forward.mean_oos_sharpe < 0",
                       "basis": "STATISTICAL_GATE", "deterministic": True,
                       "why": "The method itself stops being defensible on this name.",
                       "category": "STATISTICAL_DETERIORATION"})
    nxt = (catalysts or {}).get("next_event")
    if nxt:
        exit_conds.append({"condition": f"{nxt['event']} on {nxt['date']} resolves against the "
                                        f"thesis and price fails to reclaim the nearest level",
                           "measurable_as": "post-event close below the nearest sourced support",
                           "basis": "CATALYST", "deterministic": False,
                           "why": "Catalyst failure — the event the thesis was waiting on went "
                                  "the other way.",
                           "category": "CATALYST_FAILURE"})
    if (risk_budget.get("position_risk") or {}).get("risk_budget_pct") is not None:
        exit_conds.append({"condition": "Portfolio risk at invalidation breaches the stated "
                                        "budget and cannot be brought back by trimming",
                           "measurable_as": "position_risk.portfolio_risk_if_invalidated_pct > budget",
                           "basis": "RISK_BUDGET", "deterministic": True,
                           "why": "Risk violation.", "category": "RISK_VIOLATION"})

    # ── DO NOT ADD ───────────────────────────────────────────────────────
    do_not_add: List[Dict[str, Any]] = []
    for b in (add_analysis.get("blockers") or []):
        do_not_add.append({"condition": b, "basis": "ADD_ANALYSIS", "deterministic": True})
    if add_analysis.get("cheaper_not_better"):
        do_not_add.insert(0, {
            "condition": "The only thing that has improved is the price",
            "basis": "ADD_ANALYSIS", "deterministic": True,
            "why": add_analysis.get("rule")})
    if not do_not_add:
        do_not_add.append({"condition": "No blocking condition is currently active",
                           "basis": "ADD_ANALYSIS", "deterministic": True})

    # ── Evidence for and against the current position ────────────────────
    evidence_for = (thesis.get("supporting_evidence") or [])[:4]
    evidence_against = (thesis.get("opposing_evidence") or [])[:4]

    # ── Review date ──────────────────────────────────────────────────────
    regime = (algo_signals or {}).get("vol_regime")
    cadence = REVIEW_CADENCE_DAYS.get(regime, DEFAULT_CADENCE_DAYS)
    review_date = date.today() + timedelta(days=cadence)
    catalyst_date = (nxt or {}).get("date")
    if catalyst_date and catalyst_date < review_date.isoformat():
        review_date_str, review_reason = catalyst_date, f"{nxt['event']} falls first."
    else:
        review_date_str = review_date.isoformat()
        review_reason = (f"Volatility regime {regime or 'unknown'} → {cadence}-day review "
                         f"cadence.")

    violated = [c for c in hold if c.get("currently_true") is False]

    return {
        "position_status": owned_state,
        "hold_conditions_already_violated": violated,
        "n_hold_conditions_violated": len(violated),
        "position_status_meaning": (decision_state.get("if_owned") or {}).get("meaning"),
        "why": {
            "evidence_supporting_current_position": evidence_for,
            "evidence_against_current_position": evidence_against,
            "summary": (decision_state.get("if_owned") or {}).get("reason"),
        },
        "hold_conditions": hold,
        "add_conditions": add,
        "reduce_conditions": reduce,
        "exit_conditions": exit_conds,
        "do_not_add_conditions": do_not_add,
        "thesis_invalidation": [e for e in exit_conds
                                if e.get("category") == "THESIS_INVALIDATION"],
        "catalyst_watch": ((catalysts or {}).get("upcoming") or [])[:3],
        "review_date": review_date_str,
        "review_reason": review_reason,
        "note": ("Conditions, not instructions. Each is phrased against something this system "
                 "computes, so it can be checked later rather than merely read now."),
    }


def build_monitoring_plan(playbook: Dict[str, Any], mind_changers: Dict[str, Any],
                          catalysts: Optional[Dict[str, Any]] = None,
                          algo_signals: Optional[Dict[str, Any]] = None,
                          level_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What to check, when to check it, and what would change the state."""
    regime = (algo_signals or {}).get("vol_regime")
    cadence = REVIEW_CADENCE_DAYS.get(regime, DEFAULT_CADENCE_DAYS)

    checks: List[Dict[str, Any]] = []

    lm = level_map or {}
    if lm.get("status") == "OK":
        for label, lv in (("support", lm.get("nearest_support")),
                          ("resistance", lm.get("nearest_resistance"))):
            if lv:
                checks.append({
                    "what": f"Daily close against nearest {label} {lv['price']}",
                    "when": "Every trading day",
                    "changes_state_if": (f"A close below {lv['price']} → invalidation"
                                         if label == "support" else
                                         f"A close above {lv['price']} → confirmation"),
                    "source": "price feed",
                    "automatable": True,
                })

    checks.append({
        "what": "Statistical gate status (walk-forward, PBO, dSR, sample)",
        "when": f"Every {cadence} days, or after any material price move",
        "changes_state_if": "Level reaches HIGH → position sizing unlocks",
        "source": "agents/recommendation.py::_assess_statistical_edge",
        "automatable": True,
    })
    checks.append({
        "what": "Evidence balance across horizons",
        "when": f"Every {cadence} days",
        "changes_state_if": "Net directional weight crosses zero, or agreement ratio falls "
                            "below 55% → CONFLICTED",
        "source": "decision/horizons.py + decision/conflict.py",
        "automatable": True,
    })
    nxt = (catalysts or {}).get("next_event")
    if nxt:
        checks.append({
            "what": f"{nxt['event']} outcome",
            "when": f"On {nxt['date']} ({nxt['days_away']} days away)",
            "changes_state_if": "Result and guidance versus the consensus range already stated",
            "source": "catalyst calendar",
            "automatable": False,
        })
    checks.append({
        "what": "Live calibration sample",
        "when": "Weekly (the heartbeat already refreshes matured outcomes)",
        "changes_state_if": "Sample reaches a readable size → stated confidence becomes "
                            "measurable rather than asserted",
        "source": "data/prediction_ledger.py",
        "automatable": True,
    })

    return {
        "review_date": playbook.get("review_date"),
        "review_reason": playbook.get("review_reason"),
        "cadence_days": cadence,
        "checks": checks,
        "confirmation_triggers": (mind_changers or {}).get("confirmation", []),
        "invalidation_triggers": (mind_changers or {}).get("invalidation", []),
        "what_am_i_waiting_for": (catalysts or {}).get(
            "waiting_for", "No catalyst calendar was consulted."),
    }
