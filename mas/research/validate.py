"""A final deterministic pass over the answer, looking for self-contradiction.

Every check here corresponds to a contradiction that is individually
plausible, collectively impossible, and invisible unless something looks for
it. They are cheap, they run last, and they fail loudly — a surfaced
contradiction is a usable warning, while a silent one is a wrong answer that
reads perfectly well.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ERROR = "ERROR"
WARNING = "WARNING"

BULLISH_ACTIONS = {"BUY", "ACCUMULATE", "ADD", "ADD_CONDITIONALLY",
                   "ENTER_CONDITIONALLY"}

# Which evidence ITEMS satisfy which required evidence KIND. Stated rather
# than inferred from string shape, so a renamed item fails a test instead of
# silently satisfying nothing.
SATISFIED_BY = {
    "thesis": {"research:composite"},
    "technical_structure": {"research:composite", "research:levels"},
    "levels": {"research:levels"},
    "risk": {"risk:volatility"},
    "statistical_edge": {"stats:backtest"},
    "catalysts": {"catalyst:next"},
    "benchmark_relation": {"relative:benchmark"},
    "scenarios": {"research:scenarios"},
    "regime": {"macro:regime"},
    "option_structures": {"options:structures"},
    "forward_record": {"stats:forward_record"},
}


def _issue(code: str, severity: str, message: str, detail: str,
           fields: List[str]) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message,
            "detail": detail, "fields": fields}


def validate(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Contradictions in a finished research answer. Never raises."""
    issues: List[Dict[str, Any]] = []
    syn = answer.get("synthesis") or {}
    ev = (answer.get("evidence") or {}).get("items") or []
    plan = answer.get("plan") or {}
    decision = answer.get("decision") or {}
    fresh = answer.get("freshness") or {}

    by_key = {i.get("key"): i for i in ev}

    # 1. A directional call on top of a failed statistical gate.
    edge = by_key.get("stats:backtest") or {}
    action = str(decision.get("state") or decision.get("action") or "").upper()
    if action in BULLISH_ACTIONS and "no_demonstrated_edge" in (edge.get("flags") or []):
        issues.append(_issue(
            "ACTION_WITHOUT_EDGE", WARNING,
            "A directional action is stated while no statistical edge was demonstrated.",
            f"decision={action} but the backtest shows nothing beat holding. "
            f"Both are shown so the difference is visible rather than resolved "
            f"silently; the gate constrains size, it does not veto a view.",
            ["decision.state", "stats:backtest"]))

    # 2. Agreement claimed from a single source.
    if syn.get("consensus") in ("STRONG_AGREEMENT", "AGREEMENT") \
            and (syn.get("n_directional_sources") or 0) < 2:
        issues.append(_issue(
            "AGREEMENT_OF_ONE", ERROR,
            "Consensus is claimed from a single directional source.",
            f"consensus={syn.get('consensus')} with "
            f"{syn.get('n_directional_sources')} source(s). One source "
            f"agreeing with itself is not agreement.",
            ["synthesis.consensus", "synthesis.n_directional_sources"]))

    # 3. A probability stated without calibration.
    probs = answer.get("scenarios") or {}
    if probs.get("any_probability_stated") and not probs.get("calibrated"):
        issues.append(_issue(
            "UNCALIBRATED_PROBABILITY", ERROR,
            "A probability is shown without a calibrated basis.",
            "Scenario probabilities may only be displayed when measured. "
            "Otherwise the honest field is PROBABILITY NOT CALIBRATED.",
            ["scenarios.any_probability_stated"]))

    # 4. A catalyst outside the horizon described as bearing on it.
    cat = by_key.get("catalyst:next") or {}
    days = cat.get("value")
    hz_days = (plan.get("horizon") or {}).get("days")
    if isinstance(days, (int, float)) and isinstance(hz_days, (int, float)):
        if days > hz_days and "catalyst" in str(decision.get("why", "")).lower():
            issues.append(_issue(
                "CATALYST_OUTSIDE_HORIZON", ERROR,
                "A catalyst beyond the horizon is cited as bearing on it.",
                f"The next event is {days:.0f} days out; this plan's horizon "
                f"is {hz_days:.0f} days. It cannot move a position that will "
                f"be closed before it happens.",
                ["catalyst:next", "plan.horizon.days"]))

    # 5. A stale price presented as current.
    if fresh.get("stale") and not fresh.get("statement"):
        issues.append(_issue(
            "STALE_PRICE_UNLABELLED", ERROR,
            "The price used has drifted and nothing says so.",
            "A price the market has moved past, shown without its age, is the "
            "exact defect the freshness layer exists to make visible.",
            ["freshness"]))

    # 6. An unavailable tool represented as evidence.
    for i in ev:
        if "unavailable" in (i.get("flags") or []) and i.get("usable_for_decision"):
            issues.append(_issue(
                "UNAVAILABLE_AS_EVIDENCE", ERROR,
                "An unavailable source is being counted as evidence.",
                f"{i.get('key')} is flagged unavailable yet is marked usable "
                f"for a decision. Missing data is not a neutral reading.",
                [str(i.get("key"))]))

    # 7. An LLM item admitted to the weighing.
    for i in ev:
        if i.get("tier") in ("LLM_EXPLANATION", "INTERPRETATION") \
                and i.get("usable_for_decision"):
            issues.append(_issue(
                "PROSE_AS_DECISION_EVIDENCE", ERROR,
                "Interpretation is being weighed as decision evidence.",
                f"{i.get('key')} is tier {i.get('tier')} and must inform the "
                f"wording only. Numbers come from deterministic components.",
                [str(i.get("key"))]))

    # 8. A position question answered without a position.
    if plan.get("intent") in ("ADD_TO_POSITION", "REDUCE_POSITION",
                              "EXIT_POSITION") \
            and not (plan.get("position") or {}).get("owns") \
            and answer.get("answerable"):
        issues.append(_issue(
            "POSITION_QUESTION_WITHOUT_POSITION", ERROR,
            "A position-management question was answered without a position.",
            f"intent={plan.get('intent')} but no holding was supplied or "
            f"stated. The thing being decided about was never established.",
            ["plan.intent", "plan.position"]))

    # 9. Required evidence missing while the answer is presented as complete.
    # Satisfaction is checked through an explicit map. The first version
    # compared evidence KINDS ("thesis", "levels") against item key PREFIXES
    # ("research", "catalyst") — two different namespaces that never matched,
    # so the warning fired on every answer and therefore meant nothing.
    required = set(plan.get("required_evidence") or [])
    produced_keys = {str(i.get("key", "")) for i in ev
                     if "unavailable" not in (i.get("flags") or [])}
    unmet = sorted(
        k for k in required
        if k != "position_context"
        and not (SATISFIED_BY.get(k, set()) & produced_keys))
    if unmet and answer.get("answerable") and not answer.get("degraded"):
        issues.append(_issue(
            "REQUIRED_EVIDENCE_MISSING", WARNING,
            "Required evidence did not arrive and the answer is not marked degraded.",
            f"Missing: {', '.join(unmet)}. The answer may still be useful, but "
            f"it is thinner than the plan called for and should say so.",
            ["plan.required_evidence"]))

    n_err = sum(1 for i in issues if i["severity"] == ERROR)
    n_warn = len(issues) - n_err
    return {
        "issues": issues, "n_errors": n_err, "n_warnings": n_warn,
        "passed": n_err == 0,
        "summary": (f"{n_err} contradiction(s) and {n_warn} warning(s)."
                    if issues else
                    "Every cross-check agreed; nothing contradicts anything else."),
        "checks_run": 9,
    }
