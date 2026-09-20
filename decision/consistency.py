"""
Consistency checks (Phase 26).

Every check here corresponds to a contradiction that was observed in this app's
own live output, not to a hypothetical. They run over the assembled decision
object and emit explicit warnings, so a contradiction reaches the reader as a
labelled warning instead of as two confident numbers sitting next to each other.

Severity:
  ERROR    two outputs state incompatible things; at least one is wrong
  WARNING  the outputs are individually defensible but read as contradictory
  INFO     a known asymmetry worth naming so it is not mistaken for a bug

These checks are also the regression net for the whole layer: if a future change
reintroduces one of these contradictions, the check fires rather than the
contradiction shipping silently.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

BULLISH_STATES = ("ADD_CONDITIONALLY", "WAIT_FOR_ENTRY")
NEGATIVE_STATES = ("REDUCE", "EXIT_CONDITIONALLY", "AVOID_NEW_POSITION")


def _issue(code: str, severity: str, message: str, detail: str,
           fields: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"code": code, "severity": severity, "message": message,
            "detail": detail, "fields": fields or []}


def check_consistency(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Run every check against a fully assembled Decision Intelligence object."""
    issues: List[Dict[str, Any]] = []

    rec = decision.get("_recommendation") or {}
    state = decision.get("decision_state") or {}
    edge = decision.get("statistical_edge") or {}
    entry = decision.get("entry") or {}
    conf = decision.get("confidence") or {}
    levels = decision.get("level_map") or {}
    sizing = decision.get("sizing") or {}
    scenarios = decision.get("scenarios") or {}
    price = decision.get("current_price")

    headline = state.get("headline_state")

    # 1. A directional action while the gate says there is no demonstrated edge.
    if headline in BULLISH_STATES and not edge.get("demonstrated"):
        if headline == "ADD_CONDITIONALLY":
            issues.append(_issue(
                "ACTION_WITHOUT_EDGE", "ERROR",
                "Decision state recommends adding while the statistical gate reports no "
                "demonstrated edge.",
                f"state={headline}, statistical_edge={edge.get('level')}. Adding size on an "
                f"unproven signal is the exact configuration the gate exists to prevent.",
                ["decision_state.headline_state", "statistical_edge.level"]))
        else:
            issues.append(_issue(
                "ENTRY_PLAN_WITHOUT_EDGE", "WARNING",
                "An entry plan is stated while no statistical edge is demonstrated.",
                f"state={headline}, statistical_edge={edge.get('level')}. The plan describes "
                f"WHERE to enter if one chooses to; it does not assert that entering is "
                f"supported. Position size remains 0%.",
                ["decision_state.headline_state", "statistical_edge.level"]))

    # 2. Non-zero sizing while the sizing gate is closed.
    size_pct = sizing.get("position_size_pct")
    if sizing.get("gated") and size_pct not in (None, 0, 0.0):
        issues.append(_issue(
            "SIZING_GATE_VIOLATED", "ERROR",
            "A non-zero position size is shown while the sizing gate reports 0%.",
            f"position_size_pct={size_pct} with gated={sizing.get('gated')}.",
            ["sizing.position_size_pct", "sizing.gated"]))

    # 3. Headline confidence HIGH while calibration is insufficient.
    cal_level = ((conf.get("dimensions") or {}).get("calibration") or {}).get("level")
    if conf.get("decision_confidence") == "HIGH" and cal_level in ("INSUFFICIENT", "NONE", "LOW"):
        issues.append(_issue(
            "CONFIDENCE_WITHOUT_CALIBRATION", "ERROR",
            "Decision confidence reads HIGH while calibration is insufficient or unreliable.",
            f"decision_confidence=HIGH, calibration={cal_level}. Stated confidence has not "
            f"been measured against outcomes.",
            ["confidence.decision_confidence", "confidence.dimensions.calibration.level"]))

    # 4. An entry level at or above the current price presented as an entry.
    plan = decision.get("entry_plan") or {}
    entry_low = plan.get("entry_low")
    if entry_low is not None and price and entry_low >= price and entry.get("status") in (
            "WAIT", "OVEREXTENDED", "HIGH_RISK", "INVALIDATED"):
        issues.append(_issue(
            "ENTRY_AT_PRICE_WHILE_WAITING", "ERROR",
            "An entry level at or above the current price is shown while the entry engine "
            "says to wait.",
            f"entry_low={entry_low}, current_price={price}, entry status={entry.get('status')}. "
            f"An entry band that contains the current price answers 'is now a good entry?' "
            f"with yes by construction.",
            ["entry_plan.entry_low", "entry.status"]))

    # 5. Upside framing on a target below the current price.
    for s in scenarios.get("scenarios", []):
        if s.get("expected_direction") != "UP":
            continue
        for lv in s.get("price_levels", []):
            if lv.get("price") is not None and price and lv["price"] < price:
                issues.append(_issue(
                    "TARGET_BELOW_PRICE", "ERROR",
                    "A bull-case price reference sits below the current price.",
                    f"level={lv['price']} < current_price={price}.",
                    ["scenarios.BULL.price_levels"]))
                break

    # 6. Owned and not-owned branches identical when they should differ.
    owned = (state.get("if_owned") or {}).get("state")
    not_owned = (state.get("if_not_owned") or {}).get("state")
    if owned and not_owned and owned == not_owned and owned not in (
            "CONFLICTED", "INSUFFICIENT_DATA"):
        issues.append(_issue(
            "OWNERSHIP_BRANCHES_IDENTICAL", "WARNING",
            "The owned and not-owned branches produced the same state.",
            f"both={owned}. These are different decision problems; identical answers are "
            f"possible but worth checking rather than assuming.",
            ["decision_state.if_owned.state", "decision_state.if_not_owned.state"]))

    # 7. The legacy recommendation verb disagreeing with the decision state.
    rec_action = rec.get("action")
    if rec_action in ("BUY", "ACCUMULATE") and headline in NEGATIVE_STATES:
        issues.append(_issue(
            "REC_VERB_VS_STATE", "WARNING",
            "The composite recommendation verb and the decision state point opposite ways.",
            f"recommendation.action={rec_action} (composite {rec.get('composite')}) versus "
            f"decision state {headline}. The composite scores the security; the decision "
            f"state additionally requires a demonstrated edge and a workable entry. Both "
            f"are shown so the difference is visible rather than silently resolved.",
            ["_recommendation.action", "decision_state.headline_state"]))
    elif rec_action in ("SELL", "REDUCE") and headline in BULLISH_STATES:
        issues.append(_issue(
            "REC_VERB_VS_STATE", "ERROR",
            "The composite verb is negative while the decision state is constructive.",
            f"recommendation.action={rec_action} versus decision state {headline}.",
            ["_recommendation.action", "decision_state.headline_state"]))

    # 8. A probability shown without calibration.
    for s in scenarios.get("scenarios", []):
        if s.get("probability") is not None and s.get("probability_basis") != "CALIBRATED":
            issues.append(_issue(
                "UNCALIBRATED_PROBABILITY", "ERROR",
                "A numeric probability is displayed without a calibrated basis.",
                f"scenario={s.get('name')}, basis={s.get('probability_basis')}.",
                ["scenarios.probability"]))

    # 9. Levels presented as support/resistance that are only ATR arithmetic.
    if levels.get("status") == "OK" and levels.get("n_observed", 0) == 0:
        issues.append(_issue(
            "NO_OBSERVED_LEVELS", "WARNING",
            "No level in the ladder is a price the market has actually traded at.",
            "Every level is derived from moving averages, bands or ATR arithmetic. Treat the "
            "entry and invalidation geometry as approximate.",
            ["level_map.n_observed"]))

    # 10. A demonstrated edge claim alongside a dSR below the bar.
    dsr = ((edge.get("backtest") or {}).get("dsr"))
    if edge.get("demonstrated") and dsr is not None and dsr < 0.5:
        issues.append(_issue(
            "EDGE_CLAIM_VS_DSR", "ERROR",
            "The edge is reported as demonstrated while the deflated Sharpe is below its bar.",
            f"demonstrated=True, dsr={dsr} < 0.5.",
            ["statistical_edge.demonstrated", "statistical_edge.backtest.dsr"]))

    # 11. A reduce/exit state with no invalidation level to act on.
    if headline in ("REDUCE", "EXIT_CONDITIONALLY"):
        rb = decision.get("risk_budget") or {}
        if rb.get("invalidation_level") is None:
            issues.append(_issue(
                "EXIT_WITHOUT_LEVEL", "WARNING",
                "An exit-oriented state is recommended with no priced invalidation level.",
                "The reader is told to consider exiting but not at what price the thesis is "
                "actually broken.",
                ["decision_state.headline_state", "risk_budget.invalidation_level"]))

    # 12. INFO: the known, deliberate asymmetry between composite and state.
    if rec_action and headline and rec_action not in ("SELL", "REDUCE") \
            and headline in ("WATCH", "NO_TRADE"):
        issues.append(_issue(
            "COMPOSITE_SCORES_SECURITY_NOT_DECISION", "INFO",
            "The composite screens the security; the decision state additionally requires a "
            "demonstrated edge.",
            f"composite={rec.get('composite')} with action {rec_action}, decision state "
            f"{headline}. This is by design, not a disagreement to reconcile away.",
            ["_recommendation.composite", "decision_state.headline_state"]))

    errors = [i for i in issues if i["severity"] == "ERROR"]
    warnings_ = [i for i in issues if i["severity"] == "WARNING"]

    return {
        "issues": issues,
        "n_errors": len(errors),
        "n_warnings": len(warnings_),
        "passed": len(errors) == 0,
        "summary": ("No consistency errors." if not errors else
                    f"{len(errors)} consistency ERROR(s): "
                    + ", ".join(e["code"] for e in errors))
                   + (f" {len(warnings_)} warning(s)." if warnings_ else ""),
    }
