"""
Decision quality (Phase 18) — a structured object, not another 0-100 score.

Another composite number would be the same mistake this whole layer exists to
correct: it would let a strong component conceal a missing one, and it would
give the reader no way to act on the weakness. So quality is reported as eight
named components, each with a status, an explanation and the provenance of the
thing being judged.

Statuses are deliberately blunt: GOOD / ADEQUATE / WEAK / MISSING. A component
that could not be computed is MISSING, never a low score — "we don't know" and
"we know it is bad" are different facts and must not be averaged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

STATUSES = ("GOOD", "ADEQUATE", "WEAK", "MISSING")
_RANK = {s: i for i, s in enumerate(("MISSING", "WEAK", "ADEQUATE", "GOOD"))}

COMPONENTS = ("DATA", "EVIDENCE", "THESIS", "EDGE", "RISK", "CALIBRATION",
              "CATALYST", "POSITION_CONTEXT")


def _c(status: str, explanation: str, provenance: str) -> Dict[str, Any]:
    return {"status": status, "explanation": explanation, "provenance": provenance}


def assess_decision_quality(
    confidence: Dict[str, Any],
    edge: Dict[str, Any],
    conflict: Dict[str, Any],
    risk_budget: Dict[str, Any],
    catalysts: Optional[Dict[str, Any]],
    position_context: Dict[str, Any],
    level_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dims = confidence.get("dimensions", {})

    def from_level(level: str) -> str:
        return {"HIGH": "GOOD", "MEDIUM": "ADEQUATE", "LOW": "WEAK",
                "INSUFFICIENT": "MISSING", "NONE": "MISSING"}.get(level, "MISSING")

    components: Dict[str, Dict[str, Any]] = {}

    components["DATA"] = _c(
        from_level(dims.get("data", {}).get("level", "NONE")),
        dims.get("data", {}).get("detail") or "",
        "agents/recommendation.py::_data_confidence + decision/evidence.py data_quality")

    components["EVIDENCE"] = _c(
        from_level(dims.get("evidence", {}).get("level", "NONE")),
        dims.get("evidence", {}).get("detail") or "",
        "decision/conflict.py::analyze_conflicts")

    components["THESIS"] = _c(
        from_level(dims.get("thesis", {}).get("level", "NONE")),
        dims.get("thesis", {}).get("detail") or "",
        "decision/thesis.py::build_thesis")

    components["EDGE"] = _c(
        "GOOD" if edge.get("demonstrated") else
        ("MISSING" if edge.get("level") == "NONE" else "WEAK"),
        (edge.get("basis") or "") + " " + (edge.get("sizing_consequence") or ""),
        "agents/recommendation.py::_assess_statistical_edge")

    risk_ok = risk_budget.get("invalidation_level") is not None
    risk_basis = risk_budget.get("invalidation_basis")
    if not risk_ok:
        risk_status = "MISSING"
        risk_expl = "No invalidation level could be sourced, so downside is unpriced."
    elif risk_basis == "TRADED_PRICE":
        risk_status = "GOOD"
        risk_expl = (f"Invalidation at {risk_budget['invalidation_level']} "
                     f"({risk_budget.get('risk_to_invalidation_pct')}%), a price the market "
                     f"has actually traded at.")
    elif risk_basis == "PRICE_HISTORY_DERIVED":
        risk_status = "ADEQUATE"
        risk_expl = (f"Invalidation at {risk_budget['invalidation_level']} from a line computed "
                     f"over price history, not a traded price.")
    else:
        risk_status = "WEAK"
        risk_expl = (f"Invalidation at {risk_budget['invalidation_level']} is derived from "
                     f"today's price and moves with it — a weak marker.")
    if (risk_budget.get("position_risk") or {}).get("status") == "POSITION_SIZING_NOT_COMPUTABLE":
        risk_expl += " Portfolio-level risk is not computable without a portfolio value."
    components["RISK"] = _c(risk_status, risk_expl,
                            "decision/levels.py + decision/position.py::build_risk_budget")

    components["CALIBRATION"] = _c(
        from_level(dims.get("calibration", {}).get("level", "NONE")),
        dims.get("calibration", {}).get("detail") or "",
        "data/prediction_ledger.py::calibration_report")

    cat_status = (catalysts or {}).get("status")
    if cat_status == "OK":
        n = (catalysts or {}).get("n_upcoming", 0)
        components["CATALYST"] = _c(
            "GOOD" if n else "ADEQUATE",
            (catalysts or {}).get("waiting_for", ""),
            "decision/catalysts.py (yfinance calendar)")
    elif cat_status == "NO_EVENTS":
        components["CATALYST"] = _c(
            "ADEQUATE",
            "The calendar was read and nothing is scheduled — an answer, not a gap.",
            "decision/catalysts.py")
    else:
        components["CATALYST"] = _c(
            "MISSING",
            "No catalyst calendar could be read, so the plan has no known checkpoint.",
            "decision/catalysts.py")

    pc_status = position_context.get("status")
    if pc_status == "PROVIDED":
        missing = position_context.get("missing") or []
        components["POSITION_CONTEXT"] = _c(
            "GOOD" if not missing else "ADEQUATE",
            ("Cost basis, share count and portfolio value all supplied."
             if not missing else
             f"Cost basis supplied; {', '.join(missing)} missing, so "
             f"{'dollar risk and ' if 'shares' in missing else ''}"
             f"{'portfolio weight ' if 'portfolio_value' in missing else ''}"
             f"cannot be computed."),
            "caller-supplied position")
    else:
        components["POSITION_CONTEXT"] = _c(
            "MISSING",
            position_context.get("message", "No position information supplied."),
            "caller-supplied position")

    weakest = min(components.items(), key=lambda kv: _RANK[kv[1]["status"]])
    missing_list = [k for k, v in components.items() if v["status"] == "MISSING"]
    weak_list = [k for k, v in components.items() if v["status"] == "WEAK"]

    return {
        "components": components,
        "weakest_component": weakest[0],
        "weakest_status": weakest[1]["status"],
        "missing": missing_list,
        "weak": weak_list,
        "n_good": sum(1 for v in components.values() if v["status"] == "GOOD"),
        "n_components": len(components),
        "summary": (f"{sum(1 for v in components.values() if v['status'] == 'GOOD')}/"
                    f"{len(components)} components good; weakest is {weakest[0]} "
                    f"({weakest[1]['status']})"
                    + (f"; missing: {', '.join(missing_list)}" if missing_list else "")),
        "note": ("Deliberately not a single number. A composite would let a strong component "
                 "conceal a missing one, which is the failure this whole layer exists to fix."),
    }
