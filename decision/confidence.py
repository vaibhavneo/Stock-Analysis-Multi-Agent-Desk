"""
Decomposed decision confidence (Phase 16).

One HIGH/MEDIUM/LOW badge is worse than none, because the reader has no way to
know which part of the analysis it refers to. A dataset can be pristine while
the edge is absent; a backtest can be strong while the live calibration says the
stated confidences are unreliable. Compressing those into one word lets the
strongest dimension stand in for the weakest, which is exactly backwards — a
chain is as strong as its weakest link, and a decision is as trustworthy as its
least trustworthy input.

So seven dimensions are reported separately, and the overall decision confidence
is derived by a stated rule that CANNOT exceed the weakest decision-critical
dimension. That rule is the whole point: it is impossible for a strong
data-quality score to lift the headline while the edge is unproven.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LEVELS = ("NONE", "INSUFFICIENT", "LOW", "MEDIUM", "HIGH")
_ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}

# Dimensions that CAP the headline. The others inform it but cannot hold it
# down on their own — a thin catalyst calendar should not make a well-evidenced,
# well-validated read read as low confidence.
CAPPING_DIMENSIONS = ("data", "evidence", "statistical", "validation")


def _min_level(levels: List[str]) -> str:
    present = [l for l in levels if l in _ORDER]
    return min(present, key=lambda l: _ORDER[l]) if present else "NONE"


def _dim(level: str, basis: str, detail: Optional[str] = None) -> Dict[str, Any]:
    return {"level": level, "basis": basis, "detail": detail}


def decompose_confidence(
    rec: Dict[str, Any],
    items,
    thesis: Dict[str, Any],
    edge: Dict[str, Any],
    conflict: Dict[str, Any],
    scenarios: Optional[Dict[str, Any]] = None,
    position_context: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Seven dimensions plus a derived headline that cannot exceed the weakest
    decision-critical one."""
    conf = rec.get("confidence") or {}

    # 1. Data — reuse the recommendation's own data-confidence dimension rather
    #    than re-deriving one, then downgrade it if normalized evidence found
    #    missing or stale inputs the pillar-level check does not see.
    data_level = (conf.get("data") or {}).get("level", "LOW")
    missing = [q for q in (conflict.get("evidence_quality_issues") or [])
               if q["issue"] in ("MISSING", "STALE")]
    if missing and data_level == "HIGH":
        data_level = "MEDIUM"
    data = _dim(data_level, (conf.get("data") or {}).get("basis", ""),
                (f"{len(missing)} input(s) missing or stale: "
                 + ", ".join(q["source"] for q in missing[:4])) if missing else
                "All consulted inputs returned usable data.")

    # 2. Evidence — how much directional weight exists and how divided it is.
    ratio = conflict.get("agreement_ratio")
    total_weight = round((conflict.get("bullish_weight") or 0)
                         + (conflict.get("bearish_weight") or 0), 3)
    if ratio is None or total_weight < 0.3:
        evidence_level = "INSUFFICIENT"
    elif ratio >= 0.85 and total_weight >= 1.0:
        evidence_level = "HIGH"
    elif ratio >= 0.65:
        evidence_level = "MEDIUM"
    else:
        evidence_level = "LOW"
    evidence = _dim(evidence_level,
                    "agreement across directional evidence, weighted by reliability × magnitude",
                    f"{total_weight:.2f} total directional weight, "
                    f"{(ratio or 0):.0%} of it on the larger side, "
                    f"{conflict.get('n_conflicts', 0)} named conflict(s).")

    # 3. Thesis — how strong the directional read is on its own terms.
    thesis_level = {"STRONG": "HIGH", "MODERATE": "MEDIUM",
                    "WEAK": "LOW", "NONE": "NONE"}[thesis["strength"]]
    thesis_dim = _dim(thesis_level, "net directional weight of the evidence",
                      thesis["statement"])

    # 4. Statistical edge — the gate's verdict, verbatim.
    stat_level = edge.get("level", "NONE")
    if stat_level not in _ORDER:
        stat_level = "NONE"
    statistical = _dim(stat_level, edge.get("gate_required", ""),
                       f"{edge.get('n_gates_passed')}/{edge.get('n_gates')} gates pass. "
                       + edge.get("sizing_consequence", ""))

    # 5. Model validation — did the backtest survive out-of-sample testing?
    gates = {g["gate"]: g["passed"] for g in (edge.get("gates") or [])}
    if gates.get("walk_forward") and gates.get("pbo"):
        validation_level = "HIGH"
    elif gates.get("walk_forward") or gates.get("pbo"):
        validation_level = "MEDIUM"
    elif gates.get("min_sample") is False:
        validation_level = "INSUFFICIENT"
    else:
        validation_level = "LOW"
    validation = _dim(validation_level,
                      "walk-forward with purge+embargo, and probability of backtest overfitting",
                      f"walk_forward={gates.get('walk_forward')}, pbo={gates.get('pbo')}, "
                      f"min_sample={gates.get('min_sample')}")

    # 6. Calibration — has this engine's stated confidence been measured?
    cal = edge.get("calibration") or {}
    n_cal = cal.get("n_predictions") or 0
    ece = cal.get("ece")
    if cal.get("status") != "OK" or n_cal < 30:
        calibration_level = "INSUFFICIENT"
        cal_detail = (f"{n_cal} matured predictions — not enough to measure whether stated "
                      f"confidence matches outcomes.")
    elif ece is not None and ece < 0.1:
        calibration_level, cal_detail = "HIGH", f"Calibration error {ece:.2f} over {n_cal} predictions."
    elif ece is not None and ece < 0.2:
        calibration_level, cal_detail = "MEDIUM", f"Calibration error {ece:.2f} over {n_cal} predictions."
    else:
        calibration_level = "LOW"
        cal_detail = (f"Calibration error {ece:.2f} over {n_cal} predictions — stated "
                      f"confidence levels are unreliable.")
    calibration = _dim(calibration_level, "measured live outcomes in the prediction ledger",
                       cal_detail)

    # 7. Scenario — can the scenarios be priced against real levels?
    n_priced = 0
    n_scen = 0
    if scenarios:
        for s in scenarios.get("scenarios", []):
            n_scen += 1
            if s.get("price_levels"):
                n_priced += 1
    if n_scen == 0:
        scenario_level, scen_detail = "NONE", "No scenarios were built."
    elif n_priced == 0:
        scenario_level, scen_detail = "LOW", "No scenario could be priced against a sourced level."
    elif n_priced >= n_scen - 1:
        scenario_level = "MEDIUM"
        scen_detail = (f"{n_priced}/{n_scen} scenarios are priced against sourced levels; "
                       f"no scenario carries a calibrated probability"
                       if not scenarios.get("any_probability_stated") else
                       f"{n_priced}/{n_scen} scenarios priced, and at least one probability "
                       f"is calibrated.")
        if scenarios.get("any_probability_stated"):
            scenario_level = "HIGH"
    else:
        scenario_level = "LOW"
        scen_detail = f"Only {n_priced}/{n_scen} scenarios could be priced."
    scenario = _dim(scenario_level, "scenarios priced against sourced levels; probabilities "
                                    "shown only when calibrated", scen_detail)

    dimensions = {
        "data": data, "evidence": evidence, "thesis": thesis_dim,
        "statistical": statistical, "validation": validation,
        "calibration": calibration, "scenario": scenario,
    }

    # ── The headline, by stated rule ─────────────────────────────────────
    cap = _min_level([dimensions[d]["level"] for d in CAPPING_DIMENSIONS])
    # A decision can be confident in its OWN terms (the state is clear, the
    # evidence agrees) without the underlying edge being proven — but it can
    # never be MORE confident than the weakest decision-critical input.
    base_candidates = [dimensions["evidence"]["level"], dimensions["data"]["level"]]
    base = _min_level(base_candidates)
    headline = base if _ORDER[base] <= _ORDER[cap] else cap

    limits: List[str] = []
    for d in CAPPING_DIMENSIONS:
        if dimensions[d]["level"] == cap:
            limits.append(f"{d} ({cap})")

    return {
        "dimensions": {k: v for k, v in dimensions.items()},
        "decision_confidence": headline,
        "capped_by": limits,
        "rule": ("Decision confidence is the weakest of the decision-critical dimensions "
                 f"({', '.join(CAPPING_DIMENSIONS)}). A strong score in any other dimension "
                 "cannot lift it. This is why high data quality never reads as strong "
                 "predictive evidence."),
        "summary": " · ".join(f"{k.title()}: {v['level']}" for k, v in dimensions.items())
                   + f" → Decision: {headline}",
    }
