"""
Deterministic factor + composite scores from normalized features.

Each factor is a documented average of its normalized member features (all in
[0,1], higher = more attractive after orientation). The composite uses the
FIXED, PRE-REGISTERED weights from the ExperimentRegistry (experiments.
ranking_config) — never tuned on the evaluation period. Risk is a PENALTY and a
VETO, never a positive alpha factor. Highly-correlated features inside a factor
are averaged (not summed), which caps double-counting.

Nothing here is stochastic and no LLM touches a number: identical inputs +
identical config => identical scores => identical ranks (a fingerprint proves it).

FEATURE ORIENTATION (higher_is_better) and factor membership:
  quality   : gross_margin(+), operating_margin(+), fcf_margin(+),
              cash_conversion(+), dilution(-)
  growth    : revenue_growth_yoy(+)
  valuation : price_to_sales(-), fcf_yield(+), earnings_yield(+)
  momentum  : mom_3m(+), mom_6m(+), mom_12_1(+), mom_voladj_6m(+),
              rs_vs_bench_6m(+), trend_persistence(+)
  revisions : UNAVAILABLE (no PIT estimate history) -> factor absent, weight
              redistributed proportionally across present factors.
  risk      : realized_vol(+risky), downside_vol(+risky), max_drawdown(+risky
              i.e. deeper is riskier), beta(+risky above 1), leverage(+risky),
              gap_risk(+risky) -> risk_score in [0,1], 1 = most risky. Applied as
              composite -= risk_penalty*risk_score, plus a top-percentile veto.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from xsection import normalize as nz

# (feature_name, higher_is_better, sector_relative)
FACTOR_FEATURES = {
    "quality": [("gross_margin", True, True), ("operating_margin", True, True),
                ("fcf_margin", True, True), ("cash_conversion", True, False),
                ("dilution", False, False)],
    "growth": [("revenue_growth_yoy", True, True)],
    "valuation": [("price_to_sales", False, True), ("fcf_yield", True, True),
                  ("earnings_yield", True, True)],
    "momentum": [("mom_3m", True, False), ("mom_6m", True, False),
                 ("mom_12_1", True, False), ("mom_voladj_6m", True, False),
                 ("rs_vs_bench_6m", True, False), ("trend_persistence", True, False)],
    "revisions": [("earnings_revision", True, False)],
}
# Risk features -> higher raw = riskier (used to build a 0..1 risk score).
RISK_FEATURES = [("realized_vol", True), ("downside_vol", True), ("max_drawdown", False),
                 ("beta", True), ("leverage", True), ("gap_risk", True)]


def _raw_matrix(rows: List[Dict[str, Any]]) -> Dict[str, List[Optional[float]]]:
    """feature_name -> list of raw values aligned to rows."""
    names = set()
    for r in rows:
        for f in r["features"]:
            names.add(f["feature_name"])
    out = {}
    for name in names:
        col = []
        for r in rows:
            fv = next((f["raw_value"] for f in r["features"] if f["feature_name"] == name), None)
            col.append(fv)
        out[name] = col
    return out


def score_cross_section(feature_rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute normalized features, factor scores, risk penalty/veto, and the
    composite for a whole cross-section (one ranking date). Returns per-security
    factor/composite scores + writes normalized_value back onto each feature."""
    sectors = [r.get("sector") for r in feature_rows]
    raw = _raw_matrix(feature_rows)
    n = len(feature_rows)
    winsor = config.get("winsor_pct", 0.05)

    # Normalize every factor feature; stash back onto the feature records.
    norm: Dict[str, List[Optional[float]]] = {}
    for factor, feats in {**FACTOR_FEATURES}.items():
        for (name, hib, sec_rel) in feats:
            if name not in raw:
                continue
            norm[name] = nz.normalize_feature(raw[name], sectors, higher_is_better=hib,
                                              winsor_pct=winsor, sector_relative=sec_rel)
    # Write normalized values back into the provenance-carrying feature records.
    for i, r in enumerate(feature_rows):
        for f in r["features"]:
            if f["feature_name"] in norm:
                f["normalized_value"] = (round(norm[f["feature_name"]][i], 4)
                                         if norm[f["feature_name"]][i] is not None else None)

    # Factor scores = mean of present normalized member features (caps double-count).
    factor_scores: List[Dict[str, Optional[float]]] = []
    present_factors_all = set()
    for i in range(n):
        fs: Dict[str, Optional[float]] = {}
        for factor, feats in FACTOR_FEATURES.items():
            vals = [norm[name][i] for (name, _, _) in feats
                    if name in norm and norm[name][i] is not None]
            if vals:
                fs[factor] = round(sum(vals) / len(vals), 4)
                present_factors_all.add(factor)
            else:
                fs[factor] = None
        factor_scores.append(fs)

    # Risk score in [0,1] (1 = most risky). Higher-risk features oriented so that
    # bigger raw -> bigger risk; then percentile-ranked (robust).
    risk_components = []
    for (name, higher_is_riskier) in RISK_FEATURES:
        if name not in raw:
            continue
        pr = nz.normalize_feature(raw[name], sectors, higher_is_better=higher_is_riskier,
                                  winsor_pct=winsor, sector_relative=False)
        risk_components.append(pr)
    risk_score = []
    for i in range(n):
        vals = [rc[i] for rc in risk_components if rc[i] is not None]
        risk_score.append(round(sum(vals) / len(vals), 4) if vals else 0.5)

    # Composite: weighted sum of PRESENT alpha factors (weights renormalized over
    # present factors so an absent factor — e.g. revisions — doesn't zero a name),
    # minus the risk penalty. Weights come from the pre-registered config ONLY.
    weights = config["weights"]
    risk_penalty = config.get("risk_penalty", 0.0)
    veto_pct = config.get("risk_veto_percentile", 1.0)
    # veto threshold on the risk_score cross-section
    sorted_risk = sorted(risk_score)
    veto_thresh = sorted_risk[min(len(sorted_risk) - 1, int(veto_pct * (len(sorted_risk) - 1)))] \
        if sorted_risk else 1.0

    out_rows = []
    composites = []
    for i in range(n):
        fs = factor_scores[i]
        present = {k: weights[k] for k in weights if fs.get(k) is not None}
        wsum = sum(present.values()) or 1.0
        alpha = sum((present[k] / wsum) * fs[k] for k in present)
        rp = risk_penalty * risk_score[i]
        composite = alpha - rp
        veto = risk_score[i] >= veto_thresh and veto_pct < 1.0
        out_rows.append({
            "factor_scores": fs,
            "risk_score": risk_score[i],
            "alpha_score": round(alpha, 4),
            "risk_penalty_applied": round(rp, 4),
            "composite_raw": round(composite, 4),
            "risk_veto": bool(veto),
            "factors_present": sorted(present.keys()),
        })
        composites.append(composite)

    # Percentile rank the composite across the eligible cross-section.
    comp_pr = nz.percentile_rank(composites)
    for i, r in enumerate(out_rows):
        r["composite_percentile"] = round(comp_pr[i], 4) if comp_pr[i] is not None else None
    return {"rows": out_rows, "factors_present": sorted(present_factors_all),
            "risk_veto_threshold": round(veto_thresh, 4)}
