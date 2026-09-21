"""
The Decision Intelligence engine — one snapshot, one synthesis, many views
(Phases 2 and 20, with the performance contract of Phase 28).

Everything the UI shows is derived from ONE pass through this function. The
sections do not each re-fetch or re-derive their inputs: evidence is normalized
once, horizons are partitioned once, the level ladder is built once, and every
downstream section reads those. That is the difference between a page that
renders in a second and one that re-runs a cross-sectional ranking per panel.

Authority, stated once and enforced by construction:

  - `agents/recommendation.py` remains the ONLY producer of the composite, the
    action verb, the statistical-edge level, Kelly sizing and the ATR geometry.
    This module reads them; it never recomputes or overrides them.
  - No LLM output enters any field that a decision reads. Narrative is carried
    under `thesis.narrative` and is quoted, never parsed.
  - Sizing comes from the recommendation's gate, unchanged. If the gate says 0%,
    every section that mentions size says 0%.

`build_decision_intelligence()` is pure: given the same inputs it returns the
same output (modulo `generated_at`), which is what makes the fingerprint and the
repeatability tests meaningful.
"""
from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime
from typing import Any, Dict, List, Optional

from decision.catalysts import build_catalyst_timeline
from decision.confidence import decompose_confidence
from decision.conflict import analyze_conflicts
from decision.consistency import check_consistency
from decision.entry import assess_entry
from decision.evidence import build_decision_evidence, to_dicts
from decision.horizon_plan import build_horizon_plans
from decision.horizons import synthesize_by_horizon
from decision.levels import build_level_map
from decision.plain import build_plain_summary
from decision.playbook import build_monitoring_plan, build_playbook
from decision.position import analyze_add, build_position_context, build_risk_budget
from decision.quality import assess_decision_quality
from decision.scenarios import build_mind_changers, build_scenarios
from decision.state import build_decision_state
from decision.thesis import (build_statistical_edge, build_thesis,
                             relate_thesis_and_edge)
from decision.track_record import build_track_record

REQUIRED_REC_KEYS = ("current_price", "action", "composite", "confidence")

# Map the recommendation's own horizon (trading days) onto the forecast
# engine's labels, so the scenario section reads the horizon the recommendation
# actually operates on rather than a hardcoded default.
def _horizon_label(days: Optional[int]) -> str:
    if not days:
        return "3M"
    for cutoff, label in ((10, "1W"), (35, "1M"), (95, "3M"), (180, "6M")):
        if days <= cutoff:
            return label
    return "1Y"


def build_decision_intelligence(
    ticker: str,
    recommendation: Dict[str, Any],
    indicators: Optional[Dict[str, Any]] = None,
    algo_signals: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    regime: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    xsec_interp: Optional[Dict[str, Any]] = None,
    calibration_interp: Optional[Dict[str, Any]] = None,
    backtest_interp: Optional[Dict[str, Any]] = None,
    backtest_all: Optional[Dict[str, Any]] = None,
    calibration_by_horizon: Optional[Dict[Any, Any]] = None,
    pillar_contradictions: Optional[List[Dict[str, Any]]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, Any]] = None,
    prior_decision: Optional[Dict[str, Any]] = None,
    max_portfolio_risk_pct: Optional[float] = None,
    llm_prose: Optional[Dict[str, Any]] = None,
    fetch_catalysts: bool = False,
) -> Dict[str, Any]:
    """Assemble the full Decision Intelligence object.

    Every optional argument degrades a section rather than the whole result:
    without `historical_context` the level ladder loses its swing levels but
    keeps its moving averages; without `catalysts` the timeline reports
    UNAVAILABLE rather than empty. `fetch_catalysts=True` lets the engine pull
    the calendar itself for callers that have no other source — off by default
    so nothing does surprise network I/O.
    """
    ticker = (ticker or "").upper().strip()
    now = datetime.now().isoformat(timespec="seconds")
    rec = recommendation or {}

    missing = [k for k in REQUIRED_REC_KEYS if rec.get(k) is None]
    if missing or not ticker:
        return {
            "ticker": ticker or None,
            "generated_at": now,
            "status": "INSUFFICIENT_DATA",
            "decision_state": {
                "headline_state": "INSUFFICIENT_DATA",
                "headline_reason": ("Required recommendation inputs are missing: "
                                    + ", ".join(missing or ["ticker"])),
                "ownership": "UNKNOWN",
            },
            "missing_inputs": missing or ["ticker"],
        }

    current_price = rec.get("current_price")
    horizon_days = rec.get("time_horizon_days") or 91
    horizon_label = _horizon_label(horizon_days)

    # ── 1. Position context first: it feeds evidence and every owned branch ──
    position_context = build_position_context(current_price, position)

    # ── 2. Catalysts ─────────────────────────────────────────────────────
    if catalysts is None and fetch_catalysts:
        catalysts = build_catalyst_timeline(
            ticker, current_price, (rec.get("levels") or {}).get("atr_14"))
    catalysts = catalysts or {"status": "UNAVAILABLE", "events": [], "upcoming": [],
                              "next_event": None, "n_upcoming": 0,
                              "waiting_for": "No catalyst calendar was consulted.",
                              "flags": ["not_requested"]}

    # ── 3. ONE normalized evidence snapshot. Everything below reads this. ──
    items = build_decision_evidence(
        rec, regime=regime, historical_context=historical_context, forecast=forecast,
        xsec_interp=xsec_interp, cal_interp=calibration_interp,
        position_context=(position_context if position_context.get("status") == "PROVIDED" else None),
        catalysts=(catalysts if catalysts.get("status") == "OK" else None))

    # ── 4. Horizon partition, then conflicts over the same items ─────────
    horizon_read = synthesize_by_horizon(items)
    conflict = analyze_conflicts(items, pillar_contradictions,
                                 horizon_read.get("horizon_conflicts"))

    # ── 5. Thesis and edge, kept apart, then related explicitly ──────────
    thesis = build_thesis(items, horizon_read, llm_prose)
    edge = build_statistical_edge(rec, calibration_interp, backtest_interp)
    relation = relate_thesis_and_edge(thesis, edge)

    # ── 6. Levels → entry → risk → add. Each reads the previous. ─────────
    level_map = build_level_map(current_price, indicators, historical_context,
                                rec.get("levels"), forecast, horizon_label)
    entry = assess_entry(current_price, thesis, edge, level_map, indicators,
                         algo_signals, rec, catalysts)
    risk_budget = build_risk_budget(current_price, position_context, level_map, rec,
                                    algo_signals, historical_context, max_portfolio_risk_pct)
    add_analysis = analyze_add(thesis, edge, entry, position_context, risk_budget,
                               conflict, (prior_decision or {}).get("thesis"), catalysts)

    # ── 7. Decision state (both ownership branches) ──────────────────────
    quality_blockers: List[str] = []
    if level_map.get("status") != "OK" and entry.get("status") == "INSUFFICIENT_DATA":
        quality_blockers.append(entry.get("reason", "No price levels could be sourced."))
    decision_state = build_decision_state(thesis, edge, entry, conflict, position_context,
                                          risk_budget, add_analysis, quality_blockers)

    # ── 8. Scenarios, mind-changers, playbook, monitoring ────────────────
    scenarios = build_scenarios(thesis, edge, items, level_map, forecast,
                                calibration_interp, catalysts, horizon_days,
                                horizon_label, current_price)
    mind_changers = build_mind_changers(thesis, edge, items, level_map, catalysts, conflict)
    playbook = build_playbook(decision_state, thesis, edge, entry, conflict,
                              position_context, risk_budget, add_analysis, level_map,
                              catalysts, mind_changers, algo_signals, horizon_days)
    monitoring = build_monitoring_plan(playbook, mind_changers, catalysts,
                                       algo_signals, level_map)

    # ── 9. Confidence and quality ────────────────────────────────────────
    confidence = decompose_confidence(rec, items, thesis, edge, conflict, scenarios,
                                      position_context, catalysts)
    quality = assess_decision_quality(confidence, edge, conflict, risk_budget,
                                      catalysts, position_context, level_map)

    # ── 10. Sizing — read from the recommendation's gate, never recomputed ──
    gated = bool(rec.get("position_size_gated", True))
    sizing = {
        "position_size_pct": 0.0 if gated else rec.get("position_size_pct"),
        "gated": gated,
        "raw_kelly_pct": rec.get("raw_kelly_pct"),
        "basis": ("Half-Kelly on the core strategy's returns, capped at 10%, and gated to 0% "
                  "until the statistical edge is demonstrated."),
        "statement": ("POSITION SIZING NOT SUPPORTED — the statistical gate has not cleared, "
                      "so size is 0% regardless of how good the setup looks."
                      if gated else
                      f"{rec.get('position_size_pct')}% of the risk-budgeted allocation "
                      f"(half-Kelly, capped)."),
        "dollar_sizing": ("POSITION_SIZING_NOT_COMPUTABLE — no portfolio value or risk budget "
                          "was supplied, so no dollar amount can be stated."
                          if not max_portfolio_risk_pct or
                          position_context.get("portfolio_value") is None else None),
    }

    # ── 11. Entry plan — only where the entry engine supports one ────────
    entry_plan = _build_entry_plan(entry, level_map, decision_state, sizing)

    # ── 12. Statistical honesty strip (Phase 17) ─────────────────────────
    honesty = _build_honesty(rec, edge, conflict, scenarios, level_map, calibration_interp)

    # ── 13. What this engine has actually done. Reads the strategy library
    #        race and the per-horizon live record — both of which existed and
    #        reached no decision surface.
    track_record = build_track_record(rec, edge, backtest_all, calibration_by_horizon)

    # ── 14. Per-horizon plans — the join between horizon-tagged evidence
    #        and timeframe-tagged levels that the engine could not previously
    #        express. Computed after sizing so every plan carries the same
    #        gate verdict: a longer holding period does not unlock size.
    horizon_plans = build_horizon_plans(
        horizon_read, items, level_map, entry, edge, position_context, sizing,
        rec, catalysts)

    decision: Dict[str, Any] = {
        "ticker": ticker,
        "generated_at": now,
        "status": "OK",
        "current_price": current_price,
        "sector": rec.get("sector"),
        "data_asof": rec.get("data_asof"),
        "horizon_days": horizon_days,
        "horizon_label": horizon_label,

        "decision_state": decision_state,
        "thesis": thesis,
        "statistical_edge": edge,
        "thesis_edge_relation": relation,

        "evidence": to_dicts(items),
        "evidence_count": len(items),
        "horizon_read": horizon_read,
        "horizon_plans": horizon_plans,
        "conflict": conflict,

        "level_map": level_map,
        "entry": entry,
        "entry_plan": entry_plan,

        "position_context": position_context,
        "risk_budget": risk_budget,
        "add_analysis": add_analysis,
        "sizing": sizing,
        "playbook": playbook,

        "scenarios": scenarios,
        "mind_changers": mind_changers,
        "catalysts": catalysts,
        "monitoring": monitoring,

        "confidence": confidence,
        "quality": quality,
        "statistical_honesty": honesty,
        "track_record": track_record,

        "composite": rec.get("composite"),
        "_recommendation": {k: rec.get(k) for k in
                            ("action", "composite", "conviction", "risk_veto",
                             "position_size_pct", "position_size_gated", "levels",
                             "time_horizon_days", "expected_return_pct",
                             "decision_fingerprint", "honesty_flags")},

        "authority": {
            "numbers_from": "agents/recommendation.py (composite, action verb, gate, sizing, ATR)",
            "llm_role": "explanation only; no LLM output enters any decision field",
            "this_layer": "synthesis and separation — it originates no number",
        },
        "disclaimer": ("Research and paper-decision support built from computed outputs. Not "
                       "financial advice, and not an instruction to transact. No broker "
                       "execution exists in this system."),
    }

    decision["consistency"] = check_consistency(decision)

    # ── 15. Plain English, built LAST from the finished object. It is a
    #        rendering of the decision, never an input to it — which is why it
    #        cannot describe a state the object does not hold, and why it is
    #        deliberately excluded from the fingerprint below.
    decision["plain"] = build_plain_summary(decision)

    decision["decision_fingerprint"] = _fingerprint(decision)
    return decision


def _build_entry_plan(entry: Dict[str, Any], level_map: Dict[str, Any],
                      decision_state: Dict[str, Any], sizing: Dict[str, Any]) -> Dict[str, Any]:
    """The entry plan exists only where an entry is actually supported.

    When the engine says WAIT, the plan carries the CONDITIONS and no price
    band — showing a band around the current price while telling the reader to
    wait is the contradiction this whole module was built to remove.
    """
    status = entry.get("status")
    supported = status in ("ATTRACTIVE", "ACCEPTABLE")
    geo = entry.get("geometry") or {}

    if not supported:
        return {
            "supported": False,
            "entry_low": None,
            "entry_high": None,
            "status": status,
            "instruction": (decision_state.get("headline_meaning") or ""),
            "conditions": entry.get("conditions", []),
            "avoid_conditions": entry.get("avoid_conditions", []),
            "size": sizing["statement"],
            "why_no_band": ("No entry band is shown because the entry engine does not support "
                            "entering here. A band drawn around the current price would "
                            "always contain the current price and would answer nothing."),
        }

    support = geo.get("nearest_support")
    price = geo.get("current_price")
    # The band runs from the sourced support up to the current price: buying
    # closer to the level the market has defended is what improves reward:risk,
    # and paying above the quote is never part of a plan this engine produces.
    return {
        "supported": True,
        "entry_low": support,
        "entry_high": price,
        "status": status,
        "instruction": (f"If entering, do so between {support} and {price} — closer to "
                        f"{support} shortens the distance to invalidation."),
        "band_basis": (level_map.get("nearest_support") or {}).get("basis"),
        "reward_risk": geo.get("reward_risk_to_levels"),
        "conditions": entry.get("conditions", []),
        "avoid_conditions": entry.get("avoid_conditions", []),
        "size": sizing["statement"],
    }


def _build_honesty(rec: Dict[str, Any], edge: Dict[str, Any], conflict: Dict[str, Any],
                   scenarios: Dict[str, Any], level_map: Dict[str, Any],
                   cal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 17 — everything that must be said out loud, in one place."""
    bt = edge.get("backtest") or {}
    flags = rec.get("honesty_flags") or {}
    cal_n = (cal or {}).get("n_predictions", 0)

    return {
        "demonstrated_edge": edge.get("demonstrated"),
        "verdict": edge.get("verdict"),
        "sample_size": {
            "backtest_observations": ((edge.get("gates") or [{}])[0].get("detail") or {}).get("n_obs"),
            "required": ((edge.get("gates") or [{}])[0].get("detail") or {}).get("required"),
            "backtest_trades": bt.get("n_trades"),
            "matured_live_predictions": cal_n,
        },
        "deflated_sharpe": {
            "value": bt.get("dsr"),
            "bar": 0.5,
            "n_trials_corrected_for": bt.get("n_trials"),
            "meaning": ("Deflated Sharpe is the probability the observed Sharpe reflects skill "
                        "rather than luck after correcting for the number of strategy variants "
                        "tried. Below 0.5 it does not support an edge claim."),
            "clears_bar": (bt.get("dsr") or 0) >= 0.5,
        },
        "calibration": {
            "status": (edge.get("calibration") or {}).get("status"),
            "n": cal_n,
            "ece": (edge.get("calibration") or {}).get("ece"),
            "statement": ("INSUFFICIENT — this engine's live accuracy has not been measured "
                          f"({cal_n} matured predictions)."
                          if (edge.get("calibration") or {}).get("status") != "OK" else
                          f"Measured over {cal_n} matured predictions."),
        },
        "backtest_status": {
            "survivorship_safe": flags.get("survivorship_safe", False),
            "point_in_time_fundamentals": flags.get("pit_fundamentals", False),
            "covers_core_only": flags.get("backtest_covers_core_only", False),
            "cost_model": bt.get("cost_model"),
            "total_cost_pct": bt.get("total_cost_pct"),
        },
        "multiple_testing": {
            "corrected": True,
            "method": "deflated Sharpe against a fixed pre-registered variant count",
            "n_trials": bt.get("n_trials"),
            "interaction_independent": flags.get("interaction_independent", False),
            "statement": ("The variant count is fixed and pre-registered, so running more "
                          "backtests cannot inflate the result."),
        },
        "one_sided": {
            "long_only": True,
            "statement": ("This system evaluates long exposure only. A bearish read means "
                          "'do not own', never 'short it' — no short-side evidence is "
                          "collected or tested."),
        },
        "probabilities": {
            "any_stated": scenarios.get("any_probability_stated", False),
            "policy": scenarios.get("probability_policy"),
        },
        "levels": {
            "n_observed": level_map.get("n_observed", 0),
            "n_total": level_map.get("n_levels", 0),
            "statement": ("Only levels with basis TRADED_PRICE are prices this security has "
                          "actually traded at; the rest are computed lines."),
        },
        "statements": edge.get("honesty", []),
    }


def _fingerprint(decision: Dict[str, Any]) -> str:
    """A stable hash of the DECISION-determining fields only.

    Excludes `generated_at` and every narrative field, so two runs on identical
    inputs produce the same fingerprint. This is the repeatability contract the
    tests assert against.
    """
    d = {
        "ticker": decision["ticker"],
        "state": decision["decision_state"]["headline_state"],
        "if_owned": decision["decision_state"]["if_owned"]["state"],
        "if_not_owned": decision["decision_state"]["if_not_owned"]["state"],
        "thesis": {k: decision["thesis"][k] for k in ("direction", "strength", "net_weight")},
        "edge": decision["statistical_edge"]["level"],
        "entry": decision["entry"]["status"],
        "sizing": decision["sizing"]["position_size_pct"],
        "gated": decision["sizing"]["gated"],
        "confidence": decision["confidence"]["decision_confidence"],
        "levels": [l["price"] for l in (decision["level_map"].get("ladder") or [])],
        "horizon_stances": {p["horizon"]: p["stance"]
                            for p in (decision.get("horizon_plans") or {}).get("plans", [])},
        "recommendation_fingerprint": (decision.get("_recommendation") or {}).get(
            "decision_fingerprint"),
    }
    return hashlib.sha1(_json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]
