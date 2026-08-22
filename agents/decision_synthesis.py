"""
Decision Synthesis — one coherent BUY/HOLD/SELL decision from all Stock Agent outputs.

This is a PURE CONSUMER layer: it reads existing outputs (recommendation, backtest,
calibration, cross-sectional ranking, prediction ledger) and synthesizes one structured
decision report. It creates no new agents, strategies, factors, or databases.

The final_action is DETERMINISTIC:
  BUY  — thesis positive, data sufficient, statistical gates pass, allocation non-zero,
          no risk veto.
  HOLD — thesis positive but edge or allocation unproven.
  SELL — thesis negative or material risk veto.
  AVOID — same as SELL when no position is held (context-aware label).

LLM may explain the structured results (via `thesis` prose from the orchestrator) but
cannot change scores, confidence, sizing, gates, or the final action.
"""
from __future__ import annotations

import hashlib
import json as _json
from datetime import datetime
from typing import Any, Dict, List, Optional


def synthesize_decision(
    ticker: str,
    recommendation: Dict[str, Any],
    backtest_all: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    xsec_ranking: Optional[Dict[str, Any]] = None,
    prediction_summary: Optional[Dict[str, Any]] = None,
    orchestrator_result: Optional[Dict[str, Any]] = None,
    owns_position: bool = False,
    regime: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    analog: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    risk_profile: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce the complete structured decision report.

    Every input is an existing output dict from the Stock Agent. This function
    does arithmetic and formatting only — no LLM calls, no new data fetches.

    The six new keyword args (regime, historical_context, analog, forecast,
    risk_profile, evidence) are the intelligence/ package's outputs - all
    optional, all default None, and every existing caller that doesn't pass
    them gets byte-identical output to before these existed. `evidence` is
    intelligence/evidence_synthesis.py::build_evidence_ledger()'s result -
    stored under report["evidence_ledger"], not report["evidence"] (that key
    already means something different: claim ids from _collect_evidence()).
    `regime` is the intelligence/regime.py market-wide regime dict - stored
    under report["market_regime"], since report["regime"] already means the
    recommendation's own per-ticker volatility regime string.
    """
    ticker = ticker.upper().strip()
    rec = recommendation
    now = datetime.now().isoformat(timespec="seconds")

    # ── Final action (deterministic rules) ────────────────────────────────
    final_action, action_rationale = _determine_action(rec, owns_position)

    # ── Agent consensus ───────────────────────────────────────────────────
    agent_scores = _extract_agent_scores(rec, orchestrator_result)
    consensus = _compute_consensus(agent_scores, rec)

    # ── Backtest interpretation ───────────────────────────────────────────
    backtest_interp = _interpret_backtest(rec, backtest_all)

    # ── Calibration interpretation ────────────────────────────────────────
    calibration_interp = _interpret_calibration(calibration, prediction_summary)

    # ── Cross-sectional rank ──────────────────────────────────────────────
    xsec_interp = _interpret_xsec(ticker, xsec_ranking)

    # ── Bull/base/bear cases + catalysts + risks + thesis breakers ────────
    scenarios = _build_scenarios(rec, orchestrator_result, agent_scores, evidence)

    # ── Evidence & warnings ───────────────────────────────────────────────
    evidence_claims = _collect_evidence(rec)
    warnings_list = _collect_warnings(rec, calibration_interp, backtest_interp, xsec_interp, evidence)

    # ── Item 8: strongest evidence + what would change the call ───────────
    # Deliberately computed AFTER everything above - "generated only after
    # all evidence is assembled" (item 8's own wording).
    strongest = _strongest_evidence(evidence)
    would_change = _what_would_change_the_call(final_action, rec, evidence, risk_profile)
    conviction = _conviction_from_evidence(evidence, rec)

    report: Dict[str, Any] = {
        "ticker": ticker,
        "generated_at": now,
        "current_price": rec.get("current_price"),
        "sector": rec.get("sector"),
        "regime": rec.get("regime"),

        "final_action": final_action,
        "action_rationale": action_rationale,
        "composite_score": rec.get("composite"),

        "confidence": {
            "thesis": rec.get("confidence", {}).get("thesis", {}),
            "data": rec.get("confidence", {}).get("data", {}),
            "statistical": rec.get("confidence", {}).get("statistical_edge", {}),
            "allocation": rec.get("confidence", {}).get("allocation", {}),
        },

        "agent_scores": agent_scores,
        "consensus": consensus,

        "backtest": backtest_interp,
        "calibration": calibration_interp,
        "cross_sectional_rank": xsec_interp,

        "scenarios": scenarios,

        "levels": rec.get("levels"),
        "position_size_pct": rec.get("position_size_pct"),
        "position_size_gated": rec.get("position_size_gated"),
        "time_horizon_days": rec.get("time_horizon_days"),

        "evidence": evidence_claims,
        "warnings": warnings_list,

        "honesty_flags": rec.get("honesty_flags", {}),
        "disclaimer": ("Composite decision report synthesized from computed outputs — "
                       "not financial advice. Numbers are historical, not predictive."),

        # ── intelligence/ package outputs (all optional, all None by default -
        # every existing caller that doesn't pass them sees these as None/empty,
        # nothing else in the report changes shape or value). ────────────────
        "market_regime": regime,
        "historical_context": historical_context,
        "analog": analog,
        "forecast": forecast,
        "risk_profile": risk_profile,
        "evidence_ledger": evidence,
        "conviction": conviction,
        "strongest_evidence": strongest,
        "what_would_change_the_call": would_change,
    }
    report["decision_fingerprint"] = _fingerprint(report)
    report["recommendation_fingerprint"] = rec.get("decision_fingerprint")
    return report


# ── Action determination (deterministic) ──────────────────────────────────

def _determine_action(rec: Dict[str, Any], owns_position: bool) -> tuple:
    """Deterministic action from the recommendation's gates.

    BUY only when: thesis positive, data sufficient, statistical gates pass,
    allocation non-zero, no risk veto."""
    action = rec.get("action", "HOLD")
    conf = rec.get("confidence", {})
    stat_edge = conf.get("statistical_edge", {})
    data_conf = conf.get("data", {})
    risk_veto = rec.get("risk_veto", False)
    gated = rec.get("position_size_gated", True)
    composite = rec.get("composite", 50)

    reasons = []

    if action in ("SELL", "REDUCE") or composite < 35:
        final = "SELL" if owns_position else "AVOID"
        if risk_veto:
            reasons.append("risk veto active (high vol + expanding)")
        if composite < 35:
            reasons.append(f"composite score {composite} below SELL threshold (35)")
        if action in ("SELL", "REDUCE"):
            reasons.append(f"pillar consensus: {action}")
        return final, "; ".join(reasons) or "negative thesis"

    thesis_positive = action in ("BUY", "ACCUMULATE")
    data_sufficient = data_conf.get("level") in ("HIGH", "MEDIUM")
    stat_proven = stat_edge.get("level") == "HIGH"
    allocation_ok = not gated

    if thesis_positive and data_sufficient and stat_proven and allocation_ok and not risk_veto:
        reasons.append(f"composite {composite}, all gates pass")
        reasons.append(f"statistical edge: {stat_edge.get('level')}")
        if not gated:
            reasons.append(f"Kelly allocation: {rec.get('position_size_pct')}%")
        return "BUY", "; ".join(reasons)

    if thesis_positive and not (stat_proven and allocation_ok):
        reasons.append(f"composite {composite} suggests {action}")
        if not stat_proven:
            reasons.append(f"statistical edge: {stat_edge.get('level', 'NONE')} (need HIGH)")
        if gated:
            reasons.append("allocation gated to 0%")
        if risk_veto:
            reasons.append("risk veto active")
        watch = "HOLD" if action == "BUY" else "HOLD"
        return watch, "; ".join(reasons) or "positive thesis but edge unproven"

    reasons.append(f"composite {composite}, action={action}")
    return "HOLD", "; ".join(reasons) or "neutral"


# ── Agent scores ──────────────────────────────────────────────────────────

def _extract_agent_scores(rec: Dict[str, Any],
                          orch: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract per-agent (pillar) scores and prediction agent data."""
    pillars = rec.get("pillars", {})
    scores = {}
    for name in ("technical", "algo", "risk", "fundamentals", "research", "social"):
        p = pillars.get(name, {})
        scores[name] = {
            "score": p.get("score", 50),
            "backtestable": p.get("backtestable", False),
            "flags": p.get("flags", []),
        }

    if orch:
        pred = orch.get("prediction", {})
        scores["prediction"] = {
            "action": pred.get("action"),
            "conviction": pred.get("conviction"),
            "grounding": pred.get("grounding", "none"),
        }
    return scores


def _compute_consensus(agent_scores: Dict[str, Any],
                       rec: Dict[str, Any]) -> Dict[str, Any]:
    """How many agents agree or conflict with the final action."""
    action = rec.get("action", "HOLD")
    bullish = action in ("BUY", "ACCUMULATE")
    bearish = action in ("SELL", "REDUCE")

    agreeing = []
    conflicting = []
    neutral = []

    for name in ("technical", "algo", "fundamentals", "research", "social"):
        s = agent_scores.get(name, {}).get("score", 50)
        if s >= 60:
            if bullish:
                agreeing.append(name)
            elif bearish:
                conflicting.append(name)
            else:
                neutral.append(name)
        elif s <= 40:
            if bearish:
                agreeing.append(name)
            elif bullish:
                conflicting.append(name)
            else:
                neutral.append(name)
        else:
            neutral.append(name)

    risk_s = agent_scores.get("risk", {}).get("score", 50)
    risk_flags = agent_scores.get("risk", {}).get("flags", [])

    return {
        "agreeing": agreeing,
        "conflicting": conflicting,
        "neutral": neutral,
        "n_agreeing": len(agreeing),
        "n_conflicting": len(conflicting),
        "risk_veto": rec.get("risk_veto", False),
        "risk_score": risk_s,
        "risk_flags": risk_flags,
        "summary": _consensus_summary(agreeing, conflicting, neutral, rec),
    }


def _consensus_summary(agreeing, conflicting, neutral, rec) -> str:
    action = rec.get("action", "HOLD")
    n_agree = len(agreeing)
    n_conflict = len(conflicting)
    if n_agree >= 4:
        return f"Strong consensus: {n_agree}/5 agents agree with {action}."
    if n_agree >= 3 and n_conflict == 0:
        return f"Moderate consensus: {n_agree}/5 agents support {action}, none oppose."
    if n_conflict >= 2:
        return (f"Mixed signals: {n_agree} agent(s) agree with {action}, "
                f"but {n_conflict} conflict — conviction reduced.")
    return f"{n_agree}/5 agents lean {action}; {len(neutral)} are neutral."


# ── Backtest interpretation ───────────────────────────────────────────────

def _interpret_backtest(rec: Dict[str, Any],
                        backtest_all: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bt = rec.get("backtest", {})
    sharpe = bt.get("sharpe", 0)
    dsr = bt.get("dsr", 0)
    max_dd = bt.get("max_drawdown", 0)
    cost_model = bt.get("cost_model", "unknown")
    n_trades = bt.get("n_trades", 0)

    if sharpe <= 0:
        verdict = "The core strategy has been unprofitable historically — no edge detected."
    elif dsr < 0.5:
        verdict = (f"Sharpe {sharpe:.2f} is positive but dSR {dsr:.2f} is below 0.5 — "
                   "the edge may be luck rather than skill (after multiple-testing correction).")
    elif max_dd > 0.3:
        verdict = (f"Sharpe {sharpe:.2f} with dSR {dsr:.2f} shows an edge, but "
                   f"max drawdown {max_dd:.0%} is severe — sizing must account for this.")
    else:
        verdict = (f"Sharpe {sharpe:.2f}, dSR {dsr:.2f} (above 0.5): a credible edge "
                   f"after cost and multiple-testing correction. Max drawdown {max_dd:.0%}.")

    result: Dict[str, Any] = {
        "strategy": bt.get("strategy"),
        "sharpe": sharpe,
        "dsr": dsr,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "cost_model": cost_model,
        "total_cost_pct": bt.get("total_cost_pct"),
        "core_signal_now": bt.get("core_signal_now"),
        "n_trials": bt.get("n_trials"),
        "interpretation": verdict,
    }

    if backtest_all and "rows" in backtest_all:
        best = max(backtest_all["rows"], key=lambda r: r.get("dsr", 0), default=None)
        beating_hold = sum(1 for r in backtest_all["rows"] if r.get("beats_hold"))
        result["library_summary"] = {
            "n_strategies": len(backtest_all["rows"]),
            "beating_buy_hold": beating_hold,
            "best_strategy": best.get("strategy") if best else None,
            "best_dsr": best.get("dsr") if best else None,
            "buy_hold_sharpe": (backtest_all.get("buy_hold") or {}).get("sharpe"),
        }

    return result


# ── Calibration interpretation ────────────────────────────────────────────

def _interpret_calibration(calibration: Optional[Dict[str, Any]],
                           prediction_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not calibration:
        return {"status": "INSUFFICIENT_HISTORY",
                "interpretation": "No calibration data available — predictions have not been "
                                  "tracked long enough to measure accuracy."}

    n = (calibration.get("overall") or {}).get("n", 0)
    if n < 10:
        return {"status": "INSUFFICIENT_HISTORY",
                "interpretation": f"Only {n} matured predictions — too few for reliable calibration.",
                "n_predictions": n}

    overall = calibration.get("overall", {})
    ece = calibration.get("calibration_error_ece")
    win_rate = overall.get("win_rate")
    avg_return = overall.get("avg_raw_return_pct")
    avg_excess = overall.get("avg_excess_return_pct")

    if win_rate is not None and win_rate > 0.55:
        verdict = (f"Calibration on {n} predictions: {win_rate:.0%} win rate, "
                   f"average return {avg_return:+.1f}% — above random.")
    elif win_rate is not None and win_rate >= 0.45:
        verdict = (f"Calibration on {n} predictions: {win_rate:.0%} win rate — "
                   "near coin-flip. Edge is marginal at best.")
    elif win_rate is not None:
        verdict = (f"Calibration on {n} predictions: {win_rate:.0%} win rate, "
                   f"average return {avg_return:+.1f}% — below random, suggesting "
                   "the current model may not have predictive power.")
    else:
        verdict = f"Calibration data available ({n} predictions) but win rate not computed."

    if ece is not None:
        if ece < 0.1:
            verdict += " Confidence is well-calibrated."
        elif ece < 0.2:
            verdict += f" Confidence calibration error {ece:.2f} — moderate; stated confidence slightly off."
        else:
            verdict += f" Confidence calibration error {ece:.2f} — stated confidence levels are unreliable."

    result: Dict[str, Any] = {
        "status": "OK",
        "n_predictions": n,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "avg_excess_return_pct": avg_excess,
        "calibration_error_ece": ece,
        "interpretation": verdict,
    }

    if calibration.get("by_action"):
        result["by_action"] = calibration["by_action"]
    if calibration.get("confidence_reliability"):
        result["confidence_reliability"] = calibration["confidence_reliability"]
    if prediction_summary:
        result["total_predictions"] = prediction_summary.get("total_predictions", 0)
        result["matured_20d"] = prediction_summary.get("matured_20d", 0)

    return result


# ── Cross-sectional rank ─────────────────────────────────────────────────

def _interpret_xsec(ticker: str,
                    xsec_ranking: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not xsec_ranking or xsec_ranking.get("status") != "OK":
        return {"status": "UNAVAILABLE",
                "interpretation": "No cross-sectional ranking available for this analysis."}

    ranked = xsec_ranking.get("ranked", [])
    n_ranked = len(ranked)
    if n_ranked == 0:
        return {"status": "UNAVAILABLE", "interpretation": "No securities in the ranking."}

    ticker_entry = None
    for item in ranked:
        if (item.get("ticker_as_of") or "").upper() == ticker:
            ticker_entry = item
            break

    if not ticker_entry:
        return {
            "status": "NOT_IN_UNIVERSE",
            "n_securities": n_ranked,
            "as_of": xsec_ranking.get("as_of"),
            "universe_id": xsec_ranking.get("universe_id"),
            "interpretation": f"{ticker} is not in the ranked universe ({n_ranked} securities).",
        }

    rank = ticker_entry.get("rank", n_ranked)
    percentile = ticker_entry.get("composite_percentile", 0)
    composite = ticker_entry.get("composite_raw", 0)

    if percentile >= 0.8:
        verdict = f"{ticker} ranks #{rank}/{n_ranked} (top {100*(1-percentile):.0f}%) — a top-decile opportunity."
    elif percentile >= 0.5:
        verdict = f"{ticker} ranks #{rank}/{n_ranked} (top {100*(1-percentile):.0f}%) — above median."
    else:
        verdict = f"{ticker} ranks #{rank}/{n_ranked} (bottom {100*percentile:.0f}%) — below median vs. peers."

    return {
        "status": "OK",
        "rank": rank,
        "n_securities": n_ranked,
        "percentile": round(percentile, 3),
        "composite_score": round(composite, 3),
        "as_of": xsec_ranking.get("as_of"),
        "universe_id": xsec_ranking.get("universe_id"),
        "survivorship_safe": xsec_ranking.get("survivorship_safe", False),
        "sector_rank": ticker_entry.get("rank_in_sector"),
        "factor_scores": ticker_entry.get("factor_scores"),
        "interpretation": verdict,
    }


# ── Scenarios ─────────────────────────────────────────────────────────────

def _build_scenarios(rec: Dict[str, Any],
                     orch: Optional[Dict[str, Any]],
                     agent_scores: Dict[str, Any],
                     evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    levels = rec.get("levels") or {}
    current_price = rec.get("current_price", 0)
    # Levels may be present but carry null prices (e.g. ATR unavailable). Fall back
    # to current_price so scenario prose degrades gracefully instead of crashing —
    # this touches no scores or gates, only the narrative shown alongside them.
    target = levels.get("target_price") or current_price or 0
    stop = levels.get("stop_loss") or current_price or 0

    bull_upside = round((target / current_price - 1) * 100, 1) if current_price else 0
    bear_downside = round((stop / current_price - 1) * 100, 1) if current_price else 0

    bull_case = f"Target ${target:.2f} ({bull_upside:+.1f}%). "
    bear_case = f"Stop ${stop:.2f} ({bear_downside:+.1f}%). "
    base_case = "Composite score suggests the position has marginal edge — size accordingly."

    thesis = rec.get("thesis")
    if thesis:
        if thesis.get("bull_case"):
            bull_case += thesis["bull_case"]
        if thesis.get("bear_case"):
            bear_case += thesis["bear_case"]
        if thesis.get("summary"):
            base_case = thesis["summary"]

    catalysts = []
    risks = []
    thesis_breakers = []

    if thesis and thesis.get("key_catalysts"):
        catalysts.append(thesis["key_catalysts"])

    tech_score = agent_scores.get("technical", {}).get("score", 50)
    algo_score = agent_scores.get("algo", {}).get("score", 50)
    if tech_score >= 70 and algo_score >= 70:
        catalysts.append("Strong technical + algorithmic momentum alignment.")
    if tech_score <= 30:
        risks.append("Technical indicators are bearish.")
    if algo_score <= 30:
        risks.append("Algorithmic signals show negative momentum.")

    fund_score = agent_scores.get("fundamentals", {}).get("score", 50)
    if fund_score >= 70:
        catalysts.append("Strong fundamental quality/value scores.")
    elif fund_score <= 30:
        risks.append("Weak fundamental metrics (low quality or overvalued).")

    if rec.get("risk_veto"):
        thesis_breakers.append("RISK VETO: high and expanding volatility — reduce exposure.")
    if rec.get("position_size_gated"):
        thesis_breakers.append("NO PROVEN EDGE: statistical gate fails — allocation is zero.")

    bt = rec.get("backtest", {})
    if bt.get("max_drawdown", 0) > 0.3:
        risks.append(f"Historical max drawdown {bt['max_drawdown']:.0%} — severe.")
    if bt.get("dsr", 0) < 0.5:
        risks.append(f"Deflated Sharpe {bt.get('dsr', 0):.2f} < 0.5 — edge may be luck.")

    # Item 4: named contradictions reduce conviction and are surfaced here,
    # never silently absorbed into a single blended score. HIGH severity is a
    # thesis breaker (a real reason to distrust the call outright); MEDIUM is
    # a risk worth naming but not disqualifying.
    if evidence:
        for c in evidence.get("contradictions", []):
            line = f"{c['description']}"
            if c.get("severity") == "HIGH":
                thesis_breakers.append(f"CONTRADICTION ({c['name']}): {line}")
            else:
                risks.append(f"Contradiction ({c['name']}): {line}")

    return {
        "bull_case": bull_case,
        "base_case": base_case,
        "bear_case": bear_case,
        "catalysts": catalysts,
        "risks": risks,
        "thesis_breakers": thesis_breakers,
    }


# ── Evidence & warnings ──────────────────────────────────────────────────

def _collect_evidence(rec: Dict[str, Any]) -> Dict[str, Any]:
    claims = rec.get("claims", {})
    return {
        "claim_ids": {k: v for k, v in claims.items() if v},
        "experiment_manifest_hash": rec.get("experiment_manifest_hash"),
        "recommendation_fingerprint": rec.get("decision_fingerprint"),
        "honesty_flags": rec.get("honesty_flags", {}),
    }


def _collect_warnings(rec: Dict[str, Any],
                      cal: Dict[str, Any],
                      bt: Dict[str, Any],
                      xsec: Dict[str, Any],
                      evidence: Optional[Dict[str, Any]] = None) -> List[str]:
    warnings_list = []
    flags = rec.get("honesty_flags", {})

    if not flags.get("survivorship_safe"):
        warnings_list.append("Not survivorship-safe: backtest results exclude delisted companies.")
    if not flags.get("pit_fundamentals"):
        warnings_list.append("Fundamentals are not point-in-time (SEC EDGAR unavailable).")
    if flags.get("backtest_covers_core_only"):
        warnings_list.append("Backtest covers only the core (tech+algo) — social/research untested.")

    if cal.get("status") == "INSUFFICIENT_HISTORY":
        warnings_list.append("Prediction calibration: insufficient history to measure accuracy.")
    elif cal.get("win_rate") is not None and cal["win_rate"] < 0.45:
        warnings_list.append(f"Prediction calibration: win rate {cal['win_rate']:.0%} is below random.")

    if xsec.get("status") == "UNAVAILABLE":
        warnings_list.append("No cross-sectional ranking available for relative comparison.")

    if rec.get("risk_veto"):
        warnings_list.append("RISK VETO active: high volatility, expanding — conviction capped.")
    if rec.get("position_size_gated"):
        warnings_list.append("Position size gated to 0%: statistical edge not established.")

    if evidence:
        for c in evidence.get("contradictions", []):
            if c.get("severity") == "HIGH":
                warnings_list.append(f"Signal contradiction ({c['name'].replace('_', ' ')}): {c['description']}")

    return warnings_list


# ── Item 8: strongest evidence, conviction, what would change the call ────
# All three are pure readers of the evidence ledger (intelligence/
# evidence_synthesis.py) and the existing recommendation dict - no new
# scoring authority, no LLM call. Computed last, after every other section,
# so they can reference the assembled evidence rather than pre-empting it.

def _strongest_evidence(evidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The single strongest bullish and bearish item from the evidence
    ledger, ranked by reliability * distance-from-neutral - the item that is
    both confident AND extreme, not just the most extreme regardless of how
    little it should be trusted. None (not fabricated) when there's no
    evidence ledger or no item of that direction."""
    if not evidence or not evidence.get("evidence"):
        return {"bull": None, "bear": None}

    def strength(item: Dict[str, Any]) -> float:
        return (item.get("reliability") or 0.0) * abs((item.get("score") or 50) - 50)

    bullish = [e for e in evidence["evidence"] if e.get("signal") == "bullish"]
    bearish = [e for e in evidence["evidence"] if e.get("signal") == "bearish"]
    strongest_bull = max(bullish, key=strength, default=None)
    strongest_bear = max(bearish, key=strength, default=None)

    return {
        "bull": {"source": strongest_bull["source"], "score": strongest_bull["score"],
                 "reliability": strongest_bull["reliability"]} if strongest_bull else None,
        "bear": {"source": strongest_bear["source"], "score": strongest_bear["score"],
                 "reliability": strongest_bear["reliability"]} if strongest_bear else None,
    }


def _what_would_change_the_call(final_action: str, rec: Dict[str, Any],
                                evidence: Optional[Dict[str, Any]],
                                risk_profile: Optional[Dict[str, Any]]) -> List[str]:
    """Deterministic, explainable statements of what would flip or strengthen
    the call - item 8's explicit requirement. Always non-empty: there is
    always something that would need to change for a HIGH-conviction BUY
    with zero contradictions to be even MORE clearly right, so this never
    silently returns nothing to say."""
    changes: List[str] = []
    conf = rec.get("confidence", {}) or {}
    stat_edge = (conf.get("statistical_edge") or {}).get("level")
    data_level = (conf.get("data") or {}).get("level")

    if final_action in ("HOLD", "SELL", "AVOID"):
        if stat_edge != "HIGH":
            changes.append("A proven statistical edge (walk-forward + PBO passing at HIGH confidence) "
                           "would support a BUY.")
        if data_level not in ("HIGH", "MEDIUM"):
            changes.append("More reliable underlying data (currently LOW confidence or insufficient) "
                           "would strengthen the case either way.")
        if rec.get("risk_veto"):
            changes.append("Volatility normalizing (the risk veto clearing) would remove a hard block on sizing.")

    if evidence:
        for c in evidence.get("contradictions", []):
            changes.append(f"Resolving the '{c['name'].replace('_', ' ')}' contradiction "
                           f"(currently {c['severity']}) would raise conviction.")

    if risk_profile and (risk_profile.get("cost_basis") or {}).get("underwater"):
        recovery = risk_profile["cost_basis"].get("recovery_required_pct")
        if recovery is not None:
            changes.append(f"The price recovering {recovery:.1f}% would return the position to breakeven.")

    if not changes:
        changes.append("No specific blocking condition identified — the current read is already well-supported "
                       "by the available evidence.")

    return changes


def _conviction_from_evidence(evidence: Optional[Dict[str, Any]], rec: Dict[str, Any]) -> Optional[str]:
    """LOW/MEDIUM/HIGH from the evidence ledger's conviction_multiplier
    (itself reduced by named contradictions) combined with whether a
    statistical edge is actually proven - a confident-sounding narrative
    with no proven edge should never read as HIGH conviction. None (not a
    default HIGH/MEDIUM) when no evidence ledger was supplied."""
    if not evidence:
        return None
    multiplier = evidence.get("conviction_multiplier")
    if multiplier is None:
        return None
    stat_proven = ((rec.get("confidence") or {}).get("statistical_edge") or {}).get("level") == "HIGH"
    if multiplier >= 0.9 and stat_proven:
        return "HIGH"
    if multiplier >= 0.65:
        return "MEDIUM"
    return "LOW"


# ── Fingerprint ───────────────────────────────────────────────────────────

def _fingerprint(report: Dict[str, Any]) -> str:
    d = {
        "ticker": report["ticker"],
        "final_action": report["final_action"],
        "composite_score": report["composite_score"],
        "confidence": report["confidence"],
        "recommendation_fingerprint": report.get("recommendation_fingerprint"),
    }
    return hashlib.sha1(
        _json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
