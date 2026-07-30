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
) -> Dict[str, Any]:
    """Produce the complete structured decision report.

    Every input is an existing output dict from the Stock Agent. This function
    does arithmetic and formatting only — no LLM calls, no new data fetches."""
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
    scenarios = _build_scenarios(rec, orchestrator_result, agent_scores)

    # ── Evidence & warnings ───────────────────────────────────────────────
    evidence = _collect_evidence(rec)
    warnings_list = _collect_warnings(rec, calibration_interp, backtest_interp, xsec_interp)

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

        "evidence": evidence,
        "warnings": warnings_list,

        "honesty_flags": rec.get("honesty_flags", {}),
        "disclaimer": ("Composite decision report synthesized from computed outputs — "
                       "not financial advice. Numbers are historical, not predictive."),
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
                     agent_scores: Dict[str, Any]) -> Dict[str, Any]:
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
                      xsec: Dict[str, Any]) -> List[str]:
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

    return warnings_list


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
