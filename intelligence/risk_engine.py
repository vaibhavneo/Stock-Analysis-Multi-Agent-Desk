"""
Risk / cost-basis engine (item 7) - downside risk, volatility, drawdown
risk, ATR-based levels, and scenario risk (bull/bear downside per horizon,
from intelligence/prediction_engine.py's forecast when available). When the
caller supplies a position (avg_cost, optionally shares), adds a cost-basis
block: current gain/loss, distance from cost basis, downside to the nearest
technical support level, and recovery requirements - computed with the
correct ASYMMETRIC math (a -50% loss needs +100% to get back to breakeven,
not +50% - a common, easy-to-get-wrong mistake this module gets right by
construction, not by convention).

Reuses backtest/risk.py::compute_atr's already-computed `levels` dict
(agents/recommendation.py's entry/stop/target geometry) rather than
recomputing ATR independently - the trading path and this risk read stay
provably in agreement.

The cost-basis block is entirely ABSENT from the result (not a null-filled
key) when no position is supplied - the caller can tell "not asked for"
from "computed and empty."
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _nearest_support(historical_context: Optional[Dict[str, Any]]) -> Optional[float]:
    if not historical_context:
        return None
    sr = historical_context.get("support_resistance") or {}
    for window in ("1Y", "6M", "20D"):
        w = sr.get(window)
        if w and w.get("data_available"):
            return w.get("support")
    return None


def compute_risk_profile(
    ticker: str,
    current_price: Optional[float],
    levels: Optional[Dict[str, Any]],
    algo_signals: Dict[str, Any],
    historical_context: Optional[Dict[str, Any]] = None,
    regime: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Downside risk, volatility, drawdown risk, ATR levels, and scenario
    risk; plus an optional cost-basis block when `position={"avg_cost":...,
    "shares":...}` is supplied. Degrades each block independently and
    honestly - a missing input drops just that block's confidence
    contribution and adds a named flag, never a fabricated number.
    """
    flags: List[str] = []
    algo_signals = algo_signals or {}

    if not current_price or current_price <= 0:
        return {"ticker": ticker, "confidence": 0.0, "flags": ["no_current_price"],
                "volatility": {}, "levels": {}, "drawdown_risk": {"data_available": False},
                "scenario_risk": {}, "regime_risk": None}

    volatility = {
        "historical_volatility_20d": algo_signals.get("historical_volatility_20d"),
        "historical_volatility_60d": algo_signals.get("historical_volatility_60d"),
        "vol_regime": algo_signals.get("vol_regime"),
        "vol_expanding": algo_signals.get("vol_expanding"),
    }
    if volatility["historical_volatility_20d"] is None:
        flags.append("volatility_unavailable")

    level_risk: Dict[str, Any] = {}
    if levels and levels.get("stop_loss") is not None and levels.get("target_price") is not None:
        stop = float(levels["stop_loss"])
        target = float(levels["target_price"])
        risk = current_price - stop
        reward = target - current_price
        level_risk = {
            "stop_loss": stop,
            "target_price": target,
            "downside_to_stop_pct": round((stop / current_price - 1) * 100, 2),
            "upside_to_target_pct": round((target / current_price - 1) * 100, 2),
            "risk_reward_ratio": round(reward / risk, 2) if risk > 0 else None,
        }
        if risk <= 0:
            flags.append("stop_loss_above_current_price")
    else:
        flags.append("levels_unavailable")

    drawdown_risk: Dict[str, Any] = {"data_available": False}
    if historical_context:
        horizons = historical_context.get("horizons") or {}
        dd_candidates = [(label, h["max_drawdown_pct"]) for label, h in horizons.items()
                          if h.get("data_available") and h.get("max_drawdown_pct") is not None]
        if dd_candidates:
            worst_label, worst_dd = min(dd_candidates, key=lambda t: t[1])
            drawdown_risk = {"worst_observed_drawdown_pct": worst_dd,
                              "worst_observed_window": worst_label, "data_available": True}
    if not drawdown_risk["data_available"]:
        flags.append("drawdown_history_unavailable")

    scenario_risk: Dict[str, Any] = {}
    if forecast and forecast.get("horizons"):
        for label, h in forecast["horizons"].items():
            pr = h.get("price_range") or {}
            if pr.get("data_available"):
                scenario_risk[label] = {
                    "bear_case_downside_pct": round((pr["bear_price"] / current_price - 1) * 100, 2),
                    "bull_case_upside_pct": round((pr["bull_price"] / current_price - 1) * 100, 2),
                }
    if not scenario_risk:
        flags.append("scenario_risk_unavailable")

    regime_risk = regime.get("risk_stance") if regime else None
    if regime_risk is None:
        flags.append("regime_unavailable")

    result: Dict[str, Any] = {
        "ticker": ticker,
        "volatility": volatility,
        "levels": level_risk,
        "drawdown_risk": drawdown_risk,
        "scenario_risk": scenario_risk,
        "regime_risk": regime_risk,
        "flags": flags,
    }

    if position and position.get("avg_cost"):
        avg_cost = float(position["avg_cost"])
        shares = position.get("shares")
        underwater = current_price < avg_cost
        gain_loss_pct = round((current_price / avg_cost - 1) * 100, 2)
        # Asymmetric on purpose: recovering from a loss needs a LARGER
        # percentage gain than the loss itself (avg_cost/current - 1, not
        # -gain_loss_pct) - e.g. -50% needs +100% to get back to breakeven.
        recovery_required_pct = round((avg_cost / current_price - 1) * 100, 2) if underwater else 0.0

        cost_basis: Dict[str, Any] = {
            "avg_cost": avg_cost,
            "gain_loss_pct": gain_loss_pct,
            "underwater": underwater,
            "recovery_required_pct": recovery_required_pct,
        }
        if shares:
            cost_basis["gain_loss_dollars"] = round((current_price - avg_cost) * float(shares), 2)
        if level_risk.get("stop_loss") is not None:
            cost_basis["stop_loss_vs_cost_basis_pct"] = round((level_risk["stop_loss"] / avg_cost - 1) * 100, 2)
        support = _nearest_support(historical_context)
        if support is not None:
            cost_basis["nearest_support"] = support
            cost_basis["support_vs_cost_basis_pct"] = round((support / avg_cost - 1) * 100, 2)

        result["cost_basis"] = cost_basis

    confidence = 1.0
    if "volatility_unavailable" in flags:
        confidence -= 0.2
    if "levels_unavailable" in flags:
        confidence -= 0.2
    if "drawdown_history_unavailable" in flags:
        confidence -= 0.15
    if "scenario_risk_unavailable" in flags:
        confidence -= 0.15
    if "regime_unavailable" in flags:
        confidence -= 0.1
    result["confidence"] = round(max(0.0, confidence), 2)

    return result
