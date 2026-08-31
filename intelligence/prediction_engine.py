"""
Multi-horizon probabilistic forecast (item 6) - direction, bull/base/bear
price scenarios, and a probability/confidence per horizon (1wk/1mo/3mo/6mo/
1yr), deterministic and explainable end to end. No LLM call, no black box.

Deliberately avoids false precision:
  - p_up is scaled by the pillars' own average confidence BEFORE anything
    else, so a call with weak/missing data lands close to 0.5 (no directional
    claim) even if the raw composite score looks lopsided. This is what
    keeps "statistical edge isn't proven" from ever producing a confident-
    looking number.
  - p_up is regressed further toward 0.5 as the horizon lengthens (a today
    technical/fundamental snapshot says much less about 12 months out than
    about next week) - a simple, explainable decay, not a fitted model.
  - The bull/base/bear price range comes from ATR-implied daily volatility
    scaled by sqrt(horizon) - the standard random-walk scaling, not a
    fabricated spread - and is capped so it can never imply a negative price.

Composite score is derived from the SAME `pillars` dict (and the SAME
CORE_WEIGHTS) agents/orchestrator.py and backtest/pillars.py already use, so
this module's read of "how bullish is this stock" never silently disagrees
with the rest of the app - it's the identical inputs, not a re-scored
opinion.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from backtest.pillars import CORE_WEIGHTS

HORIZONS: Tuple[int, ...] = (5, 21, 63, 126, 252)
HORIZON_LABELS: Dict[int, str] = {5: "1W", 21: "1M", 63: "3M", 126: "6M", 252: "1Y"}

MAX_PILLAR_SWING = 0.30      # pillars alone never claim more than 80/20
REGIME_SHIFT = 0.03
ANALOG_MAX_WEIGHT = 0.30
MIN_ANALOG_SAMPLE = 5


def _composite_from_pillars(pillars: Dict[str, Any]) -> Tuple[Optional[float], float]:
    """Composite core score (technical/algo/fundamentals, CORE_WEIGHTS-weighted,
    renormalized over whichever core pillars actually have a score) and the
    average confidence across ALL pillars that reported one. None composite
    when no core pillar has a score - nothing to forecast from."""
    weighted_sum = 0.0
    weight_total = 0.0
    for name, w in CORE_WEIGHTS.items():
        p = (pillars or {}).get(name)
        if p and p.get("score") is not None:
            weighted_sum += w * p["score"]
            weight_total += w
    if weight_total == 0:
        return None, 0.0
    composite = weighted_sum / weight_total

    confidences = [p["confidence"] for p in (pillars or {}).values()
                   if p and p.get("confidence") is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return round(composite, 1), round(avg_confidence, 3)


def _p_up_base(composite: float, avg_confidence: float) -> float:
    raw = 0.5 + (composite - 50.0) / 100.0 * (2 * MAX_PILLAR_SWING)
    p = 0.5 + (raw - 0.5) * avg_confidence
    return max(0.5 - MAX_PILLAR_SWING, min(0.5 + MAX_PILLAR_SWING, p))


def _apply_regime_adjustment(p: float, regime: Optional[Dict[str, Any]]) -> float:
    if not regime or regime.get("risk_stance") is None:
        return p
    shift = {"RISK_ON": REGIME_SHIFT, "RISK_OFF": -REGIME_SHIFT, "NEUTRAL": 0.0}.get(
        regime["risk_stance"], 0.0)
    return max(0.05, min(0.95, p + shift))


def _apply_analog_adjustment(p: float, analog_result: Optional[Dict[str, Any]], h: int) -> float:
    if not analog_result or analog_result.get("status") != "ok":
        return p
    outcome = (analog_result.get("outcome_by_horizon") or {}).get(h)
    if not outcome or (outcome.get("n") or 0) < MIN_ANALOG_SAMPLE:
        return p
    analog_p = outcome.get("pct_positive")
    if analog_p is None:
        return p
    weight = ANALOG_MAX_WEIGHT * (analog_result.get("confidence") or 0.0)
    blended = p * (1 - weight) + analog_p * weight
    return max(0.05, min(0.95, blended))


def _decay_factor(h: int) -> float:
    """1.0 at the shortest horizon, decaying toward ~0.5 at 1 year - reflects
    a point-in-time snapshot mattering less the further out you project,
    without pretending to model exactly how much less."""
    return max(0.35, 1.0 - (h / 252.0) * 0.5)


def _price_range(current_price: float, atr_14: Optional[float], p_h: float, h: int) -> Dict[str, Any]:
    if not atr_14 or atr_14 <= 0 or not current_price:
        return {"data_available": False}
    daily_vol_pct = atr_14 / current_price
    horizon_vol_pct = max(0.005, min(daily_vol_pct * math.sqrt(h), 0.95))
    drift_pct = (p_h - 0.5) * 2 * min(0.5, horizon_vol_pct)
    base = current_price * (1 + drift_pct)
    bull = base * (1 + horizon_vol_pct)
    bear = max(base * (1 - horizon_vol_pct), current_price * 0.01)
    return {
        "bull_price": round(bull, 2),
        "base_price": round(base, 2),
        "bear_price": round(bear, 2),
        "implied_move_pct": round(horizon_vol_pct * 100, 1),
        "data_available": True,
    }


def _best_support_resistance(historical_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Invalidation levels: the nearest available support/resistance window
    from intelligence/historical_context.py, preferring the widest window
    that actually has data (a longer window's levels matter more for a
    multi-horizon forecast than the noisy 20-day one)."""
    if not historical_context:
        return {"data_available": False}
    sr = historical_context.get("support_resistance") or {}
    for window in ("1Y", "6M", "20D"):
        w = sr.get(window)
        if w and w.get("data_available"):
            return {
                "bullish_invalidated_below": w["support"],
                "bearish_invalidated_above": w["resistance"],
                "source_window": window,
                "data_available": True,
            }
    return {"data_available": False}


def forecast_horizons(
    ticker: str,
    current_price: Optional[float],
    pillars: Dict[str, Any],
    algo_signals: Dict[str, Any],
    regime: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    analog_result: Optional[Dict[str, Any]] = None,
    atr_14: Optional[float] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    calibrators: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Probability-oriented forecast at each horizon: direction, p_up,
    bull/base/bear price scenarios, and invalidation levels.

    Never fabricates: with no scoreable core pillar or no current price,
    returns confidence=0 and an empty horizons dict rather than a guess.

    `calibrators`: optional {horizon_days: fit_calibration() result} from
    intelligence.calibration. This is the feedback edge — it corrects stated
    probabilities using what actually happened to previous predictions at the
    same horizon. Omitted (the default) leaves p_up exactly as before, so every
    existing caller is unaffected.

    Passed IN rather than loaded here on purpose: fitting reads the ledger
    (I/O + a cross-validation pass), and this function is called per ticker
    while the calibration is portfolio-wide — the caller fits once and reuses.
    Direction is decided AFTER calibration, so a corrected probability is what
    actually drives the verdict rather than a label attached to a stale number.
    """
    composite, avg_confidence = _composite_from_pillars(pillars or {})
    if composite is None or not current_price:
        return {
            "ticker": ticker, "composite_score": None, "horizons": {},
            "invalidation": {"data_available": False},
            "confidence": 0.0, "flags": ["insufficient_pillar_data"],
        }

    flags: List[str] = []
    if avg_confidence < 0.4:
        flags.append("low_pillar_confidence")
    if not regime or regime.get("risk_stance") is None:
        flags.append("regime_unavailable")
    if not atr_14 or atr_14 <= 0:
        flags.append("atr_unavailable")
    if not analog_result or analog_result.get("status") != "ok":
        flags.append("analog_unavailable")
    if not calibrators:
        flags.append("calibration_unavailable")
    elif not any(c.get("applied") for c in calibrators.values()):
        # Distinguish "never looked" from "looked and it didn't earn its place".
        flags.append("calibration_not_applied")

    invalidation = _best_support_resistance(historical_context)

    horizon_forecasts: Dict[str, Any] = {}
    for h in horizons:
        label = HORIZON_LABELS.get(h, f"{h}d")
        p = _p_up_base(composite, avg_confidence)
        p = _apply_regime_adjustment(p, regime)
        p = _apply_analog_adjustment(p, analog_result, h)
        decay = _decay_factor(h)
        p_h = round(max(0.05, min(0.95, 0.5 + (p - 0.5) * decay)), 3)

        # Feedback edge: correct this horizon's probability using how previous
        # predictions at the SAME horizon actually resolved. Monotonic, so the
        # relative ranking of tickers is untouched - only stated confidence
        # moves. No-op unless that horizon's map beat raw probabilities
        # out-of-sample (see intelligence/calibration.fit_calibration).
        p_raw = p_h
        calibrated = False
        cal = (calibrators or {}).get(h)
        if cal:
            try:
                from intelligence.calibration import calibrate
                p_cal = round(max(0.05, min(0.95, calibrate(p_h, cal))), 3)
                calibrated = bool(cal.get("applied")) and p_cal != p_h
                p_h = p_cal
            except Exception:
                pass                     # a broken map costs the correction, not the forecast

        direction = "UP" if p_h > 0.55 else ("DOWN" if p_h < 0.45 else "NEUTRAL")
        entry = {
            "horizon_days": h,
            "p_up": p_h,
            "direction": direction,
            "price_range": _price_range(current_price, atr_14, p_h, h),
            "confidence": round(max(0.0, min(1.0, avg_confidence * decay)), 2),
        }
        if calibrated:
            # Kept so a surprising verdict can always be traced back to the
            # uncorrected number that produced it.
            entry["p_up_uncalibrated"] = p_raw
            entry["calibrated"] = True
        horizon_forecasts[label] = entry

    return {
        "ticker": ticker,
        "composite_score": composite,
        "horizons": horizon_forecasts,
        "invalidation": invalidation,
        "confidence": avg_confidence,
        "flags": flags,
    }
