"""
Adaptive orchestration (item 10) - decides which intelligence/ computations
actually run for a given request, instead of unconditionally running
everything on every call. "Spend computation where it adds information."

Uses a typed `sections` parameter with four named presets that literally
implement the four example use cases from the spec:
  - "valuation"    ("Is AMD overvalued?")           -> fundamentals (always
                                                        computed as part of
                                                        the base pillars) +
                                                        historical valuation
                                                        context only.
  - "full"          ("Should I buy AMD now?")        -> everything.
  - "price_action"  ("Why did AMD fall?")            -> historical context +
                                                        market regime.
  - "recovery"      ("Can AMD recover to my $490?")  -> historical context +
                                                        risk (which carries
                                                        the cost-basis math)
                                                        + forecast.

No chat/NL interface - a resolved, explicit judgment call: the app had zero
free-text endpoints before this session, and this deliberately doesn't add
one. `plan_sections()` auto-selects "recovery" whenever the caller supplies
a position (avg_cost) without needing to parse any question text at all -
the presence of that field already signals the intent.

run_selected() only fetches/computes what the resolved sections actually
need: SPY/VIX (the expensive extra fetches) are pulled ONLY when "regime" or
"analog" is in the resolved list, so a "valuation"-only request never pays
for them - directly serving item 12 ("preserve performance").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

PRESETS: Dict[str, List[str]] = {
    "valuation": ["historical_context"],
    "full": ["regime", "historical_context", "analog", "forecast", "risk", "evidence"],
    "price_action": ["regime", "historical_context"],
    "recovery": ["historical_context", "risk", "forecast"],
}
ALL_SECTIONS: List[str] = sorted({s for sections in PRESETS.values() for s in sections})


def plan_sections(requested: Optional[Any] = None, has_position: bool = False) -> List[str]:
    """Resolve which of the 6 gateable sections (regime, historical_context,
    analog, forecast, risk, evidence) actually need to run.

    `requested` may be a single preset name (one of PRESETS' keys) or an
    explicit list of section names (only recognized ones are kept; an
    unrecognized/empty result falls back to "full" rather than computing
    nothing, since silently returning zero sections would look identical to
    "everything succeeded but found nothing" from the caller's side).

    When `requested` is omitted: "recovery" is auto-selected whenever the
    caller has a position (has_position=True, meaning avg_cost was
    supplied) - the field's presence already signals "can this recover to
    my cost?" without any text parsing - otherwise "full".
    """
    if isinstance(requested, str):
        requested = [requested]

    if requested:
        if len(requested) == 1 and requested[0] in PRESETS:
            return list(PRESETS[requested[0]])
        valid = [s for s in requested if s in ALL_SECTIONS]
        if valid:
            return valid
        return list(PRESETS["full"])

    if has_position:
        return list(PRESETS["recovery"])
    return list(PRESETS["full"])


def _compute_levels(current_price: Optional[float], atr_14: Optional[float]) -> Optional[Dict[str, Any]]:
    """Same entry/stop/target geometry agents/recommendation.py uses (2x ATR
    stop, 2:1 reward:risk) - kept in sync deliberately so risk_engine's
    risk/reward read never disagrees with the recommendation engine's own."""
    if not current_price or not atr_14:
        return None
    stop_distance = max(2.0 * atr_14, current_price * 0.02)
    return {
        "entry_zone_low": round(current_price - 0.5 * atr_14, 2),
        "entry_zone_high": round(current_price + 0.25 * atr_14, 2),
        "stop_loss": round(current_price - stop_distance, 2),
        "target_price": round(current_price + 2.0 * stop_distance, 2),
        "atr_14": round(atr_14, 2),
    }


def run_selected(
    ticker: str,
    sections: List[str],
    avg_cost: Optional[float] = None,
    shares: Optional[float] = None,
) -> Dict[str, Any]:
    """Fetch the base data every section needs (ticker price history,
    indicators, algo signals, fundamentals, pillars - all relatively cheap
    and, for the price history, cached), then call only the intelligence/
    modules `sections` actually asks for.

    Degrades honestly on a price-history fetch failure: returns confidence
    0.0 with a flag, rather than a partially-fabricated result.
    """
    from tools.market_data import (
        fetch_price_history, fetch_fundamentals, compute_indicators,
        compute_signal_summary, compute_algo_signals,
    )
    from backtest.pillars import compute_pillar_scores
    from backtest.risk import compute_atr
    from financial_data import get_bars_df

    ticker = ticker.upper().strip()
    flags: List[str] = []

    try:
        df = fetch_price_history(ticker, period="5y")
    except Exception:
        df = None
    if df is None or df.empty:
        return {"ticker": ticker, "sections": sections, "current_price": None,
                "confidence": 0.0, "flags": ["price_history_unavailable"]}

    indicators = compute_indicators(df)
    signal_summary = compute_signal_summary(indicators)
    algo_signals = compute_algo_signals(df, indicators)

    try:
        fundamentals = fetch_fundamentals(ticker)
    except Exception:
        fundamentals = {}
        flags.append("fundamentals_unavailable")

    pillar_snapshot = compute_pillar_scores(
        ticker, indicators, signal_summary, algo_signals, fundamentals, strict_fundamentals=False)
    pillars = pillar_snapshot.get("pillars", {})

    current_price = indicators.get("current_price")
    atr_14 = compute_atr(df)
    levels = _compute_levels(current_price, atr_14)

    # SPY/VIX are the genuinely expensive extras - fetched only when a
    # section that actually needs them was requested.
    spy_df = None
    vix_series = None
    if "regime" in sections or "analog" in sections:
        spy_df = get_bars_df("SPY", period="5y")
        try:
            from intelligence.regime import _fetch_vix_series
            vix_series = _fetch_vix_series()
        except Exception:
            vix_series = None

    regime = None
    if "regime" in sections:
        from intelligence.regime import compute_market_regime
        regime = compute_market_regime()

    historical_context = None
    if "historical_context" in sections:
        from intelligence.historical_context import compute_historical_context
        historical_context = compute_historical_context(
            ticker, df, algo_signals, indicators, benchmark_df=spy_df)

    analog = None
    if "analog" in sections:
        from intelligence.analog_engine import find_historical_analogs
        analog = find_historical_analogs(ticker, df, spy_df=spy_df, vix_series=vix_series)

    forecast = None
    if "forecast" in sections:
        from intelligence.prediction_engine import forecast_horizons
        forecast = forecast_horizons(
            ticker, current_price, pillars, algo_signals, regime=regime,
            historical_context=historical_context, analog_result=analog, atr_14=atr_14)

    position = {"avg_cost": avg_cost, "shares": shares} if avg_cost else None

    risk_profile = None
    if "risk" in sections:
        from intelligence.risk_engine import compute_risk_profile
        risk_profile = compute_risk_profile(
            ticker, current_price, levels, algo_signals, historical_context=historical_context,
            regime=regime, forecast=forecast, position=position)

    evidence = None
    if "evidence" in sections:
        from intelligence.evidence_synthesis import build_evidence_ledger
        evidence = build_evidence_ledger(
            ticker, pillars, regime=regime, historical_context=historical_context,
            analog_result=analog, risk_profile=risk_profile, position=position)

    return {
        "ticker": ticker,
        "sections": sections,
        "current_price": current_price,
        "levels": levels,
        "pillars": pillars,
        "regime": regime,
        "historical_context": historical_context,
        "analog": analog,
        "forecast": forecast,
        "risk_profile": risk_profile,
        "evidence": evidence,
        "flags": flags,
    }
