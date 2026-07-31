from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from .analysis import analyze_symbol
from .models import AnalysisReport, FundamentalSnapshot, PriceBar


def _trend_slope(closes: List[float], window: int = 20) -> float:
    """Least-squares slope over last `window` closes (price per day)."""
    sample = closes[-window:] if len(closes) >= window else closes
    n = len(sample)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.fmean(sample)
    num = sum((i - x_mean) * (sample[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def price_target(
    symbol: str,
    bars: List[PriceBar],
    fundamental: Optional[FundamentalSnapshot],
    report: AnalysisReport,
) -> dict:
    ind = report.indicators
    last_price = report.last_price
    closes = [b.close for b in bars]
    vol = ind.volatility
    rsi = ind.rsi
    momentum = ind.momentum
    sma_long = ind.sma_long
    score = report.score

    # --- fundamental fair value ---
    pe = fundamental.pe_ratio if fundamental and fundamental.pe_ratio > 0 else 25.0
    eps_growth = fundamental.eps_growth if fundamental and fundamental.eps_growth else 0.10
    estimated_eps = last_price / pe
    forward_pe = _clamp(pe / (1 + eps_growth), 5.0, 200.0)
    fair_value_pe = estimated_eps * forward_pe

    # --- trend targets ---
    slope = _trend_slope(closes, 20)

    # RSI adjustment factor: buy dip / trim peak
    if rsi < 40:
        rsi_adj = 1 + (40 - rsi) / 200.0   # up to +5%
    elif rsi > 65:
        rsi_adj = 1 - (rsi - 65) / 300.0   # down to -~1.2%
    else:
        rsi_adj = 1.0

    week_target = max(0.01, (last_price + slope * 5) * rsi_adj)
    month_target = max(0.01, last_price + slope * 21)
    q_target = max(0.01, fair_value_pe * 0.70 + month_target * 0.30)

    # --- bull / bear bands ---
    band = max(0.05, vol * 2.5)
    bull_target = month_target * (1 + band)
    bear_target = month_target * (1 - band)

    # --- buy zone and stop-loss ---
    buy_lo = last_price * (1 - max(0.015, vol * 1.35))
    buy_hi = last_price * (1 + max(0.008, vol * 0.55))
    stop_loss = min(sma_long, last_price * (1 - max(0.055, vol * 3.2)))

    # --- signal ---
    if rsi < 40 and momentum > 0 and score > 0.20:
        signal = "BUY_WINDOW"
        signal_note = "RSI oversold + positive momentum: favourable entry"
    elif rsi > 65 and momentum < 0:
        signal = "TAKE_PROFIT_ZONE"
        signal_note = "RSI overbought + weakening momentum: trim position"
    elif score >= 0.60:
        signal = "BULLISH"
        signal_note = "Strong composite score supports accumulation"
    elif score <= 0.35:
        signal = "CAUTIOUS"
        signal_note = "Weak composite score; wait for confirmation"
    else:
        signal = "NEUTRAL"
        signal_note = "Mixed signals; hold existing position"

    return {
        "symbol": symbol,
        "last_price": round(last_price, 2),
        "fair_value_pe": round(fair_value_pe, 2),
        "targets": {
            "1_week": round(week_target, 2),
            "1_month": round(month_target, 2),
            "3_month": round(q_target, 2),
            "bull": round(bull_target, 2),
            "bear": round(bear_target, 2),
        },
        "buy_zone": {
            "low": round(buy_lo, 2),
            "high": round(buy_hi, 2),
        },
        "stop_loss": round(stop_loss, 2),
        "signal": signal,
        "signal_note": signal_note,
        "indicators": {
            "rsi": round(rsi, 1),
            "momentum": round(momentum, 4),
            "volatility": round(vol, 4),
            "trend_slope": round(slope, 4),
        },
        "action": report.signal,
        "composite_score": round(score, 3),
    }


def next_day_estimate(bars: List[PriceBar]) -> dict:
    """Single-session-ahead price estimate driven by trend slope + RSI reversion.
    Shared by the live predict endpoint and the introspection backtester so both
    use the exact same model — the backtest is only meaningful if it scores the
    same formula that powers the live signal."""
    report = analyze_symbol(bars)
    closes = [b.close for b in bars]
    last_price = report.last_price
    rsi = report.indicators.rsi
    slope = _trend_slope(closes, 20)

    if rsi < 40:
        rsi_adj = 1 + (40 - rsi) / 200.0
    elif rsi > 65:
        rsi_adj = 1 - (rsi - 65) / 300.0
    else:
        rsi_adj = 1.0

    predicted_close = max(0.01, (last_price + slope * 1) * rsi_adj)
    if predicted_close > last_price:
        direction = "UP"
    elif predicted_close < last_price:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return {
        "last_price": round(last_price, 4),
        "predicted_close": round(predicted_close, 4),
        "predicted_direction": direction,
        "signal": report.signal,
    }


def prediction_for_all(
    bars_by_symbol: Dict[str, List[PriceBar]],
    fundamentals: Dict[str, FundamentalSnapshot],
    reports: Optional[Dict[str, AnalysisReport]] = None,
) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for symbol, bars in bars_by_symbol.items():
        if not bars:
            continue
        try:
            if reports and symbol in reports:
                report = reports[symbol]
            else:
                report = analyze_symbol(bars)
            fundamental = fundamentals.get(symbol)
            result[symbol] = price_target(symbol, bars, fundamental, report)
        except Exception as exc:
            result[symbol] = {"symbol": symbol, "error": str(exc)}
    return result
