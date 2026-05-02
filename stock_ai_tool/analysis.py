from __future__ import annotations

import statistics
from typing import Iterable, List

from .models import AnalysisReport, IndicatorSnapshot, PriceBar


def _sma(values: List[float], window: int) -> float:
    if len(values) < window:
        return statistics.fmean(values) if values else 0.0
    return statistics.fmean(values[-window:])


def _momentum(values: List[float], lookback: int = 10) -> float:
    if len(values) <= lookback:
        if len(values) < 2 or values[0] == 0:
            return 0.0
        return (values[-1] - values[0]) / values[0]
    start = values[-lookback - 1]
    if start == 0:
        return 0.0
    return (values[-1] - start) / start


def _volatility(values: List[float], window: int = 20) -> float:
    if len(values) < 2:
        return 0.0
    sample = values[-window:] if len(values) >= window else values
    returns = []
    for i in range(1, len(sample)):
        prev = sample[i - 1]
        curr = sample[i]
        if prev != 0:
            returns.append((curr - prev) / prev)
    if len(returns) < 2:
        return 0.0
    return statistics.pstdev(returns)


def _rsi(values: List[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(len(values) - period, len(values)):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0 and gains == 0:
        return 50.0
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - (100 / (1 + rs))


def analyze_symbol(bars: Iterable[PriceBar]) -> AnalysisReport:
    bar_list = list(bars)
    if not bar_list:
        raise ValueError("No bars to analyze")

    closes = [bar.close for bar in bar_list]
    sma_short = _sma(closes, 5)
    sma_long = _sma(closes, 20)
    momentum = _momentum(closes, 10)
    rsi = _rsi(closes, 14)
    volatility = _volatility(closes, 20)

    trend_score = 1.0 if sma_short > sma_long else -1.0
    momentum_score = max(-1.0, min(1.0, momentum * 5))
    rsi_score = 0.0
    if rsi < 35:
        rsi_score = 0.7
    elif rsi > 70:
        rsi_score = -0.7
    risk_penalty = min(1.0, volatility * 20)

    raw_score = 0.5 * trend_score + 0.35 * momentum_score + 0.15 * rsi_score
    score = raw_score - 0.2 * risk_penalty
    confidence = min(0.99, max(0.05, abs(score)))

    if score >= 0.35:
        signal = "BUY"
        rationale = "Uptrend and positive momentum outweigh current risk."
    elif score <= -0.35:
        signal = "SELL"
        rationale = "Weak trend/momentum profile with elevated downside risk."
    else:
        signal = "HOLD"
        rationale = "Mixed conditions; no high-conviction edge right now."

    return AnalysisReport(
        symbol=bar_list[-1].symbol,
        signal=signal,
        confidence=round(confidence, 3),
        score=round(score, 3),
        last_price=bar_list[-1].close,
        indicators=IndicatorSnapshot(
            sma_short=round(sma_short, 4),
            sma_long=round(sma_long, 4),
            momentum=round(momentum, 4),
            rsi=round(rsi, 2),
            volatility=round(volatility, 4),
        ),
        rationale=rationale,
    )

