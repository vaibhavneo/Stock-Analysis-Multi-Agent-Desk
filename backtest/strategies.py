"""
Backtestable Strategy Library
===============================
7 concrete strategies, each `f(prices, **params) -> pd.Series` returning a
{-1, 0, 1} position signal, directly consumable by backtest.engine's
run_vectorized_backtest(). Pairs trading (cointegration-based, needs a second
ticker) is deliberately deferred — see the stock_agent upgrade plan.

Reuse discipline: wherever tools/market_data.py already computes a signal
value for the LIVE dashboard (mean_reversion_zscore, momentum_composite,
rsi_14, macd_cross), this module recomputes the exact same rolling-window
formula and thresholds as a full historical Series (not a single "today"
snapshot) — see each strategy's docstring for the exact source lines being
mirrored. This is necessary because the live functions only ever compute the
latest value; backtesting needs a value on every historical day. It is NOT a
second, independently-invented implementation of the same idea — the window
sizes and thresholds are copied verbatim from tools/market_data.py so a
strategy's signal and the live dashboard's own label agree on the same date
(see tests/test_strategies.py's consistency checks).

Sources: Hilpisch, "Python for Algorithmic Trading" (SMA crossover, p.109-113;
trend-following via SMA(252), Ch.4); Liu, "Quantitative Trading Strategies
Using Python" (momentum Ch.5 p.149-174, stat-arb Ch.8 p.225-255, RSI/MACD
p.175); existing tools/market_data.py (momentum/mean-reversion/RSI/MACD/
candlestick signal sources, reused per the discipline above).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════════════════
# 1. SMA Crossover (Hilpisch, Ch.4, p.109-113)
# ══════════════════════════════════════════════════════════════════════════

def sma_crossover_strategy(prices: pd.Series, fast: int = 42, slow: int = 252) -> pd.Series:
    """Long when the fast SMA is above the slow SMA, short otherwise —
    the book's own SMA1/SMA2 construction verbatim."""
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    signal = pd.Series(np.where(sma_fast > sma_slow, 1, -1), index=prices.index)
    signal[sma_fast.isna() | sma_slow.isna()] = 0
    return signal


# ══════════════════════════════════════════════════════════════════════════
# 2. Momentum (mirrors tools/market_data.py's momentum_composite formula:
#    average of 1M/3M/6M (21/63/126-day) % returns, thresholds ±3/±10)
# ══════════════════════════════════════════════════════════════════════════

def momentum_strategy(prices: pd.Series, bullish_threshold: float = 3.0, bearish_threshold: float = -3.0) -> pd.Series:
    ret_1m = prices.pct_change(21) * 100
    ret_3m = prices.pct_change(63) * 100
    ret_6m = prices.pct_change(126) * 100
    composite = pd.concat([ret_1m, ret_3m, ret_6m], axis=1).mean(axis=1)

    signal = pd.Series(0, index=prices.index)
    signal[composite > bullish_threshold] = 1
    signal[composite < bearish_threshold] = -1
    return signal


# ══════════════════════════════════════════════════════════════════════════
# 3. Mean Reversion / Bollinger (mirrors tools/market_data.py's
#    mean_reversion_zscore: 20-day rolling Z-score, thresholds ±0.8)
# ══════════════════════════════════════════════════════════════════════════

def mean_reversion_strategy(prices: pd.Series, window: int = 20, threshold: float = 0.8) -> pd.Series:
    """Long when price is significantly BELOW its rolling mean (buy the dip),
    short when significantly ABOVE it (fade the rally) — the classical
    mean-reversion assumption. Note this is the OPPOSITE sign convention to
    reading the Z-score itself: a negative Z-score (price below mean) is a
    BUY signal (+1)."""
    roll_mean = prices.rolling(window).mean()
    roll_std = prices.rolling(window).std()
    z = (prices - roll_mean) / roll_std

    signal = pd.Series(0, index=prices.index)
    signal[z < -threshold] = 1
    signal[z > threshold] = -1
    return signal


# ══════════════════════════════════════════════════════════════════════════
# 4. Stat-Arb (t-test on rolling return distribution vs. zero mean)
#    Liu, "Quantitative Trading Strategies Using Python", Ch.8, p.225-255
# ══════════════════════════════════════════════════════════════════════════

def stat_arb_strategy(prices: pd.Series, window: int = 60, p_threshold: float = 0.05) -> pd.Series:
    """Rolling one-sample t-test: reject H0 (mean daily return = 0) at
    p < p_threshold, then trade in the direction of the significant mean.
    A rolling t-test is computed via .rolling().apply() — this is the one
    strategy here that can't be vectorized into pure array ops, since scipy's
    ttest_1samp needs the actual window of values, not just a rolling stat."""
    log_returns = np.log(prices / prices.shift(1))

    def _signal_from_window(window_returns: np.ndarray) -> float:
        window_returns = window_returns[~np.isnan(window_returns)]
        if len(window_returns) < 10:
            return 0.0
        t_stat, p_value = scipy_stats.ttest_1samp(window_returns, popmean=0.0)
        if p_value < p_threshold:
            return 1.0 if t_stat > 0 else -1.0
        return 0.0

    signal = log_returns.rolling(window).apply(_signal_from_window, raw=True)
    return signal.fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════
# 5. Trend-Following (SMA(252) + N-day breakout)
# ══════════════════════════════════════════════════════════════════════════

def trend_following_strategy(prices: pd.Series, sma_window: int = 252, breakout_window: int = 20) -> pd.Series:
    """Long when price is above its 252-day SMA (mirrors market_data.py's
    above_sma_200 concept, using the more standard 252-trading-day year) AND
    makes a new N-day high; short on the symmetric breakdown condition."""
    sma = prices.rolling(sma_window).mean()
    rolling_high = prices.rolling(breakout_window).max()
    rolling_low = prices.rolling(breakout_window).min()

    above_trend = prices > sma
    below_trend = prices < sma
    new_high = prices >= rolling_high
    new_low = prices <= rolling_low

    signal = pd.Series(0, index=prices.index)
    signal[above_trend & new_high] = 1
    signal[below_trend & new_low] = -1
    return signal


# ══════════════════════════════════════════════════════════════════════════
# 6. Candlestick-Filtered (detect_candlestick_signal + RSI filter)
# ══════════════════════════════════════════════════════════════════════════

def candlestick_filtered_strategy(df: pd.DataFrame, rsi_oversold: float = 40.0, rsi_overbought: float = 60.0) -> pd.Series:
    """A bullish engulfing pattern only counts as a BUY if RSI isn't already
    overbought (avoids buying an exhausted rally); symmetric for bearish."""
    from tools.market_data import detect_candlestick_signal
    import ta

    candlestick_signal = detect_candlestick_signal(df)
    rsi = ta.momentum.RSIIndicator(df["Close"].astype(float), window=14).rsi()

    signal = pd.Series(0, index=df.index)
    signal[(candlestick_signal == 1) & (rsi < rsi_overbought)] = 1
    signal[(candlestick_signal == -1) & (rsi > rsi_oversold)] = -1
    return signal


# ══════════════════════════════════════════════════════════════════════════
# 7. RSI/MACD Threshold (mirrors market_data.py's rsi_signal/macd_cross)
# ══════════════════════════════════════════════════════════════════════════

def rsi_macd_strategy(prices: pd.Series, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0) -> pd.Series:
    """RSI < 30 (oversold) -> buy; RSI > 70 (overbought) -> sell; confirmed
    by MACD crossing in the same direction. Mirrors market_data.py's
    rsi_signal thresholds (30/70) and macd_cross bullish/bearish labels."""
    import ta

    rsi = ta.momentum.RSIIndicator(prices, window=14).rsi()
    macd_ind = ta.trend.MACD(prices)
    macd_line = macd_ind.macd()
    macd_signal_line = macd_ind.macd_signal()
    macd_bullish = macd_line > macd_signal_line

    signal = pd.Series(0, index=prices.index)
    signal[(rsi < rsi_oversold) & macd_bullish] = 1
    signal[(rsi > rsi_overbought) & ~macd_bullish] = -1
    return signal


# ══════════════════════════════════════════════════════════════════════════
# Registry — for the API layer (Phase 5) to enumerate available strategies
# ══════════════════════════════════════════════════════════════════════════

def seven_pillar_core(df, enter: float = 65.0, exit: float = 55.0):
    """Long-only composite of the backtestable pillar subset (technical + algo
    voting meters × rolling risk multiplier), weekly cadence + hysteresis.
    Implementation and honesty notes: backtest/pillars.py."""
    from backtest.pillars import seven_pillar_core_strategy
    return seven_pillar_core_strategy(df, enter=enter, exit=exit)


STRATEGY_REGISTRY: dict[str, Callable] = {
    "sma_crossover":         sma_crossover_strategy,
    "momentum":              momentum_strategy,
    "mean_reversion":        mean_reversion_strategy,
    "stat_arb":              stat_arb_strategy,
    "trend_following":       trend_following_strategy,
    "candlestick_filtered":  candlestick_filtered_strategy,   # needs full df, not just prices
    "rsi_macd":              rsi_macd_strategy,
    "seven_pillar_core":     seven_pillar_core,               # needs full df (OHLCV + Volume)
}

# Which strategies need the full OHLCV DataFrame vs. just the Close price Series
STRATEGIES_NEEDING_FULL_DF = {"candlestick_filtered", "seven_pillar_core"}
