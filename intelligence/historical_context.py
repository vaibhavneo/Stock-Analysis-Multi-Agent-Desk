"""
Multi-horizon historical context (item 1) - trend, momentum, volatility,
drawdown, support/resistance, per-stock regime changes, and relative
performance vs a benchmark, from price history the app already fetches
(agents/orchestrator.py already refetches 5y for grounding - this reuses
that same df, no new price fetch needed for the ticker itself).

Momentum for 1M/3M/6M/1Y reuses tools/market_data.py::compute_algo_signals'
own momentum_1w/1m/3m/6m/1y fields as-is (identical pct_return(n) formula,
n=21/63/126/252) rather than recomputing them, so this module's numbers
never silently disagree with what's already displayed elsewhere in the app.
Only 3Y/5Y are genuinely new. Regime-change detection extends
backtest/pillars.py's already-proven technical_score_series; the 6M/1Y
support/resistance windows extend tools/market_data.py's existing 20-day
rolling min/max formula to longer windows.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

HORIZON_DAYS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}
TREND_DEADBAND_PCT = 3.0


def _horizon_metrics(close: pd.Series, n: int,
                      override_return_pct: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Trend/return/volatility/drawdown over the trailing n trading days.
    None (never a fabricated number) when there isn't enough history."""
    if len(close) <= n:
        return None
    window = close.tail(n + 1)
    start, end = float(window.iloc[0]), float(window.iloc[-1])
    if start == 0:
        return None

    return_pct = round(float(override_return_pct), 2) if override_return_pct is not None \
        else round((end / start - 1) * 100, 2)

    returns = window.pct_change().dropna()
    vol_annualized = round(float(returns.std() * math.sqrt(252) * 100), 2) if len(returns) > 1 else None

    running_max = window.cummax()
    drawdown = (window - running_max) / running_max
    max_drawdown_pct = round(float(drawdown.min() * 100), 2)

    if return_pct > TREND_DEADBAND_PCT:
        trend = "UP"
    elif return_pct < -TREND_DEADBAND_PCT:
        trend = "DOWN"
    else:
        trend = "FLAT"

    return {
        "return_pct": return_pct,
        "volatility_annualized_pct": vol_annualized,
        "max_drawdown_pct": max_drawdown_pct,
        "trend": trend,
        "data_available": True,
    }


def _support_resistance(df: pd.DataFrame, n: int) -> Dict[str, Any]:
    if len(df) < n:
        return {"data_available": False}
    window = df.tail(n)
    return {
        "support": round(float(window["Low"].min()), 2),
        "resistance": round(float(window["High"].max()), 2),
        "data_available": True,
    }


def _regime_changes(close: pd.Series) -> Dict[str, Any]:
    """Per-stock technical regime persistence: how long the current
    bullish/bearish/neutral technical read has held, and how many times it
    flipped in the last 2 years - a "how choppy has this been" signal,
    distinct from intelligence/regime.py's market-wide regime."""
    from backtest.pillars import technical_score_series

    if len(close) < 20:
        return {"current_regime": None, "current_regime_days": None,
                "transitions_last_2y": None, "data_available": False}

    try:
        score = technical_score_series(pd.DataFrame({"Close": close}))
    except Exception:
        return {"current_regime": None, "current_regime_days": None,
                "transitions_last_2y": None, "data_available": False}

    label = pd.Series("NEUTRAL", index=score.index)
    label[score >= 65] = "BULLISH"
    label[score <= 35] = "BEARISH"

    changed = label.ne(label.shift(1))
    changed.iloc[0] = False
    change_dates = label.index[changed]
    current_regime_start = change_dates[-1] if len(change_dates) else label.index[0]
    current_regime_days = int((label.index[-1] - current_regime_start).days)

    two_years_ago = label.index[-1] - pd.Timedelta(days=730)
    transitions_last_2y = int(changed[label.index >= two_years_ago].sum())

    return {
        "current_regime": str(label.iloc[-1]),
        "current_regime_days": current_regime_days,
        "transitions_last_2y": transitions_last_2y,
        "data_available": True,
    }


def compute_historical_context(
    ticker: str,
    df_long: pd.DataFrame,
    algo_signals: Dict[str, Any],
    indicators: Dict[str, Any],
    benchmark_df: Optional[pd.DataFrame] = None,
    benchmark_symbol: str = "SPY",
) -> Dict[str, Any]:
    """Multi-horizon (1M/3M/6M/1Y/3Y/5Y, where data exists) trend, momentum,
    volatility, drawdown, support/resistance, per-stock regime persistence,
    and relative performance vs a benchmark.

    Degrades honestly per horizon: a horizon with insufficient history
    returns {"data_available": False} for that horizon specifically, rather
    than fabricating a number or dropping the key entirely - callers can
    tell "we don't know yet" from "this stock has 0% return."
    """
    if df_long is None or df_long.empty or "Close" not in df_long.columns:
        return {
            "horizons": {}, "support_resistance": {}, "regime_changes": None,
            "relative_performance": {}, "confidence": 0.0,
            "flags": ["no_price_history"],
        }

    close = df_long["Close"].astype(float)
    flags: List[str] = []

    algo_momentum_map = {
        "1M": algo_signals.get("momentum_1m"),
        "3M": algo_signals.get("momentum_3m"),
        "6M": algo_signals.get("momentum_6m"),
        "1Y": algo_signals.get("momentum_1y"),
    }

    horizons: Dict[str, Any] = {}
    for label, n in HORIZON_DAYS.items():
        m = _horizon_metrics(close, n, override_return_pct=algo_momentum_map.get(label))
        horizons[label] = m if m is not None else {"data_available": False}

    available_count = sum(1 for h in horizons.values() if h.get("data_available"))
    if available_count == 0:
        flags.append("insufficient_history_for_any_horizon")

    support_resistance: Dict[str, Any] = {}
    for label, n in (("20D", 20), ("6M", 126), ("1Y", 252)):
        support_resistance[label] = _support_resistance(df_long, n)

    regime_changes = _regime_changes(close)

    relative_performance: Dict[str, Any] = {}
    if benchmark_df is not None and not benchmark_df.empty and "Close" in benchmark_df.columns:
        bench_close = benchmark_df["Close"].astype(float)
        for label, n in HORIZON_DAYS.items():
            stock_m = horizons.get(label)
            bench_m = _horizon_metrics(bench_close, n)
            if not stock_m or not stock_m.get("data_available") or bench_m is None:
                relative_performance[label] = {"data_available": False}
                continue
            relative_performance[label] = {
                "stock_return_pct": stock_m["return_pct"],
                "benchmark_return_pct": bench_m["return_pct"],
                "excess_return_pct": round(stock_m["return_pct"] - bench_m["return_pct"], 2),
                "benchmark_symbol": benchmark_symbol,
                "data_available": True,
            }
    else:
        # Not a data-quality problem - benchmark_df is an OPTIONAL input the
        # caller may simply not have supplied. Noted for transparency, but
        # doesn't reduce confidence in the horizons/support-resistance/regime
        # data that WAS computed, which is an independent concern.
        flags.append("benchmark_not_supplied")

    confidence = available_count / len(HORIZON_DAYS)

    return {
        "horizons": horizons,
        "support_resistance": support_resistance,
        "regime_changes": regime_changes,
        "relative_performance": relative_performance,
        "confidence": round(confidence, 2),
        "flags": flags,
    }
