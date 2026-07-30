"""
Verification for stock_agent/backtest/strategies.py.

Run: python3 tests/test_strategies.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
from backtest.engine import run_vectorized_backtest
from tools.market_data import fetch_price_history, compute_algo_signals, compute_indicators

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"  {name:65s} {status}  {detail}")
    if not condition:
        FAILURES.append(name)


# ══════════════════════════════════════════════════════════════════════════
# 1. Every strategy: valid signal values, runs cleanly through the engine.
# ══════════════════════════════════════════════════════════════════════════

def test_all_strategies_run_cleanly():
    print("\n=== test_all_strategies_run_cleanly ===")
    df = fetch_price_history("SPY", period="3y")
    prices = df["Close"]

    for name, fn in STRATEGY_REGISTRY.items():
        signal = fn(df) if name in STRATEGIES_NEEDING_FULL_DF else fn(prices)

        valid_values = set(signal.dropna().unique()) <= {-1, 0, 1}
        check(f"{name}: signal values only in {{-1,0,1}}", valid_values, f"unique={sorted(signal.dropna().unique())}")

        no_leading_nan_bleed = not signal.iloc[-50:].isna().any()
        check(f"{name}: no NaN in recent (backtestable) window", no_leading_nan_bleed)

        try:
            result = run_vectorized_backtest(prices, signal.fillna(0))
            check(f"{name}: runs through engine without error", True, f"{result.n_trades} trades")
        except Exception as e:
            check(f"{name}: runs through engine without error", False, str(e))


# ══════════════════════════════════════════════════════════════════════════
# 2. Consistency check: momentum & mean-reversion strategy signals should
#    agree DIRECTIONALLY with compute_algo_signals()'s own live labels for
#    the same date — proves "reuse" produced one consistent number, not two
#    silently-diverging implementations of the same idea.
# ══════════════════════════════════════════════════════════════════════════

def test_consistency_with_live_signals():
    print("\n=== test_consistency_with_live_signals ===")
    df = fetch_price_history("AAPL", period="2y")
    prices = df["Close"]
    indicators = compute_indicators(df)
    algo = compute_algo_signals(df, indicators)

    from backtest.strategies import momentum_strategy, mean_reversion_strategy

    mom_signal = momentum_strategy(prices)
    mr_signal = mean_reversion_strategy(prices)

    live_mom_signal = algo.get("momentum_signal", "")
    live_mr_signal = algo.get("mean_reversion_signal", "")

    today_mom = mom_signal.iloc[-1]
    today_mr = mr_signal.iloc[-1]

    mom_agrees = (
        (today_mom == 1 and live_mom_signal in ("BULLISH", "STRONG_BULLISH")) or
        (today_mom == -1 and live_mom_signal in ("BEARISH", "STRONG_BEARISH")) or
        (today_mom == 0 and live_mom_signal == "NEUTRAL")
    )
    check(
        "momentum_strategy's today signal agrees with live momentum_signal label",
        mom_agrees,
        f"strategy={today_mom} vs live={live_mom_signal} (composite={algo.get('momentum_composite')})",
    )

    mr_agrees = (
        (today_mr == 1 and live_mr_signal in ("BUY", "STRONG_BUY")) or
        (today_mr == -1 and live_mr_signal in ("SELL", "STRONG_SELL")) or
        (today_mr == 0 and live_mr_signal == "NEUTRAL")
    )
    check(
        "mean_reversion_strategy's today signal agrees with live mean_reversion_signal label",
        mr_agrees,
        f"strategy={today_mr} vs live={live_mr_signal} (zscore={algo.get('mean_reversion_zscore')})",
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. SMA crossover sanity: on a clean synthetic uptrend, should end up long.
# ══════════════════════════════════════════════════════════════════════════

def test_sma_crossover_synthetic_trend():
    print("\n=== test_sma_crossover_synthetic_trend ===")
    from backtest.strategies import sma_crossover_strategy

    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    prices = pd.Series(np.linspace(100, 200, 400), index=dates)  # clean linear uptrend

    signal = sma_crossover_strategy(prices, fast=20, slow=100)
    check(
        "SMA crossover goes long on a clean sustained uptrend",
        signal.iloc[-1] == 1,
        f"final signal = {signal.iloc[-1]}",
    )


if __name__ == "__main__":
    test_all_strategies_run_cleanly()
    test_consistency_with_live_signals()
    test_sma_crossover_synthetic_trend()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
