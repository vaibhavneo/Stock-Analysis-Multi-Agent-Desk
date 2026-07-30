"""
Verification for stock_agent/backtest/engine.py.

Run: python3 tests/test_backtest_engine.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.engine import (
    run_vectorized_backtest,
    compute_performance_metrics,
    deflated_sharpe_ratio,
    _expected_max_sharpe,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"  {name:55s} {status}  {detail}")
    if not condition:
        FAILURES.append(name)


# ══════════════════════════════════════════════════════════════════════════
# 1. Synthetic leak test — the primary, self-contained proof of point-in-time
#    correctness. Construct a price series flat for 100 days then a scripted
#    +50% jump on day 101, with a signal that "predicts" it by going long
#    exactly one day BEFORE the jump (day 100). If the engine is leak-free,
#    that trade's position is only active starting day 101 (via the internal
#    shift), so the jump's return correctly accrues; if a signal flips to +1
#    ON THE SAME DAY as the jump itself, that trade must NOT capture it.
# ══════════════════════════════════════════════════════════════════════════

def test_synthetic_leak_free():
    print("\n=== test_synthetic_leak_free ===")
    n = 150
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.Series(100.0, index=dates)
    jump_day = 100
    prices.iloc[jump_day:] = 150.0   # +50% jump occurs ON index `jump_day`

    # Case A: signal goes long the day BEFORE the jump (correctly anticipates it)
    signal_early = pd.Series(0, index=dates)
    signal_early.iloc[jump_day - 1:] = 1
    result_a = run_vectorized_backtest(prices, signal_early, transaction_cost_bps=0)
    jump_date = dates[jump_day]
    captured_return_a = result_a.strategy_returns.get(jump_date, 0.0)
    check(
        "signal set day-before-jump captures the jump's return",
        captured_return_a > 0.3,
        f"strategy_return on jump day = {captured_return_a:.4f} (expect ~log(1.5)={np.log(1.5):.4f})",
    )

    # Case B: signal only flips to long ON the jump day itself (too late — should NOT capture it)
    signal_late = pd.Series(0, index=dates)
    signal_late.iloc[jump_day:] = 1
    result_b = run_vectorized_backtest(prices, signal_late, transaction_cost_bps=0)
    captured_return_b = result_b.strategy_returns.get(jump_date, 0.0)
    check(
        "signal set ON jump day does NOT capture that day's return (no look-ahead)",
        abs(captured_return_b) < 1e-9,
        f"strategy_return on jump day = {captured_return_b:.6f} (expect 0.0)",
    )

    # Off-by-one audit: first tradeable day has zero position (no info yet)
    check(
        "first executed_position is 0 (no information on day 0)",
        result_a.executed_position.iloc[0] == 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. Drawdown formula unit test — hand-constructed series with a known dip.
# ══════════════════════════════════════════════════════════════════════════

def test_drawdown_formula():
    print("\n=== test_drawdown_formula ===")
    # Log returns: up 10%, up 10%, down 30%, up 5%, up 5%
    log_rets = pd.Series([0.10, 0.10, -0.30, 0.05, 0.05])
    cumulative = np.exp(log_rets.cumsum())
    cummax = cumulative.cummax()
    # RELATIVE drawdown (fraction of peak), matching the engine.
    drawdown = (cummax - cumulative) / cummax
    expected_max_dd = float(drawdown.max())

    metrics = compute_performance_metrics(log_rets)
    check(
        "max_drawdown matches manual arithmetic",
        abs(metrics["max_drawdown"] - expected_max_dd) < 1e-3,  # engine rounds to 4dp
        f"got {metrics['max_drawdown']}, expected {expected_max_dd:.6f}",
    )
    check(
        "max_drawdown is a valid fraction in [0,1]",
        0.0 <= metrics["max_drawdown"] <= 1.0,
        f"got {metrics['max_drawdown']}",
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. Deflated Sharpe Ratio sanity bounds
# ══════════════════════════════════════════════════════════════════════════

def test_dsr_bounds():
    print("\n=== test_dsr_bounds ===")
    dsr_low_trials = deflated_sharpe_ratio(sharpe_hat=1.0, n_trials=1, skewness=0.0, kurtosis=3.0, n_obs=252)
    dsr_high_trials = deflated_sharpe_ratio(sharpe_hat=1.0, n_trials=1000, skewness=0.0, kurtosis=3.0, n_obs=252)

    check("dSR(1 trial) in [0,1]", 0.0 <= dsr_low_trials <= 1.0, f"{dsr_low_trials:.4f}")
    check("dSR(1000 trials) in [0,1]", 0.0 <= dsr_high_trials <= 1.0, f"{dsr_high_trials:.4f}")
    check(
        "dSR decreases as n_trials increases (fixed sharpe_hat) — more trials, same result looks less impressive",
        dsr_high_trials < dsr_low_trials,
        f"1 trial={dsr_low_trials:.4f} vs 1000 trials={dsr_high_trials:.4f}",
    )

    # Sanity anchor from the book: "if we conduct 1,000 trials, the expected
    # maximum Sharpe ratio is 3.26 even though the true Sharpe ratio is zero"
    # (López de Prado, Ch.8) — check our E[max SR] approximation is in a
    # plausible neighborhood of that stated figure for n_trials=1000.
    e_max_sr_1000 = _expected_max_sharpe(1000)
    check(
        "E[max Sharpe] at 1000 trials is in a plausible range of the book's stated 3.26",
        2.0 < e_max_sr_1000 < 4.5,
        f"got {e_max_sr_1000:.3f} (book states ~3.26)",
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. Real-data directional consistency check (Hilpisch EUR/USD SMA example)
#    — NOT an exact-decimal reproduction (see engine.py docstring for why),
#    but the qualitative conclusion (strategy beats buy-and-hold; buy-and-hold
#    is negative over this period) must hold on the actual public dataset.
# ══════════════════════════════════════════════════════════════════════════

def test_hilpisch_eurusd_directional():
    print("\n=== test_hilpisch_eurusd_directional ===")
    csv_path = "/tmp/pyalgo_eikon_eod_data.csv"
    try:
        raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("  SKIPPED — companion dataset not present at", csv_path)
        return

    prices = pd.DataFrame(raw["EUR="]).dropna()["EUR="]
    prices = prices.loc["2010-1-1":"2019-12-31"]

    sma1 = prices.rolling(42).mean()
    sma2 = prices.rolling(252).mean()
    signal = pd.Series(np.where(sma1 > sma2, 1, -1), index=prices.index)
    signal[sma1.isna() | sma2.isna()] = 0

    result = run_vectorized_backtest(prices, signal, transaction_cost_bps=0)
    buy_hold_return = float(result.log_returns.dropna().sum())
    strategy_return = float(result.strategy_returns.sum())

    check(
        "buy-and-hold cumulative log return is negative over this period (matches book's narrative)",
        buy_hold_return < 0,
        f"{buy_hold_return:.4f}",
    )
    check(
        "SMA(42,252) strategy beats buy-and-hold (matches book's stated conclusion)",
        strategy_return > buy_hold_return,
        f"strategy={strategy_return:.4f} vs buy_hold={buy_hold_return:.4f}",
    )
    print(f"  (Book states exactly: returns=-0.176731, strategy=0.253121 — see engine.py docstring")
    print(f"   for why current public dataset reproduces the same conclusion but different exact decimals)")


if __name__ == "__main__":
    test_synthetic_leak_free()
    test_drawdown_formula()
    test_dsr_bounds()
    test_hilpisch_eurusd_directional()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
