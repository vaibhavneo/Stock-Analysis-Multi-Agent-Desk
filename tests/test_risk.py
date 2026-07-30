"""
Verification for stock_agent/backtest/risk.py (Kelly sizing + vol targeting +
correlation-aware sizing).

Run: python3 tests/test_risk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.risk import (
    kelly_fraction, safe_kelly_fraction, volatility_target_scale,
    correlation_aware_position_size,
)
from tools.market_data import fetch_price_history

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    print(f"  {name:60s} {status}  {detail}")
    if not condition:
        FAILURES.append(name)


# ══════════════════════════════════════════════════════════════════════════
# 1. Book ground-truth: the S&P 500's own Kelly fraction comes out to ~4.5
#    (Hilpisch p.293), computed with a near-zero risk-free rate. Reproducing a
#    same-order-of-magnitude result against SPY is the correctness anchor.
# ══════════════════════════════════════════════════════════════════════════

def test_spy_kelly_ballpark():
    print("\n=== test_spy_kelly_ballpark ===")
    df = fetch_price_history("SPY", period="10y")
    prices = df["Close"]
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Book uses ~0 risk-free rate to get its ~4.5 figure
    kelly_rf0 = kelly_fraction(log_returns, risk_free_rate=0.0)
    check(
        "SPY buy-and-hold Kelly (r=0) is same order of magnitude as book's ~4.5",
        1.0 < kelly_rf0 < 10.0,
        f"got {kelly_rf0:.2f} (book states ~4.5)",
    )

    # Default r=0.04 should still be a sane single-digit number, just lower
    kelly_rf4 = kelly_fraction(log_returns, risk_free_rate=0.04)
    check(
        "SPY Kelly with default r=0.04 is sane (positive, single-digit, < the r=0 value)",
        0 < kelly_rf4 < kelly_rf0,
        f"got {kelly_rf4:.2f} vs r=0 value {kelly_rf0:.2f}",
    )


# ══════════════════════════════════════════════════════════════════════════
# 2. Negative-edge strategy -> negative raw Kelly (not silently clipped to 0)
# ══════════════════════════════════════════════════════════════════════════

def test_negative_edge_negative_kelly():
    print("\n=== test_negative_edge_negative_kelly ===")
    # A synthetic strategy that loses money on average
    rng = np.random.default_rng(42)
    losing = pd.Series(rng.normal(-0.001, 0.01, 500))   # mean -0.1%/day
    raw = kelly_fraction(losing)
    check(
        "losing strategy produces NEGATIVE raw Kelly (signals 'don't take it')",
        raw < 0,
        f"raw kelly = {raw:.3f}",
    )
    # ...but safe_kelly floors it at 0 (you just don't take the position)
    safe = safe_kelly_fraction(losing)
    check(
        "safe_kelly_fraction floors a negative-edge strategy to 0",
        safe == 0.0,
        f"safe kelly = {safe}",
    )


# ══════════════════════════════════════════════════════════════════════════
# 3. Half-Kelly is exactly half the raw (when the cap doesn't bind)
# ══════════════════════════════════════════════════════════════════════════

def test_half_kelly_is_half():
    print("\n=== test_half_kelly_is_half ===")
    # Deterministic positive-edge series: force the daily mean to exactly
    # +0.0004 so raw Kelly is reliably POSITIVE and this test actually
    # exercises the "half of raw" path (not the floor-to-zero path).
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.015, 5000)
    winning = pd.Series(noise - noise.mean() + 0.0004)
    raw = kelly_fraction(winning)
    safe = safe_kelly_fraction(winning, kelly_multiplier=0.5, max_position_pct=1.0)  # high cap so it doesn't bind
    check(
        "raw Kelly is positive for this positive-edge series (exercises the real path)",
        raw > 0,
        f"raw={raw:.4f}",
    )
    check(
        "safe_kelly(mult=0.5) == exactly half the raw kelly (cap not binding)",
        abs(safe - 0.5 * raw) < 1e-9,
        f"raw={raw:.4f}, safe={safe:.4f}, 0.5*raw={0.5*raw:.4f}",
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. Degenerate near-zero-variance input doesn't produce an absurd fraction
# ══════════════════════════════════════════════════════════════════════════

def test_degenerate_variance():
    print("\n=== test_degenerate_variance ===")
    near_constant = pd.Series([0.0001] * 500)   # basically no variance
    safe = safe_kelly_fraction(near_constant, max_position_pct=0.10)
    check(
        "near-zero-variance input stays within the hard cap (no 500x blowup)",
        0.0 <= safe <= 0.10,
        f"safe kelly = {safe}",
    )


# ══════════════════════════════════════════════════════════════════════════
# 5. Volatility targeting scales the right direction
# ══════════════════════════════════════════════════════════════════════════

def test_vol_targeting_direction():
    print("\n=== test_vol_targeting_direction ===")
    scale_up = volatility_target_scale(realized_vol=0.075, target_vol=0.15)    # calmer than target
    scale_down = volatility_target_scale(realized_vol=0.30, target_vol=0.15)    # wilder than target
    check("vol target scales UP when realized vol < target", scale_up > 1.0, f"scale={scale_up:.2f}")
    check("vol target scales DOWN when realized vol > target", scale_down < 1.0, f"scale={scale_down:.2f}")
    check("vol target clips absurd upscale when realized vol near zero",
          volatility_target_scale(realized_vol=0.001, target_vol=0.15) <= 3.0)


# ══════════════════════════════════════════════════════════════════════════
# 6. Correlation-aware sizing shrinks redundant (correlated) positions
# ══════════════════════════════════════════════════════════════════════════

def test_correlation_aware_sizing():
    print("\n=== test_correlation_aware_sizing ===")
    rng = np.random.default_rng(1)
    base = rng.normal(0.0005, 0.01, 300)

    # Two nearly-identical strategies (same underlying bet)
    a = pd.Series(base + rng.normal(0, 0.0005, 300))
    b = pd.Series(base + rng.normal(0, 0.0005, 300))   # ~0.99 correlated with a
    # One genuinely independent strategy
    c = pd.Series(rng.normal(0.0005, 0.01, 300))

    proposed = {"a": 0.10, "b": 0.10, "c": 0.10}
    series = {"a": a, "b": b, "c": c}
    adjusted = correlation_aware_position_size(proposed, series, corr_threshold=0.7)

    combined_ab = adjusted["a"] + adjusted["b"]
    check(
        "highly-correlated pair's COMBINED size is shrunk below the naive sum (0.20)",
        combined_ab < 0.20,
        f"combined a+b = {combined_ab:.3f} (naive would be 0.20)",
    )
    check(
        "independent strategy c is left ~unchanged",
        abs(adjusted["c"] - 0.10) < 0.02,
        f"c = {adjusted['c']:.3f} (proposed 0.10)",
    )


# ══════════════════════════════════════════════════════════════════════════
# 7. End-to-end: size a real backtested strategy sanely
# ══════════════════════════════════════════════════════════════════════════

def test_real_strategy_sizing():
    print("\n=== test_real_strategy_sizing ===")
    from backtest.strategies import momentum_strategy
    from backtest.engine import run_vectorized_backtest

    df = fetch_price_history("NVDA", period="5y")
    prices = df["Close"]
    signal = momentum_strategy(prices).fillna(0)
    result = run_vectorized_backtest(prices, signal)

    safe = safe_kelly_fraction(result.strategy_returns)
    check(
        "real NVDA momentum strategy sizes to a sane fraction in [0, 0.10]",
        0.0 <= safe <= 0.10,
        f"safe kelly = {safe:.4f}",
    )


if __name__ == "__main__":
    test_spy_kelly_ballpark()
    test_negative_edge_negative_kelly()
    test_half_kelly_is_half()
    test_degenerate_variance()
    test_vol_targeting_direction()
    test_correlation_aware_sizing()
    test_real_strategy_sizing()

    print(f"\n{'='*64}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
