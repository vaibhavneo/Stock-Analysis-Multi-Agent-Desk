"""
Verification for intelligence/historical_context.py (item 1: multi-horizon
historical context).

Run: python3 tests/test_historical_context.py

Offline/deterministic - no network. Uses both a synthetic GBM series (the
tests/test_pillars.py::make_df convention) and a hand-constructed
deterministic linear series where exact return/drawdown values can be
computed independently, to catch an off-by-one in the horizon slicing that
a random series could mask.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from intelligence.historical_context import compute_historical_context, HORIZON_DAYS

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def make_df(kind: str, n: int = 1400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = {"up": 0.0015, "down": -0.0015, "flat": 0.0}[kind]
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n))), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.006, "Low": close * 0.994, "Close": close,
        "Volume": pd.Series(rng.uniform(1e6, 2e6, n), index=idx),
    })


def make_linear_df(n: int, start: float = 100.0, step: float = 0.0) -> pd.DataFrame:
    """Deterministic (non-random) price path - lets return_pct/drawdown be
    hand-computed exactly, to catch off-by-one horizon-slicing errors a
    random series could mask."""
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    close = pd.Series([start + step * i for i in range(n)], index=idx)
    return pd.DataFrame({
        "Open": close, "High": close, "Low": close, "Close": close,
        "Volume": pd.Series(1e6, index=idx),
    })


def test_full_history_all_horizons_available():
    df = make_df("up", n=1400)
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    for label in HORIZON_DAYS:
        check(f"{label} horizon available with 1400 bars of history",
              r["horizons"][label].get("data_available") is True, str(r["horizons"][label]))
    check("confidence is 1.0 when all horizons are available", r["confidence"] == 1.0)
    check("no insufficient-history flag when everything is available",
          "insufficient_history_for_any_horizon" not in r["flags"])


def test_short_history_only_short_horizons_available():
    df = make_df("up", n=300, seed=2)  # ~14 months of trading days
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    check("1M available with 300 bars", r["horizons"]["1M"]["data_available"] is True)
    check("3Y NOT available with only 300 bars (not fabricated)",
          r["horizons"]["3Y"] == {"data_available": False})
    check("5Y NOT available with only 300 bars (not fabricated)",
          r["horizons"]["5Y"] == {"data_available": False})
    check("confidence is between 0 and 1 with partial horizon coverage",
          0.0 < r["confidence"] < 1.0, str(r["confidence"]))


def test_empty_history_is_honest_not_fabricated():
    r = compute_historical_context("TEST", pd.DataFrame(), algo_signals={}, indicators={})
    check("confidence is exactly 0.0 with no price history", r["confidence"] == 0.0)
    check("flagged no_price_history", "no_price_history" in r["flags"])
    check("horizons dict is empty, not fabricated placeholders", r["horizons"] == {})


def test_exact_return_and_drawdown_on_deterministic_series():
    # A clean linear ramp: 300 bars from 100 -> 100+299*0.5 = 249.5.
    # 1M (21 trading days) return, hand-computed against the exact window.
    df = make_linear_df(n=300, start=100.0, step=0.5)
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    close = df["Close"]
    expected_1m = round((float(close.iloc[-1]) / float(close.iloc[-22]) - 1) * 100, 2)
    check("1M return_pct matches hand-computed value on a deterministic ramp",
          r["horizons"]["1M"]["return_pct"] == expected_1m,
          f"got {r['horizons']['1M']['return_pct']}, expected {expected_1m}")
    check("a monotonic uptrend has ~0% max drawdown over 1M",
          r["horizons"]["1M"]["max_drawdown_pct"] == 0.0, str(r["horizons"]["1M"]["max_drawdown_pct"]))
    check("a monotonic uptrend is classified UP", r["horizons"]["1M"]["trend"] == "UP")


def test_drawdown_detects_a_real_dip():
    # Ramp up, then a sharp drop, then flat - the 3M window should show a
    # real, specific max drawdown, not zero.
    n = 100
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    up = [100 + i for i in range(60)]
    down = [159 - i * 2 for i in range(20)]  # drop from 159 to 121
    flat = [121] * 20
    close = pd.Series(up + down + flat, index=idx)
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                        "Volume": pd.Series(1e6, index=idx)})
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    dd = r["horizons"]["3M"]["max_drawdown_pct"] if r["horizons"]["3M"]["data_available"] else None
    # 3M horizon needs > 63 bars; with only 100 bars it IS available (100 > 63).
    check("3M horizon available with 100 bars", dd is not None, str(r["horizons"]["3M"]))
    if dd is not None:
        check("a real ~24% peak-to-trough dip is detected as a real drawdown",
              dd < -15.0, f"max_drawdown_pct={dd}")


def test_algo_signals_momentum_is_reused_verbatim():
    df = make_df("up", n=400, seed=3)
    sentinel_momentum = {"momentum_1m": 12.34, "momentum_3m": 56.78, "momentum_6m": -1.11, "momentum_1y": 9.99}
    r = compute_historical_context("TEST", df, algo_signals=sentinel_momentum, indicators={})
    check("1M return_pct is the exact algo_signals value, not recomputed",
          r["horizons"]["1M"]["return_pct"] == 12.34)
    check("3M return_pct is the exact algo_signals value, not recomputed",
          r["horizons"]["3M"]["return_pct"] == 56.78)
    check("trend label is consistent with the reused (overridden) return_pct",
          r["horizons"]["1M"]["trend"] == "UP")  # 12.34 > TREND_DEADBAND_PCT


def test_support_resistance_windows():
    n = 300
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 150, n), index=idx)
    df = pd.DataFrame({"Open": close, "High": close + 2, "Low": close - 2, "Close": close,
                        "Volume": pd.Series(1e6, index=idx)})
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    check("20D support/resistance present", r["support_resistance"]["20D"]["data_available"] is not False)
    check("6M support/resistance present", r["support_resistance"]["6M"]["data_available"] is not False)
    check("1Y support/resistance present", r["support_resistance"]["1Y"]["data_available"] is not False)
    # A monotonic uptrend: the longer window's support must be lower (or equal)
    # than the shorter window's, since it spans further back into lower prices.
    check("1Y support <= 6M support on a monotonic uptrend (wider window, lower low)",
          r["support_resistance"]["1Y"]["support"] <= r["support_resistance"]["6M"]["support"])


def test_regime_changes_detects_a_real_flip():
    n = 500
    idx = pd.date_range("2019-01-02", periods=n, freq="B")
    rng = np.random.default_rng(4)
    bear_leg = 100 * np.exp(np.cumsum(rng.normal(-0.003, 0.006, 300)))
    bull_leg = bear_leg[-1] * np.exp(np.cumsum(rng.normal(0.004, 0.006, 200)))
    close = pd.Series(np.concatenate([bear_leg, bull_leg]), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.006, "Low": close * 0.994,
                        "Close": close, "Volume": pd.Series(1e6, index=idx)})
    r = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    rc = r["regime_changes"]
    check("regime_changes reports data_available", rc["data_available"] is True, str(rc))
    check("current regime reflects the recent bull leg, not the earlier bear leg",
          rc["current_regime"] in ("BULLISH", "NEUTRAL"), str(rc))
    check("at least one transition detected in the last 2 years",
          rc["transitions_last_2y"] >= 1, str(rc))


def test_relative_performance_with_and_without_benchmark():
    df = make_df("up", n=400, seed=5)
    bench = make_df("flat", n=400, seed=6)
    r_with = compute_historical_context("TEST", df, algo_signals={}, indicators={}, benchmark_df=bench)
    check("relative_performance populated when benchmark supplied",
          r_with["relative_performance"]["1M"].get("data_available") is True, str(r_with["relative_performance"]["1M"]))
    check("excess_return_pct = stock - benchmark",
          abs(r_with["relative_performance"]["1M"]["excess_return_pct"]
              - (r_with["relative_performance"]["1M"]["stock_return_pct"]
                 - r_with["relative_performance"]["1M"]["benchmark_return_pct"])) < 1e-6)
    check("no benchmark_not_supplied flag when a benchmark is supplied",
          "benchmark_not_supplied" not in r_with["flags"])

    r_without = compute_historical_context("TEST", df, algo_signals={}, indicators={})
    check("relative_performance is empty without a benchmark", r_without["relative_performance"] == {})
    check("flagged benchmark_not_supplied when omitted", "benchmark_not_supplied" in r_without["flags"])


def test_never_uses_data_beyond_what_it_was_given():
    """No-look-ahead sanity: computing on a truncated df must match hand-
    slicing the same truncated window - proving the function only ever sees
    the rows it's actually passed, never a full/global series."""
    df_full = make_df("up", n=1400, seed=9)
    cutoff = 1000
    df_truncated = df_full.iloc[:cutoff]

    r = compute_historical_context("TEST", df_truncated, algo_signals={}, indicators={})
    close_truncated = df_truncated["Close"]
    expected_1m = round((float(close_truncated.iloc[-1]) / float(close_truncated.iloc[-22]) - 1) * 100, 2)
    check("1M return_pct on a truncated df matches hand-slicing that exact truncated window",
          r["horizons"]["1M"]["return_pct"] == expected_1m,
          f"got {r['horizons']['1M']['return_pct']}, expected {expected_1m}")

    # 5Y (1260 bars) must be unavailable with only 1000 bars, proving the
    # function didn't quietly reach past the truncation to satisfy it.
    check("5Y correctly unavailable on the truncated df (would be available on the full one)",
          r["horizons"]["5Y"] == {"data_available": False})


if __name__ == "__main__":
    test_full_history_all_horizons_available()
    test_short_history_only_short_horizons_available()
    test_empty_history_is_honest_not_fabricated()
    test_exact_return_and_drawdown_on_deterministic_series()
    test_drawdown_detects_a_real_dip()
    test_algo_signals_momentum_is_reused_verbatim()
    test_support_resistance_windows()
    test_regime_changes_detects_a_real_flip()
    test_relative_performance_with_and_without_benchmark()
    test_never_uses_data_beyond_what_it_was_given()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — historical context: exact math, honest degradation, no look-ahead")
