"""
Verification for intelligence/regime.py (item 2: market regime).

Run: python3 tests/test_regime.py

Offline/deterministic - financial_data.get/get_bars_df are patched at their
intelligence.regime binding (not the source module - the `from financial_data
import get, get_bars_df` at the top of regime.py creates a separate name
there, same lesson learned from agents/orchestrator.py's own _get_client
patching pattern this session). Synthetic OHLCV follows tests/test_pillars.py
::make_df's convention (drift dominates noise so directionality is
unambiguous).
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from intelligence.regime import compute_market_regime, compute_market_regime_series

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def make_df(kind: str, n: int = 300, seed: int = 7, growth: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = {"up": 0.002, "down": -0.002, "flat": 0.0}[kind]
    if growth:
        drift *= 1.8  # QQQ growth leg outrunning SPY for the growth_vs_value test
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n))), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.006, "Low": close * 0.994, "Close": close,
        "Volume": pd.Series(rng.uniform(1e6, 2e6, n), index=idx),
    })


def make_vix_result(level: float, n: int = 300) -> dict:
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    return {"data": [{"period_end": d.date().isoformat(), "value": level} for d in idx]}


def _bars_side_effect(spy_df, qqq_df):
    def _fn(symbol, period="1y", **kwargs):
        if symbol == "SPY":
            return spy_df
        if symbol == "QQQ":
            return qqq_df if qqq_df is not None else pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return _fn


def test_bull_market_low_vix_is_risk_on():
    spy = make_df("up", seed=1)
    qqq = make_df("up", seed=2)
    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq)), \
         patch("intelligence.regime.get", return_value=make_vix_result(12.0)):
        r = compute_market_regime()
    check("bull trend detected", r["trend"] == "BULLISH", str(r))
    check("low VIX -> LOW volatility regime", r["volatility_regime"] == "LOW")
    check("bullish + low vol -> RISK_ON", r["risk_stance"] == "RISK_ON")
    check("full data -> confidence 1.0", r["confidence"] == 1.0)
    check("no flags when everything is available", r["flags"] == [])


def test_bear_market_high_vix_is_risk_off():
    spy = make_df("down", seed=3)
    qqq = make_df("down", seed=4)
    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq)), \
         patch("intelligence.regime.get", return_value=make_vix_result(32.0)):
        r = compute_market_regime()
    check("bear trend detected", r["trend"] == "BEARISH", str(r))
    check("high VIX -> HIGH volatility regime", r["volatility_regime"] == "HIGH")
    check("bearish + high vol -> RISK_OFF", r["risk_stance"] == "RISK_OFF")


def test_missing_vix_degrades_honestly_not_fabricated():
    spy = make_df("up", seed=5)
    qqq = make_df("up", seed=6)
    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq)), \
         patch("intelligence.regime.get", return_value={"data": []}):
        r = compute_market_regime()
    check("volatility_regime is None, not fabricated", r["volatility_regime"] is None)
    check("vix_level is None", r["vix_level"] is None)
    check("still classifies trend from SPY alone", r["trend"] == "BULLISH")
    check("flags note vix_unavailable", "vix_unavailable" in r["flags"])
    check("confidence reduced below 1.0", r["confidence"] < 1.0)
    check("still produces a risk_stance from trend alone", r["risk_stance"] in ("RISK_ON", "NEUTRAL", "RISK_OFF"))


def test_missing_spy_returns_zero_confidence_no_guess():
    with patch("intelligence.regime.get_bars_df",
               return_value=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])), \
         patch("intelligence.regime.get", return_value=make_vix_result(20.0)):
        r = compute_market_regime()
    check("trend is None when SPY is unavailable (no fabrication)", r["trend"] is None)
    check("risk_stance is None when SPY is unavailable", r["risk_stance"] is None)
    check("confidence is exactly 0.0", r["confidence"] == 0.0)
    check("flagged spy_unavailable", "spy_unavailable" in r["flags"])


def test_growth_leading_and_lagging_classification():
    spy = make_df("flat", seed=8)
    qqq_leading = make_df("up", seed=9, growth=True)
    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq_leading)), \
         patch("intelligence.regime.get", return_value=make_vix_result(18.0)):
        r = compute_market_regime()
    check("QQQ outrunning flat SPY -> GROWTH_LEADING", r["growth_vs_value"] == "GROWTH_LEADING", str(r))

    qqq_lagging = make_df("down", seed=10, growth=True)
    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq_lagging)), \
         patch("intelligence.regime.get", return_value=make_vix_result(18.0)):
        r2 = compute_market_regime()
    check("QQQ falling faster than flat SPY -> GROWTH_LAGGING", r2["growth_vs_value"] == "GROWTH_LAGGING", str(r2))


def test_vectorized_series_agrees_with_live_at_last_bar():
    """The two implementations use different mechanisms (compute_signal_summary
    on the live path vs. technical_score_series bar-by-bar for the vectorized
    path) - they should still agree at the most recent bar, mirroring the
    consistency guarantee backtest/pillars.py itself documents and relies on."""
    spy = make_df("up", seed=11, n=300)
    qqq = make_df("up", seed=12, n=300)
    vix_level = 14.0
    vix_series = pd.Series(vix_level, index=spy.index)

    with patch("intelligence.regime.get_bars_df", side_effect=_bars_side_effect(spy, qqq)), \
         patch("intelligence.regime.get", return_value=make_vix_result(vix_level, n=300)):
        live = compute_market_regime()

    series = compute_market_regime_series(spy, qqq, vix_series)
    last = series.iloc[-1]

    check("vectorized trend at last bar matches the live trend",
          last["trend"] == live["trend"], f"series={last['trend']} live={live['trend']}")
    check("vectorized volatility_regime at last bar matches the live one",
          last["volatility_regime"] == live["volatility_regime"],
          f"series={last['volatility_regime']} live={live['volatility_regime']}")
    check("vectorized risk_stance at last bar matches the live one",
          last["risk_stance"] == live["risk_stance"],
          f"series={last['risk_stance']} live={live['risk_stance']}")


def test_series_handles_missing_qqq_and_vix_without_raising():
    spy = make_df("up", seed=13)
    try:
        series = compute_market_regime_series(spy, None, pd.Series(dtype=float))
        ok = True
    except Exception as e:
        ok = False
        series = None
    check("series computation never raises on missing QQQ/VIX", ok)
    if ok:
        check("growth_vs_value column exists and is all None without QQQ",
              "growth_vs_value" in series.columns and series["growth_vs_value"].isna().all())
        check("volatility_regime column exists and is all None without VIX",
              series["volatility_regime"].isna().all())


if __name__ == "__main__":
    test_bull_market_low_vix_is_risk_on()
    test_bear_market_high_vix_is_risk_off()
    test_missing_vix_degrades_honestly_not_fabricated()
    test_missing_spy_returns_zero_confidence_no_guess()
    test_growth_leading_and_lagging_classification()
    test_vectorized_series_agrees_with_live_at_last_bar()
    test_series_handles_missing_qqq_and_vix_without_raising()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — market regime: correct labels, honest degradation, live/series agree")
