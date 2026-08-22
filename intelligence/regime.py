"""
Market regime classification (item 2) - was entirely absent from this
codebase before: only a per-ticker realized-volatility "vol_regime" existed
(tools/market_data.py), nothing market-wide. This module fills that gap
using data that's already fetchable but previously unused: SPY/QQQ (already
proven as a generic symbol fetch via financial_data.get_bars_df, used
elsewhere for SPY as a benchmark) and VIX (keyless via the cboe provider,
already cached, zero prior callers).

Reuses tools/market_data.py's own compute_indicators/compute_signal_summary
for the live SPY trend classification - same functions, new input - so the
regime's "BULLISH/BEARISH/NEUTRAL" language and thresholds (score>=65/<=35)
match exactly what every other part of this app already means by those
words.

Sector-specific rotation (item 2's third bullet) is intentionally scoped to
a coarse QQQ-vs-SPY growth/value proxy rather than a full 11-sector-ETF
breakdown - a documented, accepted trade-off (see the plan), not an
oversight.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from financial_data import get, get_bars_df
from tools.market_data import compute_indicators, compute_signal_summary

VIX_HIGH = 25.0
VIX_LOW = 15.0
GROWTH_VALUE_GAP_PCT = 2.0


def _fetch_vix_series(start: Optional[str] = None, end: Optional[str] = None,
                       as_of: Optional[str] = None) -> pd.Series:
    """VIX close as a date-indexed Series. Empty (never raises) if unavailable."""
    try:
        res = get("macro", ["VIX"], start=start, end=end, as_of=as_of)
    except Exception:
        return pd.Series(dtype=float)
    rows = res.get("data") or []
    if not rows:
        return pd.Series(dtype=float)
    rows = sorted(rows, key=lambda d: d["period_end"])
    idx = [pd.Timestamp(d["period_end"]) for d in rows]
    vals = [float(d["value"]) for d in rows]
    return pd.Series(vals, index=idx).sort_index()


def compute_market_regime(as_of: Optional[str] = None) -> Dict[str, Any]:
    """Classify the current (or as-of, PIT-honored via the gateway) broad
    market regime: SPY trend, VIX-based volatility regime, a combined
    risk-on/risk-off stance, and a QQQ-vs-SPY growth/value proxy.

    Degrades honestly: a missing SPY fetch returns confidence=0 with no
    regime guessed; a missing VIX or QQQ fetch drops confidence and adds a
    flag but still returns whatever the remaining inputs support.
    """
    flags: List[str] = []
    spy_df = get_bars_df("SPY", period="1y", as_of=as_of)
    if spy_df.empty:
        return {
            "trend": None, "volatility_regime": None, "vix_level": None,
            "risk_stance": None, "growth_vs_value": None,
            "confidence": 0.0, "flags": ["spy_unavailable"], "as_of": as_of,
        }

    spy_indicators = compute_indicators(spy_df)
    spy_signal = compute_signal_summary(spy_indicators)
    trend = spy_signal["direction"]

    vix_level: Optional[float] = None
    vol_regime: Optional[str] = None
    vix_series = _fetch_vix_series(as_of=as_of)
    if not vix_series.empty:
        vix_level = round(float(vix_series.iloc[-1]), 2)
        if vix_level >= VIX_HIGH:
            vol_regime = "HIGH"
        elif vix_level <= VIX_LOW:
            vol_regime = "LOW"
        else:
            vol_regime = "MEDIUM"
    else:
        flags.append("vix_unavailable")

    qqq_df = get_bars_df("QQQ", period="1y", as_of=as_of)
    growth_vs_value: Optional[str] = None
    if not qqq_df.empty and len(qqq_df) > 21 and len(spy_df) > 21:
        qqq_1m = float(qqq_df["Close"].iloc[-1] / qqq_df["Close"].iloc[-22] - 1) * 100
        spy_1m = float(spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-22] - 1) * 100
        gap = qqq_1m - spy_1m
        if gap > GROWTH_VALUE_GAP_PCT:
            growth_vs_value = "GROWTH_LEADING"
        elif gap < -GROWTH_VALUE_GAP_PCT:
            growth_vs_value = "GROWTH_LAGGING"
        else:
            growth_vs_value = "NEUTRAL"
    else:
        flags.append("qqq_unavailable")

    if vol_regime is None:
        # No VIX - fall back to trend alone rather than guessing a vol regime.
        risk_stance = "RISK_OFF" if trend == "BEARISH" else "NEUTRAL"
    elif trend == "BULLISH" and vol_regime in ("LOW", "MEDIUM"):
        risk_stance = "RISK_ON"
    elif trend == "BEARISH" or vol_regime == "HIGH":
        risk_stance = "RISK_OFF"
    else:
        risk_stance = "NEUTRAL"

    confidence = 1.0
    if "vix_unavailable" in flags:
        confidence -= 0.3
    if "qqq_unavailable" in flags:
        confidence -= 0.15

    return {
        "trend": trend,
        "volatility_regime": vol_regime,
        "vix_level": vix_level,
        "risk_stance": risk_stance,
        "growth_vs_value": growth_vs_value,
        "confidence": round(max(0.0, confidence), 2),
        "flags": flags,
        "as_of": as_of,
    }


def compute_market_regime_series(spy_df: pd.DataFrame, qqq_df: Optional[pd.DataFrame],
                                  vix_series: pd.Series) -> pd.DataFrame:
    """Vectorized, full-history regime label per bar - for the historical
    analog engine, which needs "what was the regime on date D" for every
    candidate D, not just today.

    Mirrors compute_market_regime()'s classification using the already-proven
    vectorized technical_score_series (backtest/pillars.py, documented and
    tested to equal compute_signal_summary's score at the live bar) instead
    of recomputing indicators per historical date in a loop - the same
    score>=65/<=35 thresholds compute_signal_summary itself uses.
    """
    from backtest.pillars import technical_score_series

    tech_score = technical_score_series(spy_df)
    trend = pd.Series("NEUTRAL", index=spy_df.index)
    trend[tech_score >= 65] = "BULLISH"
    trend[tech_score <= 35] = "BEARISH"

    vix_aligned = vix_series.reindex(spy_df.index, method="ffill") if not vix_series.empty \
        else pd.Series(index=spy_df.index, dtype=float)
    vol_regime = pd.Series(None, index=spy_df.index, dtype=object)
    vol_regime[vix_aligned >= VIX_HIGH] = "HIGH"
    vol_regime[vix_aligned <= VIX_LOW] = "LOW"
    vol_regime[(vix_aligned > VIX_LOW) & (vix_aligned < VIX_HIGH)] = "MEDIUM"

    risk_stance = pd.Series("NEUTRAL", index=spy_df.index, dtype=object)
    has_vol = vol_regime.notna()
    risk_stance[has_vol & (trend == "BULLISH") & (vol_regime != "HIGH")] = "RISK_ON"
    risk_stance[(trend == "BEARISH") | (has_vol & (vol_regime == "HIGH"))] = "RISK_OFF"
    risk_stance[~has_vol & (trend == "BEARISH")] = "RISK_OFF"

    out = pd.DataFrame({
        "trend": trend,
        "volatility_regime": vol_regime,
        "vix_level": vix_aligned,
        "risk_stance": risk_stance,
    }, index=spy_df.index)

    if qqq_df is not None and not qqq_df.empty:
        qqq_close = qqq_df["Close"].reindex(spy_df.index, method="ffill")
        qqq_1m = qqq_close.pct_change(21) * 100
        spy_1m = spy_df["Close"].astype(float).pct_change(21) * 100
        gap = qqq_1m - spy_1m
        growth_vs_value = pd.Series("NEUTRAL", index=spy_df.index, dtype=object)
        growth_vs_value[gap > GROWTH_VALUE_GAP_PCT] = "GROWTH_LEADING"
        growth_vs_value[gap < -GROWTH_VALUE_GAP_PCT] = "GROWTH_LAGGING"
        growth_vs_value[gap.isna()] = None
        out["growth_vs_value"] = growth_vs_value
    else:
        out["growth_vs_value"] = None

    return out
