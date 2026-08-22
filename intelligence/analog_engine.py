"""
Historical analog / outcome engine (item 5) - did not exist anywhere in this
codebase before (confirmed by a full-repo grep during planning: zero hits for
"analog", "similar setup", "historical pattern", under any name). Finds prior
periods in a ticker's OWN history whose technical/momentum/volatility/
valuation/regime combination looked like today's, and reports what actually
happened afterward at each horizon - presented as historical evidence/
context, never as a guaranteed prediction (see the "confidence" cap and the
"never claim more" flags below).

Scope, per the plan (both explicit judgment calls the user confirmed):
  - LIVE PATH ONLY. Not wired into agents/replay.py this round - candidate
    search is single-anchor (one as_of, evaluated once), not itself replayed
    across many historical as_of dates, which would need a second, stricter
    look-ahead surface this round deliberately defers.
  - Valuation is a CHEAP PROXY: price-percentile within its own trailing
    range, not true point-in-time P/E (which would need an EDGAR fetch per
    candidate date - expensive, and against item 12's "don't add calls").
    Always labeled as a proxy in the output, never presented as real
    valuation.

No-look-ahead design: everything - feature computation, z-scoring, the
candidate pool itself - is anchored to ONE decision point, `as_of`. The
first thing this function does is truncate every input series to
`index <= as_of`; every computation after that only ever sees rows already
inside that truncated series, so nothing downstream can reach past it. This
mirrors agents/replay.py::pit_inputs()'s own "truncate first, everything
else follows" no-look-ahead pattern, and reuses backtest/pillars.py's
already-proven-no-look-ahead technical_score_series/algo_score_series
instead of re-deriving indicator logic here.

A candidate historical bar is excluded from the pool entirely if the
longest requested horizon's outcome would fall beyond `as_of` (i.e., it
isn't fully knowable yet) - not a second look-ahead guard on top of the
truncation, just data availability: the truncated series simply doesn't
contain that price yet.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intelligence.regime import compute_market_regime_series

HORIZONS: Tuple[int, ...] = (5, 21, 63, 126, 252)
MIN_HISTORY_BARS = 756          # ~3 years - the plan's stated minimum
MIN_FEATURE_LOOKBACK = 260      # feature windows (252d vol percentile, etc.) need ~1y+ to be real
NUMERIC_FEATURES = ("technical", "algo", "vol_percentile", "momentum_3m", "momentum_6m", "valuation_proxy")
REGIME_MISMATCH_PENALTY = 1.5
VALUATION_LOOKBACK_BARS = 1260  # ~5 years for the price-percentile proxy's own range


def _compute_feature_series(df: pd.DataFrame, spy_df: Optional[pd.DataFrame],
                             vix_series: pd.Series) -> pd.DataFrame:
    from backtest.pillars import technical_score_series, algo_score_series

    close = df["Close"].astype(float)

    technical = technical_score_series(df)
    algo = algo_score_series(df)

    returns = close.pct_change()
    hv20 = returns.rolling(20).std() * math.sqrt(252) * 100
    vol_percentile = hv20.rolling(252, min_periods=60).rank(pct=True) * 100

    momentum_3m = close.pct_change(63) * 100
    momentum_6m = close.pct_change(126) * 100

    roll_min = close.rolling(VALUATION_LOOKBACK_BARS, min_periods=60).min()
    roll_max = close.rolling(VALUATION_LOOKBACK_BARS, min_periods=60).max()
    span = (roll_max - roll_min).replace(0, np.nan)
    valuation_proxy = ((close - roll_min) / span * 100)

    features = pd.DataFrame({
        "technical": technical, "algo": algo, "vol_percentile": vol_percentile,
        "momentum_3m": momentum_3m, "momentum_6m": momentum_6m,
        "valuation_proxy": valuation_proxy,
    }, index=df.index)

    if spy_df is not None and not spy_df.empty:
        try:
            regime_series = compute_market_regime_series(spy_df, None, vix_series)
            features["regime"] = regime_series["risk_stance"].reindex(df.index, method="ffill")
        except Exception:
            features["regime"] = None
    else:
        features["regime"] = None

    return features


def _zscore_matrix(features: pd.DataFrame) -> pd.DataFrame:
    z = pd.DataFrame(index=features.index)
    for col in NUMERIC_FEATURES:
        s = features[col]
        mean, std = s.mean(), s.std()
        z[col] = (s - mean) / std if std and std > 0 else pd.Series(0.0, index=s.index)
    return z


def find_historical_analogs(
    ticker: str,
    df_long: pd.DataFrame,
    spy_df: Optional[pd.DataFrame] = None,
    vix_series: Optional[pd.Series] = None,
    as_of: Optional[str] = None,
    k: int = 15,
    min_separation_days: int = 63,
    horizons: Tuple[int, ...] = HORIZONS,
) -> Dict[str, Any]:
    """Find up to k historical bars in this ticker's own history whose
    technical/momentum/volatility/valuation/regime setup was closest to
    `as_of` (default: the latest available bar), and report the forward
    return distribution at each horizon across those matches.

    Never fabricates a match: below MIN_HISTORY_BARS of history, or with no
    eligible candidate at all, returns status="insufficient_history" (or the
    equivalent "no_eligible_candidates" flag) with confidence=0 and an empty
    match list - never a padded or invented result.
    """
    if df_long is None or df_long.empty or "Close" not in df_long.columns:
        return {"ticker": ticker, "status": "insufficient_history", "as_of": as_of,
                "confidence": 0.0, "matches": [], "outcome_by_horizon": {},
                "flags": ["no_price_history"]}

    df = df_long
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        df = df[df.index <= cutoff]
    if spy_df is not None and as_of is not None and not spy_df.empty:
        spy_df = spy_df[spy_df.index <= pd.Timestamp(as_of)]
    if vix_series is not None and as_of is not None and not vix_series.empty:
        vix_series = vix_series[vix_series.index <= pd.Timestamp(as_of)]

    if len(df) < MIN_HISTORY_BARS:
        return {"ticker": ticker, "status": "insufficient_history", "as_of": as_of,
                "confidence": 0.0, "matches": [], "outcome_by_horizon": {},
                "flags": [f"needs_{MIN_HISTORY_BARS}_bars_has_{len(df)}"]}

    close = df["Close"].astype(float)
    max_h = max(horizons)
    n = len(df)
    target_idx = n - 1

    features = _compute_feature_series(df, spy_df, vix_series if vix_series is not None else pd.Series(dtype=float))
    z = _zscore_matrix(features)

    target_z = z.iloc[target_idx]
    target_regime = features["regime"].iloc[target_idx]
    flags: List[str] = []
    if target_regime is None:
        flags.append("regime_context_unavailable")

    eligible: List[Tuple[int, float]] = []
    for i in range(MIN_FEATURE_LOOKBACK, n - max_h):
        row = z.iloc[i]
        if row.isna().any():
            continue
        dist = float(np.sqrt(((row - target_z) ** 2).sum()))
        if target_regime is not None and features["regime"].iloc[i] != target_regime:
            dist += REGIME_MISMATCH_PENALTY
        eligible.append((i, dist))

    if not eligible:
        return {"ticker": ticker, "status": "insufficient_history", "as_of": str(df.index[target_idx].date()),
                "confidence": 0.0, "matches": [], "outcome_by_horizon": {},
                "flags": flags + ["no_eligible_candidates"]}

    eligible.sort(key=lambda t: t[1])

    selected: List[Tuple[int, float]] = []
    selected_dates: List[pd.Timestamp] = []
    for i, dist in eligible:
        date_i = df.index[i]
        if any(abs((date_i - d).days) < min_separation_days for d in selected_dates):
            continue
        selected.append((i, dist))
        selected_dates.append(date_i)
        if len(selected) >= k:
            break

    match_records: List[Dict[str, Any]] = []
    horizon_returns: Dict[int, List[float]] = {h: [] for h in horizons}
    for i, dist in selected:
        p0 = float(close.iloc[i])
        rec: Dict[str, Any] = {"date": str(df.index[i].date()), "distance": round(dist, 3),
                                "regime_at_match": features["regime"].iloc[i]}
        for h in horizons:
            ret = (float(close.iloc[i + h]) / p0 - 1.0) * 100.0
            rec[f"return_{h}d_pct"] = round(ret, 2)
            horizon_returns[h].append(ret)
        match_records.append(rec)

    outcome_by_horizon: Dict[int, Dict[str, Any]] = {}
    for h in horizons:
        rets = horizon_returns[h]
        outcome_by_horizon[h] = {
            "n": len(rets),
            "pct_positive": round(sum(1 for r in rets if r > 0) / len(rets), 3),
            "avg_return_pct": round(sum(rets) / len(rets), 2),
            "median_return_pct": round(float(np.median(rets)), 2),
        }

    if len(selected) < k:
        flags.append("fewer_than_k_matches")

    # Deliberately capped below 1.0: even a full k=15 set of matches on a
    # SINGLE ticker's own history is a thin sample after de-correlation -
    # this should read as corroborating context, never a strong signal on
    # its own. See the plan's own stated "biggest remaining limitation."
    confidence = round(min(0.6, 0.6 * len(selected) / k), 2)

    return {
        "ticker": ticker,
        "status": "ok",
        "as_of": str(df.index[target_idx].date()),
        "confidence": confidence,
        "matches": match_records,
        "outcome_by_horizon": outcome_by_horizon,
        "valuation_note": "valuation dimension is a price-percentile-within-range proxy, not true P/E-based valuation",
        "flags": flags,
    }
