"""
Position Sizing & Risk Module
===============================
Kelly-criterion position sizing with the safety practices real practitioners
actually use (fractional Kelly, hard caps, volatility targeting), plus
correlation-aware sizing so highly-redundant strategies don't get double-counted.

Sources (verified directly against the book text):
- Hilpisch, "Python for Algorithmic Trading", Ch.10 "Automating Trading
  Operations" -> "Capital Management" (p.284-296). The Kelly formula for a
  stock is `f* = (mu - r) / sigma^2` (p.293: "the expected excess return of
  the stock over the risk-free rate divided by the variance of the returns"),
  and the book notes the S&P 500's own optimal Kelly comes out to ~4.5x
  leverage. Full ("optimal") Kelly is famously too volatile to run in
  practice, so the book recommends HALF-Kelly (f*/2) as the sane default
  (p.297) — hence `kelly_multiplier=0.5` here.

Design note on the Kelly value being > 1: an f* of 4.5 means "4.5x leverage
is theoretically growth-optimal," which is far more risk than any sane
retail sizing would take. This module's `safe_kelly_fraction()` is what
callers should actually use — it applies fractional-Kelly scaling AND a hard
cap (default 10% of capital per position), so the output is a sane position
size, never a leverage multiplier. `kelly_fraction()` returns the raw
theoretical number (which CAN be > 1 or negative) for transparency/inspection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ══════════════════════════════════════════════════════════════════════════
# Kelly criterion
# ══════════════════════════════════════════════════════════════════════════

def kelly_fraction(strategy_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Raw Kelly fraction f* = (mu - r) / sigma^2, per Hilpisch p.293, computed on
    the STRATEGY's own realized returns (its edge), not naive buy-and-hold.

    mu and sigma^2 are annualized (daily mean * 252, daily variance * 252).
    Returns the raw theoretical value, which CAN be:
      - > 1  (the strategy's edge justifies leverage — the S&P itself is ~4.5)
      - negative (the strategy has a negative expected excess return — a signal
        to NOT take the position, deliberately not clipped to 0 here so the
        information is preserved; safe_kelly_fraction() does the clipping)
    """
    returns = strategy_returns.dropna()
    if len(returns) < 2:
        return 0.0

    mean_annual = returns.mean() * TRADING_DAYS_PER_YEAR
    var_annual = returns.var() * TRADING_DAYS_PER_YEAR
    if var_annual <= 0:
        return 0.0

    return float((mean_annual - risk_free_rate) / var_annual)


def safe_kelly_fraction(
    strategy_returns: pd.Series,
    kelly_multiplier: float = 0.5,
    max_position_pct: float = 0.10,
    risk_free_rate: float = 0.04,
) -> float:
    """
    The number callers should actually use for position sizing. Applies:
      1. fractional Kelly (half-Kelly by default — full Kelly is too volatile)
      2. a hard cap (default 10% of capital per position)
      3. a floor at 0 — a strategy with no positive historical edge gets size 0
         (you simply don't take the position), rather than a negative/short size

    Returns a magnitude in [0, max_position_pct]. Direction (long vs. short)
    is NOT encoded here — that comes from the strategy's current signal in the
    grounding step (Phase 5); this function only answers "how big."

    The hard cap is also the guaranteed guard against a degenerate near-zero-
    variance return series producing an absurd raw Kelly: min(scaled, cap) can
    never exceed the cap no matter how explosive the raw value is.
    """
    raw = kelly_fraction(strategy_returns, risk_free_rate=risk_free_rate)
    if raw <= 0:
        return 0.0
    scaled = raw * kelly_multiplier
    return float(min(scaled, max_position_pct))


# ══════════════════════════════════════════════════════════════════════════
# Volatility targeting
# ══════════════════════════════════════════════════════════════════════════

def volatility_target_scale(
    realized_vol: float,
    target_vol: float = 0.15,
    min_scale: float = 0.1,
    max_scale: float = 3.0,
) -> float:
    """
    scale = target_vol / realized_vol, clipped to [min_scale, max_scale].
    Returns > 1 when the position is calmer than target (scale up), < 1 when
    it's more volatile than target (scale down). The clip prevents absurd
    leverage when realized_vol is near zero.
    """
    if realized_vol <= 0:
        return 1.0
    return float(np.clip(target_vol / realized_vol, min_scale, max_scale))


# ══════════════════════════════════════════════════════════════════════════
# Correlation-aware sizing
# ══════════════════════════════════════════════════════════════════════════

def correlation_aware_position_size(
    proposed_positions: dict[str, float],
    strategy_return_series: dict[str, pd.Series],
    corr_threshold: float = 0.7,
) -> dict[str, float]:
    """
    When several strategies fire at once for the same ticker, naively summing
    their individual Kelly sizes overstates diversification that doesn't exist
    — momentum and trend-following, for instance, are often ~0.9 correlated and
    are really the same bet twice. This scales each position DOWN by how much
    it overlaps (correlation above `corr_threshold`) with the others, so a
    cluster of near-identical strategies ends up sized close to a single
    position rather than the sum.

    The math is name-agnostic: it de-duplicates any set of return streams that
    move together. Strategies-on-one-ticker was the original use; the portfolio
    layer (agents/portfolio_brief.py) feeds it {ticker: 1.0} against per-ticker
    price returns to recover a pure haircut factor per HOLDING, which is the
    same "five names that are really one bet" problem one level up. Keep it
    generic — do not narrow it back to strategies.

    proposed_positions:     {name: sized_fraction}   (name = strategy OR ticker)
    strategy_return_series: {name: pd.Series of that name's returns}
    """
    names = list(proposed_positions.keys())
    if len(names) <= 1:
        return dict(proposed_positions)

    aligned = pd.DataFrame(
        {n: strategy_return_series[n] for n in names if n in strategy_return_series}
    ).dropna()
    if aligned.shape[1] <= 1 or len(aligned) < 10:
        return dict(proposed_positions)   # not enough overlapping data to judge correlation

    corr = aligned.corr()
    adjusted: dict[str, float] = {}
    for n in names:
        if n not in corr.columns:
            adjusted[n] = proposed_positions[n]
            continue
        overlap = sum(
            max(0.0, float(corr.loc[n, m]))
            for m in names
            if m != n and m in corr.columns and float(corr.loc[n, m]) > corr_threshold
        )
        adjusted[n] = proposed_positions[n] / (1.0 + overlap)
    return adjusted


def compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Average True Range over the last `window` bars.

    The ONE shared ATR for level-setting (stops/entry zones). Extracted from
    agents/synthesis.py so the trading path (1.5*ATR stops) and the investing
    recommendation path (2*ATR stops) provably measure the same volatility —
    two private ATRs drifting apart would make their levels incomparable.
    Falls back to 2% of price on degenerate input, never raises.
    """
    try:
        high, low, close = df["High"], df["Low"], df["Close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(window).mean().iloc[-1])
        return atr if (atr > 0 and not np.isnan(atr)) else float(close.iloc[-1]) * 0.02
    except Exception:
        return float(df["Close"].iloc[-1]) * 0.02
