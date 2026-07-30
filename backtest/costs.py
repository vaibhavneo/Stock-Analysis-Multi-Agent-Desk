"""
Realistic transaction-cost model (FIL M-F1 remainder).

The engine's default is a flat 10 bps per trade. That is a fiction with a
direction: it is roughly right for a liquid large-cap traded in size, and wildly
optimistic for a thin small-cap or a high-turnover strategy that pays the spread
every few days. A backtest that undercounts costs makes churn look free, which
is exactly how a strategy that loses money after costs shows a positive Sharpe.

This model charges the three costs a real desk actually pays, each grounded:

  1. Half-spread — you cross the bid/ask on every trade. Modeled from a
     liquidity proxy (dollar volume): tighter for liquid names, wider for thin
     ones. This is the dominant cost for most retail-scale strategies.

  2. Market impact — your own order pushes the price. The square-root law
     (impact ∝ sqrt(participation)) is the standard practitioner model
     (Almgren, and Kyle's lambda intuition): impact grows sublinearly with how
     much of the day's volume you are. Without an explicit order size in a
     whole-capital {-1,0,1} backtest we assume a fixed participation rate, so
     impact scales with 1/sqrt(dollar-volume) — the liquidity term of the law.

  3. Borrow — a short position pays a stock-loan fee (carry) EVERY day it is
     held, not just when opened. Longs do not. Ignoring this flatters every
     short strategy; hard-to-borrow names can cost tens of percent annualized.

All three are deliberately conservative-by-default and fully documented, because
the point of this module is to stop backtests from lying in the optimistic
direction — an honest cost model that is slightly too HIGH is a safe error; one
that is too low is the dangerous one this exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class CostModel:
    """Per-day cost model. All rates in basis points unless named otherwise.

    Defaults are calibrated to a liquid US large-cap at retail scale; they make
    the flat-10bps default look generous, which is the intended direction.
    """
    base_half_spread_bps: float = 3.0     # floor half-spread for a very liquid name
    impact_coef_bps: float = 8.0          # impact at ~$1M/day dollar volume (scaled by 1/sqrt)
    participation: float = 0.02           # assumed fraction of daily volume traded
    borrow_rate_annual: float = 0.005     # 0.5%/yr general-collateral short-loan fee
    min_cost_bps: float = 1.0             # a trade is never free
    max_cost_bps: float = 250.0           # sanity cap (2.5%) for pathological illiquidity

    def per_trade_cost_bps(self, dollar_volume: Optional[float]) -> float:
        """Cost in bps to cross once (half-spread + impact), given a name's
        recent dollar volume. Falls back to a liquid-name assumption when volume
        is unknown — labelled generous on purpose, never fabricated as zero."""
        if dollar_volume is None or not np.isfinite(dollar_volume) or dollar_volume <= 0:
            # No liquidity data: assume liquid (small cost). Documented as the
            # optimistic branch — callers with volume get the honest number.
            spread = self.base_half_spread_bps
            impact = self.impact_coef_bps * np.sqrt(self.participation)
        else:
            dv_millions = dollar_volume / 1_000_000.0
            # Spread widens as liquidity falls; floored at base for liquid names.
            spread = max(self.base_half_spread_bps,
                         self.base_half_spread_bps / np.sqrt(max(dv_millions, 1e-6)))
            # Square-root-law impact, liquidity term: participation of a thinner
            # book costs more. impact ∝ sqrt(participation) / sqrt(dv).
            impact = self.impact_coef_bps * np.sqrt(self.participation) / np.sqrt(max(dv_millions, 1e-6))
        return float(np.clip(spread + impact, self.min_cost_bps, self.max_cost_bps))

    def compute_costs(
        self,
        executed_position: pd.Series,
        position_change: pd.Series,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Total per-day cost as a fraction of capital (to subtract from returns).

        executed_position: the position actually held each day (already lagged)
        position_change:   day-over-day change in executed position
        prices/volume:     for the dollar-volume liquidity proxy (volume optional)

        Trading cost is charged on the days the position CHANGES (proportional to
        the size of the change). Borrow is charged EVERY day a short is held.
        """
        idx = executed_position.index

        # Dollar-volume proxy per day (rolling, so a single spike doesn't make a
        # name look permanently liquid). Missing volume -> None -> optimistic branch.
        if volume is not None:
            dollar_vol = (prices.reindex(idx) * volume.reindex(idx)).rolling(20, min_periods=1).median()
        else:
            dollar_vol = pd.Series(np.nan, index=idx)

        trade_cost = pd.Series(0.0, index=idx)
        changed = position_change.abs() > 0
        for t in idx[changed]:
            dv = dollar_vol.get(t)
            bps = self.per_trade_cost_bps(dv if (dv is not None and np.isfinite(dv)) else None)
            trade_cost.loc[t] = abs(position_change.loc[t]) * bps / 10_000.0

        # Borrow: daily carry on the short leg only (executed_position < 0).
        daily_borrow = self.borrow_rate_annual / TRADING_DAYS_PER_YEAR
        borrow_cost = executed_position.clip(upper=0.0).abs() * daily_borrow

        return (trade_cost + borrow_cost).reindex(idx).fillna(0.0)


# A single shared default so callers don't each re-specify the calibration.
DEFAULT_COST_MODEL = CostModel()
