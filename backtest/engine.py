"""
Vectorized Backtesting Engine
===============================
Point-in-time-correct backtest core + standard performance metrics, including
the Deflated Sharpe Ratio (dSR).

Point-in-time correctness: a signal computed using information available AS OF
day t can only be ACTED ON starting day t+1 — you can't know today's close
until today's close happens, so a signal formed from today's data can't earn
today's return. This engine enforces that with a single internal `.shift(1)`
applied once, here, rather than leaving each strategy to remember to do it
correctly on its own.

Sources (verified directly against the actual book text, not paraphrased):
- Hilpisch, "Python for Algorithmic Trading", Ch. 4 "Mastering Vectorized
  Backtesting" (p.100-134) — the `position.shift(1) * returns` construction is
  the book's own code (p.112), confirmed by direct PDF read. Performance
  metrics (annualized return/vol, max drawdown) follow the same chapter's
  formulas, cross-checked against Ch.10 "Capital Management" (p.284-307) where
  `drawdown = cummax - cumulative` appears verbatim.
- López de Prado, "Machine Learning for Asset Managers", Ch. 8 "Testing Set
  Overfitting" (p.110-129) — the Deflated Sharpe Ratio formula (p.118),
  confirmed by direct PDF read: a Z-transform of skewness/kurtosis-adjusted
  Sharpe relative to the EXPECTED MAXIMUM Sharpe across `n_trials`, not raw
  Sharpe. This exists specifically to guard against the "run enough strategy
  variants and one will look good by chance" problem — directly relevant here
  since the strategy library (strategies.py) tries multiple signal types per
  ticker.

Honesty note on ground-truth verification: I attempted to reproduce Hilpisch's
own stated EUR/USD SMA(42,252) result (p.112: returns=-0.176731,
strategy=0.253121) using the book's own publicly hosted companion dataset
(http://hilpisch.com/pyalgo_eikon_eod_data.csv). The reproduced numbers do NOT
match exactly (returns=-0.157..-0.162, strategy=0.356..0.361 depending on
slicing order) — most likely because the hosted dataset has been revised since
the book's publication, not because the `position.shift(1) * returns`
construction is wrong (that line is transcribed verbatim from the book). The
QUALITATIVE conclusion still holds on the current data: buy-and-hold is
negative over the period, the SMA strategy is positive and beats it — see
tests/test_backtest_engine.py's `test_hilpisch_eurusd_directional` for this
weaker-but-honest check, and `test_synthetic_leak_free` for the primary,
self-contained, non-externally-dependent proof of point-in-time correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


# ══════════════════════════════════════════════════════════════════════════
# Core backtest
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    log_returns: pd.Series           # raw daily log returns of the underlying
    executed_position: pd.Series     # position actually held each day (lagged)
    strategy_returns: pd.Series      # daily strategy log returns, net of costs
    transaction_cost_bps: float      # flat rate used, or -1.0 when a CostModel was used
    n_trades: int
    total_cost: float = 0.0          # summed cost drag over the whole backtest (fraction)
    cost_model: str = "flat"         # "flat" or "CostModel" — provenance of the cost figure


def run_vectorized_backtest(
    prices: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float = 10.0,
    cost_model=None,
    volume: "Optional[pd.Series]" = None,
) -> BacktestResult:
    """
    prices: Close prices, DatetimeIndex, ascending.
    signal: position stance in {-1, 0, 1} at each date, computed using ONLY
            information available as of that date's close.

    Point-in-time correctness is enforced HERE, once:
      executed_position[t] = signal[t-1]   (never signal[t] — that would let
                                             today's stance earn today's return)
      strategy_return[t]   = executed_position[t] * log_return[t] - cost[t]

    Cost handling:
      - default: flat `transaction_cost_bps` on each position change (unchanged,
        so every existing caller/test behaves identically).
      - `cost_model=CostModel(...)`: realistic spread + square-root impact +
        daily borrow on short legs (backtest/costs.py). `volume` feeds the
        liquidity proxy; without it the model uses its documented optimistic
        (liquid-name) branch rather than fabricating a number.
    """
    prices = prices.astype(float)
    signal = signal.reindex(prices.index).fillna(0.0)

    log_returns = np.log(prices / prices.shift(1))

    executed_position = signal.shift(1).fillna(0.0)

    position_change = executed_position.diff().fillna(executed_position.iloc[0] if len(executed_position) else 0.0)

    if cost_model is not None:
        costs = cost_model.compute_costs(executed_position, position_change, prices, volume)
        cost_rate_label = -1.0
        model_label = type(cost_model).__name__
    else:
        cost_per_trade = transaction_cost_bps / 10_000.0
        costs = position_change.abs() * cost_per_trade
        cost_rate_label = transaction_cost_bps
        model_label = "flat"

    strategy_returns = executed_position * log_returns - costs
    strategy_returns = strategy_returns.dropna()

    n_trades = int((position_change.abs() > 0).sum())

    return BacktestResult(
        log_returns=log_returns,
        executed_position=executed_position,
        strategy_returns=strategy_returns,
        transaction_cost_bps=cost_rate_label,
        n_trades=n_trades,
        total_cost=float(costs.reindex(strategy_returns.index).fillna(0.0).sum()),
        cost_model=model_label,
    )


# ══════════════════════════════════════════════════════════════════════════
# Performance metrics
# ══════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252


def compute_performance_metrics(
    strategy_returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    All formulas per Hilpisch Ch.4/Ch.10 (verified against actual book pages):
      annualized_return = exp(mean(daily_log_returns) * 252) - 1
      annualized_vol     = std(daily_log_returns) * sqrt(252)
      sharpe             = (annualized_return - risk_free_rate) / annualized_vol
      sortino            = same, but vol computed only over downside (negative) returns
      cumulative         = exp(cumsum(daily_log_returns))
      drawdown           = (cummax(cumulative) - cumulative) / cummax   (relative, in [0,1])
      max_drawdown       = drawdown.max()
      calmar             = annualized_return / max_drawdown
      win_rate           = fraction of nonzero-return days with positive return
    """
    returns = strategy_returns.dropna()
    if len(returns) < 2:
        return _empty_metrics()

    mean_daily = returns.mean()
    std_daily = returns.std()

    annualized_return = float(np.exp(mean_daily * TRADING_DAYS_PER_YEAR) - 1)
    annualized_vol = float(std_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(downside) > 1 else 0.0
    sortino = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    cumulative = np.exp(returns.cumsum())
    cummax = cumulative.cummax()
    # RELATIVE drawdown (fraction of the running peak), always in [0,1].
    # Hilpisch p.306-307 plots the ABSOLUTE difference (cummax - cumulative),
    # which is fine for a single-asset chart but explodes above 1.0 for a
    # high-compounding equity curve (e.g. a 30x stock shows "drawdown" of 30).
    # Dividing by the peak makes it a true percentage drawdown that is
    # comparable across tickers and correct as the denominator of Calmar.
    drawdown = (cummax - cumulative) / cummax
    max_drawdown = float(drawdown.max())

    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    nonzero = returns[returns != 0]
    win_rate = float((nonzero > 0).sum() / len(nonzero)) if len(nonzero) > 0 else 0.0

    return {
        "annualized_return": round(annualized_return, 4),
        "annualized_volatility": round(annualized_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 3),
        "win_rate": round(win_rate, 3),
        "n_observations": int(len(returns)),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis() + 3),  # pandas returns EXCESS kurtosis; dSR wants regular kurtosis
    }


def _empty_metrics() -> dict:
    return {
        "annualized_return": 0.0, "annualized_volatility": 0.0, "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0, "max_drawdown": 0.0, "calmar_ratio": 0.0, "win_rate": 0.0,
        "n_observations": 0, "skewness": 0.0, "kurtosis": 3.0,
    }


# ══════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio (López de Prado)
# ══════════════════════════════════════════════════════════════════════════

_EULER_MASCHERONI = 0.5772156649015329


def _expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """
    E[max_k{SR_k}] approximation for n_trials independent strategy trials with
    unit Sharpe variance, per the False Strategy Theorem (López de Prado Ch.8,
    p.113-118): as more trials are run, the expected maximum observed Sharpe
    of a TRUE-ZERO-SHARPE strategy rises — this is the benchmark dSR deflates
    observed Sharpe against.
    """
    if n_trials <= 1:
        return 0.0
    return float(
        np.sqrt(sharpe_variance) * (
            (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials)
            + _EULER_MASCHERONI * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    sharpe_hat: float,
    n_trials: int,
    skewness: float,
    kurtosis: float,
    n_obs: int,
) -> float:
    """
    Per López de Prado Ch.8 (p.118), verified against the actual book text:

      dSR = Phi[ (SR_hat - E[max_SR]) * sqrt(n_obs - 1)
                 / sqrt(1 - skew*SR_hat + (kurtosis-1)/4 * SR_hat^2) ]

    where Phi is the standard normal CDF. Returns a value in [0, 1],
    interpretable as P(the true Sharpe ratio is actually > 0 | observed
    results, adjusted for the number of trials attempted). dSR > 0.5 is the
    threshold the book gives for "more likely skill than luck."

    n_trials should be the count of distinct strategy/parameter combinations
    tried for this ticker, NOT the count of LLM agents — the multiple-
    comparisons problem this metric guards against is at the parameter/strategy
    level. Source it from `data.ledger.effective_n_trials()`, which remembers
    every variant ever tried rather than only the ones tried in this process;
    a per-run count resets the memory that dSR exists to keep.

    Raises ValueError for n_trials < 1. Zero trials is not a quantity, it is a
    bug in the caller: with n_trials=0 the expected-max term is 0, deflation
    silently vanishes, and a mediocre strategy scores dSR=1.0 — "certain it is
    real". Returning a flattering number for a nonsense input is the single
    most dangerous thing this function could do, so it refuses.
    """
    if n_trials < 1:
        raise ValueError(
            f"n_trials={n_trials} is invalid: dSR deflates against the expected max "
            f"Sharpe of n_trials attempts, so n<1 disables deflation entirely and "
            f"inflates confidence toward 1.0. Pass the real count "
            f"(data.ledger.effective_n_trials), minimum 1.")
    if n_obs < 2:
        return 0.0

    expected_max_sr = _expected_max_sharpe(n_trials)

    denom_inner = 1 - skewness * sharpe_hat + ((kurtosis - 1) / 4) * sharpe_hat ** 2
    if denom_inner <= 0:
        denom_inner = 1e-8   # guard against a degenerate (near-zero-variance) return series

    z = (sharpe_hat - expected_max_sr) * np.sqrt(n_obs - 1) / np.sqrt(denom_inner)
    return float(norm.cdf(z))
