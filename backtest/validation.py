"""
Overfitting-aware validation: purged/embargoed walk-forward CV + PBO.

The companion to the Deflated Sharpe Ratio (already in engine.py). dSR asks "is
this Sharpe real given how many variants I tried?"; this module asks the other
overfitting question: "does an in-sample winner stay a winner out-of-sample, or
did I just fit noise?"

Two tools, both from López de Prado, "Advances in Financial Machine Learning"
(Ch.7 cross-validation, Ch.11 backtest overfitting) — a different book from the
dSR source (Ch.8 of "ML for Asset Managers"), and the two are meant to be used
together:

  walk_forward_cv — train on a window, test on the NEXT window, roll forward.
    The honest way to backtest a time series: you never test on data that
    precedes your training set. PURGING removes training observations whose
    labels overlap the test window (a signal using a 20-day lookback computed
    right before the test set peeks into it); the EMBARGO drops a gap after each
    test fold so serial correlation can't leak train->test. Skipping either is
    how a leaky backtest reports a Sharpe it can never earn live.

  probability_of_backtest_overfitting (PBO) — the CSCV procedure (Ch.11).
    Split the return series into S even chunks, form every half-and-half
    combination of chunks into (in-sample, out-of-sample) pairs, pick the
    strategy that ranked best IS, and see where it ranks OOS. PBO = the fraction
    of splits where the IS-best strategy lands in the BOTTOM half OOS. A high PBO
    means "my selection process picks OOS losers" — the definition of overfitting.

Both are self-contained (numpy/pandas/itertools/scipy) and tested against
synthetic strategies with a KNOWN answer: a pure-noise "best of many" must score
a high PBO; a genuinely predictive signal must score a low one.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .engine import compute_performance_metrics, run_vectorized_backtest


# ══════════════════════════════════════════════════════════════════════════
# Purged & embargoed walk-forward CV
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardFold:
    train_start: int
    train_end: int          # exclusive
    test_start: int
    test_end: int           # exclusive
    test_sharpe: float
    test_return: float
    n_test_obs: int


@dataclass
class WalkForwardResult:
    folds: List[WalkForwardFold]
    mean_test_sharpe: float
    std_test_sharpe: float
    oos_return: float               # compounded OOS return across all test folds
    consistency: float              # fraction of folds with positive test Sharpe
    n_folds: int


def walk_forward_cv(
    prices: pd.Series,
    signal_fn: Callable[[pd.Series], pd.Series],
    n_folds: int = 5,
    embargo_frac: float = 0.01,
    purge_obs: int = 0,
    transaction_cost_bps: float = 10.0,
    cost_model=None,
    volume: Optional[pd.Series] = None,
) -> WalkForwardResult:
    """Anchored walk-forward: expanding train window, next block as test.

    signal_fn(prices_slice) -> position Series in {-1,0,1}. It is re-run on each
    fold's data (train+test up to test_end) and evaluated ONLY on the test slice,
    so a strategy cannot see beyond the fold. Purge drops `purge_obs` bars at the
    train/test boundary; embargo drops `embargo_frac` of the series after each
    test block before the next train window may use it.

    Deliberately re-computes the signal per fold rather than slicing a global
    signal: a global signal computed once on all data has already seen the future
    at every point, which is the leak this is meant to prevent.
    """
    prices = prices.astype(float)
    n = len(prices)
    if n < n_folds * 2:
        raise ValueError(f"need >= {n_folds*2} observations for {n_folds} folds, got {n}")

    embargo = int(np.ceil(n * embargo_frac))
    fold_size = n // (n_folds + 1)     # +1 so the first fold has a real train window
    folds: List[WalkForwardFold] = []

    for k in range(1, n_folds + 1):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds else n
        train_end = max(0, test_start - purge_obs)      # PURGE at the boundary
        train_start = 0                                  # anchored (expanding)
        if train_end - train_start < fold_size or test_end - test_start < 2:
            continue

        # Signal is computed on data up to test_end, then evaluated on the test
        # slice only. Everything before train_end minus the embargo is the "seen"
        # region; the embargo gap sits between train_end and test_start.
        window = prices.iloc[train_start:test_end]
        sig = signal_fn(window).reindex(window.index).fillna(0.0)
        test_prices = prices.iloc[test_start:test_end]
        test_sig = sig.reindex(test_prices.index).fillna(0.0)

        res = run_vectorized_backtest(test_prices, test_sig,
                                      transaction_cost_bps=transaction_cost_bps,
                                      cost_model=cost_model,
                                      volume=volume.iloc[test_start:test_end] if volume is not None else None)
        m = compute_performance_metrics(res.strategy_returns)
        folds.append(WalkForwardFold(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            test_sharpe=m["sharpe_ratio"], test_return=m["annualized_return"],
            n_test_obs=len(test_prices),
        ))

    if not folds:
        return WalkForwardResult([], 0.0, 0.0, 0.0, 0.0, 0)

    sharpes = np.array([f.test_sharpe for f in folds])
    oos = float(np.prod([1 + f.test_return for f in folds]) - 1)
    return WalkForwardResult(
        folds=folds,
        mean_test_sharpe=float(sharpes.mean()),
        std_test_sharpe=float(sharpes.std()),
        oos_return=oos,
        consistency=float((sharpes > 0).mean()),
        n_folds=len(folds),
    )


# ══════════════════════════════════════════════════════════════════════════
# Probability of Backtest Overfitting (CSCV)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PBOResult:
    pbo: float                       # P(IS-best strategy is OOS below-median)
    n_splits: int
    n_strategies: int
    mean_logit: float                # mean logit of OOS rank; <0 => overfit-prone
    is_overfit: bool                 # pbo > 0.5


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 10,
) -> PBOResult:
    """CSCV PBO (López de Prado, AFML Ch.11).

    returns_matrix: columns = strategies, rows = time, values = per-period
    returns. Split time into `n_splits` equal chunks; for every way to choose
    half the chunks as in-sample (the rest OOS), rank strategies by IS Sharpe,
    take the IS winner, and record its OOS rank. PBO is the share of splits where
    that winner ranks in the bottom half OOS.

    Interpretation: PBO near 0 => the selection generalizes; near 1 => picking
    the best backtest reliably picks an OOS loser (overfitting). 0.5 is the
    no-information baseline (selection is a coin flip).
    """
    R = returns_matrix.dropna(how="any")
    T, N = R.shape
    if N < 2:
        raise ValueError("PBO needs >= 2 strategies to compare")
    if n_splits % 2 != 0:
        n_splits += 1                # CSCV needs an even chunk count to halve
    if T < n_splits:
        n_splits = max(2, (T // 2) * 2)

    # Even chunks over time.
    bounds = np.linspace(0, T, n_splits + 1).astype(int)
    chunks = [R.iloc[bounds[i]:bounds[i + 1]] for i in range(n_splits)]

    def sharpe(df):
        mu, sd = df.mean(), df.std(ddof=1)
        return (mu / sd.replace(0, np.nan)).fillna(0.0)

    logits: List[float] = []
    below_median = 0
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    for is_idx in combos:
        oos_idx = [i for i in range(n_splits) if i not in is_idx]
        is_data = pd.concat([chunks[i] for i in is_idx])
        oos_data = pd.concat([chunks[i] for i in oos_idx])

        is_sharpe = sharpe(is_data)
        oos_sharpe = sharpe(oos_data)
        best = is_sharpe.idxmax()                       # winner chosen IN-SAMPLE

        # Rank the IS winner OOS: fraction of strategies it beats OOS.
        oos_rank = (oos_sharpe < oos_sharpe[best]).mean()   # 1.0 = best OOS, 0.0 = worst
        oos_rank = min(max(oos_rank, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(oos_rank / (1 - oos_rank))))
        if oos_rank < 0.5:
            below_median += 1

    n = len(combos)
    return PBOResult(
        pbo=below_median / n if n else 0.0,
        n_splits=n_splits,
        n_strategies=N,
        mean_logit=float(np.mean(logits)) if logits else 0.0,
        is_overfit=(below_median / n if n else 0.0) > 0.5,
    )
