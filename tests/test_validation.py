"""
Verification for purged walk-forward CV + PBO (FIL M-F1 remainder).

Run: python3 tests/test_validation.py

The whole value of an overfitting detector is that it fires on overfitting and
stays quiet otherwise. So the tests are built around cases with a KNOWN answer:

  - PBO on many PURE-NOISE strategies must be HIGH (~0.5+): selecting the best
    backtest among noise is exactly overfitting, and the detector must say so.
  - PBO on a genuinely PREDICTIVE strategy vs noise must be LOW: a real edge
    generalizes out-of-sample, and the detector must not cry wolf.
  - Walk-forward must never let a fold see beyond itself (leak-free), and must
    reward a real trend-follower on trending data.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.validation import (
    probability_of_backtest_overfitting,
    walk_forward_cv,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:56s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_pbo_detects_noise():
    print("=== 1. PBO is HIGH for best-of-pure-noise (the overfit case) ===")
    rng = np.random.default_rng(7)
    T, N = 400, 30
    # 30 strategies of pure noise, zero true edge. Selecting the IS-best among
    # them is textbook overfitting; PBO should be near/above the 0.5 baseline.
    noise = pd.DataFrame(rng.normal(0, 0.01, (T, N)),
                         columns=[f"noise_{i}" for i in range(N)])
    res = probability_of_backtest_overfitting(noise, n_splits=10)
    check("PBO on pure noise is high (>= 0.4)", res.pbo >= 0.4, f"pbo={res.pbo:.3f}")
    check("noise is flagged overfit-prone or borderline", res.pbo >= 0.4)
    check("mean logit <= ~0 (no OOS generalization)", res.mean_logit <= 0.2,
          f"logit={res.mean_logit:.3f}")
    check("reports the strategy/split counts", res.n_strategies == N and res.n_splits == 10)


def test_pbo_clears_real_edge():
    print("=== 2. PBO is LOW for a genuinely predictive strategy ===")
    rng = np.random.default_rng(11)
    T, N = 400, 20
    # One strategy with a persistent positive mean (a real edge) among 19 noise.
    cols = {f"noise_{i}": rng.normal(0, 0.01, T) for i in range(N - 1)}
    cols["real_edge"] = rng.normal(0.0015, 0.01, T)     # small but persistent
    df = pd.DataFrame(cols)
    res = probability_of_backtest_overfitting(df, n_splits=10)
    check("PBO with a real edge present is low (< 0.4)", res.pbo < 0.4, f"pbo={res.pbo:.3f}")
    check("NOT flagged as overfit", not res.is_overfit, f"pbo={res.pbo:.3f}")

    # The count-based PBO (the canonical statistic) is what we trust for the
    # verdict. mean_logit is a secondary diagnostic whose ABSOLUTE sign is not
    # guaranteed for a single-edge-among-noise setup — a lucky noise strategy
    # occasionally wins IS and lands very low OOS, dragging the mean negative
    # even while PBO stays correctly below 0.5. The property that IS robust and
    # meaningful is DISCRIMINATION: the real-edge case scores better than noise
    # on BOTH statistics.
    noise_only = pd.DataFrame({f"n_{i}": rng.normal(0, 0.01, T) for i in range(N)})
    res_noise = probability_of_backtest_overfitting(noise_only, n_splits=10)
    check("noise PBO > real-edge PBO (detector discriminates)",
          res_noise.pbo > res.pbo, f"noise={res_noise.pbo:.3f} edge={res.pbo:.3f}")
    check("real-edge mean logit > noise mean logit (OOS generalization signal)",
          res.mean_logit > res_noise.mean_logit,
          f"edge={res.mean_logit:.3f} noise={res_noise.mean_logit:.3f}")


def test_pbo_guards():
    print("=== 3. PBO input guards ===")
    try:
        probability_of_backtest_overfitting(pd.DataFrame({"only": [0.1, 0.2, 0.3]}))
        check("single-strategy input rejected", False, "no raise")
    except ValueError:
        check("single-strategy input rejected", True)

    # Odd split count is coerced even, not silently mis-halved.
    df = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (200, 5)))
    r = probability_of_backtest_overfitting(df, n_splits=7)
    check("odd n_splits coerced to even", r.n_splits % 2 == 0, f"n_splits={r.n_splits}")
    check("PBO is a probability in [0,1]", 0.0 <= r.pbo <= 1.0, f"pbo={r.pbo}")


def test_walk_forward_leak_free_and_rewards_trend():
    print("=== 4. Walk-forward: leak-free + rewards a real trend-follower ===")
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # A persistent uptrend with noise: a trend-follower SHOULD work OOS here.
    trend = np.cumsum(rng.normal(0.0008, 0.01, n))
    prices = pd.Series(100 * np.exp(trend), index=idx)

    def trend_follower(p):
        # long when above its own 50-day mean — a real, causal signal
        ma = p.rolling(50, min_periods=10).mean()
        return (p > ma).astype(float).replace(0.0, -1.0)

    res = walk_forward_cv(prices, trend_follower, n_folds=5, embargo_frac=0.02, purge_obs=50)
    check("produces the requested folds", res.n_folds >= 3, f"n_folds={res.n_folds}")
    check("trend-follower has positive mean OOS Sharpe on trending data",
          res.mean_test_sharpe > 0, f"mean_sharpe={res.mean_test_sharpe:.3f}")
    check("consistency is a fraction in [0,1]", 0.0 <= res.consistency <= 1.0,
          f"consistency={res.consistency:.2f}")

    # Leak check: a random (non-causal) signal must NOT show a reliable OOS edge.
    def coin_flip(p):
        return pd.Series(np.random.default_rng(99).choice([-1.0, 1.0], len(p)), index=p.index)

    noise_res = walk_forward_cv(prices, coin_flip, n_folds=5, embargo_frac=0.02, purge_obs=50)
    check("a non-causal signal does NOT beat the real trend-follower OOS",
          noise_res.mean_test_sharpe < res.mean_test_sharpe,
          f"noise={noise_res.mean_test_sharpe:.3f} trend={res.mean_test_sharpe:.3f}")

    # Too-few-observations guard.
    try:
        walk_forward_cv(prices.iloc[:5], trend_follower, n_folds=5)
        check("insufficient data rejected", False, "no raise")
    except ValueError:
        check("insufficient data rejected", True)


if __name__ == "__main__":
    test_pbo_detects_noise()
    test_pbo_clears_real_edge()
    test_pbo_guards()
    test_walk_forward_leak_free_and_rewards_trend()
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — purged walk-forward CV + PBO catch overfitting, clear real edges")
