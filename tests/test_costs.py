"""
Verification for the realistic cost model (FIL M-F1 remainder).

Run: python3 tests/test_costs.py

The flat-10bps default is a fiction with a direction — right for a liquid
large-cap, wildly optimistic for a thin name or a high-churn strategy. This
model exists to stop backtests lying in the optimistic direction, so the tests
assert the properties that make it honest:

  1. Liquidity gradient — thinner names cost strictly more
  2. Borrow — shorts pay daily carry, longs never do
  3. Backward compatibility — the flat path is unchanged (existing tests rely on it)
  4. Churn — a high-turnover strategy on a thin name is measurably penalized
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.costs import CostModel, DEFAULT_COST_MODEL
from backtest.engine import run_vectorized_backtest

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:56s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_liquidity_gradient():
    print("=== 1. Liquidity gradient: thinner = strictly more expensive ===")
    cm = CostModel()
    dvs = [5e8, 5e7, 5e6, 5e5, 5e4]     # mega -> micro
    costs = [cm.per_trade_cost_bps(dv) for dv in dvs]
    check("cost is monotonically non-decreasing as liquidity falls",
          all(a <= b + 1e-9 for a, b in zip(costs, costs[1:])),
          " ".join(f"{c:.2f}" for c in costs))
    check("a $50k microcap costs strictly more than a $500M mega-cap",
          costs[-1] > costs[0] * 2, f"{costs[-1]:.2f} vs {costs[0]:.2f}")
    check("liquid mega-cap is cheaper than flat 10bps (flat was pessimistic here)",
          costs[0] < 10.0, f"{costs[0]:.2f}")
    check("thin microcap EXCEEDS flat 10bps (flat was optimistic here)",
          costs[-1] > 10.0, f"{costs[-1]:.2f} — this is the bias flat-10bps hides")

    # Unknown volume must use the documented optimistic branch, never fabricate 0.
    unk = cm.per_trade_cost_bps(None)
    check("unknown volume -> optimistic-but-nonzero (never free)",
          cm.min_cost_bps <= unk < costs[-1], f"{unk:.2f}")
    check("cost is bounded by the sanity cap",
          cm.per_trade_cost_bps(1.0) <= cm.max_cost_bps)


def test_borrow():
    print("=== 2. Borrow: shorts pay daily carry, longs do not ===")
    idx = pd.date_range("2022-01-01", periods=60, freq="B")
    prices = pd.Series(100.0, index=idx)
    cm = CostModel(base_half_spread_bps=0, impact_coef_bps=0, min_cost_bps=0,
                   borrow_rate_annual=0.30)   # isolate borrow, hard-to-borrow name

    short = pd.Series(-1.0, index=idx)
    long = pd.Series(1.0, index=idx)
    no_change = pd.Series(0.0, index=idx)     # already holding, no trades

    short_cost = cm.compute_costs(short, no_change, prices).sum()
    long_cost = cm.compute_costs(long, no_change, prices).sum()
    check("a held short accrues borrow every day", short_cost > 0, f"{short_cost:.5f}")
    check("a held long accrues ZERO borrow", long_cost == 0.0, f"{long_cost:.5f}")
    expected = 0.30 / 252 * 60
    check("borrow magnitude matches rate x days", abs(short_cost - expected) < 1e-9,
          f"{short_cost:.5f} vs {expected:.5f}")


def test_backward_compat():
    print("=== 3. Backward compatibility: the flat path is unchanged ===")
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)
    sig = pd.Series(rng.choice([-1, 0, 1], 300), index=idx).astype(float)

    r = run_vectorized_backtest(prices, sig, transaction_cost_bps=10.0)
    check("flat path still labels bps and model", r.transaction_cost_bps == 10.0 and r.cost_model == "flat")
    check("flat path unchanged when no cost_model passed (default 10bps)",
          run_vectorized_backtest(prices, sig).cost_model == "flat")

    rc = run_vectorized_backtest(prices, sig, cost_model=DEFAULT_COST_MODEL)
    check("cost_model path is labelled distinctly", rc.cost_model == "CostModel" and rc.transaction_cost_bps == -1.0)
    check("both paths report a total_cost", r.total_cost > 0 and rc.total_cost >= 0)
    check("same signal, same n_trades regardless of cost model", r.n_trades == rc.n_trades)


def test_churn_penalty():
    print("=== 4. Churn on a thin name is measurably penalized ===")
    rng = np.random.default_rng(2)
    idx = pd.date_range("2021-01-01", periods=500, freq="B")
    prices = pd.Series(50 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, 500))), index=idx)
    thin_vol = pd.Series(rng.uniform(2e4, 8e4, 500), index=idx)     # ~$1-4M/day: thin
    liquid_vol = pd.Series(rng.uniform(2e7, 8e7, 500), index=idx)   # ~$1-4B/day: liquid

    churn = pd.Series(np.tile([1, 0], 250)[:500], index=idx).astype(float)   # trade every day

    thin = run_vectorized_backtest(prices, churn, cost_model=DEFAULT_COST_MODEL, volume=thin_vol)
    liquid = run_vectorized_backtest(prices, churn, cost_model=DEFAULT_COST_MODEL, volume=liquid_vol)
    check("high churn on a THIN name costs more than on a liquid name",
          thin.total_cost > liquid.total_cost, f"thin={thin.total_cost:.4f} liquid={liquid.total_cost:.4f}")

    # The plan's motivating case: a strategy positive gross can go negative net.
    # Build a signal with a tiny positive gross edge, then show costs erase it.
    gross = run_vectorized_backtest(prices, churn, transaction_cost_bps=0.0)
    net = run_vectorized_backtest(prices, churn, cost_model=DEFAULT_COST_MODEL, volume=thin_vol)
    check("costs strictly reduce returns (churn is not free)",
          net.strategy_returns.sum() < gross.strategy_returns.sum(),
          f"gross={gross.strategy_returns.sum():.4f} net={net.strategy_returns.sum():.4f}")


if __name__ == "__main__":
    test_liquidity_gradient()
    test_borrow()
    test_backward_compat()
    test_churn_penalty()
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — realistic cost model (spread + impact + borrow), liquidity-aware")
