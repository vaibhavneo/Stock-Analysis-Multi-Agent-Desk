"""
Position-management rule backtests (Phase 24).

The question is narrow and worth asking precisely: given that the core signal
has already decided to be long, does HOW the position is entered, added to, or
exited add anything? Seven rules are tested against the same base signal on the
same price path, so any difference between them is attributable to the rule and
nothing else.

    buy_immediately              the baseline — take the signal as-is
    wait_for_pullback            enter only when price is near its N-day low
    confirmation_entry           enter only after price makes an N-day high
    staged_entry                 half on signal, the rest on a pullback
    add_on_confirmation          start at half, add on an N-day high
    reduce_on_deterioration      cut to half when the signal weakens but has
                                 not exited
    exit_on_invalidation         exit immediately when price loses the N-day low

Every rule is evaluated with the machinery the rest of this codebase already
uses and trusts:

  - point-in-time execution (the engine's own single `.shift(1)`)
  - realistic costs: spread + square-root market impact + borrow, with volume
    feeding the liquidity proxy (backtest/costs.py)
  - walk-forward with purge and embargo (backtest/validation.py)
  - deflated Sharpe against the number of rules tried — the multiple-testing
    correction is the whole point, because seven rules on one price path will
    always produce a best one
  - probability of backtest overfitting across the rule set

The honest default is NO_DEMONSTRATED_EDGE. This module exists to find out
whether position management adds value, not to produce a rule that backtests
well; a run where nothing clears is a successful run with a negative result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.costs import DEFAULT_COST_MODEL
from backtest.engine import (compute_performance_metrics, deflated_sharpe_ratio,
                             run_vectorized_backtest)
from backtest.validation import (probability_of_backtest_overfitting, walk_forward_cv)

# Lookback for the pullback / confirmation / invalidation references. One value,
# fixed in advance, used by every rule that needs a window — so the comparison
# is between RULES, not between parameter choices. Tuning this per rule would
# turn seven trials into seventy.
LOOKBACK = 20

# Proximity to the N-day low that counts as a pullback, and above the N-day high
# that counts as confirmation. Pre-committed, not fitted.
PULLBACK_BAND = 0.03
PARTIAL_SIZE = 0.5

DSR_BAR = 0.5
PBO_MAX = 0.5
MIN_OBS = 504


@dataclass
class RuleResult:
    name: str
    sharpe: float
    dsr: float
    annualized_return_pct: float
    max_drawdown_pct: float
    n_trades: int
    total_cost_pct: float
    time_in_market_pct: float
    walk_forward: Dict[str, Any]
    beats_baseline: bool


# ── Rules. Each maps (prices, base_signal, high, low) → position series. ────
# Positions are in [0, 1]: this is a long-only system and a rule may hold a
# partial position, which is exactly what staged entry and trimming mean.

def _rolling(prices: pd.Series, n: int):
    return prices.rolling(n, min_periods=n)


def rule_buy_immediately(prices, base, high, low):
    return base.astype(float)


def rule_wait_for_pullback(prices, base, high, low):
    """Long only while the signal is on AND price is within PULLBACK_BAND of its
    N-day low. Deliberately strict: this is the "never chase" discipline in its
    purest form, and its cost (missing trends entirely) is the thing being
    measured."""
    near_low = prices <= low * (1.0 + PULLBACK_BAND)
    entered = (base > 0) & near_low
    # Once entered, stay in while the base signal holds.
    state = np.zeros(len(prices))
    in_pos = False
    e = entered.to_numpy()
    b = (base > 0).to_numpy()
    for i in range(len(prices)):
        if not b[i]:
            in_pos = False
        elif e[i]:
            in_pos = True
        state[i] = 1.0 if in_pos else 0.0
    return pd.Series(state, index=prices.index)


def rule_confirmation_entry(prices, base, high, low):
    """Long only after price has confirmed by making an N-day high while the
    signal is on. Pays a worse price for a resolved level."""
    confirmed = prices >= high
    state = np.zeros(len(prices))
    in_pos = False
    c = confirmed.to_numpy()
    b = (base > 0).to_numpy()
    for i in range(len(prices)):
        if not b[i]:
            in_pos = False
        elif c[i]:
            in_pos = True
        state[i] = 1.0 if in_pos else 0.0
    return pd.Series(state, index=prices.index)


def rule_staged_entry(prices, base, high, low):
    """Half on the signal, the rest on a pullback toward the N-day low."""
    full = rule_wait_for_pullback(prices, base, high, low)
    half = (base > 0).astype(float) * PARTIAL_SIZE
    return pd.Series(np.maximum(half.to_numpy(), full.to_numpy()), index=prices.index)


def rule_add_on_confirmation(prices, base, high, low):
    """Half on the signal, topped up to full once price confirms."""
    conf = rule_confirmation_entry(prices, base, high, low)
    half = (base > 0).astype(float) * PARTIAL_SIZE
    return pd.Series(np.maximum(half.to_numpy(), conf.to_numpy()), index=prices.index)


def rule_reduce_on_deterioration(prices, base, high, low):
    """Cut to half while the position is still on but deteriorating — price
    below its own N-day mean.

    Deliberately NOT keyed to the base signal's own moving average: the base
    signal is binary {0,1}, so its rolling mean carries almost no information
    and the rule collapsed into the baseline. Reading price against its mean is
    a real deterioration proxy that works for any signal, and the difference is
    exactly what the "rules must differ from the baseline" test now enforces.
    """
    on = (base > 0).astype(float)
    mean = prices.rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    weakening = (prices < mean).fillna(False)
    return pd.Series(on.to_numpy() * np.where(weakening.to_numpy(), PARTIAL_SIZE, 1.0),
                     index=prices.index)


def rule_exit_on_invalidation(prices, base, high, low):
    """Exit the moment price loses the N-day low, and stay out until the base
    signal cycles off and back on. A hard invalidation discipline."""
    broken = prices < low
    state = np.zeros(len(prices))
    in_pos = False
    stopped = False
    br = broken.to_numpy()
    b = (base > 0).to_numpy()
    for i in range(len(prices)):
        if not b[i]:
            in_pos, stopped = False, False
        elif stopped:
            in_pos = False
        elif br[i]:
            in_pos, stopped = False, True
        else:
            in_pos = True
        state[i] = 1.0 if in_pos else 0.0
    return pd.Series(state, index=prices.index)


RULES: Dict[str, Callable] = {
    "buy_immediately": rule_buy_immediately,
    "wait_for_pullback": rule_wait_for_pullback,
    "confirmation_entry": rule_confirmation_entry,
    "staged_entry": rule_staged_entry,
    "add_on_confirmation": rule_add_on_confirmation,
    "reduce_on_deterioration": rule_reduce_on_deterioration,
    "exit_on_invalidation": rule_exit_on_invalidation,
}

BASELINE_RULE = "buy_immediately"


def _references(df: pd.DataFrame, lookback: int = LOOKBACK):
    """N-day high/low, shifted one bar so a rule can never reference the very
    bar it is deciding on. The engine shifts execution again, so this is
    belt-and-braces against exactly the leak these rules are most prone to."""
    high = df["High"].rolling(lookback, min_periods=lookback).max().shift(1)
    low = df["Low"].rolling(lookback, min_periods=lookback).min().shift(1)
    return high, low


def evaluate_position_rules(
    df: pd.DataFrame,
    base_signal: Optional[pd.Series] = None,
    lookback: int = LOOKBACK,
    n_folds: int = 5,
    embargo_frac: float = 0.02,
    purge_obs: int = 50,
) -> Dict[str, Any]:
    """Run every rule on one ticker and report, honestly, whether any of them
    adds value over taking the signal as-is."""
    if df is None or df.empty or "Close" not in df.columns:
        return {"status": "INSUFFICIENT_DATA", "reason": "no price history", "rules": []}

    prices = df["Close"].astype(float)
    if len(prices) < MIN_OBS:
        return {"status": "INSUFFICIENT_SAMPLE",
                "reason": f"{len(prices)} bars < {MIN_OBS} minimum",
                "n_obs": len(prices), "rules": []}

    if base_signal is None:
        from backtest.pillars import seven_pillar_core_strategy
        base_signal = seven_pillar_core_strategy(df)
    base_signal = base_signal.reindex(prices.index).fillna(0.0)

    high, low = _references(df, lookback)
    volume = df["Volume"] if "Volume" in df.columns else None

    results: List[RuleResult] = []
    returns_matrix: Dict[str, pd.Series] = {}
    baseline_sharpe = None

    for name, fn in RULES.items():
        try:
            sig = fn(prices, base_signal, high, low).reindex(prices.index).fillna(0.0)
            bt = run_vectorized_backtest(prices, sig, cost_model=DEFAULT_COST_MODEL,
                                         volume=volume)
            m = compute_performance_metrics(bt.strategy_returns)
            dsr = deflated_sharpe_ratio(m["sharpe_ratio"], n_trials=len(RULES),
                                        skewness=m["skewness"], kurtosis=m["kurtosis"],
                                        n_obs=m["n_observations"])
            returns_matrix[name] = bt.strategy_returns

            wf: Dict[str, Any] = {"applied": False}
            try:
                def _sig_fn(slice_prices, _fn=fn, _lb=lookback):
                    sub = df.loc[slice_prices.index]
                    h, l = _references(sub, _lb)
                    b = base_signal.reindex(slice_prices.index).fillna(0.0)
                    return _fn(slice_prices, b, h, l).reindex(slice_prices.index).fillna(0.0)

                w = walk_forward_cv(prices, _sig_fn, n_folds=n_folds,
                                    embargo_frac=embargo_frac, purge_obs=purge_obs,
                                    cost_model=DEFAULT_COST_MODEL, volume=volume)
                wf = {"applied": True, "n_folds": w.n_folds,
                      "mean_oos_sharpe": round(w.mean_test_sharpe, 3),
                      "consistency": round(w.consistency, 2),
                      "embargo_frac": embargo_frac, "purge_obs": purge_obs}
            except Exception as e:
                wf = {"applied": False, "error": str(e)[:80]}

            if name == BASELINE_RULE:
                baseline_sharpe = m["sharpe_ratio"]

            results.append(RuleResult(
                name=name,
                sharpe=round(m["sharpe_ratio"], 3),
                dsr=round(dsr, 3),
                annualized_return_pct=round(m["annualized_return"] * 100, 2),
                max_drawdown_pct=round(m["max_drawdown"] * 100, 1),
                n_trades=bt.n_trades,
                total_cost_pct=round(bt.total_cost * 100, 2),
                time_in_market_pct=round(float((sig > 0).mean()) * 100, 1),
                walk_forward=wf,
                beats_baseline=False,
            ))
        except Exception:
            continue

    if baseline_sharpe is not None:
        for r in results:
            r.beats_baseline = r.sharpe > baseline_sharpe

    # ── Multiple-testing correction across the rule set ──────────────────
    pbo: Dict[str, Any] = {"computed": False}
    try:
        rmat = pd.DataFrame(returns_matrix).dropna(how="any")
        if rmat.shape[1] >= 2 and len(rmat) >= 40:
            p = probability_of_backtest_overfitting(rmat, n_splits=8)
            pbo = {"computed": True, "pbo": round(p.pbo, 3),
                   "is_overfit": p.is_overfit, "bar": PBO_MAX}
    except Exception as e:
        pbo = {"computed": False, "error": str(e)[:80]}

    rows = sorted((r.__dict__ for r in results), key=lambda r: -r["dsr"])
    best = rows[0] if rows else None

    demonstrated = bool(
        best
        and best["dsr"] >= DSR_BAR
        and best["walk_forward"].get("applied")
        and (best["walk_forward"].get("mean_oos_sharpe") or -1) > 0
        and pbo.get("computed") and pbo.get("pbo", 1.0) < PBO_MAX
        and best["name"] != BASELINE_RULE
        and best["beats_baseline"])

    return {
        "status": "OK",
        "n_obs": len(prices),
        "n_rules_tested": len(rows),
        "lookback": lookback,
        "baseline": BASELINE_RULE,
        "baseline_sharpe": round(baseline_sharpe, 3) if baseline_sharpe is not None else None,
        "rules": rows,
        "best_rule": best["name"] if best else None,
        "pbo": pbo,
        "verdict": "DEMONSTRATED_EDGE" if demonstrated else "NO_DEMONSTRATED_EDGE",
        "statement": (
            f"{best['name']} clears the deflated-Sharpe bar, beats taking the signal as-is, "
            f"survives walk-forward, and the rule set's PBO is below {PBO_MAX}."
            if demonstrated else
            "NO_DEMONSTRATED_EDGE: no position-management rule beat taking the signal as-is "
            "after cost, walk-forward and multiple-testing correction. Position management "
            "is not shown to add value on this sample, which is a result, not a failure."),
        "method": {
            "costs": "spread + square-root impact + borrow (backtest/costs.py)",
            "execution": "point-in-time; signal at t executes at t+1",
            "validation": f"walk-forward, {n_folds} folds, purge {purge_obs} bars, "
                          f"embargo {embargo_frac}",
            "multiple_testing": f"deflated Sharpe against n_trials={len(RULES)}, plus PBO "
                                f"across the rule set",
            "survivorship": ("Single-ticker evaluation on a surviving name — results do NOT "
                             "generalize to a universe and are not survivorship-safe."),
        },
    }


def evaluate_across_tickers(tickers: List[str], fetch_fn: Optional[Callable] = None,
                            **kwargs) -> Dict[str, Any]:
    """Aggregate the rule comparison across several names.

    A rule that wins on one ticker has won one coin flip. The aggregate is the
    only number worth reading, and it is still not survivorship-safe.
    """
    if fetch_fn is None:
        from tools.market_data import fetch_price_history

        def fetch_fn(t):                       # noqa: E306
            return fetch_price_history(t, period="5y")

    per_ticker: Dict[str, Any] = {}
    agg: Dict[str, Dict[str, List[float]]] = {}
    for t in tickers:
        try:
            res = evaluate_position_rules(fetch_fn(t), **kwargs)
        except Exception as e:
            per_ticker[t] = {"status": "ERROR", "error": str(e)[:80]}
            continue
        per_ticker[t] = res
        if res.get("status") != "OK":
            continue
        for r in res["rules"]:
            b = agg.setdefault(r["name"], {"sharpe": [], "dsr": [], "return": [],
                                           "max_dd": [], "beats": []})
            b["sharpe"].append(r["sharpe"])
            b["dsr"].append(r["dsr"])
            b["return"].append(r["annualized_return_pct"])
            b["max_dd"].append(r["max_drawdown_pct"])
            b["beats"].append(1.0 if r["beats_baseline"] else 0.0)

    def _m(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    summary = {name: {"n_tickers": len(b["sharpe"]), "mean_sharpe": _m(b["sharpe"]),
                      "mean_dsr": _m(b["dsr"]), "mean_annual_return_pct": _m(b["return"]),
                      "mean_max_drawdown_pct": _m(b["max_dd"]),
                      "beat_baseline_rate": _m(b["beats"])}
               for name, b in sorted(agg.items())}

    ok = [t for t, r in per_ticker.items() if r.get("status") == "OK"]
    winners = [n for n, s in summary.items()
               if n != BASELINE_RULE and (s["beat_baseline_rate"] or 0) > 0.5
               and (s["mean_dsr"] or 0) >= DSR_BAR]

    return {
        "n_tickers_evaluated": len(ok),
        "tickers": ok,
        "summary": summary,
        "candidate_rules": winners,
        "verdict": "CANDIDATE_RULES_FOUND" if winners else "NO_DEMONSTRATED_EDGE",
        "statement": (
            f"{len(winners)} rule(s) beat the baseline on a majority of {len(ok)} tickers AND "
            f"cleared the deflated-Sharpe bar on average. That is a CANDIDATE, not a "
            f"demonstrated edge — it still needs out-of-sample confirmation on names not used "
            f"to select it." if winners else
            f"NO_DEMONSTRATED_EDGE across {len(ok)} tickers: no position-management rule "
            f"reliably beat taking the signal as-is after costs and multiple-testing "
            f"correction."),
        "per_ticker": per_ticker,
    }
