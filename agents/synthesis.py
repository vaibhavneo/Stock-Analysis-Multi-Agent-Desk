"""
Phase 5 — Grounded Synthesis
============================
Replaces the LLM-invented entry/target/stop numbers with values computed
from backtested strategies.  The LLM prose (summary, bull_case, bear_case)
is left completely untouched; only the numeric fields are overwritten.

Flow:
  1. Run all 7 strategies against already-fetched price history.
  2. Find the strategy whose current signal agrees with the LLM's action.
  3. If found:
       entry_price  = current_price  (deterministic)
       stop_loss    = current_price - 1.5 * ATR_14   (risk anchor)
       target_price = entry + 2 * (entry - stop_loss)  (2:1 R:R)
       position_size = safe_kelly_fraction(strategy_returns)
       confidence    = get_historical_hit_rate(strategy)
  4. If no strategy agrees: downgrade conviction to LOW, flag grounding="none".
  5. Return original pred dict extended with grounding_* fields.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.engine import run_vectorized_backtest, compute_performance_metrics, deflated_sharpe_ratio
from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
from backtest.risk import safe_kelly_fraction
from data.store import get_historical_hit_rate
from data import ledger


_LONG_ACTIONS  = {"BUY", "LONG", "STRONG BUY"}
_SHORT_ACTIONS = {"SELL", "SHORT", "AVOID", "STRONG SELL"}


def _record_live_trial(ticker, name, metrics, outcome="inconclusive", dsr=None):
    """Register a live-path backtest in the TrialRegistry.

    `params` deliberately carries the coarse label "live" rather than the exact
    date range: price_df grows by one bar a day, so keying on the range would
    mint a fresh trial every morning and inflate n_trials without anyone having
    tried a new idea. The same strategy tested via the CLI over an explicit
    period IS a separate trial (different window = another shot at a flattering
    result), so those ids differ — a slight over-count.

    Over-counting is the safe direction: it deflates dSR further and makes the
    verdict more conservative. Under-counting is what turns dSR into a lie.
    """
    try:
        return ledger.record_trial(
            hypothesis=f"{name} beats buy-and-hold on {ticker}",
            params={"strategy": name, "period": "live"},
            outcome=outcome,
            family=ledger.strategy_family(ticker),
            strategy=name,
            universe=ticker,
            sharpe=metrics["sharpe_ratio"] if metrics else None,
            flags={"survivorship_safe": False, "pit_fundamentals": False,
                   "cost_model": "flat_10bps", "dsr": dsr},
        )
    except Exception:
        # The registry must never break a user's analysis. A failure here costs
        # trial-count accuracy (dSR gets less deflation than it should), so it
        # is surfaced by the count itself rather than silently swallowed:
        # effective_n_trials() still floors at the strategies tested this run.
        return None


def ground_prediction(
    ticker: str,
    current_price: float,
    llm_prediction: dict,
    price_df: pd.DataFrame,
    indicators: dict,
) -> dict:
    """
    Core grounding function.

    Parameters
    ----------
    ticker         : e.g. "AAPL"
    current_price  : latest closing price
    llm_prediction : the dict returned by run_prediction_agent()
    price_df       : full OHLCV DataFrame (from fetch_price_history)
    indicators     : dict from compute_indicators()

    Returns
    -------
    A copy of llm_prediction with numeric fields overwritten (or downgraded)
    and new grounding_* fields appended.
    """
    pred   = dict(llm_prediction)
    prices = price_df["Close"].dropna()

    raw_action = str(pred.get("action") or pred.get("recommendation") or "").upper().strip()

    if raw_action in _LONG_ACTIONS:
        wanted_signal = 1
    elif raw_action in _SHORT_ACTIONS:
        wanted_signal = -1
    else:
        wanted_signal = 0

    # ── Run all strategies, collect their current signal + returns ─────────
    # PASS 1: backtest and REGISTER each variant before scoring any of them.
    # Selecting the best-agreeing strategy out of 7 to justify a call is exactly
    # the multiple-comparisons problem dSR guards against, so these are real
    # trials and are recorded as such — idempotently, by content hash, so a user
    # clicking Analyze ten times does not inflate the count ten-fold.
    strategy_results: list[dict] = []
    for name, fn in STRATEGY_REGISTRY.items():
        try:
            signal = fn(price_df) if name in STRATEGIES_NEEDING_FULL_DF else fn(prices)
            signal = signal.fillna(0)
            if len(signal) == 0:
                continue
            result  = run_vectorized_backtest(prices, signal)
            metrics = compute_performance_metrics(result.strategy_returns)
            current_sig = int(signal.iloc[-1])
            _record_live_trial(ticker, name, metrics)
            strategy_results.append({
                "name":    name,
                "signal":  current_sig,
                "sharpe":  metrics["sharpe_ratio"],
                "metrics": metrics,
                "returns": result.strategy_returns,
                "kelly":   safe_kelly_fraction(result.strategy_returns),
            })
        except Exception:
            continue

    # THE COUNT — the IMMUTABLE pre-registered variant count (fixed = 8), so a
    # grounded prediction cannot change because more backtests were run. The
    # per-run executions are still logged (below) for the forward audit.
    from backtest import experiments
    n_trials = experiments.deflation_n()

    # PASS 2: score against the honest count and finalize each trial's verdict.
    for s in strategy_results:
        m = s.pop("metrics")
        s["dsr"] = deflated_sharpe_ratio(
            m["sharpe_ratio"], n_trials=n_trials,
            skewness=m["skewness"], kurtosis=m["kurtosis"], n_obs=m["n_observations"],
        )
        _record_live_trial(ticker, s["name"], m,
                           outcome="alive" if (s["dsr"] >= 0.5 and m["annualized_return"] > 0)
                                   else "dead",
                           dsr=round(s["dsr"], 4))

    # ── Find best agreeing strategy ────────────────────────────────────────
    # A strategy only counts as genuine support if it (a) currently points the
    # same direction as the LLM AND (b) has a POSITIVE historical edge. A
    # losing strategy (Sharpe <= 0) that happens to point the same way is not
    # evidence for the call — treating it as grounding would let a money-losing
    # rule "confirm" a BUY and keep conviction HIGH, which defeats the purpose.
    directional = [s for s in strategy_results if s["signal"] == wanted_signal and wanted_signal != 0]
    agreeing    = [s for s in directional if s["sharpe"] > 0]
    # rank by dSR then Sharpe
    agreeing.sort(key=lambda s: (s["dsr"], s["sharpe"]), reverse=True)

    best = agreeing[0] if agreeing else None
    # how many pointed the right way but had no edge (for an honest note)
    n_directional_no_edge = len(directional) - len(agreeing)

    # ── Compute ATR for stop distance ──────────────────────────────────────
    atr = _compute_atr(price_df, window=14)

    # ── Build grounded numbers ─────────────────────────────────────────────
    if best is not None:
        stop_distance  = max(1.5 * atr, current_price * 0.01)  # at least 1%
        entry_price    = round(current_price, 2)

        if wanted_signal >= 0:   # long or neutral
            stop_loss    = round(current_price - stop_distance, 2)
            target_price = round(current_price + 2 * stop_distance, 2)
        else:                    # short
            stop_loss    = round(current_price + stop_distance, 2)
            target_price = round(current_price - 2 * stop_distance, 2)

        upside_pct    = round((target_price - current_price) / current_price * 100, 1)
        position_size = best["kelly"]

        hist_hit = get_historical_hit_rate(strategy_source=best["name"], ticker=ticker)

        pred["entry_price"]  = entry_price
        pred["target_price"] = target_price
        pred["stop_loss"]    = stop_loss
        pred["upside_pct"]   = upside_pct

        grounding = {
            "grounding":                  "strategy",
            "grounding_strategy":         best["name"],
            "grounding_backtest_sharpe":  round(best["sharpe"], 3),
            "grounding_dsr":              round(best["dsr"], 3),
            "grounding_historical_hit_rate": hist_hit.get("hit_rate"),
            "grounding_hit_rate_n":       hist_hit.get("total", 0),
            "grounding_atr":              round(atr, 3),
            "position_size":              position_size,
            "n_agreeing_strategies":      len(agreeing),
            "all_strategy_signals":       {s["name"]: s["signal"] for s in strategy_results},
        }
    else:
        # No strategy agrees — downgrade conviction, flag clearly
        grounding = {
            "grounding":                  "none",
            "grounding_strategy":         None,
            "grounding_backtest_sharpe":  None,
            "grounding_dsr":              None,
            "grounding_historical_hit_rate": None,
            "grounding_hit_rate_n":       0,
            "grounding_atr":              round(atr, 3),
            "position_size":              0.0,
            "n_agreeing_strategies":      0,
            "n_directional_no_edge":      n_directional_no_edge,
            "all_strategy_signals":       {s["name"]: s["signal"] for s in strategy_results},
        }
        pred["conviction"] = "LOW"

    pred.update(grounding)
    return pred


# ── Helpers ────────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Delegates to the one shared ATR (backtest/risk.py::compute_atr) so the
    trading and investing paths measure the same volatility. Kept as a private
    alias so existing callers/tests are untouched."""
    from backtest.risk import compute_atr
    return compute_atr(df, window=window)
