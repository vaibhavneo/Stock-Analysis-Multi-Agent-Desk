"""
Command-line backtest runner — a hands-on way to see what backtest/ produces
without writing any Python.

Usage:
    python3 backtest/run_backtest.py AAPL                 # all 7 strategies on AAPL
    python3 backtest/run_backtest.py AAPL momentum        # just the momentum strategy
    python3 backtest/run_backtest.py SPY --period 5y      # longer history

It fetches real price history, runs each strategy through the vectorized
backtest engine, and prints a comparison table of the standard performance
metrics (Sharpe, Sortino, max drawdown, Calmar, win rate, Deflated Sharpe),
with a buy-and-hold row as the benchmark to beat.

NOTE: a backtest describes how a strategy WOULD have performed on past data.
It is not a prediction and not investment advice — past performance does not
guarantee future results. This tool exists to sanity-check the engine and to
see which signals have historically had an edge, not to tell you what to buy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tools.market_data import fetch_price_history
from backtest.engine import (
    run_vectorized_backtest, compute_performance_metrics, deflated_sharpe_ratio,
)
from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
from backtest.risk import safe_kelly_fraction
from data import ledger

# The bar the tool already used for "likely real, not luck" (López de Prado Ch.8).
DSR_ALIVE_BAR = 0.5


def _backtest_one(name, fn, df, prices):
    """Run one strategy. Deliberately does NOT compute dSR: dSR needs the trial
    count, and the count is not known until every strategy in this run has been
    recorded. Separating the two is what stops a strategy from being scored
    against a trial count that excludes its own siblings."""
    signal = fn(df) if name in STRATEGIES_NEEDING_FULL_DF else fn(prices)
    signal = signal.fillna(0)
    result = run_vectorized_backtest(prices, signal)
    m = compute_performance_metrics(result.strategy_returns)
    current_signal = int(signal.iloc[-1]) if len(signal) else 0   # today's stance: +1 long / -1 short / 0 flat
    kelly = safe_kelly_fraction(result.strategy_returns)          # half-Kelly, capped at 10% of capital
    return m, result.n_trades, current_signal, kelly


def _record(ticker, name, m, period, outcome, dsr=None):
    """Register one tested hypothesis in the TrialRegistry — permanently.

    Every backtest is a trial whether or not anyone liked the answer, and the
    losers are the point: they are dSR's denominator. Recording only winners
    would leave the count understating reality exactly as the old hardcoded
    n_trials=7 did.
    """
    return ledger.record_trial(
        hypothesis=f"{name} beats buy-and-hold on {ticker}",
        params={"strategy": name, "period": period},
        outcome=outcome,
        family=ledger.strategy_family(ticker),
        strategy=name,
        universe=ticker,
        sharpe=m["sharpe_ratio"] if m else None,
        # Flags travel WITH the trial so a future reader knows what this number
        # was and was not corrected for. Both are still false — M-F1/M-F3 work.
        flags={"survivorship_safe": False, "pit_fundamentals": False,
               "cost_model": "flat_10bps", "dsr": dsr},
        notes=None if m else "errored",
    )


def _plain_english_verdict(ticker, period, bh, rows):
    """Translate the backtest table into a simple-English takeaway for someone
    who doesn't read Sharpe ratios. Honest by construction: a backtest says
    whether a STRATEGY had a historical edge, which is not the same as a stock
    tip — so the verdict is careful to distinguish 'the data supports a signal'
    from 'you should buy this stock,' and always leads with the hold-vs-trade
    question a lay investor actually cares about."""
    if not rows:
        return "Not enough data to form a plain-English takeaway."

    # rows are sorted best-Sharpe-first
    best_name, best_m, best_dsr, _best_trades, best_signal, best_kelly = rows[0]
    best_sharpe = best_m["sharpe_ratio"]
    best_made_money = best_m["annualized_return"] > 0
    best_beats_hold = best_sharpe > bh["sharpe_ratio"]
    best_credible = best_dsr >= 0.5
    stock_rose = bh["annualized_return"] > 0

    dir_word = {1: "a BUY (long) position", -1: "a SELL (short/avoid) position", 0: "no position (flat)"}[best_signal]
    size_clause = (
        f" Suggested size if you act: about {best_kelly*100:.0f}% of your trading capital "
        f"(half-Kelly, capped at 10%)."
        if best_kelly > 0 and best_signal != 0 else ""
    )

    lines = []

    if best_credible and best_made_money:
        # A strategy has a statistically credible, profitable edge — safest case to give a lean.
        lines.append(
            f"The '{best_name}' strategy has a historically credible edge on {ticker} "
            f"over the last {period} (it made money AND the confidence score is {round(best_dsr,2)}/1.0, "
            f"above the 0.5 'likely real, not luck' bar)."
        )
        if best_signal == 1:
            simple = f"LEANS BUY — the strongest strategy currently holds {dir_word} on {ticker}.{size_clause}"
        elif best_signal == -1:
            simple = f"LEANS SELL / AVOID — the strongest strategy currently holds {dir_word} on {ticker}.{size_clause}"
        else:
            simple = f"HOLD / STAY OUT — the strongest strategy has an edge but sees no clear signal right now (it's flat)."
    elif best_beats_hold and best_made_money:
        # Beat buy-and-hold and made money, but the edge isn't statistically convincing.
        lines.append(
            f"The '{best_name}' strategy did best over the last {period} and beat simply holding {ticker}, "
            f"but its edge isn't statistically convincing yet (confidence {round(best_dsr,2)}/1.0, below 0.5 — "
            f"it could still be luck)."
        )
        lean = {1: "a weak BUY lean", -1: "a weak SELL lean", 0: "no clear signal"}[best_signal]
        simple = f"HOLD / LOW-CONFIDENCE {('BUY' if best_signal==1 else 'SELL' if best_signal==-1 else 'SIGNAL')} — {lean}, not strong enough to act on with confidence."
    else:
        # Nothing credibly beat just holding the stock.
        trend = "rose" if stock_rose else "fell"
        lines.append(
            f"None of the 7 strategies reliably beat simply buying and holding {ticker} over the last {period}. "
            f"Trying to time trades on this stock would have done worse than just holding it. "
            f"(The stock itself {trend} over the period.)"
        )
        simple = (
            "HOLD — if you own it, holding has historically beaten trying to trade it here. "
            "These backtests give no reliable buy-or-sell signal right now."
        )

    return lines, simple


def _print_verdict(ticker, period, bh, rows):
    result = _plain_english_verdict(ticker, period, bh, rows)
    if isinstance(result, str):
        print(result)
        return
    lines, simple = result
    print("\n" + "=" * 78)
    print("  IN PLAIN ENGLISH")
    print("=" * 78)
    for ln in lines:
        # simple word-wrap at ~74 chars
        words, line = ln.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 74:
                print("  " + line); line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            print("  " + line)
    print()
    print(f"  \U0001F449 Simple lean: {simple}")
    print()
    print("  Remember: this is based on PAST data, not a prediction of the future,")
    print("  and it is not financial advice. It tells you whether a trading strategy")
    print("  has historically worked on this stock — not what the stock will do next.")


def _buy_and_hold(prices):
    signal = pd.Series(1, index=prices.index)   # always long
    result = run_vectorized_backtest(prices, signal, transaction_cost_bps=0)
    return compute_performance_metrics(result.strategy_returns)


def main():
    # Parse args: pull out --period VALUE first, so its value isn't mistaken
    # for a positional (ticker/strategy) argument.
    raw = sys.argv[1:]
    period = "3y"
    args = []
    i = 0
    while i < len(raw):
        if raw[i] == "--period" and i + 1 < len(raw):
            period = raw[i + 1]
            i += 2
        elif raw[i].startswith("--"):
            i += 1   # ignore unknown flags
        else:
            args.append(raw[i])
            i += 1

    if not args:
        print("Usage: python3 backtest/run_backtest.py TICKER [strategy_name] [--period 3y]")
        print("Strategies:", ", ".join(STRATEGY_REGISTRY.keys()))
        sys.exit(1)

    ticker = args[0].upper()
    only_strategy = args[1] if len(args) > 1 else None

    print(f"\nFetching {ticker} ({period} of history)...")
    df = fetch_price_history(ticker, period=period)
    if df is None or len(df) < 60:
        print(f"Not enough price history for {ticker}. Try a different ticker or a longer --period.")
        sys.exit(1)
    prices = df["Close"]
    print(f"Got {len(prices)} trading days: {prices.index[0].date()} → {prices.index[-1].date()}\n")

    if only_strategy:
        if only_strategy not in STRATEGY_REGISTRY:
            print(f"Unknown strategy '{only_strategy}'. Choose from: {', '.join(STRATEGY_REGISTRY)}")
            sys.exit(1)
        to_run = {only_strategy: STRATEGY_REGISTRY[only_strategy]}
    else:
        to_run = STRATEGY_REGISTRY

    # Header
    cols = f"{'strategy':<22}{'Sharpe':>8}{'Sortino':>9}{'MaxDD':>8}{'Calmar':>8}{'WinRate':>9}{'dSR':>7}{'Trades':>8}{'Size':>8}"
    print(cols)
    print("-" * len(cols))

    # Benchmark
    bh = _buy_and_hold(prices)
    print(f"{'buy & hold (benchmark)':<22}{bh['sharpe_ratio']:>8}{bh['sortino_ratio']:>9}"
          f"{bh['max_drawdown']:>8}{bh['calmar_ratio']:>8}{bh['win_rate']:>9}{'—':>7}{'—':>8}{'—':>8}")
    print("-" * len(cols))

    # PASS 1 — run every strategy and register each as a trial BEFORE scoring
    # any of them. A trial is recorded whether it wins, loses, or errors.
    raw = []
    for name, fn in to_run.items():
        try:
            m, n_trades, current_signal, kelly = _backtest_one(name, fn, df, prices)
            _record(ticker, name, m, period, outcome="inconclusive")   # dSR not known yet
            raw.append((name, m, n_trades, current_signal, kelly))
        except Exception as e:
            _record(ticker, name, None, period, outcome="inconclusive")
            print(f"{name:<22} ERROR: {e}")

    # THE COUNT — from the registry, which remembers every variant ever tried on
    # this ticker across every session, not just the ones in this process. This
    # is what makes `run_backtest.py AAPL momentum` score against the 7 you
    # already tried yesterday instead of pretending it is the first idea ever had.
    n_trials = ledger.effective_n_trials(ticker, tested_this_run=len(raw))

    # PASS 2 — score against the honest count, then finalize each trial's verdict.
    rows = []
    for name, m, n_trades, current_signal, kelly in raw:
        dsr = deflated_sharpe_ratio(
            m["sharpe_ratio"], n_trials=n_trials,
            skewness=m["skewness"], kurtosis=m["kurtosis"], n_obs=m["n_observations"],
        )
        alive = dsr >= DSR_ALIVE_BAR and m["annualized_return"] > 0
        _record(ticker, name, m, period, outcome="alive" if alive else "dead", dsr=round(dsr, 4))
        rows.append((name, m, dsr, n_trades, current_signal, kelly))

    n_registered = ledger.n_trials(family=ledger.strategy_family(ticker))
    print(f"{'':<22}{'':>8}  (dSR deflated against n_trials={n_trials}; "
          f"{n_registered} variant(s) ever tried on {ticker})")
    print("-" * len(cols))

    # Sort by Sharpe, best first
    rows.sort(key=lambda r: r[1]["sharpe_ratio"], reverse=True)
    for name, m, dsr, n_trades, _sig, kelly in rows:
        size_str = f"{kelly*100:.1f}%" if kelly > 0 else "0%"
        print(f"{name:<22}{m['sharpe_ratio']:>8}{m['sortino_ratio']:>9}"
              f"{m['max_drawdown']:>8}{m['calmar_ratio']:>8}{m['win_rate']:>9}{round(dsr,3):>7}{n_trades:>8}{size_str:>8}")

    # ── Plain-English takeaway (for non-technical readers) ────────────────────
    _print_verdict(ticker, period, bh, rows)

    print()
    print("How to read the table above:")
    print("  Sharpe  > 1.0 is good, < 0.5 is weak (risk-adjusted return)")
    print("  MaxDD   = worst peak-to-trough drop (lower is better)")
    print("  WinRate = fraction of trading days that were positive")
    print("  dSR     = Deflated Sharpe: probability (0-1) the edge is real, not luck,")
    print("            AFTER accounting for how many strategies were tried. > 0.5 = likely real.")
    print("  Size    = suggested position size (half-Kelly, capped at 10% of capital).")
    print("            0% means the strategy has no positive historical edge — don't size it.")
    print("  A strategy that can't even beat 'buy & hold' on Sharpe isn't adding value here.")
    print()


if __name__ == "__main__":
    main()
