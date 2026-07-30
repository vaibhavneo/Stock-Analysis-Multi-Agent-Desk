"""
Backfill the TrialRegistry with hypotheses already tested before it existed.

Why this matters: dSR deflates against the number of variants tried. An empty
registry says "nobody has ever tried anything", which is false and dangerous —
it makes dSR maximally generous exactly when the accumulated multiple-testing
debt is largest. The 7 LEGACY strategies (pinned below) have been run against
these tickers repeatedly during development; that history is real and must be
counted even though nobody was writing it down at the time.

Usage:
    python3 backtest/backfill_trials.py            # dry-run: show what WOULD be recorded
    python3 backtest/backfill_trials.py --commit   # actually record

Honesty note: these trials carry outcome="inconclusive" and
notes="backfilled (reconstructed, not re-run)". We know these combinations WERE
tried; we do not have their original Sharpes, and inventing them would be
exactly the kind of fabricated evidence this whole layer exists to prevent. The
trial COUNT is what dSR needs, and the count is what we can honestly reconstruct.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import ledger

# The strategies that existed when this history was accumulated — PINNED, not
# read from the live registry. A backfill reconstructs trials that actually
# happened; iterating the registry would silently backfill strategies added
# LATER (e.g. seven_pillar_core, added 2026-07-17), fabricating trials that
# never ran — exactly the invented evidence this layer forbids.
LEGACY_STRATEGIES = ["sma_crossover", "momentum", "mean_reversion", "stat_arb",
                     "trend_following", "candlestick_filtered", "rsi_macd"]

# Tickers the 7 strategies were demonstrably run against during development —
# from the backtest CLI's own documented examples and the evaluation the user
# commissioned over PLTR/RGTI/IONQ. Conservative: only tickers with real evidence
# of having been tested, since an invented ticker would be an invented trial.
BACKFILLED_TICKERS = ["AAPL", "SPY", "PLTR", "RGTI", "IONQ"]

# Period labels actually exercised: the CLI's default and the 1-3y evaluation.
BACKFILLED_PERIODS = ["3y", "1y"]


def plan():
    rows = []
    for ticker in BACKFILLED_TICKERS:
        for period in BACKFILLED_PERIODS:
            for name in LEGACY_STRATEGIES:
                rows.append((ticker, name, period))
    return rows


def run(commit: bool = False):
    rows = plan()
    print(f"Backfill plan: {len(rows)} trials "
          f"({len(BACKFILLED_TICKERS)} tickers x {len(BACKFILLED_PERIODS)} periods "
          f"x {len(LEGACY_STRATEGIES)} strategies)\n")

    before = {t: ledger.n_trials(family=ledger.strategy_family(t)) for t in BACKFILLED_TICKERS}
    if not commit:
        print("DRY RUN — nothing written. Re-run with --commit to record.\n")
        for t in BACKFILLED_TICKERS:
            print(f"  {t:6s} registry now: {before[t]:3d}  ->  would become: "
                  f"{len(BACKFILLED_PERIODS) * len(LEGACY_STRATEGIES):3d}")
        return

    n = 0
    for ticker, name, period in rows:
        ledger.record_trial(
            hypothesis=f"{name} beats buy-and-hold on {ticker}",
            params={"strategy": name, "period": period},
            outcome="inconclusive",
            family=ledger.strategy_family(ticker),
            strategy=name,
            universe=ticker,
            sharpe=None,          # unknown, and NOT invented
            flags={"survivorship_safe": False, "pit_fundamentals": False,
                   "cost_model": "flat_10bps", "backfilled": True},
            notes="backfilled (reconstructed from development history, not re-run)",
        )
        n += 1

    print(f"Recorded {n} trials (idempotent — re-running changes nothing).\n")
    for t in BACKFILLED_TICKERS:
        after = ledger.n_trials(family=ledger.strategy_family(t))
        print(f"  {t:6s} n_trials {before[t]:3d} -> {after:3d}")
    print(f"\nTotal registry: {ledger.n_trials()} trials")
    print(f"Stats: {ledger.trial_stats()}")


if __name__ == "__main__":
    run(commit="--commit" in sys.argv)
