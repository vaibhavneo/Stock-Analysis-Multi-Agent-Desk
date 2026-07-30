"""
Verification for M2a — dSR reads the TRUE trial count.

Run: python3 tests/test_trial_honesty.py

The bug this closes, concretely: dSR deflates an observed Sharpe against the
expected maximum Sharpe of `n_trials` attempts. The count used to come from the
current process, so:

    run_backtest.py AAPL            -> 7 strategies -> n_trials=7
    run_backtest.py AAPL momentum   -> 1 strategy   -> n_trials=1 -> NO deflation

Same data, same strategy, better score — purely from re-running the winner
alone. At the Sharpe range real strategies live in (0.8-1.2) that flips the
verdict from "fails the 0.5 bar" to "certain it is real". The TrialRegistry
closes it by remembering across runs.

Groups:
  1. Engine guard — n_trials < 1 raises instead of returning a flattering 1.0
  2. effective_n_trials — the floor logic, monotone
  3. THE LOOPHOLE — re-running a winner alone can no longer inflate its dSR
  4. Producer — the registry is populated, dead trials included
  5. Backfill — reconstructs the count without inventing Sharpes
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES = []

# Throwaway DB: test fixtures must never enter the real registry, because
# n_trials is dSR's denominator (same reasoning as test_financial_data.py).
_TMPDIR = tempfile.TemporaryDirectory()
TEST_DB = Path(_TMPDIR.name) / "trials_test.db"


def check(name: str, condition: bool, detail: str = ""):
    print(f"  {name:58s} {'OK' if condition else 'FAIL'}  {detail}")
    if not condition:
        FAILURES.append(name)


def test_engine_guard():
    print("=== 1. Engine guard: n_trials < 1 must raise, not flatter ===")
    from backtest.engine import deflated_sharpe_ratio as dsr

    for bad in (0, -1):
        try:
            dsr(1.2, n_trials=bad, skewness=0.0, kurtosis=3.0, n_obs=756)
            check(f"n_trials={bad} rejected", False, "no raise — returns dSR=1.0 silently!")
        except ValueError:
            check(f"n_trials={bad} rejected", True)

    # n=1 is legitimate (a single pre-registered hypothesis has no selection
    # bias) and must still work.
    v = dsr(1.2, n_trials=1, skewness=0.0, kurtosis=3.0, n_obs=756)
    check("n_trials=1 still valid (no selection bias)", 0.0 <= v <= 1.0, f"dSR={v:.4f}")

    # BOUNDS: dSR is a probability.
    vals = [dsr(s, n_trials=n, skewness=0.0, kurtosis=3.0, n_obs=756)
            for s in (-2.0, 0.0, 1.2, 5.0) for n in (1, 7, 500)]
    check("dSR always within [0,1]", all(0.0 <= v <= 1.0 for v in vals),
          f"min={min(vals):.3f} max={max(vals):.3f}")

    # MONOTONE: more trials tried => strictly less confidence, never more.
    seq = [dsr(1.5, n_trials=n, skewness=0.0, kurtosis=3.0, n_obs=756)
           for n in (1, 5, 20, 100, 1000)]
    check("dSR is non-increasing as n_trials rises",
          all(a >= b - 1e-12 for a, b in zip(seq, seq[1:])),
          " -> ".join(f"{v:.3f}" for v in seq))


def test_effective_n_trials():
    print("=== 2. effective_n_trials: floors and monotonicity ===")
    from data import ledger
    ledger.set_db_path(TEST_DB)

    t = "TESTTICK"
    check("empty registry never yields 0 (dSR floor is 1)",
          ledger.effective_n_trials(t, tested_this_run=0) == 1)
    check("un-recorded run counts via tested_this_run",
          ledger.effective_n_trials(t, tested_this_run=7) == 7,
          "a caller that forgets to record still cannot under-count")

    for i in range(9):
        ledger.record_trial(f"strategy_{i} beats buy-and-hold on {t}",
                            {"strategy": f"s{i}", "period": "3y"}, "dead",
                            family=ledger.strategy_family(t), universe=t)
    check("registry count wins when it exceeds this run",
          ledger.effective_n_trials(t, tested_this_run=1) == 9,
          "re-running 1 strategy still deflates against all 9 ever tried")
    check("max() of the three terms", ledger.effective_n_trials(t, tested_this_run=20) == 20)

    # The property that matters: it can only ever rise.
    a = ledger.effective_n_trials(t)
    ledger.record_trial("another one", {"strategy": "s99", "period": "3y"}, "dead",
                        family=ledger.strategy_family(t), universe=t)
    check("monotone: count only rises", ledger.effective_n_trials(t) == a + 1)

    check("families are per-ticker (AAPL trials don't deflate MSFT)",
          ledger.effective_n_trials("OTHERTICK") == 1)


def test_loophole_closed():
    print("=== 3. THE LOOPHOLE: re-running a winner alone cannot inflate dSR ===")
    from backtest.engine import deflated_sharpe_ratio as dsr
    from data import ledger
    ledger.set_db_path(TEST_DB)

    t = "LOOPHOLE"
    fam = ledger.strategy_family(t)
    SHARPE = 1.2      # squarely in the range where the flip used to happen

    # Simulate the old exploit: run all 7, then re-run the winner by itself.
    for i in range(7):
        ledger.record_trial(f"s{i} beats buy-and-hold on {t}",
                            {"strategy": f"s{i}", "period": "3y"}, "dead",
                            family=fam, universe=t)

    n_all = ledger.effective_n_trials(t, tested_this_run=7)
    n_alone = ledger.effective_n_trials(t, tested_this_run=1)   # the re-run
    check("run-all-7 and re-run-alone now agree on the count",
          n_all == n_alone == 7, f"all={n_all} alone={n_alone}")

    dsr_all = dsr(SHARPE, n_trials=n_all, skewness=0.0, kurtosis=3.0, n_obs=756)
    dsr_alone = dsr(SHARPE, n_trials=n_alone, skewness=0.0, kurtosis=3.0, n_obs=756)
    check("identical dSR whether run together or alone",
          abs(dsr_all - dsr_alone) < 1e-12, f"{dsr_all:.4f} vs {dsr_alone:.4f}")

    # And prove the OLD behavior really was exploitable, so this test would have
    # caught it. Without a guard against regression, a future refactor could
    # quietly reintroduce a per-run count and nothing would complain.
    old_all = dsr(SHARPE, n_trials=7, skewness=0.0, kurtosis=3.0, n_obs=756)
    old_alone = dsr(SHARPE, n_trials=1, skewness=0.0, kurtosis=3.0, n_obs=756)
    check("(regression guard) the old per-run count WAS exploitable",
          old_alone > old_all and old_all < 0.5 <= old_alone,
          f"n=7 -> {old_all:.4f} (fails 0.5 bar) BUT n=1 -> {old_alone:.4f} (passes)")

    # Accumulated debt keeps deflating: try more variants, get less confident.
    for i in range(7, 60):
        ledger.record_trial(f"s{i} beats buy-and-hold on {t}",
                            {"strategy": f"s{i}", "period": "3y"}, "dead",
                            family=fam, universe=t)
    n_later = ledger.effective_n_trials(t, tested_this_run=1)
    dsr_later = dsr(1.9, n_trials=n_later, skewness=0.0, kurtosis=3.0, n_obs=756)
    dsr_early = dsr(1.9, n_trials=7, skewness=0.0, kurtosis=3.0, n_obs=756)
    check("more variants tried => strictly lower confidence for the same Sharpe",
          n_later == 60 and dsr_later < dsr_early,
          f"n=7 -> {dsr_early:.4f} | n=60 -> {dsr_later:.4f}")


def test_producer():
    print("=== 4. Producer: the registry actually gets populated ===")
    import numpy as np
    import pandas as pd
    from data import ledger
    ledger.set_db_path(TEST_DB)

    import backtest.run_backtest as rb
    from backtest.strategies import STRATEGY_REGISTRY

    t = "PRODUCER"
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=800, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 800))), index=idx)
    df = pd.DataFrame({"Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
                       "Close": prices, "Volume": 1e6}, index=idx)

    before = ledger.n_trials(family=ledger.strategy_family(t))
    n_ok = 0
    for name, fn in STRATEGY_REGISTRY.items():
        try:
            m, _n, _s, _k = rb._backtest_one(name, fn, df, prices)
            rb._record(t, name, m, "3y", outcome="dead")
            n_ok += 1
        except Exception:
            pass
    after = ledger.n_trials(family=ledger.strategy_family(t))
    check("running strategies registers trials", after > before, f"{before} -> {after}")
    check("every strategy that ran was recorded", after - before == n_ok,
          f"{n_ok} ran, {after-before} recorded")

    # The dead ones are the point — they are dSR's denominator.
    dead = ledger.list_trials(family=ledger.strategy_family(t), outcome="dead")
    check("losing strategies are RECORDED, not discarded", len(dead) > 0, f"{len(dead)} dead")

    # Re-running the identical sweep must not inflate the count.
    for name, fn in STRATEGY_REGISTRY.items():
        try:
            m, _n, _s, _k = rb._backtest_one(name, fn, df, prices)
            rb._record(t, name, m, "3y", outcome="dead")
        except Exception:
            pass
    check("re-running the same sweep does NOT inflate the count",
          ledger.n_trials(family=ledger.strategy_family(t)) == after,
          "dSR must not decay just because the CLI was run twice")

    # Different period = genuinely another shot at a flattering result.
    rb._record(t, "sma_crossover", {"sharpe_ratio": 0.5}, "5y", outcome="dead")
    check("a different period IS a distinct trial",
          ledger.n_trials(family=ledger.strategy_family(t)) == after + 1)


def test_backfill():
    print("=== 5. Backfill: reconstruct the count, invent nothing ===")
    from data import ledger
    ledger.set_db_path(TEST_DB)

    import backtest.backfill_trials as bf
    rows = bf.plan()
    check("plan covers tickers x periods x LEGACY strategies (pinned)",
          len(rows) == len(bf.BACKFILLED_TICKERS) * len(bf.BACKFILLED_PERIODS)
          * len(bf.LEGACY_STRATEGIES),
          f"{len(rows)} planned; list pinned so strategies added later are never backfilled")
    check("backfill list is pinned, NOT the live registry",
          "seven_pillar_core" not in bf.LEGACY_STRATEGIES,
          "backfilling a strategy added later = fabricating trials that never ran")

    t = bf.BACKFILLED_TICKERS[0]
    before = ledger.n_trials(family=ledger.strategy_family(t))
    bf.run(commit=True)
    after = ledger.n_trials(family=ledger.strategy_family(t))
    check("backfill raises the count", after > before, f"{before} -> {after}")

    trials = ledger.list_trials(family=ledger.strategy_family(t))
    backfilled = [x for x in trials if x["notes"] and "backfilled" in x["notes"]]
    check("backfilled trials are LABELLED as reconstructed", len(backfilled) > 0)
    check("backfill invents no Sharpe values (unknown stays NULL)",
          all(x["sharpe"] is None for x in backfilled),
          "a fabricated Sharpe would be exactly the evidence this layer forbids")
    check("backfill is idempotent", (bf.run(commit=True) or
          ledger.n_trials(family=ledger.strategy_family(t))) == after)


if __name__ == "__main__":
    test_engine_guard()
    test_effective_n_trials()
    test_loophole_closed()
    test_producer()
    test_backfill()

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — dSR reads the true trial count; shopping loophole closed")
