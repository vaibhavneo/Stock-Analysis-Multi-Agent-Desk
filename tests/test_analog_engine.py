"""
Verification for intelligence/analog_engine.py (item 5: historical analog
engine) - the highest-risk new module this session, since a subtle bug here
means quietly leaking future information into what's presented as historical
evidence. Extra scrutiny relative to the other new modules, per the plan.

Run: python3 tests/test_analog_engine.py

Offline/deterministic, no network. Two no-look-ahead tests reuse the exact
tricks already proven in this codebase:
  - Test A mirrors tests/test_replay.py::test_look_ahead - multiply every
    price strictly after as_of, confirm the result is byte-identical.
  - Test B is the negative control that same test uses - change as_of to an
    earlier date and confirm the result DOES change, proving test A isn't
    vacuously passing (e.g. because of a bug that ignores as_of entirely).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from intelligence.analog_engine import find_historical_analogs, MIN_HISTORY_BARS

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _random_block(n, drift, vol, start_price, seed):
    rng = np.random.default_rng(seed)
    return start_price * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def _shaped_block(log_returns, start_price):
    """Applies a FIXED, pre-generated log-return shape on top of a starting
    price - reused verbatim for every occurrence of the engineered pattern,
    so all three declines (and both recoveries) are the SAME shape in
    percentage terms, not three independent random draws from the same
    distribution. With only 63 bars per segment, independent random draws
    have enough path-to-path variance (confirmed by an earlier failed
    version of this fixture) to make "the same pattern repeated" not
    actually look similar under RSI/momentum - reusing one fixed shape
    removes that noise entirely, since technical/momentum readings are
    scale-invariant (same % path -> same RSI/momentum regardless of the
    absolute starting price)."""
    return start_price * np.exp(np.cumsum(log_returns))


# Fixed once, reused for every occurrence - see _shaped_block's docstring.
_DECLINE_SHAPE = np.random.default_rng(42).normal(-0.0035, 0.007, 63)   # ~-18.6% over 63 bars
_RECOVERY_SHAPE = np.random.default_rng(43).normal(0.0026, 0.006, 63)   # ~+10.9% over 63 bars


def make_recall_fixture():
    """~7 years of daily bars: long calm 'filler' stretches (near-zero drift,
    low vol - the algorithm should find these UNLIKE today's setup) punctuated
    by three occurrences of the SAME engineered pattern (the fixed
    _DECLINE_SHAPE) immediately followed, for the first two occurrences only,
    by the fixed _RECOVERY_SHAPE - a KNOWN, consistent ~+11% outcome over the
    next 63 bars, which is what the recall test checks for. The third
    occurrence's decline is the LAST thing in the series - that's `as_of`:
    today's setup, whose own future the engine must NOT be able to see.
    Decline length matches the 63-day momentum lookback exactly, so the
    momentum reading at each decline's last bar is driven entirely by that
    decline, not diluted by the preceding filler.

    Returns (df, filler+pattern boundary metadata) for building `as_of` and
    verifying which dates SHOULD be found as matches.
    """
    price = 100.0
    seed = 100
    segments = []
    pattern_end_dates_idx = []   # bar index of each decline's LAST bar (a "setup" date)

    def add_filler(n):
        nonlocal price, seed
        seed += 1
        block = _random_block(n, 0.0001, 0.004, price, seed)
        segments.append(block)
        price = float(block[-1])

    def add_shaped(log_returns):
        nonlocal price
        block = _shaped_block(log_returns, price)
        segments.append(block)
        price = float(block[-1])

    FILLER_N = 500

    add_filler(FILLER_N)
    add_shaped(_DECLINE_SHAPE)
    pattern_end_dates_idx.append(sum(len(s) for s in segments) - 1)
    add_shaped(_RECOVERY_SHAPE)

    add_filler(FILLER_N)
    add_shaped(_DECLINE_SHAPE)
    pattern_end_dates_idx.append(sum(len(s) for s in segments) - 1)
    add_shaped(_RECOVERY_SHAPE)

    add_filler(FILLER_N)
    add_shaped(_DECLINE_SHAPE)
    today_idx = sum(len(s) for s in segments) - 1   # as_of lands exactly here - no recovery yet

    close = np.concatenate(segments)
    n = len(close)
    idx = pd.bdate_range("2018-01-02", periods=n)
    close = pd.Series(close, index=idx)
    df = pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.006, "Low": close * 0.994, "Close": close,
        "Volume": pd.Series(np.random.default_rng(1).uniform(1e6, 2e6, n), index=idx),
    })
    return df, pattern_end_dates_idx, today_idx


def test_recall_finds_the_engineered_matches_with_the_known_outcome():
    df, pattern_dates_idx, today_idx = make_recall_fixture()
    as_of = str(df.index[today_idx].date())
    known_setup_dates = {str(df.index[i].date()) for i in pattern_dates_idx}

    # k=10, well above the 2 true prior occurrences: confirms the engine
    # ranks BOTH known setups ahead of the ~1500 unrelated filler candidates
    # (top-2 by distance, not just "found somewhere in a big list").
    r_wide = find_historical_analogs("TEST", df, as_of=as_of, k=10, min_separation_days=63)
    check("status is ok with ~7 years of history", r_wide["status"] == "ok", str(r_wide.get("flags")))
    check("at least one match found", len(r_wide["matches"]) >= 1, str(r_wide))

    if r_wide["status"] == "ok" and r_wide["matches"]:
        matched_dates = [m["date"] for m in r_wide["matches"]]
        top2 = set(matched_dates[:2])
        check("both earlier engineered setups rank as the two CLOSEST matches out of ~1500 candidates",
              top2 == known_setup_dates, f"top2={top2} known={known_setup_dates} all={matched_dates}")

    # k=2, matching the true occurrence count exactly: isolates the known-
    # outcome check from dilution by irrelevant lower-ranked filler matches.
    r_tight = find_historical_analogs("TEST", df, as_of=as_of, k=2, min_separation_days=63)
    matched_dates_tight = {m["date"] for m in r_tight["matches"]}
    check("k=2 returns exactly the two known engineered setups",
          matched_dates_tight == known_setup_dates, f"got={matched_dates_tight} known={known_setup_dates}")

    outcome_63 = r_tight["outcome_by_horizon"].get(63)
    check("63-day outcome is available", outcome_63 is not None)
    if outcome_63:
        check("aggregate 63-day return matches the known engineered recovery (~+10.9%) almost exactly",
              9.0 < outcome_63["avg_return_pct"] < 13.0, f"avg_return_pct={outcome_63['avg_return_pct']}")
        check("both matches show a positive 63-day outcome",
              outcome_63["pct_positive"] == 1.0, str(outcome_63))


def test_insufficient_history_is_honest():
    n = MIN_HISTORY_BARS - 100
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(5)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n))), index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                        "Volume": pd.Series(1e6, index=idx)})
    r = find_historical_analogs("TEST", df)
    check("status is insufficient_history below the minimum bar count", r["status"] == "insufficient_history")
    check("confidence is exactly 0.0", r["confidence"] == 0.0)
    check("matches list is empty, nothing fabricated", r["matches"] == [])
    check("outcome_by_horizon is empty", r["outcome_by_horizon"] == {})


def test_empty_history_is_honest():
    r = find_historical_analogs("TEST", pd.DataFrame())
    check("empty df -> insufficient_history, not an exception", r["status"] == "insufficient_history")
    check("flagged no_price_history", "no_price_history" in r["flags"])


def _big_fixture(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": pd.Series(rng.uniform(1e6, 2e6, n), index=idx),
    })


def test_look_ahead_A_future_prices_never_change_the_result():
    df = _big_fixture()
    as_of = str(df.index[1500].date())

    r1 = find_historical_analogs("TEST", df, as_of=as_of, k=10)

    df_spiked = df.copy()
    future_mask = df_spiked.index > pd.Timestamp(as_of)
    for col in ("Open", "High", "Low", "Close"):
        df_spiked.loc[future_mask, col] = df_spiked.loc[future_mask, col] * 5.0

    r2 = find_historical_analogs("TEST", df_spiked, as_of=as_of, k=10)

    check("result is byte-identical when only FUTURE (post-as_of) prices are altered",
          r1 == r2, "output changed despite no pre-as_of data being touched - LOOK-AHEAD LEAK")


def test_look_ahead_B_negative_control_earlier_as_of_does_change_result():
    df = _big_fixture()
    as_of_late = str(df.index[1500].date())
    as_of_early = str(df.index[900].date())

    r_late = find_historical_analogs("TEST", df, as_of=as_of_late, k=10)
    r_early = find_historical_analogs("TEST", df, as_of=as_of_early, k=10)

    check("a genuinely different as_of DOES change the result (proves test A isn't vacuous)",
          r_late != r_early, "identical results for two different as_of dates - suspicious")


def test_look_ahead_C_no_recent_bars_and_no_self_match():
    df = _big_fixture()
    as_of_pos = 1500
    as_of = str(df.index[as_of_pos].date())
    horizons = (5, 21, 63, 126, 252)
    max_h = max(horizons)

    r = find_historical_analogs("TEST", df, as_of=as_of, k=15, horizons=horizons)
    check("status ok for this fixture", r["status"] == "ok", str(r.get("flags")))

    if r["status"] == "ok":
        as_of_ts = pd.Timestamp(as_of)
        earliest_allowed_idx = as_of_pos - max_h   # candidates must resolve their longest-horizon outcome by as_of
        for m in r["matches"]:
            match_ts = pd.Timestamp(m["date"])
            check(f"match {m['date']} is not the trivial self-match (as_of itself)",
                  match_ts != as_of_ts)
            match_pos = df.index.get_loc(match_ts)
            check(f"match {m['date']} is far enough before as_of for its {max_h}d outcome to be knowable",
                  match_pos <= earliest_allowed_idx,
                  f"match_pos={match_pos} earliest_allowed={earliest_allowed_idx}")


if __name__ == "__main__":
    test_recall_finds_the_engineered_matches_with_the_known_outcome()
    test_insufficient_history_is_honest()
    test_empty_history_is_honest()
    test_look_ahead_A_future_prices_never_change_the_result()
    test_look_ahead_B_negative_control_earlier_as_of_does_change_result()
    test_look_ahead_C_no_recent_bars_and_no_self_match()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — analog engine: recalls real matches, provably no look-ahead")
