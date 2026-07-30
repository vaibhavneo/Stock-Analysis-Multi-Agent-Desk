"""
Verification for the seven-pillar composite (backtest/pillars.py).

Run: python3 tests/test_pillars.py

All offline/deterministic (synthetic OHLCV). The properties tested are the ones
that make the composite honest rather than merely plausible:

  1. Bounds & directionality — scores live in [0,100]; an uptrend scores high,
     a downtrend low, on the backtestable pillars.
  2. Last-bar consistency — the vectorized series equal the LIVE meters on the
     final bar (same `ta` calls, same rounding), so the backtest tests the same
     thing the dashboard shows.
  3. Risk is a brake, not an engine — veto caps the composite into [35,65];
     the multiplier can only shrink distance from neutral.
  4. Modifiers are bounded — social+research can tilt at most ±10 points.
  5. Hysteresis + weekly cadence strictly reduce trading vs a naive threshold.
  6. No look-ahead — perturbing the final bar never changes earlier signals.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest.pillars import (
    ENTER_THRESHOLD, EXIT_THRESHOLD, MODIFIER_MAX_PTS,
    action_for, algo_score_series, compute_pillar_scores,
    risk_mult_series, seven_pillar_core_strategy, technical_score_series,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def make_df(kind: str, n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV: 'up' (steady climb), 'down' (steady fall), 'flat'.

    Drift/vol chosen so the trend DOMINATES local noise (20-day drift ~4% vs
    sigma ~2.7%): the directionality tests ask "does a clear trend score as
    one?", and a fixture whose tail is noise would test the question badly —
    the first version of this file did exactly that (drift .0012, vol .01) and
    its 'uptrend' genuinely ended in a pullback that the live meter, correctly,
    scored bearish. Consistency (series == live) held throughout; the fixture
    was the liar, not the code.
    """
    rng = np.random.default_rng(seed)
    drift = {"up": 0.002, "down": -0.002, "flat": 0.0}[kind]
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n))), index=idx)
    return pd.DataFrame({
        "Open": close.shift(1).fillna(close.iloc[0]),
        "High": close * 1.006, "Low": close * 0.994, "Close": close,
        "Volume": pd.Series(rng.uniform(1e6, 2e6, n), index=idx),
    })


def live_meters(df):
    from tools.market_data import (compute_algo_signals, compute_indicators,
                                   compute_signal_summary)
    ind = compute_indicators(df)
    return ind, compute_signal_summary(ind), compute_algo_signals(df, ind)


def test_bounds_and_direction():
    print("=== 1. Bounds + directionality ===")
    up, down = make_df("up"), make_df("down")
    for name, series_fn in (("technical", technical_score_series), ("algo", algo_score_series)):
        su, sd = series_fn(up), series_fn(down)
        check(f"{name} series within [0,100]",
              bool(su.between(0, 100).all() and sd.between(0, 100).all()))
        check(f"{name}: uptrend tail > 60", su.tail(60).mean() > 60,
              f"mean={su.tail(60).mean():.1f}")
        check(f"{name}: downtrend tail < 40", sd.tail(60).mean() < 40,
              f"mean={sd.tail(60).mean():.1f}")
    rm = risk_mult_series(up["Close"])
    check("risk multiplier within [0.5, 1.0]",
          bool(rm.between(0.5, 1.0).all()), f"[{rm.min():.2f},{rm.max():.2f}]")


def test_last_bar_consistency():
    print("=== 2. Last-bar consistency: series == live meters ===")
    for kind in ("up", "down", "flat"):
        df = make_df(kind)
        ind, ss, algo = live_meters(df)
        t_last, a_last = technical_score_series(df).iloc[-1], algo_score_series(df).iloc[-1]
        check(f"technical series last == live score ({kind})",
              t_last == ss["score"], f"series={t_last} live={ss['score']}")
        check(f"algo series last == live score ({kind})",
              a_last == algo["algo_score"], f"series={a_last} live={algo['algo_score']}")


def _snap(df, fundamentals=None, **kw):
    ind, ss, algo = live_meters(df)
    return compute_pillar_scores("TEST", ind, ss, algo, fundamentals or {}, **kw)


def test_risk_is_a_brake():
    print("=== 3. Risk: multiplier + veto can only shrink conviction ===")
    df = make_df("up")
    ind, ss, algo = live_meters(df)

    # Veto: force HIGH+expanding vol; composite must land inside [35,65] even
    # though every other pillar screams BUY.
    algo_veto = dict(algo)
    algo_veto["vol_regime"], algo_veto["vol_expanding"] = "HIGH", True
    algo_veto["historical_volatility_20d"] = 55.0
    bulls = {"trailingPE": 12, "profitMargins": 0.3, "returnOnEquity": 0.4,
             "revenueGrowth": 0.3, "recommendationMean": 1.2,
             "targetMeanPrice": ind["current_price"] * 1.4, "numberOfAnalystOpinions": 40}
    r = compute_pillar_scores("TEST", ind, ss, algo_veto, bulls)
    check("veto flag set", r["risk_veto"] is True)
    check("veto caps composite into [35,65]", 35.0 <= r["composite"] <= 65.0,
          f"composite={r['composite']}")
    check("veto flagged on the risk pillar",
          "risk_veto_active" in r["pillars"]["risk"]["flags"])

    # Multiplier: same signals, higher vol -> composite strictly closer to 50.
    algo_hi = dict(algo)
    algo_hi["historical_volatility_20d"] = 55.0
    calm = compute_pillar_scores("TEST", ind, ss, algo, bulls)
    stressed = compute_pillar_scores("TEST", ind, ss, algo_hi, bulls)
    check("higher vol pulls composite toward 50 (never away)",
          abs(stressed["composite"] - 50) <= abs(calm["composite"] - 50) + 1e-9,
          f"calm={calm['composite']} stressed={stressed['composite']}")


def test_modifiers_bounded():
    print("=== 4. Modifiers: social + research tilt is bounded ===")
    df = make_df("flat")
    euphoric = {"recommendationMean": 1.0, "targetMeanPrice": 999.0,
                "numberOfAnalystOpinions": 50}
    reddit = {"sentiment_score": 100, "mention_count": 100}
    st = {"sentiment_ratio": 100.0, "total": 500, "bullish": 500, "bearish": 0}
    base = _snap(df)
    hyped = _snap(df, fundamentals=euphoric, reddit=reddit, stocktwits=st)
    check("modifier points never exceed +/- 2*MODIFIER_MAX_PTS",
          abs(hyped["modifier_pts"]) <= 2 * MODIFIER_MAX_PTS + 1e-9,
          f"mod={hyped['modifier_pts']} (cap {2*MODIFIER_MAX_PTS})")
    check("maximum hype moves composite by a bounded amount",
          abs(hyped["composite"] - base["composite"]) <= 2 * MODIFIER_MAX_PTS + 1e-6,
          f"{base['composite']} -> {hyped['composite']}")
    check("non-backtestable pillars are flagged",
          "not_backtestable" in hyped["pillars"]["social"]["flags"]
          and "not_backtestable" in hyped["pillars"]["research"]["flags"])

    # Tiny sample: 3 furious reddit posts must NOT move the needle.
    small = _snap(df, reddit={"sentiment_score": 100, "mention_count": 3})
    check("reddit sample <5 mentions is neutralized + flagged",
          small["pillars"]["social"]["score"] == 50.0
          and "reddit_sample_too_small" in small["pillars"]["social"]["flags"])


def test_hysteresis_and_cadence():
    print("=== 5. Hysteresis + weekly cadence reduce churn ===")
    df = make_df("flat", n=600, seed=11)   # choppy: worst case for churn

    sig = seven_pillar_core_strategy(df)
    check("strategy output is {0,1} long-only", set(sig.unique()) <= {0.0, 1.0},
          str(sorted(sig.unique())))

    # Naive comparator: daily decisions, single threshold (enter==exit) —
    # what the strategy would be WITHOUT its two churn brakes.
    from backtest.pillars import risk_mult_series as rms
    core = 0.5 * technical_score_series(df) + 0.5 * algo_score_series(df)
    score = 50.0 + (core - 50.0) * rms(df["Close"])
    naive = (score >= ENTER_THRESHOLD).astype(float)
    n_naive = int((naive.diff().abs() > 0).sum())
    n_strat = int((sig.diff().abs() > 0).sum())
    check("weekly+hysteresis trades <= naive daily threshold trades",
          n_strat <= n_naive, f"strategy={n_strat} naive={n_naive}")
    check("thresholds actually differ (hysteresis exists)",
          ENTER_THRESHOLD > EXIT_THRESHOLD)


def test_no_lookahead():
    print("=== 6. No look-ahead: the future cannot edit the past ===")
    df = make_df("up", n=400)
    base = seven_pillar_core_strategy(df)
    # Nuke the final bar with a catastrophic value; every EARLIER signal must
    # be byte-identical. (The engine's own .shift(1) additionally delays the
    # final bar's decision beyond the sample — tested in the engine suite.)
    df2 = df.copy()
    df2.iloc[-1, df2.columns.get_loc("Close")] = df2["Close"].iloc[-1] * 0.5
    df2.iloc[-1, df2.columns.get_loc("Low")] = df2["Low"].iloc[-1] * 0.5
    pert = seven_pillar_core_strategy(df2)
    check("perturbing the last bar never changes earlier signals",
          bool((base.iloc[:-1] == pert.iloc[:-1]).all()))
    for fn in (technical_score_series, algo_score_series):
        a, b = fn(df), fn(df2)
        check(f"{fn.__name__}: earlier bars unchanged",
              bool((a.iloc[:-1] == b.iloc[:-1]).all()))


def test_actions_and_registry():
    print("=== 7. Action bands + registry integration ===")
    for score, verb in ((80, "BUY"), (65, "ACCUMULATE"), (50, "HOLD"),
                        (40, "REDUCE"), (20, "SELL")):
        check(f"action_for({score}) == {verb}", action_for(score) == verb,
              action_for(score))

    from backtest.strategies import STRATEGIES_NEEDING_FULL_DF, STRATEGY_REGISTRY
    check("seven_pillar_core registered", "seven_pillar_core" in STRATEGY_REGISTRY)
    check("marked as needing full df", "seven_pillar_core" in STRATEGIES_NEEDING_FULL_DF)

    # It must run through the actual engine like any other strategy.
    from backtest.engine import compute_performance_metrics, run_vectorized_backtest
    df = make_df("up")
    res = run_vectorized_backtest(df["Close"], STRATEGY_REGISTRY["seven_pillar_core"](df))
    m = compute_performance_metrics(res.strategy_returns)
    check("runs through the engine end-to-end", "sharpe_ratio" in m,
          f"sharpe={m['sharpe_ratio']} trades={res.n_trades}")

    # ATR refactor: one shared ATR, synthesis delegates to it.
    from agents.synthesis import _compute_atr
    from backtest.risk import compute_atr
    check("shared ATR: synthesis == risk.compute_atr",
          abs(_compute_atr(df) - compute_atr(df)) < 1e-12)


if __name__ == "__main__":
    test_bounds_and_direction()
    test_last_bar_consistency()
    test_risk_is_a_brake()
    test_modifiers_bounded()
    test_hysteresis_and_cadence()
    test_no_lookahead()
    test_actions_and_registry()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — seven-pillar composite: consistent, bounded, leak-free")
