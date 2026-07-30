"""
Seven-Pillar Composite Investing Strategy — the deterministic layer.

Each of the Stock Agent's analysis sub-agents narrates one domain. This module
turns those six domains into DETERMINISTIC 0-100 pillar scores computed from the
same machine data each agent reads — the LLM prose stays as the explainer, the
formulas here are the only number source (charter P4). The 7th agent
(prediction) is deliberately NOT a pillar: it is the COMBINER. Feeding its LLM
scores back in would double-count the other six domains through a stochastic
blender; anyone tempted to "add the 7th pillar" later should reread this line.

Honesty architecture (the part that matters):
  BACKTESTABLE CORE   technical + algo + risk — fully derivable from price
                      history, so their combination can be tested through the
                      honest stack (CostModel, purged walk-forward, PBO,
                      ledger-denominated dSR). Registered in STRATEGY_REGISTRY.
  TRACKED-FORWARD     fundamentals (restated snapshot; EDGAR PIT upgrades the
  MODIFIERS           live pillar but PIT is not in the backtest path yet),
                      research (analyst consensus), social (reddit/stocktwits).
                      Free data has NO history for these — a "backtest" of them
                      would be fiction. They get small BOUNDED weight (±5 pts
                      each), explicit flags, and forward tracking via the
                      recommendations/outcomes ledger instead.

Weights are FIXED, TRANSPARENT v1 PRIORS — not fitted. 80% of the core goes to
the two fully-backtested meters, 20% to the partially-evidence-backed one, and
the never-backtested pillars can only tilt, never drive. Fitting weights is
allowed ONLY through walk_forward_cv with one ledger.record_trial per weight
vector tried; anything else is weight-shopping, the exact overfitting the
TrialRegistry exists to count.

Stance is LONG-ONLY (user decision): the strategy is invested or in cash.
Verbs: BUY >=70 · ACCUMULATE 60-70 · HOLD 45-60 · REDUCE 35-45 · SELL <35,
with hysteresis (enter >=65, exit <55) and a weekly decision cadence so the
composite doesn't churn away its edge against the realistic cost model.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ── Fixed v1 weights (see module docstring before touching these) ──────────
CORE_WEIGHTS = {"technical": 0.40, "algo": 0.40, "fundamentals": 0.20}
MODIFIER_MAX_PTS = 5.0          # per modifier pillar, max tilt in points
ENTER_THRESHOLD = 65.0          # hysteresis: go long at/above
EXIT_THRESHOLD = 55.0           # hysteresis: back to cash below
VETO_BAND = (35.0, 65.0)        # composite clamp when the risk veto fires

ACTION_BANDS = [                # (min_score_inclusive, verb) — investing verbs
    (70.0, "BUY"), (60.0, "ACCUMULATE"), (45.0, "HOLD"), (35.0, "REDUCE"), (-1.0, "SELL"),
]


def action_for(score: float) -> str:
    for lo, verb in ACTION_BANDS:
        if score >= lo:
            return verb
    return "HOLD"


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(min(hi, max(lo, v)))


# ══════════════════════════════════════════════════════════════════════════
# Snapshot pillar scores (live path — pure arithmetic, keyless)
# ══════════════════════════════════════════════════════════════════════════

def _pillar(score: Optional[float], confidence: float, backtestable: bool,
            formula: str, inputs: Dict[str, Any], flags: Optional[list] = None) -> Dict[str, Any]:
    return {
        "score": round(_clip(score), 1) if score is not None else 50.0,
        "confidence": round(_clip(confidence, 0.0, 1.0), 2),
        "backtestable": backtestable,
        "formula": formula,
        "inputs": inputs,
        "flags": flags or [],
    }


def _risk_score(hv20: Optional[float], beta: Optional[float],
                price: Optional[float], hi52: Optional[float], lo52: Optional[float],
                short_ratio: Optional[float]) -> Dict[str, Any]:
    """Inverse-risk 0-100: 100 = calm, 0 = dangerous. Sub-scores averaged over
    what's available; missing inputs reported, never guessed."""
    subs: Dict[str, float] = {}
    if hv20 is not None:
        # 15% annualized vol -> 100 … 60% -> 0, linear
        subs["volatility"] = _clip((60.0 - float(hv20)) / (60.0 - 15.0) * 100.0)
    if beta is not None:
        # |beta-1| of 0 -> 100 … 1.5 -> 0
        subs["beta"] = _clip((1.5 - abs(float(beta) - 1.0)) / 1.5 * 100.0)
    if price and hi52 and lo52 and hi52 > lo52:
        # Mid-range is calm; the extremes are where blowups and euphoria live.
        pos = (float(price) - float(lo52)) / (float(hi52) - float(lo52))
        subs["range_position"] = _clip(100.0 - abs(pos - 0.5) * 2.0 * 60.0)  # 0.5->100, edges->40
    if short_ratio is not None and float(short_ratio) > 8.0:
        subs["short_interest_penalty"] = 20.0     # crowded short = squeeze/thesis risk
    if not subs:
        return {"score": None, "subs": subs}
    return {"score": sum(subs.values()) / len(subs), "subs": subs}


def compute_pillar_scores(
    ticker: str,
    indicators: Dict[str, Any],
    signal_summary: Dict[str, Any],
    algo_signals: Dict[str, Any],
    fundamentals: Dict[str, Any],
    reddit: Optional[Dict[str, Any]] = None,
    stocktwits: Optional[Dict[str, Any]] = None,
    pit: Optional[Dict[str, Any]] = None,
    strict_fundamentals: bool = False,
) -> Dict[str, Any]:
    """The six pillar scores + composite for RIGHT NOW (live snapshot).

    Inputs are the exact dicts the LLM agents receive (market_data outputs);
    `pit` is an optional analyze_fundamentals_pit() result — when present and
    available, the margin/ROE/D-E sub-scores use EDGAR as-reported values and
    the caller's ledger claims can cite SEC-accession datums.

    `strict_fundamentals` (the RECOMMENDATION path sets it True): the
    fundamentals pillar is sourced ONLY through the FinancialDataGateway (EDGAR
    PIT) — the legacy yfinance accounting fields (profitMargins/returnOnEquity/
    totalDebt/trailingPE/…) are NOT used. When EDGAR is unavailable (non-filer),
    the pillar is neutral+flagged and the caller's data-confidence drops, rather
    than silently substituting restated yfinance numbers. This is the guarantee
    that a recommendation's fundamentals trace to SEC filings, not Yahoo.
    """
    fundamentals = fundamentals or {}
    pillars: Dict[str, Dict[str, Any]] = {}

    # 1. TECHNICAL — the existing 0-100 voting meter, verbatim. Its vote logic
    # is mirrored bar-by-bar in technical_score_series() below (backtestable).
    pillars["technical"] = _pillar(
        signal_summary.get("score"), 1.0, True,
        "signal_summary.score (RSI/MACD/SMA/BB voting, tools/market_data.py)",
        {"score": signal_summary.get("score"), "direction": signal_summary.get("direction")},
        [] if signal_summary.get("score") is not None else ["no_technical_data"])

    # 2. ALGO — the existing quant voting meter, verbatim (mirrored vectorized).
    pillars["algo"] = _pillar(
        algo_signals.get("algo_score"), 1.0, True,
        "algo_signals.algo_score (z-score/momentum/linreg/volume-price voting)",
        {"algo_score": algo_signals.get("algo_score"),
         "momentum_composite": algo_signals.get("momentum_composite"),
         "mean_reversion_zscore": algo_signals.get("mean_reversion_zscore")},
        [] if algo_signals.get("algo_score") is not None else ["no_algo_data"])

    # 3. RISK — inverse-risk score; consumed as a MULTIPLIER + veto, never as
    # additive points: risk can shrink conviction, it can never buy the stock.
    rs = _risk_score(
        algo_signals.get("historical_volatility_20d"), fundamentals.get("beta"),
        indicators.get("current_price"), fundamentals.get("fiftyTwoWeekHigh"),
        fundamentals.get("fiftyTwoWeekLow"), fundamentals.get("shortRatio"))
    veto = bool(algo_signals.get("vol_regime") == "HIGH" and algo_signals.get("vol_expanding"))
    pillars["risk"] = _pillar(
        rs["score"], 1.0 if rs["score"] is not None else 0.0, True,
        "mean(vol 15-60%% inverse, |beta-1| inverse, 52w-range position, shortRatio>8 penalty); "
        "multiplier = 0.5 + 0.5*score/100; veto when vol HIGH and expanding",
        {"subs": {k: round(v, 1) for k, v in rs["subs"].items()},
         "vol_regime": algo_signals.get("vol_regime"),
         "vol_expanding": algo_signals.get("vol_expanding")},
        (["risk_veto_active"] if veto else []) + ([] if rs["score"] is not None else ["no_risk_data"]))

    # 4. FUNDAMENTALS — quality from ratio bands (stated priors, not fitted).
    subs: Dict[str, float] = {}
    flags: list = []
    pit_ratios = (pit or {}).get("ratios") if (pit or {}).get("available") else None
    fundamentals_source = "unavailable"

    if pit_ratios:
        # EDGAR PIT, as-reported: the only source used in strict mode, and the
        # preferred source otherwise. Claims cite SEC accession numbers.
        fundamentals_source = "sec-edgar"
        flags.append("pit_fundamentals_used")
        if "net_margin" in pit_ratios:
            subs["margin"] = _clip(float(pit_ratios["net_margin"]["value"]) * 400.0)   # 25% -> 100
        if "return_on_equity" in pit_ratios:
            subs["roe"] = _clip(float(pit_ratios["return_on_equity"]["value"]) * 250.0)  # 40% -> 100
        if "debt_to_equity" in pit_ratios:
            subs["de"] = _clip(100.0 - float(pit_ratios["debt_to_equity"]["value"]) * 33.3)  # 0->100, 3->0
        if "operating_margin" in pit_ratios:
            subs["op_margin"] = _clip(float(pit_ratios["operating_margin"]["value"]) * 300.0)

    if not strict_fundamentals:
        # Non-strict (general dashboard use): value ratios + a yfinance quality
        # fallback are allowed. The recommendation path does NOT take this branch
        # — its fundamentals are SEC-only (goal: not legacy yfinance fundamentals).
        pe = fundamentals.get("trailingPE")
        if pe is not None and pe > 0:
            subs["pe"] = _clip(100.0 - (float(pe) - 10.0) * (50.0 / 25.0))  # 10->100, 35->50
        pb = fundamentals.get("priceToBook")
        if pb is not None and pb > 0:
            subs["pb"] = _clip(100.0 - (float(pb) - 1.0) * (50.0 / 4.0))    # 1->100, 5->50
        fcf, mcap = fundamentals.get("freeCashflow"), fundamentals.get("marketCap")
        if fcf is not None and mcap:
            subs["fcf_yield"] = _clip(float(fcf) / float(mcap) * 100.0 * 10.0 + 30.0)
        if not pit_ratios:
            flags.append("restated_snapshot_fundamentals")
            pm = fundamentals.get("profitMargins")
            if pm is not None:
                subs["margin"] = _clip(float(pm) * 400.0)
            roe = fundamentals.get("returnOnEquity")
            if roe is not None:
                subs["roe"] = _clip(float(roe) * 250.0)
            debt = fundamentals.get("totalDebt")
            if debt is not None and mcap:
                subs["de"] = _clip(100.0 - (float(debt) - float(fundamentals.get("totalCash") or 0))
                                   / float(mcap) * 100.0)
        rg = fundamentals.get("revenueGrowth")
        if rg is not None:
            subs["growth"] = _clip(50.0 + float(rg) * 250.0)
    elif not pit_ratios:
        # STRICT + no EDGAR (non-filer): neutral + flagged, never yfinance.
        flags.append("fundamentals_no_sec_source")

    fund_score = sum(subs.values()) / len(subs) if subs else None
    pillars["fundamentals"] = _pillar(
        fund_score, min(1.0, len(subs) / (4.0 if strict_fundamentals else 5.0)),
        bool(pit_ratios),      # backtestable only when EDGAR-sourced (PIT)
        ("mean(EDGAR PIT margin/roe/de/op_margin) — SEC-sourced, stated priors"
         if strict_fundamentals else
         "mean(pe/pb/fcf_yield + margin/roe/de/growth) — stated priors"),
        {"subs": {k: round(v, 1) for k, v in subs.items()}, "coverage": len(subs),
         "source": fundamentals_source},
        flags if subs else flags + ["no_fundamental_data"])

    # 5. RESEARCH — deterministic analyst-consensus proxy. The LLM's actual news
    # reading stays prose. NOT backtestable from free data: tracked forward.
    subs_r: Dict[str, float] = {}
    rec_mean = fundamentals.get("recommendationMean")
    if rec_mean is not None:
        subs_r["consensus"] = _clip((5.0 - float(rec_mean)) / 4.0 * 100.0)   # 1->100, 5->0
    tgt, price = fundamentals.get("targetMeanPrice"), indicators.get("current_price")
    if tgt and price:
        gap = (float(tgt) / float(price) - 1.0) * 100.0
        subs_r["target_gap"] = _clip(50.0 + gap * 2.5)                      # +20% -> 100, -20% -> 0
    n_analysts = fundamentals.get("numberOfAnalystOpinions") or 0
    conf_r = min(1.0, float(n_analysts) / 10.0)
    res_score = sum(subs_r.values()) / len(subs_r) if subs_r else None
    pillars["research"] = _pillar(
        res_score, conf_r, False,
        "mean(recommendationMean 1-5 inverted, analyst target gap); confidence = n_analysts/10",
        {"subs": {k: round(v, 1) for k, v in subs_r.items()}, "n_analysts": n_analysts},
        ["not_backtestable"] + ([] if subs_r else ["no_analyst_data"]))

    # 6. SOCIAL — reddit + stocktwits composite, confidence-shrunk toward 50 by
    # sample size (5 angry posts are noise, not signal). NOT backtestable.
    subs_s: Dict[str, float] = {}
    flags_s: list = ["not_backtestable"]
    if reddit and reddit.get("sentiment_score") is not None:
        mentions = int(reddit.get("mention_count") or 0)
        shrink = 1.0 if mentions >= 20 else (0.5 if mentions >= 5 else 0.0)
        if shrink == 0.0:
            flags_s.append("reddit_sample_too_small")
        raw = (float(reddit["sentiment_score"]) + 100.0) / 2.0            # -100..100 -> 0..100
        subs_s["reddit"] = 50.0 + (raw - 50.0) * shrink
    if stocktwits and stocktwits.get("total"):
        total = int(stocktwits["total"])
        shrink = min(1.0, total / 20.0)
        subs_s["stocktwits"] = 50.0 + (float(stocktwits.get("sentiment_ratio") or 50.0) - 50.0) * shrink
    if not subs_s:
        flags_s.append("no_social_data")
    soc_score = sum(subs_s.values()) / len(subs_s) if subs_s else None
    pillars["social"] = _pillar(
        soc_score, min(1.0, len(subs_s) / 2.0), False,
        "mean(reddit sentiment remapped + shrunk by mentions, stocktwits ratio shrunk by volume)",
        {"subs": {k: round(v, 1) for k, v in subs_s.items()}}, flags_s)

    # ── COMBINE (the prediction agent's seat at this table) ────────────────
    core = sum(CORE_WEIGHTS[k] * pillars[k]["score"] for k in CORE_WEIGHTS)
    modifiers = 0.0
    for name in ("social", "research"):
        p = pillars[name]
        modifiers += (p["score"] - 50.0) / 50.0 * MODIFIER_MAX_PTS * p["confidence"]
    risk_mult = 0.5 + 0.5 * pillars["risk"]["score"] / 100.0
    composite = 50.0 + (_clip(core + modifiers) - 50.0) * risk_mult
    if veto:
        composite = _clip(composite, VETO_BAND[0], VETO_BAND[1])
    composite = round(_clip(composite), 1)

    return {
        "ticker": ticker.upper(),
        "pillars": pillars,
        "core_score": round(core, 1),
        "modifier_pts": round(modifiers, 2),
        "risk_multiplier": round(risk_mult, 3),
        "risk_veto": veto,
        "composite": composite,
        "action": action_for(composite),
        "weights": dict(CORE_WEIGHTS),
        "modifier_max_pts": MODIFIER_MAX_PTS,
    }


# ══════════════════════════════════════════════════════════════════════════
# Vectorized backtestable core (technical + algo + risk over full history)
# ══════════════════════════════════════════════════════════════════════════
# These mirror the LIVE vote logic bar-by-bar using the same `ta` library the
# live path uses, so the last bar of each series equals the live meter — a
# consistency test enforces that. Rounding mirrors the live path (values are
# rounded before threshold comparison) so boundary bars can't diverge.

def technical_score_series(df: pd.DataFrame) -> pd.Series:
    """Bar-by-bar mirror of compute_signal_summary's voting (the vote-bearing
    subset: RSI, MACD cross+hist, SMA20/50/200, golden/death, BB %B — the
    volume/ADX lines in the live function add prose but zero votes today)."""
    import ta

    close = df["Close"].astype(float)
    n = len(close)
    idx = df.index

    rsi = ta.momentum.RSIIndicator(close, 14).rsi().round(4)
    macd_ind = ta.trend.MACD(close, 26, 12, 9)
    macd, macd_sig = macd_ind.macd().round(4), macd_ind.macd_signal().round(4)
    macd_hist = macd_ind.macd_diff().round(4)
    sma20 = ta.trend.sma_indicator(close, 20).round(4)
    sma50 = ta.trend.sma_indicator(close, 50).round(4)
    sma200 = ta.trend.sma_indicator(close, 200).round(4) if n >= 200 else pd.Series(np.nan, index=idx)
    bb = ta.volatility.BollingerBands(close, 20, 2)
    bb_up, bb_lo = bb.bollinger_hband().round(4), bb.bollinger_lband().round(4)

    bull = pd.Series(0.0, index=idx)
    bear = pd.Series(0.0, index=idx)

    # RSI votes (live: truthiness guard means NaN -> no vote)
    r_ok = rsi.notna()
    bull += (r_ok & (rsi < 30)) * 2 + (r_ok & (rsi > 55) & (rsi <= 70)) * 1
    bear += (r_ok & (rsi > 70)) * 2 + (r_ok & (rsi < 45) & (rsi >= 30)) * 1

    # MACD cross (live sets the key only when both values truthy)
    m_ok = macd.notna() & macd_sig.notna()
    bull += (m_ok & (macd > macd_sig)) * 2
    bear += (m_ok & (macd <= macd_sig)) * 2
    h_ok = macd_hist.notna()
    bull += (h_ok & (macd_hist > 0)) * 1
    bear += (h_ok & (macd_hist < 0)) * 1

    # SMA votes. Live quirk mirrored deliberately: a missing sma20/50 key makes
    # `if indicators.get(...)` falsy -> the ELSE branch fires -> a bear vote.
    bull += (sma20.notna() & (close > sma20)) * 1
    bear += (~(sma20.notna() & (close > sma20))) * 1
    bull += (sma50.notna() & (close > sma50)) * 1
    bear += (~(sma50.notna() & (close > sma50))) * 1
    # sma200 uses if/elif-is-False: NaN -> genuinely no vote either way.
    s200_ok = sma200.notna()
    bull += (s200_ok & (close > sma200)) * 2
    bear += (s200_ok & (close <= sma200)) * 2

    # Golden/death cross
    gd_ok = sma50.notna() & sma200.notna()
    bull += (gd_ok & (sma50 > sma200)) * 2
    bear += (gd_ok & (sma50 <= sma200)) * 2

    # Bollinger %B extremes
    bb_ok = bb_up.notna() & bb_lo.notna() & ((bb_up - bb_lo) > 0)
    bb_pct = ((close - bb_lo) / (bb_up - bb_lo) * 100).round(1)
    bull += (bb_ok & (bb_pct < 10)) * 1
    bear += (bb_ok & (bb_pct > 90)) * 1

    total = bull + bear
    score = (bull / total.replace(0, np.nan) * 100).fillna(50.0)
    # Live does int() truncation; mirror it for exact last-bar equality.
    return score.apply(math.floor).astype(float)


def algo_score_series(df: pd.DataFrame) -> pd.Series:
    """Bar-by-bar mirror of compute_algo_signals' composite vote (z-score,
    momentum composite, 30-bar linreg slope, volume-price divergence). Monte
    Carlo and candlestick patterns contribute zero votes in the live scorer and
    are therefore excluded. STRONG/plain signal variants share vote sets, so
    votes depend only on the thresholds mirrored here."""
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    idx = df.index

    # Mean reversion z-score (20d), live rounds to 3
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    z = ((close - roll_mean) / roll_std).round(3).fillna(0.0)

    # Momentum composite = mean of available 1m/3m/6m %returns (live rounds 2)
    moms = []
    for nper in (21, 63, 126):
        m = ((close - close.shift(nper)) / close.shift(nper) * 100).round(2)
        moms.append(m)
    mom_df = pd.concat(moms, axis=1)
    mom_comp = mom_df.mean(axis=1, skipna=True).round(2)
    mom_any = mom_df.notna().any(axis=1)

    # Rolling 30-bar linreg slope as % of window mean per day (live rounds 4).
    # Vote thresholds are ±0.1 only (STRONG variants share the same vote sets).
    win = 30
    x = np.arange(win, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()

    def _slope_pct(w: np.ndarray) -> float:
        y_mean = w.mean()
        if y_mean == 0:
            return 0.0
        slope = ((x - x_mean) * (w - y_mean)).sum() / ss_xx
        return round(slope / y_mean * 100, 4)

    slope_pct = close.rolling(win).apply(_slope_pct, raw=True)

    # Volume-price divergence (5d price change vs volume vs its 20d mean)
    price_chg = (close - close.shift(5)) / close.shift(5)
    vol_ma = volume.rolling(20).mean()
    vol_chg = (volume - vol_ma) / vol_ma
    vp_ok = price_chg.notna() & vol_chg.notna()
    breakout = vp_ok & (price_chg > 0.02) & (vol_chg > 0.5)
    breakdown = vp_ok & (price_chg < -0.02) & (vol_chg > 0.5)

    bull = pd.Series(0.0, index=idx)
    bear = pd.Series(0.0, index=idx)
    bull += (z < -0.8) * 2
    bear += (z > 0.8) * 2
    bull += (mom_any & (mom_comp > 3)) * 2
    bear += (mom_any & (mom_comp < -3)) * 2
    s_ok = slope_pct.notna()
    bull += (s_ok & (slope_pct > 0.1)) * 1
    bear += (s_ok & (slope_pct < -0.1)) * 1
    bull += breakout * 2
    bear += breakdown * 2

    total = bull + bear
    score = (bull / total.replace(0, np.nan) * 100).fillna(50.0)
    return score.apply(math.floor).astype(float)


def risk_mult_series(prices: pd.Series) -> pd.Series:
    """Rolling risk multiplier in [0.5, 1.0] from 20d annualized volatility
    (the series version uses vol only; the snapshot pillar adds beta/52w/short
    ratio, which have no free history — documented asymmetry)."""
    returns = prices.astype(float).pct_change()
    hv20 = (returns.rolling(20).std() * math.sqrt(252) * 100).round(2)
    score = ((60.0 - hv20) / 45.0 * 100.0).clip(0.0, 100.0)
    return (0.5 + 0.5 * score / 100.0).fillna(0.75)   # unknown vol -> middle


def seven_pillar_core_strategy(df: pd.DataFrame, enter: float = ENTER_THRESHOLD,
                               exit: float = EXIT_THRESHOLD) -> pd.Series:
    """The BACKTESTABLE core as a {0,1} long-only signal for the engine.

    core = (0.5*technical + 0.5*algo) scaled around 50 by the risk multiplier —
    the snapshot's fundamentals term (0.20 weight) is ABSENT here because PIT
    fundamentals are not in the backtest path yet; tech/algo renormalize to
    0.5/0.5. This means "the backtested composite" always refers to THIS core,
    never the full snapshot composite (which includes untestable modifiers).

    Weekly decision cadence: the score is sampled at each week's last bar and
    held. Hysteresis: enter at >= `enter`, stay until < `exit`. The engine's
    own .shift(1) then delays execution to the NEXT bar, so a decision made on
    Friday's close earns returns starting Monday — point-in-time correct.

    Defaults (65/55, weekly) are pre-committed priors. If they backtest badly,
    that result is REPORTED (a dead trial in the registry), not tuned away.
    """
    tech = technical_score_series(df)
    algo = algo_score_series(df)
    core = 0.5 * tech + 0.5 * algo
    mult = risk_mult_series(df["Close"])
    score = 50.0 + (core - 50.0) * mult

    # Weekly cadence: keep only each week's last observation, then hold (ffill).
    weekly_last = score.groupby(pd.Grouper(freq="W-FRI")).tail(1)
    decision = weekly_last.reindex(score.index).ffill()

    # Hysteresis state machine (long-only): enter >= enter, exit < exit.
    pos = np.zeros(len(decision))
    holding = 0.0
    vals = decision.to_numpy()
    for i in range(len(vals)):
        v = vals[i]
        if np.isnan(v):
            pos[i] = holding
            continue
        if holding == 0.0 and v >= enter:
            holding = 1.0
        elif holding == 1.0 and v < exit:
            holding = 0.0
        pos[i] = holding
    return pd.Series(pos, index=df.index)
