#!/usr/bin/env python3
"""Asset class is the correctness boundary, not a routing convenience.

Every analytic in this repo was written for a US equity. Three of those
assumptions are wrong for anything else and all three fail SILENTLY:

  - `sqrt(252)` annualization understates a 24/7 asset's volatility by 20%
  - `_pillar` substitutes 50.0 for a missing score, and the composite WEIGHTS
    that substitute, so a coin gets 10 points of fabricated fundamentals
  - the 40/20 volatility bands make every coin permanently HIGH (firing the
    risk veto) and every currency permanently LOW

These tests exist to make each of those failures loud. The two that matter
most are the neutrality tests: if classifying broke equities, the feature
would be worse than the bug.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas import asset_class as ac
from mas.asset_class import classify, spec_for, days_per_year, vol_regime

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


# ── Shape classification ──────────────────────────────────────────────────

def test_crypto_pairs_classify_by_quote_currency():
    for sym in ("BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-EUR"):
        c = classify(sym)
        check(f"{sym} is CRYPTO", c["asset_class"] == ac.CRYPTO, c["asset_class"])
        check(f"{sym} classified from shape", c["basis"] == "SYMBOL_SHAPE")


def test_share_class_suffix_is_not_a_crypto_pair():
    """The collision that makes the dash rule dangerous. BRK-B is an equity;
    reading its '-B' as a quote currency would route Berkshire to the crypto
    calendar and drop its fundamentals pillar."""
    for sym in ("BRK-B", "BF-B", "BRK-A"):
        c = classify(sym)
        check(f"{sym} is EQUITY not CRYPTO", c["asset_class"] == ac.EQUITY, c["asset_class"])


def test_index_fx_and_futures_shapes():
    check("^GSPC is INDEX", classify("^GSPC")["asset_class"] == ac.INDEX)
    check("^VIX is INDEX", classify("^VIX")["asset_class"] == ac.INDEX)
    check("EURUSD=X is FX", classify("EURUSD=X")["asset_class"] == ac.FX)
    check("USDJPY=X is FX", classify("USDJPY=X")["asset_class"] == ac.FX)
    check("GC=F is COMMODITY", classify("GC=F")["asset_class"] == ac.COMMODITY)
    check("CL=F is COMMODITY", classify("CL=F")["asset_class"] == ac.COMMODITY)


def test_empty_symbol_is_unknown_not_equity():
    c = classify("")
    check("empty -> UNKNOWN", c["asset_class"] == ac.UNKNOWN, c["asset_class"])
    spec = spec_for(ac.UNKNOWN)
    check("UNKNOWN claims no capabilities",
          not any([spec.has_fundamentals, spec.has_earnings, spec.has_sec_filings,
                   spec.has_analyst_ratings]))


def test_bare_ticker_is_assumed_and_says_so():
    """A bare ticker can't be separated from an ETF without metadata. The
    answer is allowed to be EQUITY — but it must NOT claim confidence."""
    c = classify("AAPL")
    check("bare ticker -> EQUITY", c["asset_class"] == ac.EQUITY)
    check("bare ticker basis is ASSUMED", c["basis"] == "ASSUMED", c["basis"])
    check("bare ticker is not marked confident", c["confident"] is False)
    check("the assumption is explained", len(c["reason"]) > 20)


def test_metadata_separates_equity_from_etf():
    c = classify("SPY", {"quoteType": "ETF"})
    check("metadata -> ETF", c["asset_class"] == ac.ETF, c["asset_class"])
    check("metadata basis", c["basis"] == "METADATA")
    check("ETF is confident", c["confident"] is True)


def test_shape_beats_wrong_metadata():
    """A provider that labels BTC-USD an equity is wrong and the shape is not.
    If metadata could override shape, one bad vendor field would silently put
    a coin back on the 252-day calendar."""
    c = classify("BTC-USD", {"quoteType": "EQUITY", "sector": "Technology"})
    check("shape wins over wrong metadata", c["asset_class"] == ac.CRYPTO, c["asset_class"])


# ── The calendar ──────────────────────────────────────────────────────────

def test_calendar_per_class():
    check("crypto is 365", days_per_year("BTC-USD") == 365)
    check("equity is 252", days_per_year("AAPL") == 252)
    check("fx is 260", days_per_year("EURUSD=X") == 260)
    check("index is 252", days_per_year("^GSPC") == 252)


def test_the_understatement_is_the_documented_factor():
    """The specific number the docstring claims: 1.204."""
    factor = math.sqrt(365 / 252)
    check("crypto/equity annualization ratio is ~1.204",
          abs(factor - 1.204) < 0.005, f"{factor:.4f}")
    ratio = math.sqrt(days_per_year("BTC-USD")) / math.sqrt(days_per_year("AAPL"))
    check("the specs reproduce it", abs(ratio - factor) < 1e-9)


# ── Volatility bands ──────────────────────────────────────────────────────

def test_bands_are_class_relative():
    check("50% vol is HIGH for an equity", vol_regime(50, "AAPL") == "HIGH")
    check("50% vol is only MEDIUM for a coin", vol_regime(50, "BTC-USD") == "MEDIUM")
    check("9% vol is LOW for an equity", vol_regime(9, "AAPL") == "LOW")
    check("9% vol is MEDIUM for a currency", vol_regime(9, "EURUSD=X") == "MEDIUM")


def test_missing_volatility_is_none_not_low():
    """'Not measured' must never read as 'calm' — entry.py branches on HIGH
    and position.py sizes on the regime."""
    check("None vol -> None regime", vol_regime(None, "AAPL") is None)


# ── Neutrality: the feature must not move equities ────────────────────────

def test_equity_pillar_weights_are_unchanged():
    from backtest.pillars import CORE_WEIGHTS
    spec = spec_for(ac.EQUITY)
    check("equity has no inapplicable pillars", spec.inapplicable_pillars == ())
    check("nominal weights still sum to 1",
          abs(sum(CORE_WEIGHTS.values()) - 1.0) < 1e-9)


def test_composite_renormalizes_instead_of_scoring_a_missing_pillar():
    from backtest.pillars import compute_pillar_scores
    ind = {"current_price": 100.0}
    sig = {"score": 72.0}
    algo = {"composite_score": 68.0, "vol_regime": "NORMAL", "vol_expanding": False}
    fund = {"fiftyTwoWeekHigh": 120, "fiftyTwoWeekLow": 80}

    crypto = compute_pillar_scores("BTC-USD", ind, sig, algo, fund, asset_class="CRYPTO")
    equity = compute_pillar_scores("BTC-USD", ind, sig, algo, fund, asset_class="EQUITY")

    check("fundamentals dropped from crypto weights",
          "fundamentals" not in crypto["weights"], crypto["weights"])
    check("surviving weights renormalize to 1",
          abs(sum(crypto["weights"].values()) - 1.0) < 1e-9, crypto["weights"])
    check("fundamentals pillar is present but marked inapplicable",
          crypto["pillars"]["fundamentals"]["applicable"] is False)
    check("the flag names the class",
          any("not_applicable_to_crypto" in f
              for f in crypto["pillars"]["fundamentals"]["flags"]))
    check("equity mask still weights fundamentals",
          "fundamentals" in equity["weights"])
    check("the two masks give different composites on identical inputs",
          crypto["composite"] != equity["composite"],
          f"{crypto['composite']} vs {equity['composite']}")


def test_default_asset_class_preserves_legacy_behaviour():
    """Every pre-existing caller omits asset_class. If the default moved, 579
    passing tests would be measuring a different engine than production runs."""
    from backtest.pillars import compute_pillar_scores, CORE_WEIGHTS
    ind = {"current_price": 100.0}
    sig = {"score": 72.0}
    algo = {"composite_score": 68.0}
    fund = {"profitMargins": 0.2, "returnOnEquity": 0.3, "trailingPE": 25}
    default = compute_pillar_scores("AAPL", ind, sig, algo, fund)
    explicit = compute_pillar_scores("AAPL", ind, sig, algo, fund, asset_class="EQUITY")
    check("default == EQUITY composite", default["composite"] == explicit["composite"])
    check("default weights are the nominal ones",
          {k: round(v, 4) for k, v in CORE_WEIGHTS.items()} == default["weights"],
          default["weights"])


def test_zero_volume_series_degrades_rather_than_crashing():
    """Spot FX carries volume=0 on every bar. The unguarded 20-day-mean
    division raised ZeroDivisionError — a crash, not a degraded answer."""
    import pandas as pd
    from tools.market_data import compute_algo_signals, compute_indicators
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series([1.10 + 0.001 * (i % 7) for i in range(n)], index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.001, "Low": close * 0.999,
                       "Close": close, "Volume": [0] * n}, index=idx)
    ind = compute_indicators(df)
    out = compute_algo_signals(df, ind, asset_class="FX")
    check("no crash on zero volume", isinstance(out, dict))
    check("volume signal is None, not a fabricated NEUTRAL",
          out.get("volume_price_signal") is None, out.get("volume_price_signal"))
    check("and it says why", "no volume" in (out.get("volume_price_reason") or ""))


def test_annualization_days_are_reported_with_the_number():
    """A volatility figure without its calendar cannot be checked by anyone
    downstream — including the option pricer that consumes it."""
    import pandas as pd
    from tools.market_data import compute_algo_signals, compute_indicators
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series([100 + (i % 5) for i in range(n)], index=idx)
    df = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                       "Close": close, "Volume": [1_000_000] * n}, index=idx)
    ind = compute_indicators(df)
    eq = compute_algo_signals(df, ind, asset_class="EQUITY")
    cr = compute_algo_signals(df, ind, asset_class="CRYPTO")
    check("equity reports 252", eq["annualization_days"] == 252)
    check("crypto reports 365", cr["annualization_days"] == 365)
    check("crypto hv is higher on identical bars",
          cr["historical_volatility_20d"] > eq["historical_volatility_20d"])
    ratio = cr["historical_volatility_20d"] / eq["historical_volatility_20d"]
    check("by exactly the calendar ratio", abs(ratio - math.sqrt(365 / 252)) < 0.001,
          f"{ratio:.4f}")
    check("bands ride along", cr["vol_bands"]["asset_class"] == "CRYPTO")


# ── ETF identification ────────────────────────────────────────────────────

def test_identity_fields_are_fetched_so_etfs_can_be_identified():
    """SPY and AAPL are indistinguishable by shape. Without quoteType in the
    fetched field list, classify() can only ever return ASSUMED, and an ETF
    is scored on the equity mask forever."""
    import inspect
    from tools import market_data
    src = inspect.getsource(market_data.fetch_fundamentals)
    for field in ('"quoteType"', '"fundFamily"', '"navPrice"'):
        check(f"{field} is fetched", field in src, "missing from the key list")


def test_fund_metadata_classifies_an_etf():
    for meta in ({"quoteType": "ETF"},
                 {"quoteType": "MUTUALFUND"},
                 {"fundFamily": "SPDR"},
                 {"navPrice": 512.3}):
        c = classify("SPY", meta)
        check(f"{sorted(meta)} -> ETF", c["asset_class"] == ac.ETF, c["asset_class"])
        check("and is confident", c["confident"] is True)


def test_an_etf_excludes_the_fundamentals_pillar():
    """A fund has holdings, not earnings. The pillar was being fed the
    PORTFOLIO's valuation aggregates and weighted as company quality: a gold
    ETF scored 83 on 'fundamentals', which lifted its composite by ~9 points
    and held it one action band higher than the price action warranted."""
    from backtest.pillars import compute_pillar_scores
    ind = {"current_price": 100.0}
    sig = {"score": 40.0}
    algo = {"composite_score": 38.0}
    fund = {"trailingPE": 18, "priceToBook": 2.0}
    etf = compute_pillar_scores("SPY", ind, sig, algo, fund, asset_class="ETF")
    eq = compute_pillar_scores("SPY", ind, sig, algo, fund, asset_class="EQUITY")
    check("fundamentals excluded for the fund",
          "fundamentals" not in etf["weights"], etf["weights"])
    check("weights renormalize", abs(sum(etf["weights"].values()) - 1.0) < 1e-9)
    check("research excluded too (a fund has no analyst coverage)",
          "research" in etf["inapplicable_pillars"], etf["inapplicable_pillars"])
    check("the equity mask still weights it", "fundamentals" in eq["weights"])
    check("a weak tape is no longer propped up by a fund valuation score",
          etf["composite"] < eq["composite"],
          f'{etf["composite"]} vs {eq["composite"]}')


def test_core_read_resolves_an_assumed_class_before_scoring():
    """classify() returns ASSUMED for a bare ticker. core_read must resolve
    that with metadata BEFORE it scores, or the resolution is decorative."""
    import inspect
    from mas import core
    src = inspect.getsource(core.core_read)
    check("core_read re-classifies on an ASSUMED basis", "ASSUMED" in src)
    check("and it happens before compute_pillar_scores",
          src.index("ASSUMED") < src.index("compute_pillar_scores"))


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
