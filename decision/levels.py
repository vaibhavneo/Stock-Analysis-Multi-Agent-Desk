"""
Conditional price paths — the level ladder (Phase 6).

The app's existing entry/target/stop are one formula applied to every security:
entry `[p − 0.5·ATR, p + 0.25·ATR]`, stop `p − max(2·ATR, 2%)`, target
`p + 2·stop_distance`. That geometry is internally consistent and completely
uninformative about *this* stock: the entry zone always brackets the current
price (so "is now a good entry?" is answered yes by construction), and the
target is a restatement of the stop distance at a fixed 2:1, so its implied
upside varies only with volatility.

This module replaces that single manufactured band with a ladder of levels that
each come from an actual observation in the data, and it refuses to emit a level
it cannot source. Every entry carries:

    price, kind (SUPPORT/RESISTANCE/REFERENCE), source, timeframe,
    confidence, rationale, distance_pct

Sources used, all already computed elsewhere in this codebase:
  - prior swing low / high over 20D, 6M, 1Y   (intelligence/historical_context.py)
  - 52-week high / low                         (tools/market_data.py::compute_indicators)
  - SMA 20 / 50 / 200                          (same)
  - Bollinger upper / middle / lower           (same)
  - ATR volatility band around price           (backtest/risk.py::compute_atr)
  - forecast bull/base/bear price range        (intelligence/prediction_engine.py)

Confidence per source is a STATED PRIOR, not a fitted number, and is documented
inline. The ordering principle is that a level many independent observers can
see (a 1-year swing low, the 200-day average) is more likely to be defended than
one only this engine computes (an ATR multiple).

Nothing here is a prediction. A support level is "a price at which buying
appeared before", not "a price the stock will reach".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Stated priors. Higher = more independently observable / more widely watched.
# These order the ladder; they are never multiplied into a score.
SOURCE_CONFIDENCE = {
    "swing_low_1y": 0.85, "swing_high_1y": 0.85,
    "52w_low": 0.85, "52w_high": 0.85,
    "sma_200": 0.80,
    "swing_low_6m": 0.75, "swing_high_6m": 0.75,
    "sma_50": 0.70,
    "bb_lower": 0.60, "bb_upper": 0.60,
    "sma_20": 0.60, "bb_middle": 0.55,
    "swing_low_20d": 0.55, "swing_high_20d": 0.55,
    "atr_band_low": 0.50, "atr_band_high": 0.50,
    "forecast_bear": 0.40, "forecast_bull": 0.40, "forecast_base": 0.40,
}

SOURCE_RATIONALE = {
    "swing_low_1y":  "Lowest traded price of the last 12 months — the deepest level buyers have defended within a year.",
    "swing_high_1y": "Highest traded price of the last 12 months — supply has appeared here before.",
    "52w_low":       "52-week low — a widely published, widely watched reference.",
    "52w_high":      "52-week high — a widely published breakout reference.",
    "sma_200":       "200-day moving average — the most commonly cited long-term trend line.",
    "swing_low_6m":  "Lowest traded price of the last 6 months.",
    "swing_high_6m": "Highest traded price of the last 6 months.",
    "sma_50":        "50-day moving average — the common medium-term trend reference.",
    "bb_lower":      "Lower Bollinger band (20-day mean − 2σ) — statistically stretched to the downside.",
    "bb_upper":      "Upper Bollinger band (20-day mean + 2σ) — statistically stretched to the upside.",
    "sma_20":        "20-day moving average — short-term mean.",
    "bb_middle":     "20-day mean — the centre of the recent range.",
    "swing_low_20d": "20-day low — the nearest short-term floor.",
    "swing_high_20d":"20-day high — the nearest short-term ceiling.",
    "atr_band_low":  "Two 14-day ATRs below price — a volatility-scaled distance, not an observed level.",
    "atr_band_high": "Two 14-day ATRs above price — a volatility-scaled distance, not an observed level.",
    "forecast_bear": "Model bear scenario (ATR × √horizon random-walk scaling) — a dispersion estimate, not an observed level.",
    "forecast_bull": "Model bull scenario (ATR × √horizon random-walk scaling) — a dispersion estimate, not an observed level.",
    "forecast_base": "Model base scenario — drift-adjusted centre of the range.",
}

SOURCE_TIMEFRAME = {
    "swing_low_1y": "1Y", "swing_high_1y": "1Y", "52w_low": "1Y", "52w_high": "1Y",
    "sma_200": "1Y", "swing_low_6m": "6M", "swing_high_6m": "6M", "sma_50": "3M",
    "bb_lower": "1M", "bb_upper": "1M", "sma_20": "1M", "bb_middle": "1M",
    "swing_low_20d": "1M", "swing_high_20d": "1M",
    "atr_band_low": "current", "atr_band_high": "current",
    "forecast_bear": "model", "forecast_bull": "model", "forecast_base": "model",
}

# How far from the current price a level can be and still be decision-relevant.
# A 1-year low 73% below today is a genuine level and stays in the ladder, but
# calling it "major support" for an entry decision would be absurd — nothing
# about today's plan turns on it. The cap is generous (a third) so that a real
# drawdown level still qualifies, and it is applied only to the *headline*
# picks, never to filter the ladder itself.
DECISION_RELEVANT_DISTANCE_PCT = 33.0

# Levels closer together than this (as a fraction of price) are treated as one
# cluster. Reporting "support at 98.40, 98.55 and 98.70" as three independent
# levels overstates precision that daily OHLC data does not carry.
CLUSTER_TOLERANCE_PCT = 0.015

# How a level came to exist. The distinction matters because it is the
# difference between "the market has defended this price" and "this is
# arithmetic on today's close", and a reader deserves to know which they are
# looking at before deciding to act on it.
#
#   TRADED_PRICE           an actual historical high/low — the market was here
#   PRICE_HISTORY_DERIVED  a line computed FROM history (a moving average, a
#                          Bollinger band): not a traded price, but it does
#                          carry information beyond today
#   CURRENT_PRICE_DERIVED  arithmetic on today's close (an ATR multiple):
#                          it moves whenever the price moves, so it cannot be
#                          a level the price is "approaching"
#   MODEL                  a forecast-engine dispersion estimate
LEVEL_BASIS = {
    "swing_low_1y": "TRADED_PRICE", "swing_high_1y": "TRADED_PRICE",
    "52w_low": "TRADED_PRICE", "52w_high": "TRADED_PRICE",
    "swing_low_6m": "TRADED_PRICE", "swing_high_6m": "TRADED_PRICE",
    "swing_low_20d": "TRADED_PRICE", "swing_high_20d": "TRADED_PRICE",
    "sma_200": "PRICE_HISTORY_DERIVED", "sma_50": "PRICE_HISTORY_DERIVED",
    "sma_20": "PRICE_HISTORY_DERIVED", "bb_upper": "PRICE_HISTORY_DERIVED",
    "bb_lower": "PRICE_HISTORY_DERIVED", "bb_middle": "PRICE_HISTORY_DERIVED",
    "atr_band_low": "CURRENT_PRICE_DERIVED", "atr_band_high": "CURRENT_PRICE_DERIVED",
    "forecast_bear": "MODEL", "forecast_bull": "MODEL", "forecast_base": "MODEL",
}
BASIS_RANK = {"TRADED_PRICE": 3, "PRICE_HISTORY_DERIVED": 2,
              "CURRENT_PRICE_DERIVED": 1, "MODEL": 0}

# A level is "observed" only if the market actually traded there.
OBSERVED_SOURCES = frozenset(
    k for k, v in LEVEL_BASIS.items() if v == "TRADED_PRICE")


def _add(levels: List[Dict[str, Any]], name: str, price: Optional[float],
         current_price: float) -> None:
    """Append a level if it is real, finite and positive. Never fabricates."""
    if price is None:
        return
    try:
        p = float(price)
    except (TypeError, ValueError):
        return
    if p <= 0 or p != p:            # non-positive or NaN
        return
    levels.append({
        "price": round(p, 2),
        "source": name,
        "kind": "SUPPORT" if p < current_price else ("RESISTANCE" if p > current_price else "AT_PRICE"),
        "timeframe": SOURCE_TIMEFRAME.get(name, "unknown"),
        "confidence": SOURCE_CONFIDENCE.get(name, 0.4),
        "rationale": SOURCE_RATIONALE.get(name, ""),
        "distance_pct": round((p / current_price - 1) * 100, 2),
        "observed": name in OBSERVED_SOURCES,
        "basis": LEVEL_BASIS.get(name, "MODEL"),
    })


def _cluster(levels: List[Dict[str, Any]], current_price: float) -> List[Dict[str, Any]]:
    """Merge levels within CLUSTER_TOLERANCE_PCT into one zone.

    A cluster is STRONGER than any of its members: several independent methods
    identifying the same price is the clearest evidence a level is real. That is
    reflected by taking the max member confidence and naming every contributing
    source, not by summing confidences into a number above 1.
    """
    if not levels:
        return []
    ordered = sorted(levels, key=lambda l: l["price"])
    clusters: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for lv in ordered[1:]:
        ref = clusters[-1][0]["price"]
        if abs(lv["price"] - ref) / current_price <= CLUSTER_TOLERANCE_PCT:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])

    out: List[Dict[str, Any]] = []
    for group in clusters:
        prices = [g["price"] for g in group]
        low, high = min(prices), max(prices)
        mid = round(sum(prices) / len(prices), 2)
        best = max(group, key=lambda g: g["confidence"])
        observed = any(g["observed"] for g in group)
        # The cluster's basis is its STRONGEST member's: one traded price among
        # three moving averages means the market really was here.
        basis = max((g["basis"] for g in group), key=lambda b: BASIS_RANK[b])
        out.append({
            "price": mid,
            "zone_low": low,
            "zone_high": high,
            "kind": "SUPPORT" if mid < current_price else ("RESISTANCE" if mid > current_price else "AT_PRICE"),
            "sources": [g["source"] for g in group],
            "n_sources": len(group),
            "timeframe": best["timeframe"],
            "confidence": round(min(1.0, best["confidence"] + 0.05 * (len(group) - 1)), 2),
            "rationale": (best["rationale"] if len(group) == 1 else
                          f"{len(group)} independent methods identify this zone: "
                          + "; ".join(SOURCE_RATIONALE.get(g["source"], g["source"]) for g in group)),
            "distance_pct": round((mid / current_price - 1) * 100, 2),
            "observed": observed,
            "basis": basis,
        })
    return out


def build_level_map(
    current_price: Optional[float],
    indicators: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    rec_levels: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    forecast_horizon_label: str = "3M",
) -> Dict[str, Any]:
    """The full level ladder around the current price.

    Returns `{"status": "INSUFFICIENT_DATA"}` with an empty ladder when there is
    no price or no sourceable level, rather than falling back to the ATR
    geometry — an invented level is worse than an absent one.
    """
    if not current_price or current_price <= 0:
        return {"status": "INSUFFICIENT_DATA", "reason": "no current price",
                "current_price": None, "supports": [], "resistances": [], "ladder": []}

    ind = indicators or {}
    raw: List[Dict[str, Any]] = []

    sr = (historical_context or {}).get("support_resistance") or {}
    for window, lo_name, hi_name in (("20D", "swing_low_20d", "swing_high_20d"),
                                     ("6M", "swing_low_6m", "swing_high_6m"),
                                     ("1Y", "swing_low_1y", "swing_high_1y")):
        w = sr.get(window) or {}
        if w.get("data_available"):
            _add(raw, lo_name, w.get("support"), current_price)
            _add(raw, hi_name, w.get("resistance"), current_price)

    _add(raw, "52w_low", ind.get("52w_low"), current_price)
    _add(raw, "52w_high", ind.get("52w_high"), current_price)
    _add(raw, "sma_200", ind.get("sma_200"), current_price)
    _add(raw, "sma_50", ind.get("sma_50"), current_price)
    _add(raw, "sma_20", ind.get("sma_20"), current_price)
    _add(raw, "bb_lower", ind.get("bb_lower"), current_price)
    _add(raw, "bb_middle", ind.get("bb_middle"), current_price)
    _add(raw, "bb_upper", ind.get("bb_upper"), current_price)

    atr = (rec_levels or {}).get("atr_14") or ind.get("atr_14")
    if atr:
        _add(raw, "atr_band_low", current_price - 2 * float(atr), current_price)
        _add(raw, "atr_band_high", current_price + 2 * float(atr), current_price)

    if forecast and forecast.get("horizons"):
        h = (forecast["horizons"] or {}).get(forecast_horizon_label)
        pr = (h or {}).get("price_range") or {}
        if pr.get("data_available"):
            _add(raw, "forecast_bear", pr.get("bear_price"), current_price)
            _add(raw, "forecast_base", pr.get("base_price"), current_price)
            _add(raw, "forecast_bull", pr.get("bull_price"), current_price)

    if not raw:
        return {"status": "INSUFFICIENT_DATA",
                "reason": "no level could be sourced from price history or indicators",
                "current_price": round(current_price, 2),
                "supports": [], "resistances": [], "ladder": []}

    # A ladder built ONLY from arithmetic on today's close cannot answer "is
    # this a good entry": every such level moves with the price, so the answer
    # would be the same at any quote. That is the exact defect this module
    # replaces, so it is reported as its own status rather than dressed up as a
    # working ladder.
    if all(LEVEL_BASIS.get(l["source"]) in ("CURRENT_PRICE_DERIVED", "MODEL")
           for l in raw):
        return {"status": "DERIVED_ONLY",
                "reason": ("Every candidate level is arithmetic on today's close or a model "
                           "dispersion estimate — no moving average, band, or traded high/low "
                           "was available. Such levels move with the price, so they cannot "
                           "locate an entry."),
                "current_price": round(current_price, 2),
                "supports": [], "resistances": [], "ladder": [],
                "n_levels": 0, "n_observed": 0}

    clustered = _cluster(raw, current_price)
    supports = sorted([c for c in clustered if c["kind"] == "SUPPORT"],
                      key=lambda c: -c["price"])       # nearest first
    resistances = sorted([c for c in clustered if c["kind"] == "RESISTANCE"],
                         key=lambda c: c["price"])     # nearest first

    def _near(levels):
        return [l for l in levels if abs(l["distance_pct"]) <= DECISION_RELEVANT_DISTANCE_PCT]

    near_supports, near_resistances = _near(supports), _near(resistances)
    # Prefer a traded price over a computed line at equal reach; fall back to
    # any nearby level, then (only if nothing is nearby) to the nearest at all,
    # so the field is never silently empty when levels do exist.
    def _headline(near, all_levels):
        traded = [l for l in near if l["observed"]]
        pool = traded or near or all_levels
        return max(pool, key=lambda l: l["confidence"]) if pool else None

    return {
        "status": "OK",
        "current_price": round(current_price, 2),
        "supports": supports,
        "resistances": resistances,
        "nearest_support": supports[0] if supports else None,
        "nearest_resistance": resistances[0] if resistances else None,
        "major_support": _headline(near_supports, supports),
        "major_resistance": _headline(near_resistances, resistances),
        "decision_relevant_supports": near_supports,
        "decision_relevant_resistances": near_resistances,
        "ladder": sorted(clustered, key=lambda c: -c["price"]),
        "n_levels": len(clustered),
        "n_observed": sum(1 for c in clustered if c["observed"]),
        "basis_legend": {k: v for k, v in (
            ("TRADED_PRICE", "an actual historical high/low — the market traded here"),
            ("PRICE_HISTORY_DERIVED", "a line computed from history (moving average, Bollinger band)"),
            ("CURRENT_PRICE_DERIVED", "arithmetic on today's close — it moves when the price moves"),
            ("MODEL", "a forecast-engine dispersion estimate, not an observed level"),
        )},
        "note": ("Levels are observations, not predictions. Read `basis` before acting: "
                 "only TRADED_PRICE levels are prices this security has actually "
                 "traded at."),
    }
