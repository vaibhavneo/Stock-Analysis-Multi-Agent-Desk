"""
Verification for intelligence/prediction_engine.py (item 6: multi-horizon
probabilistic forecast).

Run: python3 tests/test_prediction_engine.py

Offline/deterministic, no network - pure function of the dicts it's handed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.prediction_engine import forecast_horizons, HORIZONS

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def pillars(technical=70, algo=70, fundamentals=60, confidence=0.9):
    return {
        "technical": {"score": technical, "confidence": confidence},
        "algo": {"score": algo, "confidence": confidence},
        "fundamentals": {"score": fundamentals, "confidence": confidence},
        "risk": {"score": 60, "confidence": confidence},
        "research": {"score": 55, "confidence": confidence},
        "social": {"score": 50, "confidence": confidence},
    }


def test_bull_base_bear_ordering_always_holds():
    cases = [
        (pillars(90, 90, 90, 0.9), {"risk_stance": "RISK_ON"}, 15.0, 150.0),
        (pillars(10, 10, 10, 0.9), {"risk_stance": "RISK_OFF"}, 15.0, 150.0),
        (pillars(50, 50, 50, 0.3), None, 2.0, 50.0),
        (pillars(70, 30, 60, 0.6), {"risk_stance": "NEUTRAL"}, 8.0, 300.0),
    ]
    for i, (p, regime, atr, price) in enumerate(cases):
        r = forecast_horizons("T", price, p, {}, regime=regime, atr_14=atr)
        for label, h in r["horizons"].items():
            pr = h["price_range"]
            if pr.get("data_available"):
                check(f"case {i} {label}: bull > base > bear",
                      pr["bull_price"] > pr["base_price"] > pr["bear_price"],
                      f"{pr}")


def test_low_confidence_clamps_near_half():
    p = pillars(technical=95, algo=95, fundamentals=95, confidence=0.05)
    r = forecast_horizons("T", 100.0, p, {}, regime={"risk_stance": "RISK_ON"}, atr_14=2.0)
    for label, h in r["horizons"].items():
        check(f"{label}: p_up stays close to 0.5 despite an extreme score, given near-zero pillar confidence",
              abs(h["p_up"] - 0.5) < 0.15, f"p_up={h['p_up']}")


def test_monotonicity_in_composite_score():
    low = pillars(technical=30, algo=30, fundamentals=30, confidence=0.9)
    high = pillars(technical=90, algo=90, fundamentals=90, confidence=0.9)
    r_low = forecast_horizons("T", 100.0, low, {}, regime={"risk_stance": "NEUTRAL"}, atr_14=2.0)
    r_high = forecast_horizons("T", 100.0, high, {}, regime={"risk_stance": "NEUTRAL"}, atr_14=2.0)
    for label in r_low["horizons"]:
        check(f"{label}: higher composite score -> strictly higher p_up, all else equal",
              r_high["horizons"][label]["p_up"] > r_low["horizons"][label]["p_up"],
              f"low={r_low['horizons'][label]['p_up']} high={r_high['horizons'][label]['p_up']}")


def test_missing_atr_degrades_honestly():
    r = forecast_horizons("T", 100.0, pillars(), {}, atr_14=None)
    check("atr_unavailable flagged", "atr_unavailable" in r["flags"])
    for label, h in r["horizons"].items():
        check(f"{label}: price_range honestly unavailable, not fabricated",
              h["price_range"] == {"data_available": False})
        check(f"{label}: p_up/direction still computed despite missing ATR",
              h["p_up"] is not None and h["direction"] in ("UP", "DOWN", "NEUTRAL"))


def test_insufficient_pillar_data_is_honest():
    r = forecast_horizons("T", 100.0, {}, {})
    check("confidence is exactly 0.0 with no scoreable pillar", r["confidence"] == 0.0)
    check("horizons dict is empty, not fabricated", r["horizons"] == {})
    check("flagged insufficient_pillar_data", "insufficient_pillar_data" in r["flags"])

    r2 = forecast_horizons("T", None, pillars(), {})
    check("confidence is exactly 0.0 with no current_price", r2["confidence"] == 0.0)


def test_regime_adjustment_shifts_p_up():
    p = pillars(technical=60, algo=60, fundamentals=55, confidence=0.7)
    r_on = forecast_horizons("T", 100.0, p, {}, regime={"risk_stance": "RISK_ON"}, atr_14=2.0)
    r_off = forecast_horizons("T", 100.0, p, {}, regime={"risk_stance": "RISK_OFF"}, atr_14=2.0)
    for label in r_on["horizons"]:
        check(f"{label}: RISK_ON p_up > RISK_OFF p_up with identical pillars",
              r_on["horizons"][label]["p_up"] > r_off["horizons"][label]["p_up"],
              f"on={r_on['horizons'][label]['p_up']} off={r_off['horizons'][label]['p_up']}")


def test_analog_adjustment_blends_toward_historical_outcome():
    p = pillars(technical=50, algo=50, fundamentals=50, confidence=0.5)  # neutral pillar read
    strong_bearish_analog = {
        "status": "ok", "confidence": 1.0,
        "outcome_by_horizon": {h: {"pct_positive": 0.05, "n": 20} for h in HORIZONS},
    }
    r_no_analog = forecast_horizons("T", 100.0, p, {}, atr_14=2.0)
    r_with_analog = forecast_horizons("T", 100.0, p, {}, atr_14=2.0, analog_result=strong_bearish_analog)
    for label in r_no_analog["horizons"]:
        check(f"{label}: a strongly bearish analog history pulls p_up down from the neutral baseline",
              r_with_analog["horizons"][label]["p_up"] < r_no_analog["horizons"][label]["p_up"],
              f"no_analog={r_no_analog['horizons'][label]['p_up']} with_analog={r_with_analog['horizons'][label]['p_up']}")


def test_analog_with_too_few_samples_is_ignored():
    p = pillars(technical=80, algo=80, fundamentals=80, confidence=0.8)
    thin_analog = {"status": "ok", "confidence": 1.0,
                   "outcome_by_horizon": {h: {"pct_positive": 0.01, "n": 2} for h in HORIZONS}}
    r_no_analog = forecast_horizons("T", 100.0, p, {}, atr_14=2.0)
    r_thin = forecast_horizons("T", 100.0, p, {}, atr_14=2.0, analog_result=thin_analog)
    for label in r_no_analog["horizons"]:
        check(f"{label}: an analog with too few samples (n<5) is ignored, not blended in",
              r_no_analog["horizons"][label]["p_up"] == r_thin["horizons"][label]["p_up"])


def test_decay_toward_neutral_at_longer_horizons():
    p = pillars(technical=95, algo=95, fundamentals=95, confidence=0.9)
    r = forecast_horizons("T", 100.0, p, {}, regime={"risk_stance": "RISK_ON"}, atr_14=2.0)
    dist_1w = abs(r["horizons"]["1W"]["p_up"] - 0.5)
    dist_1y = abs(r["horizons"]["1Y"]["p_up"] - 0.5)
    check("1W p_up is further from 0.5 than 1Y p_up (confidence decays with horizon, not fabricated precision)",
          dist_1w > dist_1y, f"1W dist={dist_1w} 1Y dist={dist_1y}")
    check("per-horizon confidence also decays with horizon length",
          r["horizons"]["1W"]["confidence"] > r["horizons"]["1Y"]["confidence"])


def test_invalidation_prefers_the_widest_available_window():
    hc_only_20d = {"support_resistance": {"20D": {"support": 90.0, "resistance": 110.0, "data_available": True},
                                          "6M": {"data_available": False}, "1Y": {"data_available": False}}}
    hc_full = {"support_resistance": {"20D": {"support": 95.0, "resistance": 105.0, "data_available": True},
                                      "6M": {"data_available": False},
                                      "1Y": {"support": 70.0, "resistance": 130.0, "data_available": True}}}
    r1 = forecast_horizons("T", 100.0, pillars(), {}, historical_context=hc_only_20d)
    r2 = forecast_horizons("T", 100.0, pillars(), {}, historical_context=hc_full)
    check("falls back to 20D when nothing wider is available",
          r1["invalidation"]["source_window"] == "20D", str(r1["invalidation"]))
    check("prefers 1Y over 20D when both are available",
          r2["invalidation"]["source_window"] == "1Y", str(r2["invalidation"]))
    check("no historical_context at all -> invalidation honestly unavailable",
          forecast_horizons("T", 100.0, pillars(), {})["invalidation"] == {"data_available": False})


if __name__ == "__main__":
    test_bull_base_bear_ordering_always_holds()
    test_low_confidence_clamps_near_half()
    test_monotonicity_in_composite_score()
    test_missing_atr_degrades_honestly()
    test_insufficient_pillar_data_is_honest()
    test_regime_adjustment_shifts_p_up()
    test_analog_adjustment_blends_toward_historical_outcome()
    test_analog_with_too_few_samples_is_ignored()
    test_decay_toward_neutral_at_longer_horizons()
    test_invalidation_prefers_the_widest_available_window()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — prediction engine: monotonic, honestly bounded, no false precision")
