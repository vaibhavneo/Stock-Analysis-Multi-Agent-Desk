"""
Verification for intelligence/risk_engine.py (item 7: risk / cost-basis
engine).

Run: python3 tests/test_risk_engine.py

Offline/deterministic, no network - pure function of the dicts it's handed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.risk_engine import compute_risk_profile

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


ALGO = {"historical_volatility_20d": 22.5, "historical_volatility_60d": 19.0,
        "vol_regime": "MEDIUM", "vol_expanding": False}
LEVELS = {"entry_zone_low": 98.0, "entry_zone_high": 101.0, "stop_loss": 90.0,
          "target_price": 120.0, "atr_14": 4.0}


def test_recovery_required_is_asymmetric_not_symmetric():
    cases = [
        (100.0, 50.0, 100.0),    # -50% loss -> +100% to recover
        (100.0, 80.0, 25.0),     # -20% loss -> +25% to recover
        (100.0, 90.0, 11.11),    # -10% loss -> +11.11% to recover
        (100.0, 66.6667, 50.0),  # -1/3 loss -> +50% to recover
    ]
    for avg_cost, current, expected_recovery in cases:
        r = compute_risk_profile("T", current, LEVELS, ALGO,
                                  position={"avg_cost": avg_cost})
        got = r["cost_basis"]["recovery_required_pct"]
        check(f"avg_cost={avg_cost} current={current}: recovery_required_pct={expected_recovery}",
              abs(got - expected_recovery) < 0.02, f"got {got}")
        check(f"avg_cost={avg_cost} current={current}: recovery is NOT the naive symmetric -gain_loss_pct",
              abs(got) != abs(r["cost_basis"]["gain_loss_pct"]) or avg_cost == current,
              f"recovery={got} gain_loss={r['cost_basis']['gain_loss_pct']}")


def test_gain_case_recovery_required_is_zero():
    r = compute_risk_profile("T", 120.0, LEVELS, ALGO, position={"avg_cost": 100.0})
    check("no recovery needed when already above cost basis",
          r["cost_basis"]["recovery_required_pct"] == 0.0)
    check("underwater is False", r["cost_basis"]["underwater"] is False)
    check("gain_loss_pct is +20%", r["cost_basis"]["gain_loss_pct"] == 20.0)


def test_underwater_flag_and_gain_loss_pct():
    r = compute_risk_profile("T", 70.0, LEVELS, ALGO, position={"avg_cost": 100.0})
    check("underwater is True at a loss", r["cost_basis"]["underwater"] is True)
    check("gain_loss_pct is -30%", r["cost_basis"]["gain_loss_pct"] == -30.0)


def test_cost_basis_absent_not_null_filled_when_no_position():
    r = compute_risk_profile("T", 100.0, LEVELS, ALGO)
    check("cost_basis KEY is absent entirely, not present as None",
          "cost_basis" not in r, str(r.keys()))

    r2 = compute_risk_profile("T", 100.0, LEVELS, ALGO, position={})
    check("cost_basis absent when position dict has no avg_cost", "cost_basis" not in r2)

    r3 = compute_risk_profile("T", 100.0, LEVELS, ALGO, position=None)
    check("cost_basis absent when position is None", "cost_basis" not in r3)


def test_gain_loss_dollars_only_when_shares_given():
    r_no_shares = compute_risk_profile("T", 120.0, LEVELS, ALGO, position={"avg_cost": 100.0})
    check("gain_loss_dollars absent when shares not supplied", "gain_loss_dollars" not in r_no_shares["cost_basis"])

    r_with_shares = compute_risk_profile("T", 120.0, LEVELS, ALGO,
                                          position={"avg_cost": 100.0, "shares": 10})
    check("gain_loss_dollars = (current - avg_cost) * shares",
          r_with_shares["cost_basis"]["gain_loss_dollars"] == 200.0,
          str(r_with_shares["cost_basis"]))


def test_stop_loss_and_support_vs_cost_basis():
    r = compute_risk_profile("T", 95.0, LEVELS, ALGO, position={"avg_cost": 100.0})
    check("stop_loss_vs_cost_basis_pct computed", "stop_loss_vs_cost_basis_pct" in r["cost_basis"])
    expected = round((90.0 / 100.0 - 1) * 100, 2)
    check("stop_loss_vs_cost_basis_pct is correct", r["cost_basis"]["stop_loss_vs_cost_basis_pct"] == expected)

    hc = {"support_resistance": {"1Y": {"support": 85.0, "resistance": 130.0, "data_available": True}}}
    r2 = compute_risk_profile("T", 95.0, LEVELS, ALGO, historical_context=hc, position={"avg_cost": 100.0})
    check("nearest_support surfaced in cost_basis when historical_context has it",
          r2["cost_basis"].get("nearest_support") == 85.0)
    check("support_vs_cost_basis_pct correct",
          r2["cost_basis"]["support_vs_cost_basis_pct"] == round((85.0/100.0 - 1)*100, 2))


def test_levels_unavailable_degrades_honestly():
    r = compute_risk_profile("T", 100.0, None, ALGO)
    check("levels dict is empty when not supplied", r["levels"] == {})
    check("flagged levels_unavailable", "levels_unavailable" in r["flags"])
    check("confidence reduced but not zero (other blocks still compute)",
          0.0 < r["confidence"] < 1.0, str(r["confidence"]))


def test_no_current_price_is_zero_confidence():
    r = compute_risk_profile("T", None, LEVELS, ALGO)
    check("confidence is exactly 0.0 with no current price", r["confidence"] == 0.0)
    check("flagged no_current_price", "no_current_price" in r["flags"])
    r2 = compute_risk_profile("T", 0.0, LEVELS, ALGO)
    check("confidence is exactly 0.0 with a zero current price", r2["confidence"] == 0.0)


def test_drawdown_risk_picks_the_worst_across_horizons():
    hc = {"horizons": {
        "1M": {"data_available": True, "max_drawdown_pct": -5.0},
        "3M": {"data_available": True, "max_drawdown_pct": -22.0},
        "1Y": {"data_available": True, "max_drawdown_pct": -14.0},
        "3Y": {"data_available": False},
    }}
    r = compute_risk_profile("T", 100.0, LEVELS, ALGO, historical_context=hc)
    check("picks the single worst (most negative) drawdown across all available windows",
          r["drawdown_risk"]["worst_observed_drawdown_pct"] == -22.0, str(r["drawdown_risk"]))
    check("reports which window produced the worst drawdown",
          r["drawdown_risk"]["worst_observed_window"] == "3M")

    r_none = compute_risk_profile("T", 100.0, LEVELS, ALGO)
    check("drawdown_risk honestly unavailable with no historical_context",
          r_none["drawdown_risk"] == {"data_available": False})
    check("flagged drawdown_history_unavailable", "drawdown_history_unavailable" in r_none["flags"])


def test_scenario_risk_from_forecast():
    forecast = {"horizons": {
        "1M": {"price_range": {"bull_price": 115.0, "base_price": 105.0, "bear_price": 92.0, "data_available": True}},
        "1Y": {"price_range": {"data_available": False}},
    }}
    r = compute_risk_profile("T", 100.0, LEVELS, ALGO, forecast=forecast)
    check("1M scenario risk computed", "1M" in r["scenario_risk"])
    check("bear_case_downside_pct correct", r["scenario_risk"]["1M"]["bear_case_downside_pct"] == -8.0)
    check("bull_case_upside_pct correct", r["scenario_risk"]["1M"]["bull_case_upside_pct"] == 15.0)
    check("1Y excluded since its price_range was unavailable", "1Y" not in r["scenario_risk"])


def test_risk_reward_ratio():
    r = compute_risk_profile("T", 100.0, LEVELS, ALGO)
    # risk = 100-90=10, reward = 120-100=20 -> ratio 2.0
    check("risk_reward_ratio computed correctly from levels", r["levels"]["risk_reward_ratio"] == 2.0)

    bad_levels = {"stop_loss": 105.0, "target_price": 120.0}  # stop ABOVE current price - degenerate
    r2 = compute_risk_profile("T", 100.0, bad_levels, ALGO)
    check("degenerate stop-above-price is flagged, not silently divided",
          "stop_loss_above_current_price" in r2["flags"])
    check("risk_reward_ratio is None in the degenerate case", r2["levels"]["risk_reward_ratio"] is None)


if __name__ == "__main__":
    test_recovery_required_is_asymmetric_not_symmetric()
    test_gain_case_recovery_required_is_zero()
    test_underwater_flag_and_gain_loss_pct()
    test_cost_basis_absent_not_null_filled_when_no_position()
    test_gain_loss_dollars_only_when_shares_given()
    test_stop_loss_and_support_vs_cost_basis()
    test_levels_unavailable_degrades_honestly()
    test_no_current_price_is_zero_confidence()
    test_drawdown_risk_picks_the_worst_across_horizons()
    test_scenario_risk_from_forecast()
    test_risk_reward_ratio()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — risk engine: correct asymmetric math, honest degradation")
