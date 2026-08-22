"""
Verification for intelligence/orchestration.py (item 10: adaptive
orchestration - deciding which analyses actually run per request).

Run: python3 tests/test_orchestration_adaptive.py

Offline/deterministic - every fetch/compute function run_selected() calls
is patched at ITS SOURCE module (intelligence.regime.compute_market_regime,
tools.market_data.fetch_price_history, etc.), not at orchestration's own
binding - run_selected() uses LAZY imports inside each conditional branch
(`from intelligence.regime import compute_market_regime` executes at CALL
time, inside the `if "regime" in sections:` block), so patching the source
attribute before calling run_selected() correctly affects what that lazy
import resolves to. Mirrors tests/test_orchestrator_parallel.py's
patch-and-count style for proving unneeded work is never done.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from intelligence.orchestration import plan_sections, run_selected, PRESETS

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


# ── plan_sections: pure logic, no mocking needed ────────────────────────

def test_preset_names_resolve_exactly():
    for name, expected in PRESETS.items():
        check(f"preset '{name}' resolves to its exact section list",
              plan_sections(name) == expected, str(plan_sections(name)))


def test_valuation_preset_is_the_cheap_one():
    check("'valuation' preset excludes regime/analog (the expensive SPY/VIX fetches)",
          "regime" not in PRESETS["valuation"] and "analog" not in PRESETS["valuation"])


def test_price_action_preset_matches_why_did_it_fall():
    sections = plan_sections("price_action")
    check("'price_action' includes historical_context (price behavior) and regime (market context)",
          "historical_context" in sections and "regime" in sections)
    check("'price_action' excludes the deep analog/forecast/risk/evidence stack",
          not any(s in sections for s in ("analog", "forecast", "risk", "evidence")))


def test_recovery_preset_matches_cost_basis_question():
    sections = plan_sections("recovery")
    check("'recovery' includes risk (carries the cost-basis math) and forecast",
          "risk" in sections and "forecast" in sections)


def test_recovery_auto_selected_when_position_supplied_with_no_explicit_request():
    check("no explicit request + has_position=True -> recovery preset",
          plan_sections(None, has_position=True) == PRESETS["recovery"])
    check("no explicit request + has_position=False -> full preset",
          plan_sections(None, has_position=False) == PRESETS["full"])


def test_explicit_section_list_is_honored():
    sections = plan_sections(["regime", "risk"])
    check("an explicit multi-section list is honored as given",
          sorted(sections) == ["regime", "risk"], str(sections))


def test_unrecognized_request_falls_back_to_full_not_nothing():
    sections = plan_sections(["not_a_real_section", "also_fake"])
    check("an unrecognized request falls back to 'full' rather than computing nothing",
          sections == PRESETS["full"], str(sections))


def test_single_string_request_is_treated_as_one_preset():
    check("a bare string request is treated as a single preset name",
          plan_sections("recovery") == PRESETS["recovery"])


# ── run_selected: mocked fetches, patch-and-count for gating proof ─────

def _fake_df():
    idx = pd.bdate_range("2023-01-02", periods=300)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0,
                          "Volume": 1e6}, index=idx)


def _fake_indicators(df):
    return {"current_price": 100.0}


def _fake_signal_summary(indicators):
    return {"score": 60, "direction": "BULLISH"}


def _fake_algo_signals(df, indicators):
    return {"algo_score": 55, "historical_volatility_20d": 20.0, "vol_regime": "MEDIUM"}


def _fake_pillar_scores(ticker, indicators, signal_summary, algo_signals, fundamentals, **kwargs):
    p = {"score": 55, "confidence": 0.8, "flags": []}
    return {"pillars": {k: dict(p) for k in
                        ("technical", "algo", "fundamentals", "risk", "research", "social")}}


def _run_with_full_mocks(sections, avg_cost=None):
    with patch("tools.market_data.fetch_price_history", return_value=_fake_df()), \
         patch("tools.market_data.fetch_fundamentals", return_value={}), \
         patch("tools.market_data.compute_indicators", side_effect=_fake_indicators), \
         patch("tools.market_data.compute_signal_summary", side_effect=_fake_signal_summary), \
         patch("tools.market_data.compute_algo_signals", side_effect=_fake_algo_signals), \
         patch("backtest.pillars.compute_pillar_scores", side_effect=_fake_pillar_scores), \
         patch("backtest.risk.compute_atr", return_value=2.0), \
         patch("financial_data.get_bars_df", return_value=_fake_df()) as mock_get_bars, \
         patch("intelligence.regime._fetch_vix_series", return_value=pd.Series(dtype=float)), \
         patch("intelligence.regime.compute_market_regime", return_value={"risk_stance": "NEUTRAL"}) as mock_regime, \
         patch("intelligence.historical_context.compute_historical_context",
               return_value={"confidence": 1.0, "flags": []}) as mock_hist, \
         patch("intelligence.analog_engine.find_historical_analogs",
               return_value={"status": "ok", "confidence": 0.5}) as mock_analog, \
         patch("intelligence.prediction_engine.forecast_horizons",
               return_value={"horizons": {}}) as mock_forecast, \
         patch("intelligence.risk_engine.compute_risk_profile",
               return_value={"confidence": 0.8}) as mock_risk, \
         patch("intelligence.evidence_synthesis.build_evidence_ledger",
               return_value={"weighted_score": 55.0}) as mock_evidence:
        result = run_selected("TEST", sections, avg_cost=avg_cost)
        mocks = {
            "regime": mock_regime, "historical_context": mock_hist, "analog": mock_analog,
            "forecast": mock_forecast, "risk": mock_risk, "evidence": mock_evidence,
            "get_bars_df": mock_get_bars,
        }
        return result, mocks


def test_narrow_preset_never_calls_unneeded_modules():
    result, mocks = _run_with_full_mocks(PRESETS["valuation"])   # ["historical_context"] only
    check("valuation preset: historical_context WAS called", mocks["historical_context"].called)
    for name in ("regime", "analog", "forecast", "risk", "evidence"):
        check(f"valuation preset: {name} was NEVER called (call_count == 0)",
              mocks[name].call_count == 0, f"call_count={mocks[name].call_count}")


def test_valuation_preset_never_fetches_spy():
    result, mocks = _run_with_full_mocks(PRESETS["valuation"])
    check("valuation preset never calls get_bars_df at all (no SPY/VIX fetch needed)",
          mocks["get_bars_df"].call_count == 0, f"call_count={mocks['get_bars_df'].call_count}")


def test_full_preset_calls_every_module():
    result, mocks = _run_with_full_mocks(PRESETS["full"])
    for name in ("regime", "historical_context", "analog", "forecast", "risk", "evidence"):
        check(f"full preset: {name} WAS called", mocks[name].called)
    check("full preset DOES fetch SPY (regime/analog both need it)",
          mocks["get_bars_df"].called)


def test_recovery_preset_calls_only_its_three_modules():
    result, mocks = _run_with_full_mocks(PRESETS["recovery"], avg_cost=490.0)
    for name in ("historical_context", "risk", "forecast"):
        check(f"recovery preset: {name} WAS called", mocks[name].called)
    for name in ("regime", "analog", "evidence"):
        check(f"recovery preset: {name} was NEVER called", mocks[name].call_count == 0)
    check("result carries the sections actually used", result["sections"] == PRESETS["recovery"])


def test_price_history_failure_degrades_honestly():
    with patch("tools.market_data.fetch_price_history", side_effect=RuntimeError("network down")):
        result = run_selected("TEST", PRESETS["full"])
    check("confidence is exactly 0.0 when price history can't be fetched", result["confidence"] == 0.0)
    check("flagged price_history_unavailable", "price_history_unavailable" in result["flags"])


if __name__ == "__main__":
    test_preset_names_resolve_exactly()
    test_valuation_preset_is_the_cheap_one()
    test_price_action_preset_matches_why_did_it_fall()
    test_recovery_preset_matches_cost_basis_question()
    test_recovery_auto_selected_when_position_supplied_with_no_explicit_request()
    test_explicit_section_list_is_honored()
    test_unrecognized_request_falls_back_to_full_not_nothing()
    test_single_string_request_is_treated_as_one_preset()
    test_narrow_preset_never_calls_unneeded_modules()
    test_valuation_preset_never_fetches_spy()
    test_full_preset_calls_every_module()
    test_recovery_preset_calls_only_its_three_modules()
    test_price_history_failure_degrades_honestly()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — adaptive orchestration: correct presets, provably no wasted computation")
