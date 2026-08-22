"""
Verification for intelligence-context grounding in the prediction-agent
prompt (agents/stock_agents.py's _intelligence_block + _build_prediction_
prompt's/run_prediction_agent's optional `intelligence_context` param).

This is the ONLY place intelligence/ context reaches an LLM this session -
enriching the existing single prediction-agent call's prompt, exactly the
way _pillar_block() already grounds the 4 analysis agents' prompts. No new
LLM call is added anywhere.

Run: python3 tests/test_intelligence_prompt_grounding.py

Offline/deterministic - no network, no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stock_agents import _build_prediction_prompt, _intelligence_block

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


SIGNAL_SUMMARY = {"score": 62, "direction": "BULLISH"}
ARGS = ("AAPL", "Apple Inc.", 190.0, "fund text", "tech text", SIGNAL_SUMMARY, "social text", "algo text")


def test_omitting_context_is_byte_identical_to_before():
    _, user_without_param = _build_prediction_prompt(*ARGS)
    _, user_explicit_none = _build_prediction_prompt(*ARGS, intelligence_context=None)
    _, user_empty_dict = _build_prediction_prompt(*ARGS, intelligence_context={})
    check("omitting intelligence_context vs explicit None: byte-identical prompt",
          user_without_param == user_explicit_none)
    check("an empty dict also produces a byte-identical prompt (no block added)",
          user_without_param == user_empty_dict)
    check("no DETERMINISTIC MARKET CONTEXT block appears when nothing is supplied",
          "DETERMINISTIC MARKET CONTEXT" not in user_without_param)


def test_regime_block_appears_with_correct_values():
    ctx = {"regime": {"trend": "BULLISH", "volatility_regime": "LOW", "risk_stance": "RISK_ON"}}
    _, user = _build_prediction_prompt(*ARGS, intelligence_context=ctx)
    check("regime block appears", "DETERMINISTIC MARKET CONTEXT" in user)
    check("trend value present", "BULLISH trend" in user, user)
    check("volatility regime value present", "LOW volatility" in user, user)
    check("risk stance value present", "stance RISK_ON" in user, user)


def test_regime_with_no_risk_stance_is_silently_skipped():
    ctx = {"regime": {"trend": None, "volatility_regime": None, "risk_stance": None}}
    _, user = _build_prediction_prompt(*ARGS, intelligence_context=ctx)
    check("a regime dict with no risk_stance produces no block at all (nothing meaningful to say)",
          "DETERMINISTIC MARKET CONTEXT" not in user)


def test_contradictions_appear_and_are_not_hidden():
    ctx = {"contradictions": [
        {"name": "strong_fundamentals_bearish_technicals",
         "description": "Fundamentals look strong but technicals are bearish."},
    ]}
    _, user = _build_prediction_prompt(*ARGS, intelligence_context=ctx)
    check("contradiction name appears in the prompt", "strong_fundamentals_bearish_technicals" in user)
    check("contradiction description appears in the prompt",
          "Fundamentals look strong but technicals are bearish." in user)
    check("instructs the agent to explain contradictions, not hide them",
          "don't hide them" in user)


def test_analog_summary_appears_when_status_ok():
    ctx = {"analog": {"status": "ok", "outcome_by_horizon": {
        63: {"n": 12, "pct_positive": 0.75, "avg_return_pct": 8.4}}}}
    _, user = _build_prediction_prompt(*ARGS, intelligence_context=ctx)
    check("analog match count appears", "12 similar past setups" in user, user)
    check("analog positive rate appears", "75%" in user, user)
    check("analog avg return appears", "+8.4%" in user, user)
    check("analog is framed as context, not a guarantee", "not a guarantee" in user)


def test_analog_with_insufficient_history_produces_no_block():
    ctx = {"analog": {"status": "insufficient_history", "outcome_by_horizon": {}}}
    _, user = _build_prediction_prompt(*ARGS, intelligence_context=ctx)
    check("an insufficient-history analog result contributes nothing to the prompt",
          "DETERMINISTIC MARKET CONTEXT" not in user)


def test_all_three_sections_combine_in_one_block():
    ctx = {
        "regime": {"trend": "BEARISH", "volatility_regime": "HIGH", "risk_stance": "RISK_OFF"},
        "contradictions": [{"name": "bullish_stock_bearish_regime", "description": "desc here"}],
        "analog": {"status": "ok", "outcome_by_horizon": {63: {"n": 5, "pct_positive": 0.2, "avg_return_pct": -6.0}}},
    }
    block = _intelligence_block(ctx)
    check("regime, contradiction, and analog all appear in one combined block",
          "RISK_OFF" in block and "bullish_stock_bearish_regime" in block and "5 similar past setups" in block,
          block)


def test_existing_pillar_grounding_prompt_still_unaffected():
    """Sanity check that this new block doesn't interfere with the
    prediction schema instructions already in the prompt."""
    _, user = _build_prediction_prompt(*ARGS)
    check("the JSON schema instruction is still present and intact",
          "Output ONLY valid JSON matching this schema exactly" in user)


if __name__ == "__main__":
    test_omitting_context_is_byte_identical_to_before()
    test_regime_block_appears_with_correct_values()
    test_regime_with_no_risk_stance_is_silently_skipped()
    test_contradictions_appear_and_are_not_hidden()
    test_analog_summary_appears_when_status_ok()
    test_analog_with_insufficient_history_produces_no_block()
    test_all_three_sections_combine_in_one_block()
    test_existing_pillar_grounding_prompt_still_unaffected()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — intelligence context grounds the prediction prompt, byte-identical when opted out")
