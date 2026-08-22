"""
Verification that agents/orchestrator.py::analyze_stock() actually PASSES
intelligence context to the prediction agent.

Why this file exists: _intelligence_block() and run_prediction_agent's
intelligence_context kwarg were built and unit-tested (see
tests/test_intelligence_prompt_grounding.py) but nothing in production
passed them - the contradiction narration was dormant code. Unit tests on
the prompt builder alone could never catch that; only a test at the call
site can.

Run: python3 tests/test_orchestrator_intelligence_wiring.py

Offline/deterministic - no network, no API key. Everything analyze_stock()
touches is patched at its `agents.orchestrator` binding (the module imports
these names directly, so patching the source module wouldn't affect
orchestrator's already-bound references - the same lesson
tests/test_orchestrator_parallel.py documents).

What must hold:
  1. Regime + contradictions reach run_prediction_agent as
     intelligence_context.
  2. A failure ANYWHERE in that enrichment degrades to
     intelligence_context=None - the user still gets their analysis.
  3. analyze_stock()'s return dict keys are unchanged by any of this.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


# The exact top-level keys analyze_stock() returned BEFORE this session's
# work. Hardcoded deliberately: this is the contract the plan promised to
# keep byte-identical, so it must be asserted against a frozen list, not
# against whatever the code happens to produce today.
EXPECTED_KEYS = {
    "ticker", "company_name", "sector", "current_price", "analyzed_at", "elapsed_s",
    "fundamentals", "indicators", "signal_summary", "algo_signals", "reddit", "stocktwits",
    "fundamentals_analysis", "technical_analysis", "social_analysis", "algo_analysis",
    "prediction", "recommendation",
}

FAKE_DF = pd.DataFrame(
    {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6},
    index=pd.bdate_range("2023-01-02", periods=300))

FAKE_PILLARS = {
    "technical": {"score": 20, "confidence": 0.9, "flags": []},     # bearish technicals
    "algo": {"score": 55, "confidence": 0.9, "flags": []},
    "fundamentals": {"score": 85, "confidence": 0.9, "flags": []},  # strong fundamentals
    "risk": {"score": 60, "confidence": 0.8, "flags": []},
    "research": {"score": 50, "confidence": 0.5, "flags": []},
    "social": {"score": 50, "confidence": 0.4, "flags": []},
}
FAKE_PREDICTION = {"action": "HOLD", "conviction": "LOW", "summary": "s", "scores": {}}


def _run_analyze(regime_side_effect=None, contradictions_side_effect=None):
    """Run analyze_stock() with everything stubbed, returning the captured
    intelligence_context that reached run_prediction_agent."""
    captured = {}

    def _capture_prediction(*args, **kwargs):
        captured["intelligence_context"] = kwargs.get("intelligence_context")
        return dict(FAKE_PREDICTION)

    regime_val = {"risk_stance": "RISK_OFF", "trend": "BEARISH",
                  "volatility_regime": "HIGH", "confidence": 0.9, "flags": []}

    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.fetch_price_history", return_value=FAKE_DF), \
         patch("agents.orchestrator.fetch_fundamentals", return_value={}), \
         patch("agents.orchestrator.fetch_earnings", return_value={}), \
         patch("agents.orchestrator.fetch_analyst_ratings", return_value={}), \
         patch("agents.orchestrator.compute_indicators", return_value={"current_price": 100.0}), \
         patch("agents.orchestrator.compute_signal_summary", return_value={"score": 50, "direction": "NEUTRAL"}), \
         patch("agents.orchestrator.compute_algo_signals", return_value={"algo_score": 55}), \
         patch("agents.orchestrator.fetch_reddit_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_stocktwits_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_web_forum_sentiment", return_value={}), \
         patch("agents.orchestrator.compute_pillar_scores", return_value={"pillars": FAKE_PILLARS}), \
         patch("agents.orchestrator._run_analysis_agents", return_value={
             "fundamentals_analysis": "f", "technical_analysis": "t",
             "social_analysis": "s", "algo_analysis": "a"}), \
         patch("agents.orchestrator.run_prediction_agent", side_effect=_capture_prediction), \
         patch("agents.orchestrator.ground_prediction", side_effect=lambda *a, **k: dict(FAKE_PREDICTION)), \
         patch("intelligence.regime.compute_market_regime",
               side_effect=regime_side_effect or (lambda *a, **k: regime_val)), \
         patch("intelligence.evidence_synthesis.detect_contradictions",
               side_effect=contradictions_side_effect) if contradictions_side_effect \
         else patch("intelligence.evidence_synthesis.detect_contradictions",
                    wraps=__import__("intelligence.evidence_synthesis",
                                     fromlist=["detect_contradictions"]).detect_contradictions):
        from agents.orchestrator import analyze_stock
        result = analyze_stock("TEST", api_key="fake-key")
    return result, captured.get("intelligence_context")


def test_intelligence_context_actually_reaches_the_prediction_agent():
    result, ctx = _run_analyze()
    check("intelligence_context is NOT None - the wiring is live, not dormant",
          ctx is not None, str(ctx))
    if ctx:
        check("regime reaches the prediction agent",
              (ctx.get("regime") or {}).get("risk_stance") == "RISK_OFF", str(ctx.get("regime")))
        names = [c["name"] for c in (ctx.get("contradictions") or [])]
        check("the engineered strong-fundamentals/bearish-technicals contradiction is detected and passed",
              "strong_fundamentals_bearish_technicals" in names, str(names))
        check("the bullish-stock/bearish-regime rule correctly does NOT fire (stock isn't bullish here)",
              "bullish_stock_bearish_regime" not in names, str(names))


def test_return_contract_is_unchanged():
    result, _ = _run_analyze()
    check("analyze_stock's return keys are exactly the pre-session contract",
          set(result.keys()) == EXPECTED_KEYS,
          f"unexpected={set(result.keys()) - EXPECTED_KEYS} missing={EXPECTED_KEYS - set(result.keys())}")


def test_regime_failure_degrades_to_none_not_a_broken_analysis():
    def _boom(*a, **k):
        raise RuntimeError("regime service exploded")

    result, ctx = _run_analyze(regime_side_effect=_boom)
    check("a regime failure degrades intelligence_context to None", ctx is None, str(ctx))
    check("the analysis still completes and returns the full contract",
          set(result.keys()) == EXPECTED_KEYS)
    check("the prediction is still present despite the regime failure",
          result.get("prediction") is not None)


def test_contradiction_failure_degrades_to_none_not_a_broken_analysis():
    def _boom(*a, **k):
        raise RuntimeError("contradiction engine exploded")

    result, ctx = _run_analyze(contradictions_side_effect=_boom)
    check("a contradiction-engine failure degrades intelligence_context to None", ctx is None, str(ctx))
    check("the analysis still completes and returns the full contract",
          set(result.keys()) == EXPECTED_KEYS)


def test_horizon_probabilities_are_attached_at_the_ledgers_own_horizons():
    """The forecast engine's default horizons (5/21/63/126/252) deliberately
    differ from the ledger's (1/5/20/60/126/252). Storing probabilities at
    21 and 63 would journal numbers the ledger can never score, so the
    orchestrator must forecast at the LEDGER's horizons - asserted here
    against the ledger's own constant, not a hardcoded copy of it."""
    from data.prediction_ledger import HORIZONS as LEDGER_HORIZONS

    captured = {}

    def _capture_log(rec):
        captured["horizon_probabilities"] = rec.get("horizon_probabilities")
        return 1

    fake_rec = {
        "ticker": "TEST", "current_price": 100.0, "action": "HOLD", "composite": 55,
        "pillars": {k: {"score": 60, "confidence": 0.8, "flags": []}
                    for k in ("technical", "algo", "fundamentals", "risk", "research", "social")},
        "levels": {"atr_14": 3.0, "stop_loss": 94.0, "target_price": 112.0},
        "confidence": {}, "conviction": "LOW", "time_horizon_days": 91,
    }

    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.fetch_price_history", return_value=FAKE_DF), \
         patch("agents.orchestrator.fetch_fundamentals", return_value={}), \
         patch("agents.orchestrator.fetch_earnings", return_value={}), \
         patch("agents.orchestrator.fetch_analyst_ratings", return_value={}), \
         patch("agents.orchestrator.compute_indicators", return_value={"current_price": 100.0}), \
         patch("agents.orchestrator.compute_signal_summary", return_value={"score": 50, "direction": "NEUTRAL"}), \
         patch("agents.orchestrator.compute_algo_signals", return_value={"algo_score": 55}), \
         patch("agents.orchestrator.fetch_reddit_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_stocktwits_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_web_forum_sentiment", return_value={}), \
         patch("agents.orchestrator.compute_pillar_scores", return_value={"pillars": FAKE_PILLARS}), \
         patch("agents.orchestrator._run_analysis_agents", return_value={
             "fundamentals_analysis": "f", "technical_analysis": "t",
             "social_analysis": "s", "algo_analysis": "a"}), \
         patch("agents.orchestrator.run_prediction_agent", return_value=dict(FAKE_PREDICTION)), \
         patch("agents.orchestrator.ground_prediction", side_effect=lambda *a, **k: dict(FAKE_PREDICTION)), \
         patch("intelligence.regime.compute_market_regime",
               return_value={"risk_stance": "RISK_ON", "trend": "BULLISH",
                             "volatility_regime": "LOW", "confidence": 0.9, "flags": []}), \
         patch("agents.recommendation.build_recommendation", return_value=fake_rec), \
         patch("agents.recommendation.log_composite_recommendation", side_effect=_capture_log), \
         patch("agents.fundamentals_pit.analyze_fundamentals_pit", return_value=None):
        from agents.orchestrator import analyze_stock
        analyze_stock("TEST", api_key="fake-key")

    probs = captured.get("horizon_probabilities")
    check("horizon_probabilities are attached to the rec before it is journaled",
          probs is not None and len(probs) > 0, str(probs))
    if probs:
        check("probability keys match the ledger's OWN horizons exactly (not the engine's defaults)",
              tuple(sorted(probs)) == tuple(sorted(LEDGER_HORIZONS)),
              f"got {tuple(sorted(probs))} vs ledger {tuple(sorted(LEDGER_HORIZONS))}")
        check("no probability is stored at horizon 21 or 63 (which the ledger never evaluates)",
              21 not in probs and 63 not in probs, str(sorted(probs)))
        check("every stored probability is a real number in (0,1)",
              all(isinstance(v, float) and 0.0 < v < 1.0 for v in probs.values()), str(probs))


def test_no_regime_and_no_contradictions_passes_nothing():
    """When the regime is genuinely unavailable AND no contradictions exist,
    there is nothing meaningful to tell the agent - the prompt must stay
    exactly as it was rather than carrying an empty scaffold."""
    empty_regime = {"risk_stance": None, "trend": None, "volatility_regime": None,
                    "confidence": 0.0, "flags": ["spy_unavailable"]}
    aligned_pillars = {k: {"score": 55, "confidence": 0.8, "flags": []}
                       for k in ("technical", "algo", "fundamentals", "risk", "research", "social")}

    captured = {}

    def _capture_prediction(*args, **kwargs):
        captured["ctx"] = kwargs.get("intelligence_context")
        return dict(FAKE_PREDICTION)

    with patch("agents.orchestrator._get_client", return_value=object()), \
         patch("agents.orchestrator.fetch_price_history", return_value=FAKE_DF), \
         patch("agents.orchestrator.fetch_fundamentals", return_value={}), \
         patch("agents.orchestrator.fetch_earnings", return_value={}), \
         patch("agents.orchestrator.fetch_analyst_ratings", return_value={}), \
         patch("agents.orchestrator.compute_indicators", return_value={"current_price": 100.0}), \
         patch("agents.orchestrator.compute_signal_summary", return_value={"score": 50, "direction": "NEUTRAL"}), \
         patch("agents.orchestrator.compute_algo_signals", return_value={"algo_score": 55}), \
         patch("agents.orchestrator.fetch_reddit_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_stocktwits_sentiment", return_value={}), \
         patch("agents.orchestrator.fetch_web_forum_sentiment", return_value={}), \
         patch("agents.orchestrator.compute_pillar_scores", return_value={"pillars": aligned_pillars}), \
         patch("agents.orchestrator._run_analysis_agents", return_value={
             "fundamentals_analysis": "f", "technical_analysis": "t",
             "social_analysis": "s", "algo_analysis": "a"}), \
         patch("agents.orchestrator.run_prediction_agent", side_effect=_capture_prediction), \
         patch("agents.orchestrator.ground_prediction", side_effect=lambda *a, **k: dict(FAKE_PREDICTION)), \
         patch("intelligence.regime.compute_market_regime", return_value=empty_regime):
        from agents.orchestrator import analyze_stock
        analyze_stock("TEST", api_key="fake-key")

    check("no regime + no contradictions -> intelligence_context stays None (prompt unchanged)",
          captured.get("ctx") is None, str(captured.get("ctx")))


if __name__ == "__main__":
    test_intelligence_context_actually_reaches_the_prediction_agent()
    test_return_contract_is_unchanged()
    test_regime_failure_degrades_to_none_not_a_broken_analysis()
    test_contradiction_failure_degrades_to_none_not_a_broken_analysis()
    test_horizon_probabilities_are_attached_at_the_ledgers_own_horizons()
    test_no_regime_and_no_contradictions_passes_nothing()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — intelligence context is genuinely wired into the live pipeline")
