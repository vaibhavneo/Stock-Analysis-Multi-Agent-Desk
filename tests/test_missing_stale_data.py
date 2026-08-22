"""
Cross-cutting verification for item 9's honesty requirement: "Never
fabricate missing data. Reduce confidence explicitly when data is
unavailable or stale."

Every intelligence/ module is fed deliberately incomplete input here. The
bar each must clear is the same three things:
  1. No exception - a missing input degrades, it doesn't crash the request.
  2. Confidence explicitly reduced (and 0.0 when nothing could be computed).
  3. A NAMED flag, or an explicit data_available: False / None - never a
     plausible-looking number invented to fill the gap.

The individual module test files each cover their own degradation paths;
this file exists to catch the case where a NEW module gets added later
without the honesty contract, by testing all of them against one shared
standard in one place.

Run: python3 tests/test_missing_stale_data.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from intelligence.analog_engine import find_historical_analogs
from intelligence.common import is_stale
from intelligence.evidence_synthesis import build_evidence_ledger
from intelligence.historical_context import compute_historical_context
from intelligence.prediction_engine import forecast_horizons
from intelligence.regime import compute_market_regime
from intelligence.risk_engine import compute_risk_profile

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


EMPTY_DF = pd.DataFrame()
EMPTY_OHLCV = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _tiny_df(n=50):
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                          "Close": close, "Volume": pd.Series(1e6, index=idx)})


def _assert_honest(label, result, expect_zero_confidence=True):
    """The shared standard every module must meet on missing input."""
    conf = result.get("confidence")
    check(f"{label}: returns a dict with an explicit confidence", conf is not None, str(result)[:120])
    if conf is not None and expect_zero_confidence:
        check(f"{label}: confidence is exactly 0.0 (nothing computable)", conf == 0.0, f"confidence={conf}")
    flags = result.get("flags")
    check(f"{label}: carries at least one named flag explaining the gap",
          isinstance(flags, list) and len(flags) > 0, str(flags))


def test_regime_with_everything_unavailable():
    with patch("intelligence.regime.get_bars_df", return_value=EMPTY_OHLCV), \
         patch("intelligence.regime.get", return_value={"data": []}):
        try:
            r = compute_market_regime()
            raised = False
        except Exception as e:
            r, raised = {}, e
    check("regime: no exception when SPY/QQQ/VIX are all unavailable", raised is False, str(raised))
    if not raised:
        _assert_honest("regime", r)
        for field in ("trend", "volatility_regime", "vix_level", "risk_stance"):
            check(f"regime: {field} is None, not a fabricated value", r.get(field) is None)


def test_historical_context_with_empty_and_tiny_history():
    try:
        r_empty = compute_historical_context("T", EMPTY_DF, algo_signals={}, indicators={})
        raised = False
    except Exception as e:
        r_empty, raised = {}, e
    check("historical_context: no exception on an empty df", raised is False, str(raised))
    if not raised:
        _assert_honest("historical_context (empty)", r_empty)
        check("historical_context (empty): horizons dict is empty, not filled with zeros",
              r_empty.get("horizons") == {})

    r_tiny = compute_historical_context("T", _tiny_df(50), algo_signals={}, indicators={})
    check("historical_context (50 bars): no exception", True)
    check("historical_context (50 bars): long horizons marked data_available False, not fabricated",
          r_tiny["horizons"]["1Y"] == {"data_available": False}
          and r_tiny["horizons"]["5Y"] == {"data_available": False})
    check("historical_context (50 bars): confidence reduced below 1.0", r_tiny["confidence"] < 1.0,
          str(r_tiny["confidence"]))


def test_analog_with_empty_and_insufficient_history():
    try:
        r_empty = find_historical_analogs("T", EMPTY_DF)
        raised = False
    except Exception as e:
        r_empty, raised = {}, e
    check("analog: no exception on an empty df", raised is False, str(raised))
    if not raised:
        _assert_honest("analog (empty)", r_empty)
        check("analog (empty): status is insufficient_history", r_empty.get("status") == "insufficient_history")
        check("analog (empty): matches list is empty, nothing invented", r_empty.get("matches") == [])

    r_tiny = find_historical_analogs("T", _tiny_df(200))
    check("analog (200 bars, under the 3y minimum): status insufficient_history",
          r_tiny["status"] == "insufficient_history")
    check("analog (200 bars): confidence exactly 0.0", r_tiny["confidence"] == 0.0)
    check("analog (200 bars): no fabricated matches", r_tiny["matches"] == [])


def test_prediction_engine_with_no_pillars_and_no_price():
    try:
        r_no_pillars = forecast_horizons("T", 100.0, {}, {})
        r_no_price = forecast_horizons("T", None, {"technical": {"score": 70, "confidence": 0.9}}, {})
        raised = False
    except Exception as e:
        r_no_pillars = r_no_price = {}
        raised = e
    check("prediction_engine: no exception with empty pillars or a missing price", raised is False, str(raised))
    if not raised:
        _assert_honest("prediction_engine (no pillars)", r_no_pillars)
        _assert_honest("prediction_engine (no price)", r_no_price)
        check("prediction_engine (no pillars): horizons empty, no fabricated forecast",
              r_no_pillars.get("horizons") == {})


def test_risk_engine_with_no_price_and_no_levels():
    try:
        r_no_price = compute_risk_profile("T", None, None, {})
        r_no_levels = compute_risk_profile("T", 100.0, None, {})
        raised = False
    except Exception as e:
        r_no_price = r_no_levels = {}
        raised = e
    check("risk_engine: no exception with a missing price or missing levels", raised is False, str(raised))
    if not raised:
        _assert_honest("risk_engine (no price)", r_no_price)
        _assert_honest("risk_engine (no levels)", r_no_levels, expect_zero_confidence=False)
        check("risk_engine (no levels): confidence reduced but non-zero (other blocks still computed)",
              0.0 < r_no_levels["confidence"] < 1.0, str(r_no_levels.get("confidence")))
        check("risk_engine: cost_basis key absent entirely when no position supplied",
              "cost_basis" not in r_no_levels)


def test_evidence_synthesis_with_no_evidence_at_all():
    try:
        r = build_evidence_ledger("T", {})
        raised = False
    except Exception as e:
        r, raised = {}, e
    check("evidence_synthesis: no exception with zero pillars", raised is False, str(raised))
    if not raised:
        check("evidence_synthesis: flags the absence explicitly",
              "no_evidence_available" in (r.get("flags") or []), str(r.get("flags")))
        check("evidence_synthesis: weighted_score is None, not a fabricated 50",
              r.get("weighted_score") is None)
        check("evidence_synthesis: evidence list is empty", r.get("evidence") == [])


def test_pillars_with_none_scores_are_skipped_not_defaulted():
    """A pillar that exists but has score=None (its data source failed) must
    be EXCLUDED from the weighted evidence, not silently treated as 50."""
    pillars = {
        "technical": {"score": None, "confidence": 0.9, "flags": ["no_technical_data"]},
        "algo": {"score": 80, "confidence": 0.9, "flags": []},
    }
    r = build_evidence_ledger("T", pillars)
    sources = [e["source"] for e in r["evidence"]]
    check("a score=None pillar is excluded from the evidence ledger, not defaulted to 50",
          "pillar:technical" not in sources and "pillar:algo" in sources, str(sources))
    check("the weighted score reflects only the pillar that actually had data",
          r["weighted_score"] == 80.0, str(r["weighted_score"]))


def test_zero_confidence_pillar_does_not_poison_the_weighted_score():
    """A pillar with confidence 0.0 contributes zero weight - it must not
    drag the weighted score toward its value, and must not cause a
    divide-by-zero when it's the only pillar present."""
    only_zero_conf = {"social": {"score": 90, "confidence": 0.0, "flags": ["no_social_data"]}}
    try:
        r = build_evidence_ledger("T", only_zero_conf)
        raised = False
    except Exception as e:
        r, raised = {}, e
    check("a sole zero-confidence pillar doesn't cause a divide-by-zero", raised is False, str(raised))
    if not raised:
        check("weighted_score is None when total reliability weight is zero",
              r["weighted_score"] is None, str(r["weighted_score"]))
        check("flagged no_reliable_evidence", "no_reliable_evidence" in r["flags"], str(r["flags"]))


def test_staleness_helper_treats_unparseable_as_stale():
    for bad in (None, "", "garbage", "2024-13-45"):
        check(f"is_stale({bad!r}) is True - unparseable timestamps are stale, never assumed fresh",
              is_stale(bad, max_age_days=30) is True)


if __name__ == "__main__":
    test_regime_with_everything_unavailable()
    test_historical_context_with_empty_and_tiny_history()
    test_analog_with_empty_and_insufficient_history()
    test_prediction_engine_with_no_pillars_and_no_price()
    test_risk_engine_with_no_price_and_no_levels()
    test_evidence_synthesis_with_no_evidence_at_all()
    test_pillars_with_none_scores_are_skipped_not_defaulted()
    test_zero_confidence_pillar_does_not_poison_the_weighted_score()
    test_staleness_helper_treats_unparseable_as_stale()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — every intelligence module degrades honestly, nothing fabricated")
