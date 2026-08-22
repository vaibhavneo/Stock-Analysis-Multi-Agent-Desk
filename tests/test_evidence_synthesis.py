"""
Verification for intelligence/evidence_synthesis.py (items 3 + 4: evidence
ledger + contradiction engine).

Run: python3 tests/test_evidence_synthesis.py

Offline/deterministic, no network - pure function of the dicts it's handed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.evidence_synthesis import build_evidence_ledger, detect_contradictions

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def p(score, confidence=0.8):
    return {"score": score, "confidence": confidence, "flags": []}


def test_contradiction_1_strong_fundamentals_bearish_technicals():
    pillars = {"fundamentals": p(80), "technical": p(20), "algo": p(50), "social": p(50)}
    c = detect_contradictions(pillars)
    names = [x["name"] for x in c]
    check("strong_fundamentals_bearish_technicals detected", "strong_fundamentals_bearish_technicals" in names, str(c))
    match = next(x for x in c if x["name"] == "strong_fundamentals_bearish_technicals")
    check("severity is HIGH for an 80-vs-20 gap", match["severity"] == "HIGH")


def test_contradiction_2_bullish_sentiment_deteriorating_fundamentals():
    pillars = {"fundamentals": p(20), "technical": p(50), "algo": p(50), "social": p(80)}
    c = detect_contradictions(pillars)
    names = [x["name"] for x in c]
    check("bullish_sentiment_deteriorating_fundamentals detected",
          "bullish_sentiment_deteriorating_fundamentals" in names, str(c))


def test_contradiction_3_strong_momentum_excessive_valuation():
    pillars = {"fundamentals": p(20), "technical": p(50), "algo": p(80), "social": p(50)}
    c = detect_contradictions(pillars)
    names = [x["name"] for x in c]
    check("strong_momentum_excessive_valuation detected", "strong_momentum_excessive_valuation" in names, str(c))


def test_contradiction_4_bullish_stock_bearish_regime():
    pillars = {"fundamentals": p(50), "technical": p(75), "algo": p(75), "social": p(50)}
    regime = {"risk_stance": "RISK_OFF", "confidence": 0.9}
    c = detect_contradictions(pillars, regime=regime)
    names = [x["name"] for x in c]
    check("bullish_stock_bearish_regime detected", "bullish_stock_bearish_regime" in names, str(c))

    regime_on = {"risk_stance": "RISK_ON", "confidence": 0.9}
    c_on = detect_contradictions(pillars, regime=regime_on)
    check("no bullish_stock_bearish_regime when the regime is actually RISK_ON",
          "bullish_stock_bearish_regime" not in [x["name"] for x in c_on])


def test_no_contradictions_when_aligned():
    pillars = {"fundamentals": p(65), "technical": p(65), "algo": p(65), "social": p(65)}
    regime = {"risk_stance": "RISK_ON", "confidence": 0.9}
    c = detect_contradictions(pillars, regime=regime)
    check("a fully aligned bullish picture produces zero contradictions", c == [], str(c))

    pillars_bear = {"fundamentals": p(20), "technical": p(20), "algo": p(20), "social": p(20)}
    c2 = detect_contradictions(pillars_bear, regime={"risk_stance": "RISK_OFF", "confidence": 0.9})
    check("a fully aligned bearish picture also produces zero contradictions", c2 == [], str(c2))


def test_weighted_score_diverges_from_simple_average():
    # One high-reliability BEARISH signal + three low-reliability BULLISH
    # signals: the simple average leans bullish, but reliability-weighting
    # should correctly lean bearish, since the bearish signal is the only
    # one worth trusting.
    pillars = {
        "technical": {"score": 10, "confidence": 0.95, "flags": []},   # strong bearish, HIGH reliability
        "algo": {"score": 90, "confidence": 0.05, "flags": []},        # bullish, near-zero reliability
        "fundamentals": {"score": 90, "confidence": 0.05, "flags": []},
        "social": {"score": 90, "confidence": 0.05, "flags": []},
    }
    r = build_evidence_ledger("T", pillars)
    check("simple average of these 4 scores is bullish (>60)", r["simple_average_score"] > 60,
          f"simple_average={r['simple_average_score']}")
    check("reliability-weighted score is bearish (<40), the opposite of the simple average",
          r["weighted_score"] < 40, f"weighted={r['weighted_score']}")
    check("weighted_score and simple_average_score are meaningfully different, proving this is not just an average",
          abs(r["weighted_score"] - r["simple_average_score"]) > 20,
          f"weighted={r['weighted_score']} simple={r['simple_average_score']}")


def test_conviction_multiplier_reduced_by_contradictions():
    pillars_aligned = {"fundamentals": p(65), "technical": p(65), "algo": p(65), "social": p(65)}
    r_aligned = build_evidence_ledger("T", pillars_aligned, regime={"risk_stance": "RISK_ON", "confidence": 0.9})
    check("no contradictions -> full conviction multiplier (1.0)", r_aligned["conviction_multiplier"] == 1.0)

    pillars_conflicted = {"fundamentals": p(80), "technical": p(20), "algo": p(50), "social": p(80)}
    r_conflicted = build_evidence_ledger("T", pillars_conflicted)
    check("contradictions present -> conviction multiplier reduced below 1.0",
          r_conflicted["conviction_multiplier"] < 1.0, str(r_conflicted["conviction_multiplier"]))
    check("contradictions are listed explicitly, never hidden",
          len(r_conflicted["contradictions"]) >= 1)


def test_cost_basis_absent_without_risk_profile():
    pillars = {"fundamentals": p(65), "technical": p(65)}
    r = build_evidence_ledger("T", pillars)
    check("cost_basis key absent with no risk_profile", "cost_basis" not in r)

    r2 = build_evidence_ledger("T", pillars, risk_profile={"cost_basis": {
        "gain_loss_pct": -12.5, "underwater": True, "recovery_required_pct": 14.29}})
    check("cost_basis surfaced when risk_profile has one", r2.get("cost_basis", {}).get("gain_loss_pct") == -12.5)


def test_empty_pillars_is_honest():
    r = build_evidence_ledger("T", {})
    check("weighted_score is None with no evidence", r["weighted_score"] is None)
    check("evidence list is empty", r["evidence"] == [])
    check("flagged no_evidence_available", "no_evidence_available" in r["flags"])


def test_analog_evidence_folds_into_the_ledger():
    pillars = {"technical": p(55), "algo": p(55)}
    analog_bearish = {"status": "ok", "confidence": 0.5,
                       "outcome_by_horizon": {63: {"n": 10, "pct_positive": 0.1, "avg_return_pct": -12.0}}}
    r = build_evidence_ledger("T", pillars, analog_result=analog_bearish)
    sources = [e["source"] for e in r["evidence"]]
    check("historical_analogs appears as its own evidence line", "historical_analogs" in sources, str(sources))
    analog_item = next(e for e in r["evidence"] if e["source"] == "historical_analogs")
    check("a strongly bearish analog outcome (10% positive) reads as a bearish signal",
          analog_item["signal"] == "bearish", str(analog_item))

    r_insufficient = build_evidence_ledger("T", pillars,
                                            analog_result={"status": "insufficient_history", "confidence": 0.0})
    sources2 = [e["source"] for e in r_insufficient["evidence"]]
    check("an insufficient-history analog result contributes no evidence line (not fabricated)",
          "historical_analogs" not in sources2)


if __name__ == "__main__":
    test_contradiction_1_strong_fundamentals_bearish_technicals()
    test_contradiction_2_bullish_sentiment_deteriorating_fundamentals()
    test_contradiction_3_strong_momentum_excessive_valuation()
    test_contradiction_4_bullish_stock_bearish_regime()
    test_no_contradictions_when_aligned()
    test_weighted_score_diverges_from_simple_average()
    test_conviction_multiplier_reduced_by_contradictions()
    test_cost_basis_absent_without_risk_profile()
    test_empty_pillars_is_honest()
    test_analog_evidence_folds_into_the_ledger()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — evidence synthesis: reliability-weighted, contradictions named not hidden")
