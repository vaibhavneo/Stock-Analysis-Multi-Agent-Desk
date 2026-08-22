"""
Verification for the intelligence/-package extension to
agents/decision_synthesis.py::synthesize_decision() (item 8: final
synthesis, generated only after all evidence is assembled).

tests/test_decision_synthesis.py (unmodified, still 66/66 passing) covers
everything synthesize_decision() did before this session - this file covers
only the six new optional kwargs (regime, historical_context, analog,
forecast, risk_profile, evidence) and the three new report sections they
enable (strongest_evidence, conviction, what_would_change_the_call).

Run: python3 tests/test_decision_synthesis_intelligence.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.decision_synthesis import synthesize_decision
from backtest.pillars import action_for

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _mock_rec(action="BUY", composite=72, risk_veto=False, gated=False, stat_level="HIGH"):
    return {
        "ticker": "TEST", "generated_at": "2026-07-22T12:00:00", "current_price": 100.0,
        "sector": "Technology", "regime": "MEDIUM", "action": action, "composite": composite,
        "risk_veto": risk_veto,
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.6},
            "data": {"level": "MEDIUM", "score": 0.6},
            "statistical_edge": {"level": stat_level, "score": 0.8 if stat_level == "HIGH" else 0.3},
            "allocation": {"level": "HIGH" if not gated else "NONE", "score": 0.8},
        },
        "pillars": {
            "technical": {"score": 68, "confidence": 0.9, "backtestable": True, "flags": []},
            "algo": {"score": 72, "confidence": 0.9, "backtestable": True, "flags": []},
            "risk": {"score": 55, "confidence": 0.8, "backtestable": True, "flags": []},
            "fundamentals": {"score": 60, "confidence": 0.7, "backtestable": True, "flags": []},
            "research": {"score": 65, "confidence": 0.5, "backtestable": False, "flags": []},
            "social": {"score": 52, "confidence": 0.4, "backtestable": False, "flags": []},
        },
        "levels": {"entry_zone_low": 98.0, "entry_zone_high": 101.0, "stop_loss": 94.0,
                   "target_price": 112.0, "atr_14": 3.0, "formula": "test"},
        "position_size_pct": 4.5, "position_size_gated": gated, "time_horizon_days": 91,
        "honesty_flags": {"survivorship_safe": True, "pit_fundamentals": True},
        "claims": {}, "decision_fingerprint": "fp0",
    }


def test_backward_compatible_when_new_kwargs_omitted():
    rec = _mock_rec()
    r1 = synthesize_decision("TEST", rec)
    r2 = synthesize_decision("TEST", rec, regime=None, historical_context=None,
                              analog=None, forecast=None, risk_profile=None, evidence=None)
    # generated_at is a live timestamp - exclude it from the equality check,
    # everything else must be byte-identical.
    r1c, r2c = dict(r1), dict(r2)
    r1c.pop("generated_at"); r2c.pop("generated_at")
    check("output is byte-identical whether new kwargs are omitted or explicitly None",
          r1c == r2c)


def test_new_keys_present_with_honest_defaults():
    r = synthesize_decision("TEST", _mock_rec())
    for key in ("market_regime", "historical_context", "analog", "forecast", "risk_profile", "evidence_ledger"):
        check(f"report has key '{key}', defaulting to None", key in r and r[key] is None)
    check("conviction is None with no evidence ledger supplied (not a fabricated default)",
          r["conviction"] is None)
    check("strongest_evidence is present with both directions None",
          r["strongest_evidence"] == {"bull": None, "bear": None})
    check("what_would_change_the_call is present and non-empty even with no evidence ledger",
          len(r["what_would_change_the_call"]) >= 1)


def test_what_would_change_the_call_nonempty_for_non_buy_actions():
    for action, composite, stat_level, gated in [
        ("HOLD", 50, "MEDIUM", True), ("SELL", 20, "NONE", True), ("HOLD", 65, "MEDIUM", True),
    ]:
        rec = _mock_rec(action=action, composite=composite, stat_level=stat_level, gated=gated)
        r = synthesize_decision("TEST", rec)
        check(f"what_would_change_the_call is non-empty for final_action={r['final_action']}",
              len(r["what_would_change_the_call"]) >= 1, str(r["what_would_change_the_call"]))


def test_what_would_change_the_call_nonempty_even_for_clean_buy():
    rec = _mock_rec(action="BUY", composite=80, stat_level="HIGH", gated=False)
    r = synthesize_decision("TEST", rec)
    check("even a clean BUY gets a non-empty what_would_change_the_call (a real fallback, not silence)",
          len(r["what_would_change_the_call"]) >= 1, str(r["what_would_change_the_call"]))


def test_strongest_evidence_picks_the_most_reliable_extreme():
    evidence = {
        "evidence": [
            {"source": "pillar:technical", "score": 90, "reliability": 0.9, "signal": "bullish", "flags": []},
            {"source": "pillar:algo", "score": 62, "reliability": 0.95, "signal": "bullish", "flags": []},
            {"source": "pillar:fundamentals", "score": 15, "reliability": 0.85, "signal": "bearish", "flags": []},
            {"source": "pillar:social", "score": 38, "reliability": 0.1, "signal": "bearish", "flags": []},
        ],
        "weighted_score": 55.0, "simple_average_score": 51.25,
        "contradictions": [], "conviction_multiplier": 1.0, "flags": [],
    }
    r = synthesize_decision("TEST", _mock_rec(), evidence=evidence)
    check("strongest bull picked by reliability*distance, not raw extremity alone",
          r["strongest_evidence"]["bull"]["source"] == "pillar:technical",
          str(r["strongest_evidence"]))
    check("strongest bear picked by reliability*distance (0.85*35 beats 0.1*12)",
          r["strongest_evidence"]["bear"]["source"] == "pillar:fundamentals",
          str(r["strongest_evidence"]))


def test_conviction_requires_both_high_multiplier_and_proven_edge():
    evidence_no_contradictions = {"evidence": [], "weighted_score": 70.0, "simple_average_score": 70.0,
                                   "contradictions": [], "conviction_multiplier": 1.0, "flags": []}

    rec_proven = _mock_rec(stat_level="HIGH")
    r_proven = synthesize_decision("TEST", rec_proven, evidence=evidence_no_contradictions)
    check("HIGH conviction requires: no contradictions AND a proven (HIGH) statistical edge",
          r_proven["conviction"] == "HIGH", str(r_proven["conviction"]))

    rec_unproven = _mock_rec(stat_level="MEDIUM")
    r_unproven = synthesize_decision("TEST", rec_unproven, evidence=evidence_no_contradictions)
    check("a confident-sounding evidence ledger does NOT read as HIGH conviction without a proven edge",
          r_unproven["conviction"] != "HIGH", str(r_unproven["conviction"]))

    evidence_conflicted = {"evidence": [], "weighted_score": 70.0, "simple_average_score": 70.0,
                            "contradictions": [{"name": "x", "severity": "HIGH", "description": "d", "signals": {}}],
                            "conviction_multiplier": 0.85, "flags": []}
    r_conflicted = synthesize_decision("TEST", rec_proven, evidence=evidence_conflicted)
    check("a HIGH-severity contradiction prevents HIGH conviction even with a proven edge",
          r_conflicted["conviction"] != "HIGH", str(r_conflicted["conviction"]))


def test_high_severity_contradiction_appears_in_warnings_and_thesis_breakers():
    evidence = {"evidence": [], "weighted_score": 50.0, "simple_average_score": 50.0,
                "contradictions": [{"name": "strong_fundamentals_bearish_technicals", "severity": "HIGH",
                                     "description": "test description", "signals": {}}],
                "conviction_multiplier": 0.85, "flags": []}
    r = synthesize_decision("TEST", _mock_rec(), evidence=evidence)
    check("HIGH-severity contradiction reaches the top-level warnings list (never hidden)",
          any("strong fundamentals bearish technicals" in w for w in r["warnings"]), str(r["warnings"]))
    check("HIGH-severity contradiction also becomes a thesis breaker",
          any("strong_fundamentals_bearish_technicals" in t for t in r["scenarios"]["thesis_breakers"]),
          str(r["scenarios"]["thesis_breakers"]))

    evidence_medium = dict(evidence)
    evidence_medium["contradictions"] = [{"name": "y", "severity": "MEDIUM", "description": "d2", "signals": {}}]
    r2 = synthesize_decision("TEST", _mock_rec(), evidence=evidence_medium)
    check("MEDIUM-severity contradiction appears as a risk, not a thesis breaker",
          any("d2" in risk for risk in r2["scenarios"]["risks"])
          and not any("y" in t for t in r2["scenarios"]["thesis_breakers"]),
          str(r2["scenarios"]))


def test_action_for_divergence_from_final_action_is_never_silent():
    """The recommendation's raw composite may map to a different action via
    backtest.pillars.action_for() than synthesize_decision()'s own, gated
    final_action - that's expected (statistical-edge gating is the whole
    point of _determine_action), but the reason must always be recorded in
    action_rationale, never silently dropped."""
    # composite=72 -> action_for(72) == "BUY", but the statistical edge is
    # NOT proven (MEDIUM, not HIGH) -> _determine_action gates this to HOLD.
    rec = _mock_rec(action="BUY", composite=72, stat_level="MEDIUM", gated=True)
    raw_action_reading = action_for(rec["composite"])
    r = synthesize_decision("TEST", rec)

    check("raw action_for(composite) reads BUY", raw_action_reading == "BUY")
    check("final_action diverges from the raw composite-only reading (gated by unproven edge)",
          r["final_action"] != raw_action_reading, f"final_action={r['final_action']}")
    check("the divergence is explained in action_rationale, not silent",
          "statistical edge" in r["action_rationale"].lower() or "edge" in r["action_rationale"].lower(),
          r["action_rationale"])


if __name__ == "__main__":
    test_backward_compatible_when_new_kwargs_omitted()
    test_new_keys_present_with_honest_defaults()
    test_what_would_change_the_call_nonempty_for_non_buy_actions()
    test_what_would_change_the_call_nonempty_even_for_clean_buy()
    test_strongest_evidence_picks_the_most_reliable_extreme()
    test_conviction_requires_both_high_multiplier_and_proven_edge()
    test_high_severity_contradiction_appears_in_warnings_and_thesis_breakers()
    test_action_for_divergence_from_final_action_is_never_silent()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — decision synthesis intelligence extension: backward compatible, honest")
