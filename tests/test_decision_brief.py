#!/usr/bin/env python3
"""Tests for Decision Brief v2 — pure-logic verification on synthetic inputs."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.decision_brief import build_decision_brief

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


def _mock_rec(action="BUY", composite=72, risk_veto=False, gated=False,
              stat_level="HIGH", dsr=0.7, sharpe=0.5, max_dd=0.15,
              pillars=None):
    return {
        "ticker": "TEST",
        "generated_at": "2026-07-22T12:00:00",
        "current_price": 100.0,
        "sector": "Technology",
        "regime": "MEDIUM",
        "action": action,
        "conviction": stat_level,
        "composite": composite,
        "core_score": 65.0,
        "risk_multiplier": 0.85,
        "risk_veto": risk_veto,
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.6, "basis": "test"},
            "data": {"level": "MEDIUM", "score": 0.6, "basis": "test"},
            "statistical_edge": {
                "level": stat_level, "score": 0.8, "basis": "test",
                "checks": {
                    "min_sample": {"pass": True},
                    "net_cost_positive": {"pass": True},
                    "dsr": {"pass": dsr >= 0.5},
                    "walk_forward": {"pass": True},
                    "pbo": {"pass": True},
                },
            },
            "allocation": {"level": "HIGH" if not gated else "NONE",
                           "score": 0.8, "basis": "test"},
        },
        "pillars": pillars or {
            "technical": {"score": 68, "confidence": 0.9, "backtestable": True, "flags": []},
            "algo": {"score": 72, "confidence": 0.9, "backtestable": True, "flags": []},
            "risk": {"score": 55, "confidence": 0.8, "backtestable": True, "flags": []},
            "fundamentals": {"score": 60, "confidence": 0.7, "backtestable": True, "flags": []},
            "research": {"score": 65, "confidence": 0.5, "backtestable": False, "flags": []},
            "social": {"score": 52, "confidence": 0.4, "backtestable": False, "flags": []},
        },
        "levels": {
            "entry_zone_low": 98.0, "entry_zone_high": 101.0,
            "stop_loss": 94.0, "target_price": 112.0, "atr_14": 3.0, "formula": "test",
        },
        "position_size_pct": 4.5,
        "position_size_gated": gated,
        "raw_kelly_pct": 4.5,
        "time_horizon_days": 91,
        "backtest": {
            "strategy": "seven_pillar_core", "sharpe": sharpe, "dsr": dsr,
            "n_trials": 8, "n_trials_basis": "pre-registered", "max_drawdown": max_dd,
            "n_trades": 42, "cost_model": "CostModel", "total_cost_pct": 0.35,
            "core_signal_now": 1,
        },
        "hit_rate": {"total": 10, "correct": 6, "hit_rate": 0.6},
        "experiment_manifest_hash": "abc123",
        "honesty_flags": {
            "survivorship_safe": False, "pit_fundamentals": True,
            "fundamentals_source": "sec-edgar", "cost_model": "CostModel",
            "social_research_tracked_forward_only": True,
            "backtest_covers_core_only": True, "interaction_independent": True,
        },
        "claims": {"pillar_technical": "c1", "pillar_algo": "c2", "composite": "c3"},
        "thesis": {
            "summary": "Test thesis", "bull_case": "Test bull case",
            "bear_case": "Test bear case", "key_catalysts": "Earnings growth",
        },
        "disclaimer": "test disclaimer",
        "decision_fingerprint": "abc123def456",
    }


ACTIONS = {"BUY", "ACCUMULATE", "HOLD", "WATCH", "REDUCE", "AVOID"}


def test_owned_vs_not_owned_guidance():
    """Both guidance branches are always present, and the headline follows owns_position."""
    rec = _mock_rec(action="SELL", composite=28)
    brief = build_decision_brief("TEST", rec, owns_position=True)
    check("owned branch present", brief["guidance"]["if_owned"]["action"] in ACTIONS)
    check("not-owned branch present", brief["guidance"]["if_not_owned"]["action"] in ACTIONS)
    check("owned headline is REDUCE", brief["final_action"] == "REDUCE")
    check("not-owned action is AVOID (never SELL)",
          brief["guidance"]["if_not_owned"]["action"] == "AVOID")

    brief2 = build_decision_brief("TEST", rec, owns_position=False)
    check("headline follows owns_position=False", brief2["final_action"] == "AVOID")

    brief3 = build_decision_brief("TEST", rec, owns_position=None)
    check("unknown ownership treated as not-owned (AVOID, never SELL)",
          brief3["final_action"] == "AVOID")
    check("no SELL anywhere in vocabulary", "SELL" not in (
        brief3["final_action"], brief3["guidance"]["if_owned"]["action"],
        brief3["guidance"]["if_not_owned"]["action"]))


def test_strong_company_but_hold():
    """Thesis positive (BUY from the underlying engine) but statistical edge unproven
    -> neutral tier -> HOLD (owned) / WATCH (not owned), never BUY."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="LOW", gated=True, dsr=0.2)
    brief_owned = build_decision_brief("TEST", rec, owns_position=True)
    brief_not_owned = build_decision_brief("TEST", rec, owns_position=False)
    check("owned -> HOLD", brief_owned["final_action"] == "HOLD")
    check("not owned -> WATCH", brief_not_owned["final_action"] == "WATCH")
    check("no price entry shown for WATCH", brief_not_owned["action_plan"]["entry_low"] is None)
    check("max exposure marked unsupported",
          "Unsupported" in brief_not_owned["action_plan"]["max_exposure_note"])


def test_insufficient_statistical_history():
    """No calibration data -> calibration maturity says insufficient history, and this
    shows up as one of the three decisive insights."""
    rec = _mock_rec(action="BUY", composite=75, stat_level="HIGH", gated=False, dsr=0.7)
    brief = build_decision_brief("TEST", rec, calibration=None)
    check("calibration maturity is insufficient history",
          brief["evidence_status"]["calibration_maturity"] == "insufficient history")
    check("insufficient history appears in an insight",
          any("track record" in i.lower() or "measured" in i.lower()
              for i in brief["decisive_insights"]))

    rec_missing = {"current_price": None, "action": None, "composite": None,
                   "confidence": None, "levels": None}
    brief_missing = build_decision_brief("TEST", rec_missing)
    check("missing required inputs -> INSUFFICIENT_EVIDENCE",
          brief_missing["final_action"] == "INSUFFICIENT_EVIDENCE")
    check("missing_inputs lists all five",
          len(brief_missing["missing_inputs"]) == 5)


def test_risk_veto():
    """Risk veto blocks BUY/ACCUMULATE even with a strong composite, appears in
    insights and triggers, and the guidance reflects it."""
    rec = _mock_rec(action="BUY", composite=80, stat_level="HIGH", gated=False, risk_veto=True)
    brief = build_decision_brief("TEST", rec, owns_position=True)
    check("risk veto blocks BUY", brief["final_action"] not in ("BUY", "ACCUMULATE"))
    check("risk veto mentioned in verdict", "veto" in brief["one_line_verdict"].lower())
    check("risk veto in an insight",
          any("veto" in i.lower() for i in brief["decisive_insights"]))
    check("risk veto clearing is an upgrade trigger",
          any("veto" in t.lower() for t in brief["triggers"]["upgrade"]))


def test_contradictory_agents():
    """Conflicting pillar scores surface as a decisive insight and are visible in
    the underlying consensus, without silently averaging them into one number."""
    conflicting_pillars = {
        "technical":    {"score": 82, "confidence": 0.9, "backtestable": True, "flags": []},
        "algo":         {"score": 78, "confidence": 0.9, "backtestable": True, "flags": []},
        "risk":         {"score": 20, "confidence": 0.8, "backtestable": True, "flags": []},
        "fundamentals": {"score": 25, "confidence": 0.7, "backtestable": True, "flags": []},
        "research":     {"score": 18, "confidence": 0.5, "backtestable": False, "flags": []},
        "social":       {"score": 52, "confidence": 0.4, "backtestable": False, "flags": []},
    }
    rec = _mock_rec(action="BUY", composite=60, stat_level="MEDIUM",
                     pillars=conflicting_pillars)
    brief = build_decision_brief("TEST", rec)
    check("consensus reflects conflict",
          any("mixed" in i.lower() or "conflict" in i.lower()
              for i in brief["decisive_insights"]))


def test_claim_deduplication():
    """Risk veto + zero-allocation gating both trace to 'no proven edge' — the three
    insights must not repeat the same underlying claim under two different labels."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="LOW", gated=True,
                     risk_veto=True, dsr=0.1)
    brief = build_decision_brief("TEST", rec)
    check("exactly 3 insights", len(brief["decisive_insights"]) == 3)
    check("insights are unique strings",
          len(set(brief["decisive_insights"])) == len(brief["decisive_insights"]))


def test_deterministic_action_invariance():
    """Calling the brief twice on identical inputs yields the identical action and
    fingerprint — no hidden randomness or LLM call in the decision path."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="HIGH", gated=False)
    b1 = build_decision_brief("TEST", rec, owns_position=True)
    b2 = build_decision_brief("TEST", rec, owns_position=True)
    check("same final_action", b1["final_action"] == b2["final_action"])
    check("same fingerprint", b1["brief_fingerprint"] == b2["brief_fingerprint"])
    check("same insights", b1["decisive_insights"] == b2["decisive_insights"])
    check("same action plan", b1["action_plan"] == b2["action_plan"])


def test_basic_buy_all_fields():
    rec = _mock_rec(action="BUY", composite=80, stat_level="HIGH", gated=False, dsr=0.7)
    brief = build_decision_brief("TEST", rec, owns_position=False)
    check("final_action is BUY", brief["final_action"] == "BUY")
    check("has one_line_verdict", len(brief["one_line_verdict"]) > 0)
    check("has exactly 3 insights", len(brief["decisive_insights"]) == 3)
    check("has action_plan", "instruction" in brief["action_plan"])
    check("entry shown for BUY", brief["action_plan"]["entry_low"] == 98.0)
    check("max exposure supported", brief["action_plan"]["max_exposure_pct"] == 4.5)
    check("has triggers", "upgrade" in brief["triggers"] and "downgrade" in brief["triggers"])
    check("has evidence_status 5 keys", len(brief["evidence_status"]) == 5)
    check("has alternatives", "status" in brief["alternatives"])
    check("has evidence_links", "claim_ids" in brief["evidence_links"])
    check("has fingerprint", len(brief["brief_fingerprint"]) == 16)
    check("has recommendation_fingerprint", brief["recommendation_fingerprint"] == "abc123def456")


def test_alternatives_from_xsec():
    rec = _mock_rec(action="HOLD", composite=55, stat_level="MEDIUM")
    xsec = {
        "status": "OK", "as_of": "2026-06-30", "universe_id": "production-pilot",
        "survivorship_safe": False,
        "ranked": [
            {"ticker_as_of": "BETTER1", "rank": 1, "composite_percentile": 0.98},
            {"ticker_as_of": "BETTER2", "rank": 2, "composite_percentile": 0.95},
            {"ticker_as_of": "TEST", "rank": 5, "composite_percentile": 0.80},
        ],
    }
    brief = build_decision_brief("TEST", rec, xsec_ranking=xsec)
    check("alternatives status OK", brief["alternatives"]["status"] == "OK")
    check("two better-ranked names found", len(brief["alternatives"]["items"]) == 2)
    check("best alternative first", brief["alternatives"]["items"][0]["ticker"] == "BETTER1")


def test_position_context_computed_from_avg_cost():
    """Supplying a position (avg cost) produces a position_context with the
    correct unrealized P&L, and implies ownership without an explicit flag."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="HIGH", gated=False)
    rec["current_price"] = 120.0
    brief = build_decision_brief("TEST", rec, position={"avg_cost": 100.0, "shares": 10})
    pc = brief["position_context"]
    check("avg_cost recorded", pc["avg_cost"] == 100.0)
    check("unrealized pl pct is +20%", pc["unrealized_pl_pct"] == 20.0)
    check("unrealized pl dollar is +200", pc["unrealized_pl_dollar"] == 200.0)
    check("ownership implied by position -> owned branch used",
          brief["final_action"] == brief["guidance"]["if_owned"]["action"])


def test_position_aware_profit_taking_overrides_hold_to_reduce():
    """The architecture's core claim: objective verdict (HOLD, neutral tier) +
    user position context (large unrealized gain) -> a DIFFERENT final decision
    (REDUCE) than the position-agnostic signal alone would give."""
    rec = _mock_rec(action="BUY", composite=60, stat_level="LOW", gated=True, dsr=0.2)
    rec["current_price"] = 130.0
    objective_brief = build_decision_brief("TEST", rec, owns_position=True)
    check("objective (no position) verdict is HOLD", objective_brief["final_action"] == "HOLD")

    position_brief = build_decision_brief(
        "TEST", rec, position={"avg_cost": 100.0, "shares": 5})
    check("position-aware verdict overrides to REDUCE",
          position_brief["final_action"] == "REDUCE")
    check("objective_action preserved for transparency",
          position_brief["objective_action"] == "HOLD")
    check("one_line_verdict cites the unrealized gain",
          "30.0%" in position_brief["one_line_verdict"]
          or "+30.0%" in position_brief["one_line_verdict"])
    check("position_note explains the profit-taking override",
          "lock in profit" in position_brief["guidance"]["if_owned"]["position_note"].lower())


def test_position_aware_averaging_down_note_on_buy():
    """A positive (BUY/ACCUMULATE) signal keeps its action when the user is
    deeply underwater, but the note explains the averaging-down context —
    the action doesn't flip, only the framing does."""
    rec = _mock_rec(action="BUY", composite=80, stat_level="HIGH", gated=False)
    rec["current_price"] = 80.0
    brief = build_decision_brief("TEST", rec, position={"avg_cost": 100.0, "shares": 10})
    check("action stays BUY while underwater", brief["final_action"] == "BUY")
    check("note mentions averaging down",
          "underwater" in brief["guidance"]["if_owned"]["position_note"].lower())
    check("unrealized pl is -20%",
          brief["position_context"]["unrealized_pl_pct"] == -20.0)


def test_position_aware_reduce_note_reflects_gain_vs_loss():
    """A negative (REDUCE) signal's note differs depending on whether the user
    is still in profit or already at a loss — same action, different urgency."""
    rec_negative = _mock_rec(action="SELL", composite=28)

    rec_profit = dict(rec_negative)
    rec_profit["current_price"] = 120.0
    profit_brief = build_decision_brief(
        "TEST", rec_profit, position={"avg_cost": 100.0, "shares": 10})
    check("REDUCE while in profit -> protect gains note",
          "protect" in profit_brief["guidance"]["if_owned"]["position_note"].lower()
          or "profit" in profit_brief["guidance"]["if_owned"]["position_note"].lower())

    rec_loss = dict(rec_negative)
    rec_loss["current_price"] = 80.0
    loss_brief = build_decision_brief(
        "TEST", rec_loss, position={"avg_cost": 100.0, "shares": 10})
    check("REDUCE while at a loss -> cut losses note",
          "loss" in loss_brief["guidance"]["if_owned"]["position_note"].lower())
    check("both stay REDUCE (owned, negative tier)",
          profit_brief["final_action"] == "REDUCE" and loss_brief["final_action"] == "REDUCE")


def test_no_position_matches_prior_objective_behavior():
    """Regression guard: without a `position` argument, behavior is identical
    to the pre-position-awareness implementation (no position_context, no note,
    objective_action == final_action for the owned branch)."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="HIGH", gated=False)
    brief = build_decision_brief("TEST", rec, owns_position=True)
    check("no position_context", brief["position_context"] is None)
    check("no position_note", brief["guidance"]["if_owned"]["position_note"] is None)
    check("objective_action equals final_action with no position override",
          brief["objective_action"] == brief["final_action"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()

    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
