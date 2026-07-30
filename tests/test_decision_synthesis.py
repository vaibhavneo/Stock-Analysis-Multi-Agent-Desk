#!/usr/bin/env python3
"""Tests for the DecisionSynthesis service — pure-logic verification on synthetic inputs."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.decision_synthesis import synthesize_decision

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
              stat_level="HIGH", dsr=0.7, sharpe=0.5, max_dd=0.15):
    return {
        "ticker": "TEST",
        "generated_at": "2026-07-22T12:00:00",
        "run_id": "test-run",
        "current_price": 100.0,
        "sector": "Technology",
        "regime": "MEDIUM",
        "data_asof": "2026-07-21",
        "benchmark": "SPY",
        "expected_return_pct": 10.0,
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
        "pillars": {
            "technical": {"score": 68, "confidence": 0.9, "backtestable": True, "flags": []},
            "algo": {"score": 72, "confidence": 0.9, "backtestable": True, "flags": []},
            "risk": {"score": 55, "confidence": 0.8, "backtestable": True, "flags": []},
            "fundamentals": {"score": 60, "confidence": 0.7, "backtestable": True,
                             "flags": ["pit_fundamentals_used"]},
            "research": {"score": 65, "confidence": 0.5, "backtestable": False, "flags": []},
            "social": {"score": 52, "confidence": 0.4, "backtestable": False,
                       "flags": ["no_social_data"]},
        },
        "levels": {
            "entry_zone_low": 98.0,
            "entry_zone_high": 101.0,
            "stop_loss": 94.0,
            "target_price": 112.0,
            "atr_14": 3.0,
            "formula": "test",
        },
        "position_size_pct": 4.5,
        "position_size_gated": gated,
        "raw_kelly_pct": 4.5,
        "time_horizon_days": 91,
        "backtest": {
            "strategy": "seven_pillar_core",
            "sharpe": sharpe,
            "dsr": dsr,
            "n_trials": 8,
            "n_trials_basis": "pre-registered",
            "max_drawdown": max_dd,
            "n_trades": 42,
            "cost_model": "CostModel",
            "total_cost_pct": 0.35,
            "core_signal_now": 1,
        },
        "hit_rate": {"total": 10, "correct": 6, "hit_rate": 0.6},
        "experiment_manifest_hash": "abc123",
        "honesty_flags": {
            "survivorship_safe": False,
            "pit_fundamentals": True,
            "fundamentals_source": "sec-edgar",
            "cost_model": "CostModel",
            "social_research_tracked_forward_only": True,
            "backtest_covers_core_only": True,
            "interaction_independent": True,
        },
        "claims": {"pillar_technical": "c1", "pillar_algo": "c2", "composite": "c3"},
        "thesis": {
            "summary": "Test thesis",
            "bull_case": "Test bull case",
            "bear_case": "Test bear case",
            "key_catalysts": "Earnings growth",
        },
        "disclaimer": "test disclaimer",
        "decision_fingerprint": "abc123def456",
    }


def test_basic_buy():
    """BUY when all gates pass."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="HIGH", gated=False)
    report = synthesize_decision("TEST", rec)
    check("final_action is BUY", report["final_action"] == "BUY")
    check("composite_score present", report["composite_score"] == 72)
    check("has ticker", report["ticker"] == "TEST")
    check("has generated_at", "generated_at" in report)
    check("has decision_fingerprint", len(report.get("decision_fingerprint", "")) == 16)
    check("has recommendation_fingerprint", report["recommendation_fingerprint"] == "abc123def456")
    check("has levels", "levels" in report and report["levels"]["target_price"] == 112.0)
    check("has confidence 4 dims", len(report["confidence"]) == 4)
    check("has agent_scores", "technical" in report["agent_scores"])
    check("has consensus", "agreeing" in report["consensus"])
    check("has backtest interp", "interpretation" in report["backtest"])
    check("has scenarios", "bull_case" in report["scenarios"])
    check("has evidence", "claim_ids" in report["evidence"])
    check("has warnings list", isinstance(report["warnings"], list))
    check("has honesty_flags", report["honesty_flags"]["pit_fundamentals"] is True)
    check("has disclaimer", "not financial advice" in report["disclaimer"])


def test_hold_when_edge_unproven():
    """HOLD when thesis is positive but statistical edge is LOW."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="LOW", gated=True, dsr=0.3)
    report = synthesize_decision("TEST", rec)
    check("HOLD when edge unproven", report["final_action"] == "HOLD")
    check("rationale mentions edge", "edge" in report["action_rationale"].lower()
          or "need HIGH" in report["action_rationale"])


def test_sell_when_negative():
    """SELL when thesis is negative and user owns."""
    rec = _mock_rec(action="SELL", composite=28)
    report = synthesize_decision("TEST", rec, owns_position=True)
    check("SELL when thesis negative (owns)", report["final_action"] == "SELL")


def test_avoid_when_no_position():
    """AVOID instead of SELL when user has no position."""
    rec = _mock_rec(action="SELL", composite=28)
    report = synthesize_decision("TEST", rec, owns_position=False)
    check("AVOID when no position", report["final_action"] == "AVOID")


def test_sell_becomes_sell_when_owns():
    """SELL stays SELL when user owns the position."""
    rec = _mock_rec(action="SELL", composite=28)
    report = synthesize_decision("TEST", rec, owns_position=True)
    check("SELL when owns position", report["final_action"] == "SELL")


def test_risk_veto_blocks_buy():
    """Risk veto prevents BUY even with positive composite."""
    rec = _mock_rec(action="BUY", composite=72, stat_level="HIGH",
                    gated=False, risk_veto=True)
    report = synthesize_decision("TEST", rec)
    check("risk veto blocks BUY", report["final_action"] in ("HOLD", "SELL", "AVOID"))
    check("risk veto in warnings", any("VETO" in w.upper() for w in report["warnings"]))


def test_consensus_counts():
    """Agent consensus counts agreeing/conflicting pillars."""
    rec = _mock_rec(action="BUY", composite=72)
    report = synthesize_decision("TEST", rec)
    cons = report["consensus"]
    check("agreeing + conflicting + neutral = 5",
          cons["n_agreeing"] + cons["n_conflicting"] + len(cons["neutral"]) == 5)
    check("has summary", len(cons["summary"]) > 0)


def test_backtest_interpretation():
    """Backtest interpretation is plain English."""
    rec = _mock_rec(dsr=0.3, sharpe=0.4)
    report = synthesize_decision("TEST", rec)
    interp = report["backtest"]["interpretation"]
    check("interpretation mentions dSR < 0.5", "0.5" in interp or "luck" in interp.lower())

    rec2 = _mock_rec(dsr=0.7, sharpe=0.6, max_dd=0.12)
    report2 = synthesize_decision("TEST", rec2)
    interp2 = report2["backtest"]["interpretation"]
    check("good backtest mentions credible edge", "credible" in interp2.lower() or "edge" in interp2.lower())


def test_calibration_insufficient():
    """Calibration shows INSUFFICIENT_HISTORY when no data."""
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec, calibration=None)
    check("calibration insufficient without data",
          report["calibration"]["status"] == "INSUFFICIENT_HISTORY")


def test_calibration_with_data():
    """Calibration interprets real data."""
    cal = {
        "overall": {"n": 50, "win_rate": 0.58, "avg_raw_return_pct": 1.5,
                    "avg_excess_return_pct": 0.3, "brier": 0.22},
        "calibration_error_ece": 0.08,
    }
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec, calibration=cal)
    check("calibration OK", report["calibration"]["status"] == "OK")
    check("win rate present", report["calibration"]["win_rate"] == 0.58)
    check("interpretation mentions win rate", "58%" in report["calibration"]["interpretation"]
          or "0.58" in report["calibration"]["interpretation"])
    check("calibration well-calibrated", "well-calibrated" in report["calibration"]["interpretation"])


def test_xsec_unavailable():
    """Cross-sectional returns UNAVAILABLE when not provided."""
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec, xsec_ranking=None)
    check("xsec unavailable", report["cross_sectional_rank"]["status"] == "UNAVAILABLE")


def test_xsec_with_ranking():
    """Cross-sectional ranking finds the ticker and interprets."""
    xsec = {
        "status": "OK",
        "as_of": "2026-06-30",
        "universe_id": "production-pilot",
        "survivorship_safe": False,
        "ranked": [
            {"ticker_as_of": "TEST", "rank": 5, "composite_percentile": 0.85,
             "composite_raw": 0.72, "rank_in_sector": 2,
             "factor_scores": {"quality": 0.7, "momentum": 0.8}},
        ] + [{"ticker_as_of": f"OTHER{i}", "rank": i+6, "composite_percentile": 0.5,
              "composite_raw": 0.5} for i in range(50)],
    }
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec, xsec_ranking=xsec)
    check("xsec OK", report["cross_sectional_rank"]["status"] == "OK")
    check("rank is 5", report["cross_sectional_rank"]["rank"] == 5)
    check("interpretation mentions top", "top" in report["cross_sectional_rank"]["interpretation"].lower())


def test_scenarios_have_content():
    """Scenarios include bull/base/bear + catalysts + risks."""
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec)
    s = report["scenarios"]
    check("has bull case", len(s["bull_case"]) > 0)
    check("has base case", len(s["base_case"]) > 0)
    check("has bear case", len(s["bear_case"]) > 0)
    check("catalysts list", isinstance(s["catalysts"], list))
    check("risks list", isinstance(s["risks"], list))
    check("thesis_breakers list", isinstance(s["thesis_breakers"], list))


def test_thesis_breakers_on_veto():
    """Risk veto and gated allocation produce thesis breakers."""
    rec = _mock_rec(risk_veto=True, gated=True, stat_level="LOW")
    report = synthesize_decision("TEST", rec)
    tb = report["scenarios"]["thesis_breakers"]
    check("risk veto is a thesis breaker", any("VETO" in t.upper() for t in tb))
    check("gated is a thesis breaker", any("EDGE" in t.upper() or "GATED" in t.upper() for t in tb))


def test_fingerprint_stable():
    """Same inputs produce the same fingerprint."""
    rec = _mock_rec()
    r1 = synthesize_decision("TEST", rec)
    r2 = synthesize_decision("TEST", rec)
    check("fingerprint stable", r1["decision_fingerprint"] == r2["decision_fingerprint"])


def test_warnings_include_survivorship():
    """Honesty warnings always mention survivorship when not safe."""
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec)
    check("survivorship warning", any("survivorship" in w.lower() for w in report["warnings"]))


def test_backtest_all_enrichment():
    """Backtest-all enriches the library summary."""
    rec = _mock_rec()
    bt_all = {
        "ticker": "TEST",
        "rows": [
            {"strategy": "sma_crossover", "dsr": 0.6, "sharpe": 0.5, "beats_hold": True},
            {"strategy": "momentum", "dsr": 0.3, "sharpe": 0.2, "beats_hold": False},
        ],
        "buy_hold": {"sharpe": 0.4},
    }
    report = synthesize_decision("TEST", rec, backtest_all=bt_all)
    ls = report["backtest"].get("library_summary")
    check("library_summary present", ls is not None)
    check("n_strategies is 2", ls["n_strategies"] == 2)
    check("beating_buy_hold is 1", ls["beating_buy_hold"] == 1)
    check("best_strategy is sma_crossover", ls["best_strategy"] == "sma_crossover")


def test_hold_composite():
    """HOLD action with mid-range composite."""
    rec = _mock_rec(action="HOLD", composite=52, stat_level="LOW", gated=True)
    report = synthesize_decision("TEST", rec)
    check("HOLD stays HOLD", report["final_action"] == "HOLD")


def test_all_required_keys():
    """The report has every key the mission specifies."""
    rec = _mock_rec()
    report = synthesize_decision("TEST", rec)
    required = [
        "final_action", "composite_score", "confidence", "agent_scores",
        "consensus", "backtest", "calibration", "cross_sectional_rank",
        "scenarios", "evidence", "warnings", "decision_fingerprint",
        "levels", "position_size_pct", "time_horizon_days",
    ]
    for key in required:
        check(f"has {key}", key in report)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()

    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
