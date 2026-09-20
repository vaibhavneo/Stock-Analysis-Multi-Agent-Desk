#!/usr/bin/env python3
"""Adversarial tests for the Decision Intelligence layer (Phase 25).

Thirty named scenarios, each one a configuration where a plausible
implementation gets the answer wrong in a way that would matter. They run on
synthetic inputs so every case is exactly reproducible and no network is
touched.

The mutation tests at the bottom are the more important half. A test that
passes on correct code proves nothing about whether the code is *consulted* —
these deliberately break a piece of evidence and assert that the decision
actually changes. That is the only way to keep "computed but never consulted"
from creeping back in, which is the failure mode this whole layer was built to
correct.
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decision.engine import build_decision_intelligence

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    # Fails the process, not just the transcript.
    assert condition, f"{label} {detail}"


# ── Fixtures ──────────────────────────────────────────────────────────────

def mock_rec(action="BUY", composite=72.0, risk_veto=False, gated=True,
             stat_level="LOW", dsr=0.0, sharpe=0.3, max_dd=0.2, n_obs=1200,
             price=100.0, pillars=None, gates=None, horizon=91):
    default_gates = {"min_sample": n_obs >= 504, "net_cost_positive": sharpe > 0,
                     "dsr": dsr >= 0.5, "walk_forward": stat_level in ("MEDIUM", "HIGH"),
                     "pbo": stat_level == "HIGH"}
    g = {**default_gates, **(gates or {})}
    return {
        "ticker": "TEST", "generated_at": "2026-09-19T10:00:00",
        "current_price": price, "sector": "Technology", "regime": "MEDIUM",
        "data_asof": "2026-09-18",
        "action": action, "conviction": stat_level, "composite": composite,
        "core_score": composite, "risk_multiplier": 0.85, "risk_veto": risk_veto,
        "time_horizon_days": horizon,
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.6, "basis": "test"},
            "data": {"level": "MEDIUM", "score": 0.6, "basis": "test"},
            "statistical_edge": {
                "level": stat_level, "score": sum(g.values()) / len(g),
                "basis": "test basis", "gate_required": "walk-forward + embargo + PBO",
                "checks": {
                    "min_sample": {"pass": g["min_sample"], "n_obs": n_obs, "required": 504},
                    "net_cost_positive": {"pass": g["net_cost_positive"]},
                    "dsr": {"pass": g["dsr"], "value": dsr, "bar": 0.5},
                    "walk_forward": {"pass": g["walk_forward"], "applied": True,
                                     "mean_oos_sharpe": 0.2},
                    "pbo": {"pass": g["pbo"], "computed": True, "pbo": 0.3},
                },
            },
            "allocation": {"level": "NONE" if gated else "HIGH", "score": 0.5},
        },
        "pillars": pillars or {
            "technical": {"score": 70.0, "confidence": 0.9, "backtestable": True, "flags": []},
            "algo": {"score": 68.0, "confidence": 0.9, "backtestable": True, "flags": []},
            "fundamentals": {"score": 60.0, "confidence": 0.7, "backtestable": False, "flags": []},
            "research": {"score": 55.0, "confidence": 0.5, "backtestable": False, "flags": []},
            "social": {"score": 50.0, "confidence": 0.3, "backtestable": False, "flags": []},
            "risk": {"score": 60.0, "confidence": 0.8, "backtestable": True, "flags": []},
        },
        "levels": {"entry_zone_low": price * 0.98, "entry_zone_high": price * 1.01,
                   "stop_loss": price * 0.9, "target_price": price * 1.2, "atr_14": price * 0.02},
        "position_size_pct": 0.0 if gated else 5.0,
        "position_size_gated": gated, "raw_kelly_pct": 5.0,
        "backtest": {"strategy": "seven_pillar_core", "sharpe": sharpe, "dsr": dsr,
                     "n_trials": 8, "max_drawdown": max_dd, "n_trades": 40,
                     "cost_model": "CostModel", "total_cost_pct": 1.0, "core_signal_now": 1},
        "honesty_flags": {"survivorship_safe": False, "pit_fundamentals": True,
                          "backtest_covers_core_only": True, "interaction_independent": True},
        "decision_fingerprint": "abc123",
        "expected_return_pct": 20.0,
    }


def mock_indicators(price=100.0, rsi=55.0, bb_pct=0.5):
    return {"current_price": price, "rsi_14": rsi, "bb_pct": bb_pct,
            "sma_20": price * 0.98, "sma_50": price * 0.95, "sma_200": price * 0.88,
            "bb_upper": price * 1.04, "bb_middle": price * 0.98, "bb_lower": price * 0.92,
            "52w_high": price * 1.30, "52w_low": price * 0.60, "atr_14": price * 0.02,
            "volume_ratio": 1.1}


def mock_algo(vol_regime="MEDIUM", vol_expanding=False):
    return {"vol_regime": vol_regime, "vol_expanding": vol_expanding,
            "historical_volatility_20d": 25.0, "historical_volatility_60d": 24.0}


def mock_hist(price=100.0):
    return {
        "horizons": {"1M": {"data_available": True, "return_pct": 5.0, "max_drawdown_pct": -8.0,
                            "trend": "UP", "volatility_annualized_pct": 25.0},
                     "1Y": {"data_available": True, "return_pct": 20.0, "max_drawdown_pct": -22.0,
                            "trend": "UP", "volatility_annualized_pct": 28.0}},
        "support_resistance": {
            "20D": {"support": price * 0.94, "resistance": price * 1.05, "data_available": True},
            "6M": {"support": price * 0.82, "resistance": price * 1.18, "data_available": True},
            "1Y": {"support": price * 0.62, "resistance": price * 1.28, "data_available": True}},
        "relative_performance": {
            "1M": {"stock_return_pct": 5.0, "benchmark_return_pct": 1.0,
                   "excess_return_pct": 4.0, "benchmark_symbol": "SPY", "data_available": True},
            "1Y": {"stock_return_pct": 20.0, "benchmark_return_pct": 12.0,
                   "excess_return_pct": 8.0, "benchmark_symbol": "SPY", "data_available": True}},
        "regime_changes": {"current_regime": "BULLISH", "data_available": True},
        "confidence": 0.8, "flags": [],
    }


def mock_calibration(n=200, win_rate=0.55, ece=0.08, status="OK"):
    if status != "OK":
        return {"status": "INSUFFICIENT_HISTORY", "n_predictions": n,
                "interpretation": "not enough history"}
    return {"status": "OK", "n_predictions": n, "win_rate": win_rate,
            "calibration_error_ece": ece, "avg_return_pct": 1.0,
            "interpretation": f"Calibration on {n} predictions."}


def mock_catalysts(days_away=30, status="OK"):
    if status != "OK":
        return {"status": status, "events": [], "upcoming": [], "next_event": None,
                "n_upcoming": 0, "waiting_for": "nothing", "flags": []}
    ev = {"date": "2026-10-19", "days_away": days_away, "event": "Quarterly earnings report",
          "category": "EARNINGS", "scheduled": True,
          "relevance": "HIGH" if days_away <= 14 else "MEDIUM",
          "implied_move_to_event_pct": 8.0, "bull_implication": "bull", "bear_implication": "bear",
          "uncertainty": "unknown", "probability": None, "probability_basis": "NOT_CALIBRATED",
          "history": None, "source": "test", "provenance": {"module": "test"}}
    return {"status": "OK", "events": [ev], "upcoming": [ev], "next_event": ev,
            "next_scheduled_event": ev, "n_upcoming": 1,
            "waiting_for": f"earnings in {days_away} days", "flags": [],
            "recent_analyst_actions": [], "as_of": "2026-09-19",
            "disclaimer": "test"}


def build(**kw):
    """Build a decision with sensible defaults, overridable per test."""
    price = kw.pop("price", 100.0)
    rec = kw.pop("rec", None) or mock_rec(price=price, **kw.pop("rec_kw", {}))
    return build_decision_intelligence(
        kw.pop("ticker", "TEST"), rec,
        indicators=kw.pop("indicators", mock_indicators(price)),
        algo_signals=kw.pop("algo_signals", mock_algo()),
        historical_context=kw.pop("historical_context", mock_hist(price)),
        calibration_interp=kw.pop("calibration_interp", mock_calibration()),
        catalysts=kw.pop("catalysts", mock_catalysts()),
        **kw)


# ══════════════════════════════════════════════════════════════════════════
# 1-5. Thesis / statistics / data-quality interactions
# ══════════════════════════════════════════════════════════════════════════

def test_01_bullish_thesis_weak_statistics():
    d = build(rec=mock_rec(action="BUY", stat_level="LOW", dsr=0.0, gated=True))
    check("thesis is bullish", d["thesis"]["direction"] == "BULLISH")
    check("edge is not demonstrated", d["statistical_edge"]["demonstrated"] is False)
    check("relation names the split",
          d["thesis_edge_relation"]["relation"] == "THESIS_WITHOUT_EDGE")
    check("sizing stays 0%", d["sizing"]["position_size_pct"] == 0.0)
    check("no ACTION_WITHOUT_EDGE error", d["consistency"]["n_errors"] == 0)


def test_02_bearish_thesis_strong_statistics():
    """A demonstrated edge must never turn a bearish read into a long."""
    bearish = {k: {**v, "score": 25.0} for k, v in mock_rec()["pillars"].items()}
    d = build(rec=mock_rec(action="SELL", composite=28.0, stat_level="HIGH", dsr=0.8,
                           gated=False, pillars=bearish))
    check("thesis is bearish", d["thesis"]["direction"] == "BEARISH")
    check("edge demonstrated", d["statistical_edge"]["demonstrated"] is True)
    check("not-owned avoids", d["decision_state"]["if_not_owned"]["state"] == "AVOID_NEW_POSITION")
    check("owned reduces/exits",
          d["decision_state"]["if_owned"]["state"] in ("REDUCE", "EXIT_CONDITIONALLY"))
    check("entry invalidated, not attractive", d["entry"]["status"] == "INVALIDATED")
    check("long-only stated", d["statistical_honesty"]["one_sided"]["long_only"] is True)


def test_03_technical_bullish_fundamentals_bearish():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["technical"]["score"] = 90.0
    pillars["algo"]["score"] = 85.0
    pillars["fundamentals"] = {"score": 20.0, "confidence": 0.9, "backtestable": False, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    names = [c["name"] for c in d["conflict"]["conflicts"]]
    check("a directional conflict is named",
          any("fundamentals" in n for n in names), names)
    check("horizons disagree or are separately reported",
          d["horizon_read"]["bands"]["SHORT"]["direction"] != d["horizon_read"]["bands"]["LONG"]["direction"])


def test_04_strong_thesis_insufficient_data():
    d = build(rec=mock_rec(n_obs=100, stat_level="NONE", gated=True))
    check("edge verdict is INSUFFICIENT_SAMPLE",
          d["statistical_edge"]["verdict"] == "INSUFFICIENT_SAMPLE")
    check("honesty names the sample shortfall",
          any("insufficient" in s.lower() or "sample" in s.lower()
              for s in d["statistical_honesty"]["statements"]))
    check("sizing 0%", d["sizing"]["position_size_pct"] == 0.0)


def test_05_strong_thesis_poor_calibration():
    d = build(calibration_interp=mock_calibration(n=200, ece=0.35))
    check("calibration dimension is LOW",
          d["confidence"]["dimensions"]["calibration"]["level"] == "LOW")
    check("headline confidence is not HIGH", d["confidence"]["decision_confidence"] != "HIGH")
    check("no CONFIDENCE_WITHOUT_CALIBRATION error",
          not any(i["code"] == "CONFIDENCE_WITHOUT_CALIBRATION"
                  for i in d["consistency"]["issues"]))


# ══════════════════════════════════════════════════════════════════════════
# 6-8. Price position relative to the entry zone
# ══════════════════════════════════════════════════════════════════════════

def test_06_price_below_entry_zone():
    """Price near sourced support: the best reward:risk available."""
    d = build(price=100.0, indicators={**mock_indicators(100.0), "rsi_14": 35.0},
              historical_context=mock_hist(100.0))
    check("entry is attractive or acceptable",
          d["entry"]["status"] in ("ATTRACTIVE", "ACCEPTABLE"), d["entry"]["status"])
    check("range position is computed",
          d["entry"]["geometry"]["range_position"] is not None)


def test_07_price_inside_entry_zone():
    d = build()
    check("reward:risk is measured to sourced levels",
          d["entry"]["geometry"]["reward_risk_to_levels"] is not None)
    check("entry band, when shown, never sits above the quote",
          (not d["entry_plan"]["supported"]) or
          d["entry_plan"]["entry_high"] <= d["current_price"] + 1e-9)


def test_08_price_above_entry_zone_overextended():
    ind = mock_indicators(100.0, rsi=82.0, bb_pct=1.1)
    # Push price to the top of the sourced range.
    hist = mock_hist(100.0)
    hist["support_resistance"]["20D"] = {"support": 80.0, "resistance": 100.5,
                                         "data_available": True}
    d = build(price=100.0, indicators=ind, historical_context=hist)
    check("overextended or waiting, never attractive",
          d["entry"]["status"] in ("OVEREXTENDED", "WAIT"), d["entry"]["status"])
    check("no entry band is offered", d["entry_plan"]["supported"] is False)
    check("a reason is given for withholding the band",
          bool(d["entry_plan"].get("why_no_band")))


# ══════════════════════════════════════════════════════════════════════════
# 9-14. Ownership and position context
# ══════════════════════════════════════════════════════════════════════════

def test_09_owned_position():
    d = build(position={"avg_cost": 80.0, "shares": 50, "portfolio_value": 100000},
              max_portfolio_risk_pct=2.0)
    check("position recognized", d["position_context"]["status"] == "PROVIDED")
    check("ownership is OWNED", d["decision_state"]["ownership"] == "OWNED")
    check("headline uses the owned branch",
          d["decision_state"]["headline_state"] == d["decision_state"]["if_owned"]["state"])
    check("unrealized P&L computed", d["position_context"]["unrealized_pl_pct"] == 25.0)


def test_10_unowned_position():
    d = build()
    check("ownership unknown", d["decision_state"]["ownership"] == "UNKNOWN")
    check("headline uses the not-owned branch",
          d["decision_state"]["headline_state"] == d["decision_state"]["if_not_owned"]["state"])


def test_11_missing_position_context():
    d = build()
    check("status is POSITION_CONTEXT_NOT_PROVIDED",
          d["position_context"]["status"] == "POSITION_CONTEXT_NOT_PROVIDED")
    check("ownership is never invented", d["position_context"]["owns"] is None)
    check("the message says so explicitly",
          "not provided" in d["position_context"]["message"].lower()
          or "unknown" in d["position_context"]["message"].lower())
    check("add analysis refuses rather than guessing",
          d["add_analysis"]["status"] == "POSITION_CONTEXT_NOT_PROVIDED")
    check("dollar sizing refuses", "NOT_COMPUTABLE" in (d["sizing"]["dollar_sizing"] or ""))


def test_12_large_existing_loss():
    d = build(position={"avg_cost": 160.0, "shares": 100, "portfolio_value": 100000})
    pc = d["position_context"]
    check("underwater", pc["underwater"] is True)
    check("recovery math is asymmetric", pc["recovery_required_pct"] == 60.0,
          pc["recovery_required_pct"])
    check("a large loss is named as an observation",
          any("loss" in o.lower() for o in pc["observations"]))
    check("the loss is NOT itself treated as evidence",
          any(e["source"] == "position:cost_basis" and e["direction"] == "NOT_DIRECTIONAL"
              for e in d["evidence"]))


def test_13_large_existing_gain():
    d = build(position={"avg_cost": 50.0, "shares": 100, "portfolio_value": 100000})
    pc = d["position_context"]
    check("large gain observed", any("gain" in o.lower() for o in pc["observations"]))
    check("a gain alone does not force a REDUCE",
          d["decision_state"]["if_owned"]["state"] != "REDUCE"
          or (d["risk_budget"]["position_risk"] or {}).get("within_risk_budget") is False)


def test_14_averaging_down_temptation():
    """The headline case: price has fallen, nothing else has improved."""
    d = build(position={"avg_cost": 140.0, "shares": 100, "portfolio_value": 100000})
    add = d["add_analysis"]
    check("classified as averaging down", add["add_kind"] == "AVERAGING_DOWN")
    check("cheaper_not_better is set", add["cheaper_not_better"] is True)
    check("verdict is DO_NOT_ADD", add["verdict"] == "DO_NOT_ADD")
    check("headline leads with cheaper-not-better",
          "CHEAPER" in add["headline"].upper())
    check("the rule is stated", "NEVER A REASON TO ADD" in add["rule"].upper())


# ══════════════════════════════════════════════════════════════════════════
# 15-16. Thesis versus price movement
# ══════════════════════════════════════════════════════════════════════════

def test_15_thesis_deterioration_without_price_decline():
    weak = copy.deepcopy(mock_rec()["pillars"])
    for k in ("technical", "algo", "fundamentals"):
        weak[k]["score"] = 30.0
    d = build(rec=mock_rec(action="REDUCE", composite=32.0, pillars=weak),
              position={"avg_cost": 60.0, "shares": 100, "portfolio_value": 100000})
    check("thesis is bearish despite the gain", d["thesis"]["direction"] == "BEARISH")
    check("owned branch reduces or exits",
          d["decision_state"]["if_owned"]["state"] in ("REDUCE", "EXIT_CONDITIONALLY"))


def test_16_price_decline_without_thesis_deterioration():
    d = build(position={"avg_cost": 130.0, "shares": 100, "portfolio_value": 100000})
    checks = {c["question"]: c["answer"] for c in d["add_analysis"]["checks"]}
    check("'has price merely fallen' is answered YES",
          checks["Has price merely fallen?"] == "YES")
    check("cause is attributed to price movement, not new information",
          checks["Is the move caused by new information or simply price movement?"]
          == "PRICE_MOVEMENT_ONLY")


# ══════════════════════════════════════════════════════════════════════════
# 17-18. Catalysts
# ══════════════════════════════════════════════════════════════════════════

def test_17_catalyst_approaching():
    d = build(catalysts=mock_catalysts(days_away=5))
    check("an imminent-event avoid condition appears",
          any("Imminent" in c["label"] for c in d["entry"]["avoid_conditions"]))
    check("the catalyst appears in the monitoring plan",
          any("earnings" in c["what"].lower() for c in d["monitoring"]["checks"]))
    check("no probability is attached to the catalyst",
          all(s.get("probability") is None for s in d["scenarios"]["scenarios"]
              if s["name"] == "CATALYST"))


def test_18_catalyst_absent():
    d = build(catalysts={"status": "NO_EVENTS", "events": [], "upcoming": [],
                         "next_event": None, "n_upcoming": 0,
                         "waiting_for": "Nothing scheduled.", "flags": []})
    check("quality distinguishes 'none scheduled' from 'could not read'",
          d["quality"]["components"]["CATALYST"]["status"] == "ADEQUATE")
    check("no catalyst scenario is fabricated",
          not any(s["name"] == "CATALYST" for s in d["scenarios"]["scenarios"]))


# ══════════════════════════════════════════════════════════════════════════
# 19-20. Conflicting evidence
# ══════════════════════════════════════════════════════════════════════════

def test_19_conflicting_horizons():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["technical"] = {"score": 95.0, "confidence": 1.0, "backtestable": True, "flags": []}
    pillars["fundamentals"] = {"score": 5.0, "confidence": 1.0, "backtestable": False, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    short = d["horizon_read"]["bands"]["SHORT"]["direction"]
    long_ = d["horizon_read"]["bands"]["LONG"]["direction"]
    check("short and long are reported separately", short != long_, f"{short}/{long_}")
    check("a horizon conflict is named", len(d["horizon_read"]["horizon_conflicts"]) >= 1)
    check("the conflict is not averaged away",
          any(c["kind"] == "HORIZON" for c in d["conflict"]["conflicts"]))


def test_20_contradictory_agents():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["technical"] = {"score": 95.0, "confidence": 1.0, "backtestable": True, "flags": []}
    pillars["algo"] = {"score": 5.0, "confidence": 1.0, "backtestable": True, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    check("evidence reads as conflicted",
          d["conflict"]["consensus"] in ("CONFLICTED", "LEANING"), d["conflict"]["consensus"])
    check("a directional conflict is named",
          any(c["kind"] == "DIRECTIONAL" for c in d["conflict"]["conflicts"]))
    check("conflict answers 'does it change the decision'",
          all("changes_decision" in c for c in d["conflict"]["conflicts"]))
    if d["conflict"]["agreement_ratio"] <= 0.55:
        check("a genuine split refuses a direction",
              d["decision_state"]["headline_state"] == "CONFLICTED")


# ══════════════════════════════════════════════════════════════════════════
# 21-24. Statistical honesty
# ══════════════════════════════════════════════════════════════════════════

def test_21_insufficient_historical_sample():
    d = build(rec=mock_rec(n_obs=200, stat_level="NONE"))
    check("sample shortfall is reported",
          d["statistical_honesty"]["sample_size"]["backtest_observations"] == 200)
    check("validation dimension is INSUFFICIENT",
          d["confidence"]["dimensions"]["validation"]["level"] == "INSUFFICIENT")


def test_22_one_sided_backtest():
    d = build()
    check("long-only is stated explicitly",
          "long exposure only" in d["statistical_honesty"]["one_sided"]["statement"])


def test_23_multiple_testing_failure():
    d = build(rec=mock_rec(stat_level="MEDIUM", dsr=0.2,
                           gates={"pbo": False, "dsr": False}))
    check("dSR is reported against its bar",
          d["statistical_honesty"]["deflated_sharpe"]["clears_bar"] is False)
    check("multiple-testing correction is disclosed",
          d["statistical_honesty"]["multiple_testing"]["corrected"] is True)
    check("honesty names the dSR shortfall",
          any("deflated sharpe" in s.lower() for s in d["statistical_honesty"]["statements"]))


def test_24_no_demonstrated_edge_is_the_default():
    d = build()
    check("verdict is NO_DEMONSTRATED_EDGE",
          d["statistical_edge"]["verdict"] == "NO_DEMONSTRATED_EDGE")
    check("sizing consequence is stated",
          "0%" in d["statistical_edge"]["sizing_consequence"])


# ══════════════════════════════════════════════════════════════════════════
# 25-29. Missing and stale inputs
# ══════════════════════════════════════════════════════════════════════════

def test_25_missing_price_history():
    d = build(historical_context=None,
              indicators={"current_price": 100.0})   # no SMAs, no bands, no ATR
    check("level map refuses to present a usable ladder",
          d["level_map"]["status"] in ("INSUFFICIENT_DATA", "DERIVED_ONLY"),
          d["level_map"]["status"])
    check("the refusal says why", bool(d["level_map"].get("reason")))
    check("entry refuses rather than inventing a band",
          d["entry"]["status"] == "INSUFFICIENT_DATA")
    check("state is INSUFFICIENT_DATA",
          d["decision_state"]["headline_state"] == "INSUFFICIENT_DATA")


def test_26_missing_fundamentals():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["fundamentals"] = {"score": 50.0, "confidence": 0.0, "backtestable": False,
                               "flags": ["fundamentals_unavailable"]}
    d = build(rec=mock_rec(pillars=pillars))
    check("the missing input is surfaced as a quality issue",
          any("fundamentals" in q["source"] for q in d["conflict"]["evidence_quality_issues"]))
    check("a zero-reliability pillar carries no weight",
          all(e["weight"] == 0.0 for e in d["evidence"] if e["source"] == "pillar:fundamentals"))


def test_27_missing_social_data():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["social"] = {"score": 50.0, "confidence": 0.0, "backtestable": False,
                         "flags": ["social_unavailable"]}
    d = build(rec=mock_rec(pillars=pillars))
    check("still produces a decision", d["status"] == "OK")
    check("social carries no weight",
          all(e["weight"] == 0.0 for e in d["evidence"] if e["source"] == "pillar:social"))


def test_28_missing_calibration():
    d = build(calibration_interp=mock_calibration(n=3, status="INSUFFICIENT"))
    check("calibration dimension is INSUFFICIENT",
          d["confidence"]["dimensions"]["calibration"]["level"] == "INSUFFICIENT")
    check("quality marks calibration MISSING",
          d["quality"]["components"]["CALIBRATION"]["status"] == "MISSING")
    check("stated explicitly in the honesty strip",
          "INSUFFICIENT" in d["statistical_honesty"]["calibration"]["statement"])


def test_29_stale_data():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["technical"]["flags"] = ["stale_price_data_last_bar_2026-01-01"]
    d = build(rec=mock_rec(pillars=pillars))
    check("staleness is carried onto the evidence item",
          any(e["data_quality"] == "STALE" for e in d["evidence"]))
    check("staleness is surfaced as a quality issue",
          any(q["issue"] == "STALE" for q in d["conflict"]["evidence_quality_issues"]))


def test_30_invalidated_thesis():
    weak = {k: {**v, "score": 15.0, "confidence": 0.9} for k, v in mock_rec()["pillars"].items()}
    d = build(rec=mock_rec(action="SELL", composite=20.0, pillars=weak))
    check("thesis is bearish", d["thesis"]["direction"] == "BEARISH")
    check("entry is invalidated", d["entry"]["status"] == "INVALIDATED")
    check("exit conditions exist", len(d["playbook"]["exit_conditions"]) > 0)
    check("bearish is not read as a short instruction",
          "not a short" in d["decision_state"]["if_not_owned"]["reason"].lower()
          or "short" in d["statistical_honesty"]["one_sided"]["statement"].lower())


# ══════════════════════════════════════════════════════════════════════════
# Structural guarantees
# ══════════════════════════════════════════════════════════════════════════

def test_repeatability_same_inputs_same_fingerprint():
    rec = mock_rec()
    a = build(rec=rec)
    b = build(rec=copy.deepcopy(rec))
    check("fingerprints match", a["decision_fingerprint"] == b["decision_fingerprint"])


def test_no_probability_without_calibration():
    d = build()
    for s in d["scenarios"]["scenarios"]:
        check(f"{s['name']} states no uncalibrated probability",
              s["probability"] is None or s["probability_basis"] == "CALIBRATED")
    check("no UNCALIBRATED_PROBABILITY issue",
          not any(i["code"] == "UNCALIBRATED_PROBABILITY" for i in d["consistency"]["issues"]))


def test_insufficient_inputs_short_circuit():
    d = build_decision_intelligence("TEST", {"current_price": None})
    check("status is INSUFFICIENT_DATA", d["status"] == "INSUFFICIENT_DATA")
    check("missing inputs are named", len(d["missing_inputs"]) > 0)


def test_llm_prose_cannot_change_the_decision():
    """The narrative is quoted, never consulted."""
    prose = {"summary": "This stock will certainly triple. Target $500.",
             "bull_case": "Guaranteed upside.", "bear_case": ""}
    a = build(rec=mock_rec())
    b = build(rec=mock_rec(), llm_prose=prose)
    check("fingerprint is unchanged by prose",
          a["decision_fingerprint"] == b["decision_fingerprint"])
    check("prose is carried verbatim", b["thesis"]["narrative"]["summary"] == prose["summary"])
    check("authority is stated",
          "no LLM output enters any decision field" in b["authority"]["llm_role"])


# ══════════════════════════════════════════════════════════════════════════
# MUTATION TESTS — break the evidence, assert the decision moves.
# ══════════════════════════════════════════════════════════════════════════

def _mutate(label, mutator, field):
    """Assert that mutating a decision-relevant input changes `field`.

    This is what proves the input is actually consulted. A section that renders
    an input but never lets it change the output is the failure this layer
    exists to fix, and it is invisible to ordinary tests.
    """
    base = build()
    mutated = mutator()
    a, b = _get(base, field), _get(mutated, field)
    check(f"MUTATION {label}: {field} moves ({a!r} -> {b!r})", a != b)


def _get(d, path):
    cur = d
    for part in path.split("."):
        cur = (cur or {}).get(part)
    return cur


def test_mutation_pillar_scores_move_the_thesis():
    def m():
        p = copy.deepcopy(mock_rec()["pillars"])
        for k in ("technical", "algo", "fundamentals"):
            p[k]["score"] = 15.0
        return build(rec=mock_rec(pillars=p))
    _mutate("pillars flipped bearish", m, "thesis.direction")


def test_mutation_statistical_gate_moves_sizing():
    def m():
        return build(rec=mock_rec(stat_level="HIGH", dsr=0.8, gated=False,
                                  gates={"dsr": True, "pbo": True, "walk_forward": True}))
    _mutate("gate cleared", m, "sizing.position_size_pct")


def test_mutation_risk_veto_moves_the_state():
    def m():
        return build(rec=mock_rec(risk_veto=True),
                     algo_signals=mock_algo("HIGH", True))
    _mutate("risk veto fires", m, "decision_state.if_not_owned.state")


def test_mutation_support_level_moves_invalidation():
    def m():
        h = mock_hist(100.0)
        h["support_resistance"]["20D"]["support"] = 70.0
        h["support_resistance"]["6M"]["support"] = 68.0
        ind = mock_indicators(100.0)
        ind.update({"sma_20": 71.0, "sma_50": 70.5, "bb_lower": 69.0, "bb_middle": 71.0})
        return build(historical_context=h, indicators=ind)
    _mutate("support levels moved", m, "risk_budget.invalidation_level")


def test_mutation_calibration_moves_confidence():
    def m():
        return build(calibration_interp=mock_calibration(n=5, status="INSUFFICIENT"))
    _mutate("calibration removed", m, "confidence.dimensions.calibration.level")


def test_mutation_position_context_moves_the_headline():
    def m():
        return build(position={"avg_cost": 150.0, "shares": 100, "portfolio_value": 20000},
                     max_portfolio_risk_pct=1.0)
    _mutate("position supplied", m, "decision_state.headline_state")


def test_mutation_catalyst_proximity_moves_entry_avoidance():
    base = build(catalysts=mock_catalysts(days_away=60))
    near = build(catalysts=mock_catalysts(days_away=3))
    a = len(base["entry"]["avoid_conditions"])
    b = len(near["entry"]["avoid_conditions"])
    check(f"MUTATION catalyst moved close: avoid conditions grow ({a} -> {b})", b > a)


def test_mutation_conflict_moves_the_evidence_confidence():
    def m():
        p = copy.deepcopy(mock_rec()["pillars"])
        p["technical"] = {"score": 95.0, "confidence": 1.0, "backtestable": True, "flags": []}
        p["algo"] = {"score": 5.0, "confidence": 1.0, "backtestable": True, "flags": []}
        return build(rec=mock_rec(pillars=p))
    _mutate("agents contradict", m, "confidence.dimensions.evidence.level")


def test_mutation_volatility_regime_moves_review_cadence():
    def m():
        return build(algo_signals=mock_algo("HIGH", True))
    _mutate("volatility regime raised", m, "monitoring.cadence_days")


def test_mutation_horizon_conflict_is_reported():
    base = build()
    def m():
        p = copy.deepcopy(mock_rec()["pillars"])
        p["technical"] = {"score": 98.0, "confidence": 1.0, "backtestable": True, "flags": []}
        p["fundamentals"] = {"score": 2.0, "confidence": 1.0, "backtestable": False, "flags": []}
        return build(rec=mock_rec(pillars=p))
    a = len(base["horizon_read"]["horizon_conflicts"])
    b = len(m()["horizon_read"]["horizon_conflicts"])
    check(f"MUTATION horizons split: conflicts grow ({a} -> {b})", b > a)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
