#!/usr/bin/env python3
"""Tests for Portfolio Decision Brief v2 — deterministic weight/level logic on synthetic recs."""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.portfolio_brief import build_portfolio_brief

PASS = 0
FAIL = 0


def check(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


def _rec(ticker="TEST", action="BUY", composite=72, risk_veto=False, gated=False,
         stat_level="HIGH", alloc_level="HIGH", dsr=0.7, current_price=100.0,
         size_pct=6.0, levels=True, target=112.0, stop=94.0):
    """Minimal recommendation dict shaped like agents/recommendation.build_recommendation."""
    lv = (None if levels is False else {
        "entry_zone_low": 98.0, "entry_zone_high": 101.0,
        "stop_loss": stop, "target_price": target, "atr_14": 3.0, "formula": "test",
    })
    if levels == "null_prices":   # dict present but numeric fields missing
        lv = {"entry_zone_low": None, "entry_zone_high": None,
              "stop_loss": None, "target_price": None, "atr_14": None}
    return {
        "ticker": ticker, "current_price": current_price, "sector": "Technology",
        "regime": "MEDIUM", "action": action, "conviction": stat_level,
        "composite": composite, "risk_veto": risk_veto,
        "confidence": {
            "thesis": {"level": "MEDIUM", "score": 0.6},
            "data": {"level": "HIGH", "score": 1.0},
            "statistical_edge": {"level": stat_level, "score": 0.8,
                                 "checks": {"dsr": {"pass": dsr >= 0.5}}},
            "allocation": {"level": alloc_level, "score": 0.8},
        },
        "pillars": {
            "technical": {"score": 68, "backtestable": True, "flags": []},
            "algo": {"score": 72, "backtestable": True, "flags": []},
            "risk": {"score": 55, "backtestable": True, "flags": []},
            "fundamentals": {"score": 60, "backtestable": True, "flags": []},
            "research": {"score": 65, "backtestable": False, "flags": []},
            "social": {"score": 52, "backtestable": False, "flags": []},
        },
        "levels": lv,
        "position_size_pct": size_pct, "position_size_gated": gated,
        "time_horizon_days": 91,
        "backtest": {"strategy": "seven_pillar_core", "sharpe": 0.5, "dsr": dsr,
                     "max_drawdown": 0.15, "n_trades": 42, "cost_model": "CostModel",
                     "n_trials": 8},
        "honesty_flags": {"survivorship_safe": False, "pit_fundamentals": True,
                          "backtest_covers_core_only": True},
        "claims": {"composite": "c1"},
        "thesis": None,
        "decision_fingerprint": "fp" + ticker,
    }


def _holding(ticker, shares, avg_cost, rec):
    return {"ticker": ticker, "shares": shares, "avg_cost": avg_cost, "recommendation": rec}


def _find(result, ticker):
    return next(h for h in result["holdings"] if h["ticker"] == ticker)


def test_overweight_positive_becomes_trim_not_add():
    """A strong (proven-edge) stock that already exceeds its max weight must be
    TRIM (or HOLD), never ADD — concentration control beats a positive thesis."""
    strong = _rec("AAA", action="BUY", composite=80, stat_level="HIGH",
                  alloc_level="HIGH", gated=False, current_price=100.0)
    tiny = _rec("BBB", action="HOLD", composite=50, stat_level="LOW",
                alloc_level="NONE", gated=True, current_price=100.0)
    # AAA is ~95% of the book -> far over the 25% cap.
    result = build_portfolio_brief(
        [_holding("AAA", 190, 50.0, strong), _holding("BBB", 10, 50.0, tiny)],
        max_weight_pct=25.0)
    aaa = _find(result, "AAA")
    check("overweight strong stock is TRIM", aaa["position_action"] == "TRIM")
    check("overweight flag set", aaa["overweight"] is True)
    check("not ADD despite proven edge", aaa["position_action"] != "ADD")


def test_losing_position_not_automatically_held():
    """A weak stock (negative tier) is EXIT even when it's below cost — the
    decision is not anchored to recovering the cost basis."""
    weak = _rec("WEAK", action="SELL", composite=28, current_price=70.0)
    result = build_portfolio_brief([_holding("WEAK", 100, 120.0, weak)])
    h = _find(result, "WEAK")
    check("weak losing position exits, not holds", h["position_action"] == "EXIT")
    check("exit level is the ATR stop, not cost basis",
          h["levels"]["exit_level"] == 94.0 and h["levels"]["exit_level"] != 120.0)


def test_different_cost_bases_preserve_objective_scores():
    """Objective stock verdict (and composite) is identical regardless of the
    user's entry price — position context only changes the position-aware layer."""
    rec = _rec("SAME", action="BUY", composite=72, stat_level="MEDIUM")
    r_low = build_portfolio_brief([_holding("SAME", 10, 50.0, copy.deepcopy(rec))])
    r_high = build_portfolio_brief([_holding("SAME", 10, 200.0, copy.deepcopy(rec))])
    a, b = _find(r_low, "SAME"), _find(r_high, "SAME")
    check("objective_action invariant to cost basis",
          a["objective_action"] == b["objective_action"])
    check("composite invariant to cost basis",
          a["detail_brief"]["composite_score"] == b["detail_brief"]["composite_score"])


def test_risk_veto_blocks_add():
    """A risk veto must prevent ADD even with an otherwise-proven edge."""
    veto = _rec("VETO", action="BUY", composite=78, stat_level="HIGH",
                alloc_level="HIGH", gated=False, risk_veto=True, current_price=100.0)
    result = build_portfolio_brief([_holding("VETO", 5, 90.0, veto)])
    h = _find(result, "VETO")
    check("risk veto blocks ADD", h["position_action"] != "ADD")
    check("risk-veto add zone is not offered", h["levels"].get("add_zone") is None)


def test_missing_price_evidence_level_unavailable():
    """A holding whose price levels can't be computed shows LEVEL_UNAVAILABLE and
    a would-be add falls back to WAIT rather than inventing a price."""
    noprice = _rec("NOPX", action="BUY", composite=75, stat_level="HIGH",
                   alloc_level="HIGH", gated=False, levels="null_prices")
    filler = _rec("CASH", action="HOLD", composite=50, stat_level="LOW",
                  alloc_level="NONE", gated=True, current_price=100.0)
    # Keep NOPX underweight (~9%) so the LEVEL_UNAVAILABLE -> WAIT path is exercised,
    # not the overweight -> TRIM path.
    result = build_portfolio_brief(
        [_holding("NOPX", 10, 90.0, noprice), _holding("CASH", 100, 90.0, filler)],
        max_weight_pct=25.0)
    h = _find(result, "NOPX")
    check("levels marked unavailable", h["levels"]["status"] == "LEVEL_UNAVAILABLE")
    check("would-be add becomes WAIT", h["position_action"] == "WAIT")


def test_portfolio_action_invariant_to_llm_wording():
    """Changing the LLM thesis prose must not change any deterministic action."""
    base = _rec("PROSE", action="BUY", composite=72, stat_level="MEDIUM",
                current_price=100.0)
    r1 = build_portfolio_brief([_holding("PROSE", 10, 90.0, copy.deepcopy(base))])
    with_prose = copy.deepcopy(base)
    with_prose["thesis"] = {"summary": "To the moon! Definitely a screaming buy right now!!!",
                            "bull_case": "x", "bear_case": "y", "key_catalysts": "z"}
    r2 = build_portfolio_brief([_holding("PROSE", 10, 90.0, with_prose)])
    a, b = _find(r1, "PROSE"), _find(r2, "PROSE")
    check("position_action unchanged by prose", a["position_action"] == b["position_action"])
    check("levels unchanged by prose", a["levels"] == b["levels"])
    check("target weight unchanged by prose",
          a["target_weight_pct"] == b["target_weight_pct"])


def test_concentration_limits_enforced():
    """Overweight positions are flagged in the summary and counted; the stance
    reflects concentration."""
    big = _rec("BIG", action="BUY", composite=80, stat_level="HIGH",
               alloc_level="HIGH", gated=False, current_price=100.0)
    small = _rec("SML", action="HOLD", composite=55, stat_level="LOW",
                 alloc_level="NONE", gated=True, current_price=100.0)
    result = build_portfolio_brief(
        [_holding("BIG", 90, 50.0, big), _holding("SML", 10, 50.0, small)],
        max_weight_pct=25.0)
    port = result["portfolio"]
    check("overweight holding listed", any(o["ticker"] == "BIG"
          for o in port["overweight_holdings"]))
    check("overweight count >= 1", port["counts"]["overweight"] >= 1)
    check("BIG is a concentration risk", any(c["ticker"] == "BIG" and c["over_max"]
          for c in port["concentration_risks"]))


def test_portfolio_summary_shape_and_top3():
    """The summary exposes stance, top-3 actions, best add, top trim/exit, and
    confidence — and never averages the underlying scores into the stance."""
    recs = [
        _holding("OW", 90, 50.0, _rec("OW", action="BUY", composite=82, stat_level="HIGH",
                                       alloc_level="HIGH", gated=False, current_price=100.0)),
        _holding("EX", 30, 120.0, _rec("EX", action="SELL", composite=25, current_price=70.0)),
        _holding("AD", 2, 40.0, _rec("AD", action="BUY", composite=70, stat_level="HIGH",
                                     alloc_level="HIGH", gated=False, size_pct=8.0,
                                     current_price=50.0, target=60.0)),
    ]
    result = build_portfolio_brief(recs, max_weight_pct=25.0)
    port = result["portfolio"]
    check("has stance label", "label" in port["stance"])
    check("top_3_actions present", isinstance(port["top_3_actions"], list)
          and len(port["top_3_actions"]) >= 1)
    check("top action sorted by urgency (highest first)",
          port["top_3_actions"] == sorted(port["top_3_actions"],
                                           key=lambda a: a["urgency"], reverse=True))
    check("has data + calibration confidence",
          "data_confidence" in port and "calibration_confidence" in port)
    check("total value computed", port["total_value"] > 0)


def test_add_fires_when_underweight_and_proven():
    """An underweight, proven-edge stock with a supported add zone is ADD."""
    proven = _rec("GROW", action="BUY", composite=78, stat_level="HIGH",
                  alloc_level="HIGH", gated=False, size_pct=10.0, current_price=100.0)
    filler = _rec("CASH", action="HOLD", composite=50, stat_level="LOW",
                  alloc_level="NONE", gated=True, current_price=100.0)
    # GROW ~9% of book, target 10% -> room to add.
    result = build_portfolio_brief(
        [_holding("GROW", 9, 90.0, proven), _holding("CASH", 91, 90.0, filler)],
        max_weight_pct=25.0)
    h = _find(result, "GROW")
    check("underweight proven stock is ADD", h["position_action"] == "ADD")
    check("add zone supplied", h["levels"]["add_zone"] is not None)


# ── Cross-position correlation haircut ─────────────────────────────────────

def _returns(seed, n=120, shared=None, rho=1.0):
    """Synthetic daily return series. `shared` + independent noise lets a test
    dial the correlation between two names precisely."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    noise = rng.normal(0, 0.01, n)
    if shared is None:
        return pd.Series(noise, index=idx)
    return pd.Series(rho * shared.to_numpy() + (1 - rho) * noise, index=idx)


def _add_setup():
    """Two underweight, proven-edge names — both would ADD on their own."""
    a = _rec("AAA", action="BUY", composite=78, stat_level="HIGH",
             alloc_level="HIGH", gated=False, size_pct=10.0, current_price=100.0)
    b = _rec("BBB", action="BUY", composite=78, stat_level="HIGH",
             alloc_level="HIGH", gated=False, size_pct=10.0, current_price=100.0)
    filler = _rec("CASH", action="HOLD", composite=50, stat_level="LOW",
                  alloc_level="NONE", gated=True, current_price=100.0)
    return [_holding("AAA", 5, 90.0, a), _holding("BBB", 5, 90.0, b),
            _holding("CASH", 90, 90.0, filler)]


def test_correlation_absent_reproduces_previous_behaviour():
    """No returns supplied -> byte-identical to the pre-correlation output."""
    before = build_portfolio_brief(_add_setup(), max_weight_pct=25.0)
    after = build_portfolio_brief(_add_setup(), max_weight_pct=25.0, returns=None)
    check("omitting returns changes nothing",
          [h["target_weight_pct"] for h in before["holdings"]]
          == [h["target_weight_pct"] for h in after["holdings"]])
    check("correlation field present but null when unjudged",
          all(h.get("correlation") is None for h in after["holdings"]))
    check("no correlation risks reported", after["portfolio"]["correlation_risks"] == [])


def test_highly_correlated_holdings_get_add_headroom_cut():
    """Two names that move together are one bet - their add headroom shrinks."""
    base = _returns(1)
    rets = {"AAA": base, "BBB": _returns(2, shared=base, rho=0.98),
            "CASH": _returns(3)}
    uncorr = build_portfolio_brief(_add_setup(), max_weight_pct=25.0)
    corr = build_portfolio_brief(_add_setup(), max_weight_pct=25.0, returns=rets)
    a_un, a_corr = _find(uncorr, "AAA"), _find(corr, "AAA")
    check("correlated target weight is lower than standalone",
          a_corr["target_weight_pct"] < a_un["target_weight_pct"])
    check("haircut recorded on the holding", (a_corr.get("correlation") or {}).get("haircut") is not None)
    check("pre-correlation target preserved for audit",
          (a_corr.get("correlation") or {}).get("pre_correlation_target_pct")
          == a_un["target_weight_pct"])
    check("portfolio surfaces the correlation risk",
          any(r["ticker"] == "AAA" for r in corr["portfolio"]["correlation_risks"]))
    check("correlated weight reported", corr["portfolio"]["correlated_weight_pct"] > 0)


def test_uncorrelated_holdings_keep_full_size():
    """Independent names must NOT be penalised - the haircut is not a blanket tax."""
    rets = {"AAA": _returns(11), "BBB": _returns(22), "CASH": _returns(33)}
    plain = build_portfolio_brief(_add_setup(), max_weight_pct=25.0)
    out = build_portfolio_brief(_add_setup(), max_weight_pct=25.0, returns=rets)
    check("uncorrelated target unchanged",
          _find(out, "AAA")["target_weight_pct"] == _find(plain, "AAA")["target_weight_pct"])
    check("no correlation note for independent names",
          _find(out, "AAA").get("correlation") is None)


def test_correlation_never_forces_a_sell():
    """The haircut may shrink ADD headroom but must never push target below what
    is already held - that would manufacture a TRIM/EXIT from an unbacktested
    correlation estimate."""
    base = _returns(4)
    rets = {"AAA": base, "BBB": _returns(5, shared=base, rho=0.99), "CASH": _returns(6)}
    out = build_portfolio_brief(_add_setup(), max_weight_pct=25.0, returns=rets)
    for t in ("AAA", "BBB"):
        h = _find(out, t)
        check(f"{t} target not pushed below current weight",
              h["target_weight_pct"] >= h["current_weight_pct"])
        check(f"{t} not turned into a sell by correlation",
              h["position_action"] not in ("TRIM", "EXIT"))


def test_too_little_history_means_no_haircut():
    """Under MIN_CORR_OBS overlapping bars, correlation is noise - report nothing
    rather than acting on an estimate that isn't there."""
    base = _returns(7, n=12)
    rets = {"AAA": base, "BBB": _returns(8, n=12, shared=base, rho=0.99),
            "CASH": _returns(9, n=12)}
    plain = build_portfolio_brief(_add_setup(), max_weight_pct=25.0)
    out = build_portfolio_brief(_add_setup(), max_weight_pct=25.0, returns=rets)
    check("short history leaves sizing untouched",
          _find(out, "AAA")["target_weight_pct"] == _find(plain, "AAA")["target_weight_pct"])


def test_malformed_returns_degrade_silently():
    """A broken returns payload must never cost the user their brief."""
    out = build_portfolio_brief(_add_setup(), max_weight_pct=25.0,
                                returns={"AAA": "not a series", "BBB": None})
    check("brief still produced", len(out["holdings"]) == 3)
    check("no correlation risks claimed", out["portfolio"]["correlation_risks"] == [])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
