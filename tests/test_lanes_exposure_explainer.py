#!/usr/bin/env python3
"""Tests for build items 3-5: explicit signal lanes, portfolio-level exposure,
and the LLM decision explainer.

The load-bearing property across all three is the same one this codebase keeps
everywhere else: prose and unvalidated evidence may describe a decision, but
they must never be able to change one.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.portfolio_brief import DEFAULT_MAX_SECTOR_PCT, build_portfolio_brief
from backtest.pillars import (CORE_WEIGHTS, MODIFIER_MAX_PTS, TRACKED_FORWARD_LANE,
                              VALIDATED_LANE, compute_pillar_scores, signal_lanes)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _pillars_fixture():
    return compute_pillar_scores(
        "TEST",
        indicators={"current_price": 100.0},
        signal_summary={"score": 70, "direction": "BULLISH"},
        algo_signals={"algo_score": 65, "historical_volatility_20d": 25.0,
                      "vol_regime": "MEDIUM", "vol_expanding": False},
        fundamentals={"trailingPE": 18.0, "profitMargins": 0.15,
                      "recommendationMean": 2.0, "numberOfAnalystOpinions": 12,
                      "targetMeanPrice": 115.0, "beta": 1.1},
        reddit={"sentiment_score": 20, "mention_count": 40},
        stocktwits={"total": 50, "sentiment_ratio": 60},
    )


# ── Item 3: signal lanes ───────────────────────────────────────────────────

def test_lanes_present_and_partition_the_pillars():
    snap = _pillars_fixture()
    lanes = snap["lanes"]
    v = [p["pillar"] for p in lanes["validated"]["pillars"]]
    t = [p["pillar"] for p in lanes["tracked_forward"]["pillars"]]
    check("validated lane is exactly the price-derived pillars",
          set(v) == set(VALIDATED_LANE), str(v))
    check("tracked lane is exactly the slow/narrative pillars",
          set(t) == set(TRACKED_FORWARD_LANE), str(t))
    check("lanes do not overlap", not (set(v) & set(t)))


def test_lane_weights_make_the_guarantee_checkable():
    """A lane label is just a word unless it reports how much it can move."""
    lanes = _pillars_fixture()["lanes"]
    check("validated lane reports its core weight",
          lanes["validated"]["core_weight"]
          == round(sum(CORE_WEIGHTS.get(n, 0.0) for n in VALIDATED_LANE), 2))
    check("tracked lane reports a bounded modifier ceiling",
          lanes["tracked_forward"]["max_modifier_pts"] == 2 * MODIFIER_MAX_PTS)
    check("tracked lane's tilt is small next to the validated lane's weight",
          lanes["tracked_forward"]["max_modifier_pts"] <= 10.0)


def test_lanes_state_the_fundamentals_caveat_explicitly():
    """fundamentals is tracked-forward evidence holding core weight - the one
    place lane and influence disagree. It must be named, not averaged away."""
    lanes = _pillars_fixture()["lanes"]
    check("caveat names fundamentals", "fundamentals" in lanes["caveat"])
    check("caveat states social/research cannot drive",
          "tilt" in lanes["caveat"] and "drive" in lanes["caveat"])
    check("each lane carries its evidence basis",
          bool(lanes["validated"]["basis"]) and bool(lanes["tracked_forward"]["basis"]))


def test_lanes_degrade_with_missing_pillars():
    lanes = signal_lanes({"technical": {"score": 60, "confidence": 1.0}})
    check("only present pillars are listed",
          [p["pillar"] for p in lanes["validated"]["pillars"]] == ["technical"])
    check("absent tracked pillars yield an empty lane, not an error",
          lanes["tracked_forward"]["pillars"] == [])
    check("modifier ceiling reflects absent pillars",
          lanes["tracked_forward"]["max_modifier_pts"] == 0.0)


# ── Item 4: portfolio exposure ─────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from test_portfolio_brief import _holding, _rec  # noqa: E402


def _book(sectors):
    """One holding per sector, equal share count."""
    hs = []
    for i, (tk, sector) in enumerate(sectors.items()):
        r = _rec(tk, action="HOLD", composite=55, current_price=100.0)
        r["sector"] = sector
        hs.append(_holding(tk, 10, 90.0, r))
    return hs


def test_exposure_reports_gross_net_and_cash():
    out = build_portfolio_brief(_book({"AAA": "Technology", "BBB": "Energy"}))
    ex = out["portfolio"]["exposure"]
    check("gross exposure reported", ex["gross_exposure_pct"] > 0)
    check("long-only: net equals gross",
          ex["net_exposure_pct"] == ex["gross_exposure_pct"])
    check("cash is the complement",
          abs(ex["cash_pct"] - max(0.0, 100.0 - ex["gross_exposure_pct"])) < 0.01)
    check("total value carried", ex["total_value"] > 0)


def test_sector_breach_detected_when_every_position_is_individually_legal():
    """THE case the per-position ceiling cannot see: four tech names at 25% each
    all pass the position cap while the book is one sector bet."""
    out = build_portfolio_brief(
        _book({"A": "Technology", "B": "Technology", "C": "Technology", "D": "Technology"}),
        max_weight_pct=25.0)
    ex = out["portfolio"]["exposure"]
    tech = next(s for s in ex["by_sector"] if s["sector"] == "Technology")
    check("all four positions are individually under the cap",
          all(not h["overweight"] for h in out["holdings"]),
          str([h["current_weight_pct"] for h in out["holdings"]]))
    check("sector weight is 100%", abs(tech["weight_pct"] - 100.0) < 0.01)
    check("sector breach flagged", tech["over_max"] is True)
    check("breach surfaced in sector_breaches",
          any(s["sector"] == "Technology" for s in ex["sector_breaches"]))


def test_diversified_book_reports_no_breach():
    out = build_portfolio_brief(
        _book({"A": "Technology", "B": "Energy", "C": "Healthcare", "D": "Utilities"}))
    ex = out["portfolio"]["exposure"]
    check("no sector breach in a spread book", ex["sector_breaches"] == [])
    check("all sectors under the ceiling",
          all(s["weight_pct"] <= DEFAULT_MAX_SECTOR_PCT for s in ex["by_sector"]))


def test_unclassified_weight_is_reported_not_hidden():
    """An unknown-sector book must not read as diversified."""
    out = build_portfolio_brief(_book({"A": "", "B": ""}))
    ex = out["portfolio"]["exposure"]
    check("unclassified weight surfaced", ex["unclassified_weight_pct"] > 0)
    check("unknown bucket is not counted as a real sector breach",
          all(s["sector"] != "Unknown" for s in ex["sector_breaches"]))


# ── Item 5: decision explainer ─────────────────────────────────────────────

def _full_rec():
    r = _rec("EXPL", action="BUY", composite=72, current_price=100.0)
    r["pillars"] = {"technical": {"score": 70, "confidence": 1.0},
                    "algo": {"score": 68, "confidence": 1.0},
                    "social": {"score": 55, "confidence": 0.5}}
    r["honesty_flags"] = {"social_research_tracked_forward_only": True}
    r["lanes"] = signal_lanes(r["pillars"])
    return r


def test_explainer_prompt_carries_the_decision_as_fixed():
    from agents.stock_agents import _build_explain_prompt
    system, user = _build_explain_prompt(_full_rec())
    check("system forbids deciding", "not deciding anything" in system)
    check("system forbids inventing numbers", "Invent nothing" in system)
    check("system forbids proposing a different action",
          "Never recommend a different action" in system)
    check("decision is stated as already final", "do not revise it" in user)
    check("the actual action is in the prompt", "BUY" in user)
    check("pillar numbers are supplied", "70" in user and "68" in user)


def test_explainer_surfaces_evidence_quality_and_forecast():
    from agents.stock_agents import _build_explain_prompt
    forecast = {"horizons": {"1M": {"horizon_days": 20, "p_up": 0.61,
                                    "direction": "UP", "calibrated": True}}}
    _, user = _build_explain_prompt(_full_rec(), forecast)
    check("lane weights reach the prompt", "core weight" in user)
    check("fundamentals caveat reaches the prompt", "Caveat:" in user)
    check("forecast horizon included", "p(up) = 0.61" in user)
    check("calibration provenance is stated",
          "calibrated from past outcomes" in user)


def test_explainer_cannot_change_any_number():
    """The whole safety argument: the explainer is a leaf. Running it must leave
    the recommendation byte-identical."""
    import copy

    from agents.stock_agents import run_decision_explainer

    class _Msg:
        content = "The system says BUY. (This prose must not matter.)"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]
        usage = None

    class _Completions:
        def create(self, **kw):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    rec = _full_rec()
    before = copy.deepcopy(rec)
    out = run_decision_explainer(_Client(), rec)
    check("explanation returned", isinstance(out, str) and len(out) > 0)
    check("recommendation is completely unchanged", rec == before)


def test_explainer_is_not_in_the_default_pipeline():
    """It is a 6th LLM call - it must stay opt-in, or the documented 5-call
    cost of an analysis silently becomes 6."""
    import inspect

    from agents import orchestrator
    src = inspect.getsource(orchestrator)
    check("orchestrator does not call the explainer",
          "run_decision_explainer" not in src)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
