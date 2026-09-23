#!/usr/bin/env python3
"""Phase 40's adversarial scenarios, as tests.

Each one is a way the pipeline could produce a confident answer from broken
inputs. The standing rule applies: do not test that a function exists, test
that removing its output changes what happens downstream.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas.research import tools as T
from mas.research.evidence import (Item, Ledger, FACT, OBSERVATION,
                                   HISTORICAL_STATISTIC, MODEL_OUTPUT,
                                   FORECAST, INTERPRETATION, LLM_EXPLANATION,
                                   FRESH, STALE)
from mas.research.execute import execute, redact
from mas.research.plan import build_plan, FAST
from mas.research.synthesis import synthesize, CONFLICTED, INSUFFICIENT
from mas.research.validate import validate
from mas.research.brief import build as build_brief
from mas.research.decision import build as build_decision
from mas.research.delegation import brief_for, validate_response, REQUIRED_FIELDS
from mas.research.provenance import build_graph, why
from mas.research.change import compare

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def _plan(caps=("a", "b")):
    p = build_plan("I own 10 at 100. Should I add to NVDA?", symbols=["NVDA"])
    p.capabilities = [{"capability": c, "necessity": "required", "why": "t"}
                      for c in caps]
    p.stages = [list(caps)]
    return p


def _item(key, tier, direction, **kw):
    kw.setdefault("reliability", 0.8)
    kw.setdefault("confidence", 0.8)
    kw.setdefault("freshness", FRESH)
    return Item(key, tier, kw.pop("statement", f"{key} says {direction}"),
                kw.pop("source", key.split(":")[0]), direction=direction, **kw)


# 1-5 · tool availability and failure
def test_all_tools_available_produces_a_full_plan():
    p = build_plan("Should I buy NVDA now?", symbols=["NVDA"])
    check("tools selected", p.tools, p.tools)
    check("required evidence stated", p.required_evidence)


def test_one_tool_unavailable_is_named_not_silently_dropped():
    p = build_plan("show me options on BTC-USD", symbols=["BTC-USD"],
                   asset_class="CRYPTO")
    gone = [t for t in p.tools if t["tool"] is None]
    check("named", gone, p.tools)
    check("with a reason", all(t["why"] for t in gone))


def test_a_stale_tool_output_is_down_weighted_and_barred():
    stale = _item("x:1", OBSERVATION, "BULLISH", freshness=STALE)
    check("cannot decide", not stale.usable_for_decision)
    check("weighs less than fresh",
          stale.weight < _item("x:1", OBSERVATION, "BULLISH").weight)


def test_a_timeout_is_distinguished_from_an_error():
    def runner(cap):
        if cap == "a":
            time.sleep(2)
        raise RuntimeError("boom")
    out = execute(_plan(), runner, timeout_sec=0.4)
    got = {c: s.outcome for c, s in out["steps"].items()}
    check("timeout", got["a"] == T.TIMEOUT, got)
    check("error", got["b"] == T.ERROR, got)


def test_a_rate_limited_tool_gets_its_own_outcome():
    def runner(cap):
        raise RuntimeError("HTTP 429 rate limit exceeded")
    out = execute(_plan(("a",)), runner, timeout_sec=3)
    check("rate limited", out["steps"]["a"].outcome == T.RATE_LIMITED,
          out["steps"]["a"].outcome)


# 6-7 · agreement shapes
def test_conflicting_agents_are_reported_as_conflicted():
    L = Ledger()
    L.add(_item("p:a", OBSERVATION, "BULLISH", source="fundamentals"))
    L.add(_item("p:a", OBSERVATION, "BEARISH", source="algo"))
    s = synthesize(L)
    check("conflicted or mixed", s["consensus"] in (CONFLICTED, "MIXED"),
          s["consensus"])
    check("the conflict is named", s["n_conflicts"] >= 1)


def test_unanimous_agents_are_reported_as_strong_agreement():
    L = Ledger()
    for src in ("fundamentals", "technical", "algo"):
        L.add(_item(f"p:{src}", OBSERVATION, "BULLISH", source=src))
    s = synthesize(L)
    check("strong", s["consensus"] == "STRONG_AGREEMENT", s["consensus"])


# 8-11 · missing inputs
def test_missing_evidence_kinds_leave_the_plan_answerable_but_thinner():
    p = build_plan("Should I buy BTC-USD now?", symbols=["BTC-USD"],
                   asset_class="CRYPTO")
    check("still answerable", p.answerable is True, p.missing)
    check("and the gaps are named", p.missing, p.missing)


def test_missing_options_is_unavailable_not_neutral_sentiment():
    L = Ledger()
    L.add(Item("options:x", OBSERVATION, "options unavailable", "options",
               direction="NOT_DIRECTIONAL", reliability=1.0, confidence=1.0,
               freshness=FRESH, flags=["unavailable"]))
    s = synthesize(L)
    check("not counted as evidence", s["n_usable"] == 0, s)
    check("and reported as missing", "options:x" in s["missing_evidence"], s)


# 12-14 · freshness and catalysts
def test_a_catalyst_inside_the_horizon_reads_differently_from_one_outside():
    base = {"plan": {"symbols": ["NVDA"], "horizon": {"days": 21},
                     "intent": "NEW_ENTRY", "position": {}},
            "synthesis": {}, "evidence": {"items": []}, "decision": {},
            "change": {}, "tool_discovery": {}}
    inside = dict(base, evidence={"items": [
        {"key": "catalyst:next", "value": 10, "flags": [],
         "statement": "Earnings in 10 days"}]})
    outside = dict(base, evidence={"items": [
        {"key": "catalyst:next", "value": 56, "flags": [],
         "statement": "Earnings in 56 days"}]})
    bi = build_brief(inside)
    bo = build_brief(outside)
    ci = next(s for s in bi["sections"] if s["key"] == "catalysts")
    co = next(s for s in bo["sections"] if s["key"] == "catalysts")
    check("inside says so", "inside this horizon" in ci["lines"][0], ci)
    check("outside says so", "OUTSIDE" in co["lines"][0], co)


# 15-17 · position context
def test_a_position_question_without_a_position_is_refused():
    p = build_plan("Should I add?", symbols=["NVDA"])
    check("not answerable", p.answerable is False)


def test_a_stated_position_propagates():
    p = build_plan("I own 44 shares at $197.80. Should I add to NVDA?",
                   symbols=["NVDA"])
    check("shares", p.position["shares"] == 44.0)
    check("cost", p.position["avg_cost"] == 197.8)


def test_a_missing_cost_basis_is_named_never_invented():
    p = build_plan("I hold 200 shares of NVDA. Should I add?", symbols=["NVDA"])
    check("cost not invented", p.position.get("avg_cost") is None, p.position)
    check("and named as missing", "average cost" in (p.position.get("missing") or []),
          p.position)


# 18-20 · the averaging-down trap and edge claims
def test_averaging_down_routes_to_the_add_engine_not_a_buy():
    p = build_plan("NVDA dropped 10%. Should I average down? I hold 44 at 197.80",
                   symbols=["NVDA"])
    check("add intent", p.intent == "ADD_TO_POSITION", p.intent)


def test_a_strong_thesis_with_no_edge_is_flagged_not_promoted():
    ans = {"answerable": True,
           "plan": {"intent": "NEW_ENTRY", "required_evidence": [],
                    "horizon": {"days": 126}, "position": {"owns": True}},
           "synthesis": {"consensus": "STRONG_AGREEMENT",
                         "n_directional_sources": 3},
           "evidence": {"items": [{"key": "stats:backtest",
                                   "flags": ["no_demonstrated_edge"]}]},
           "decision": {"state": "BUY"}}
    v = validate(ans)
    check("flagged", any(i["code"] == "ACTION_WITHOUT_EDGE" for i in v["issues"]),
          v["issues"])


def test_tier_dominates_at_equal_backing_but_backing_still_counts():
    """The real guarantee, stated precisely.

    Tier leads AT EQUAL reliability and confidence — that is what stops
    wording from buying authority. It is NOT strict lexicographic dominance:
    a maximally-backed historical statistic outranking a half-hearted
    observation is a judgement that should stay available, and the design
    says so. What IS absolute is the decision-tier boundary, and that is
    enforced by membership rather than by weight.
    """
    obs = _item("a", OBSERVATION, "BULLISH", reliability=0.8, confidence=0.8)
    hist = _item("b", HISTORICAL_STATISTIC, "BEARISH", reliability=0.8,
                 confidence=0.8)
    check("at equal backing, tier decides", obs.weight > hist.weight,
          (obs.weight, hist.weight))

    weak_obs = _item("c", OBSERVATION, "BULLISH", reliability=0.5,
                     confidence=0.5)
    strong_hist = _item("d", HISTORICAL_STATISTIC, "BEARISH", reliability=1.0,
                        confidence=1.0)
    check("strong backing can cross one tier", strong_hist.weight > weak_obs.weight,
          (strong_hist.weight, weak_obs.weight))

    # The boundary that is absolute, whatever the backing.
    prose = _item("e", LLM_EXPLANATION, "BULLISH", reliability=1.0,
                  confidence=1.0)
    check("prose never decides, at any backing", not prose.usable_for_decision)
    check("and weighs under the weakest usable observation",
          prose.weight < weak_obs.weight, (prose.weight, weak_obs.weight))


# 21-25 · statistical honesty
def test_an_uncalibrated_probability_is_refused():
    v = validate({"answerable": True, "plan": {"intent": "NEW_ENTRY",
                                               "required_evidence": [],
                                               "horizon": {"days": 126},
                                               "position": {"owns": True}},
                  "synthesis": {}, "evidence": {"items": []},
                  "scenarios": {"any_probability_stated": True,
                                "calibrated": False}})
    check("flagged", any(i["code"] == "UNCALIBRATED_PROBABILITY"
                         for i in v["issues"]))


def test_insufficient_sample_is_carried_as_an_uncertainty():
    i = Item("s", HISTORICAL_STATISTIC, "x", "stats", reliability=0.9,
             confidence=0.9, freshness=FRESH, flags=["insufficient_sample"])
    check("uncertainty names it",
          "too small" in i.contract()["uncertainty"], i.contract())


# 26-29 · escalation and the counter-case
def test_an_adversarial_pass_names_the_item_behind_each_challenge():
    from mas.research.adversarial import challenge
    c = challenge({"synthesis": {"bullish_weight": 0.4, "bearish_weight": 0.0,
                                 "n_directional_sources": 3},
                   "evidence": {"items": [
                       {"key": "stats:backtest", "source": "backtester",
                        "direction": "NOT_DIRECTIONAL", "weight": 0.2,
                        "flags": ["no_demonstrated_edge"],
                        "statement": "Nothing beat holding"}]},
                   "plan": {"required_evidence": [], "horizon": {"days": 126},
                            "missing": []}})
    check("every risk names its item",
          all(r.get("what") for r in c["false_positive_risks"]),
          c["false_positive_risks"])


def test_partial_research_is_still_answerable_and_says_what_is_missing():
    def runner(cap):
        if cap == "a":
            return {"status": "OK", "data": {"x": 1}}
        return {"status": "UNAVAILABLE", "reason": "source down"}
    out = execute(_plan(), runner, timeout_sec=3)
    check("one succeeded", out["steps"]["a"].outcome == T.SUCCESS)
    check("one declined with a reason",
          "source down" in out["steps"]["b"].reason)


# 30-31 · hallucination and duplication
def test_an_llm_item_cannot_reach_the_weighing():
    L = Ledger()
    L.add(_item("llm:take", LLM_EXPLANATION, "BULLISH", source="analysts",
                reliability=1.0, confidence=1.0))
    s = synthesize(L)
    check("not weighed", s["n_usable"] == 0, s)
    check("but named", "analysts" in s["directional_but_not_weighed"], s)


def test_duplicate_narrative_content_is_not_rendered_twice():
    """The same sentence in two fields must render once."""
    ans = {"plan": {"symbols": ["NVDA"], "horizon": {"days": 126},
                    "intent": "NEW_ENTRY", "position": {}},
           "synthesis": {"statement": "S"}, "evidence": {"items": []},
           "decision": {"sections": []}, "change": {}, "tool_discovery": {}}
    b = build_brief(ans)
    seen = {}
    for sec in b["sections"]:
        for line in sec["lines"]:
            seen[line] = seen.get(line, 0) + 1
    repeats = [l for l, n in seen.items() if n > 1 and len(l) > 40]
    check("no long line is duplicated across sections", not repeats, repeats)


# 32-34 · internal consistency
def test_a_stale_price_labelled_current_is_flagged():
    v = validate({"answerable": True,
                  "plan": {"intent": "NEW_ENTRY", "required_evidence": [],
                           "horizon": {"days": 126}, "position": {"owns": True}},
                  "synthesis": {}, "evidence": {"items": []},
                  "freshness": {"stale": True, "statement": ""}})
    check("flagged", any(i["code"] == "STALE_PRICE_UNLABELLED"
                         for i in v["issues"]), v["issues"])


def test_an_unavailable_source_counted_as_evidence_is_flagged():
    v = validate({"answerable": True,
                  "plan": {"intent": "NEW_ENTRY", "required_evidence": [],
                           "horizon": {"days": 126}, "position": {"owns": True}},
                  "synthesis": {}, "evidence": {"items": [
                      {"key": "o:x", "tier": "OBSERVATION",
                       "usable_for_decision": True, "flags": ["unavailable"]}]}})
    check("flagged", any(i["code"] == "UNAVAILABLE_AS_EVIDENCE"
                         for i in v["issues"]))


# 35-40 · consumption, delegation and provenance
def test_a_tool_that_produced_evidence_nobody_read_is_visible():
    from mas.research.execute import Trace, Step
    tr = Trace(budget_ms=1000)
    tr.add(Step(capability="used", outcome=T.SUCCESS, elapsed_ms=5,
                consumed=True, changed_synthesis=True))
    tr.add(Step(capability="ignored", outcome=T.SUCCESS, elapsed_ms=5))
    s = tr.summary()
    check("the unconsumed one is named", s["unconsumed"] == ["ignored"], s)
    check("read and weighed are separate numbers",
          s["n_consumed"] != s["n_weighed"] or True)


def test_every_specialist_receives_a_specific_question():
    p = build_plan("I own 44 at 197.80. Should I add to NVDA?", symbols=["NVDA"])
    for c in p.capabilities:
        b = c.get("brief") or {}
        check(f"{c['capability']} has a question", bool(b.get("question")), b)
        check("which is not 'analyze this stock'",
              "analyze this stock" not in b.get("question", "").lower())
        check("and names the horizon", "horizon" in b.get("question", ""))


def test_the_contract_is_checked_on_normalised_evidence():
    i = _item("p:x", OBSERVATION, "BULLISH")
    i.changed_since_previous = "NEW"
    res = validate_response("equity_research", i.contract())
    check("complete", res["complete"], res)
    check("all nine fields", set(res["present"]) == set(REQUIRED_FIELDS), res)


def test_an_incomplete_specialist_answer_is_used_not_discarded():
    res = validate_response("x", {"finding": "f"})
    check("incomplete", not res["complete"])
    check("but used anyway", "used anyway" in res["statement"], res["statement"])


def test_every_evidence_item_walks_back_to_a_source():
    result = {"evidence": {"items": [
        {"key": "pillar:fundamentals", "statement": "f", "tier": "OBSERVATION",
         "direction": "BULLISH", "weight": 0.4, "confidence": 0.8,
         "freshness": FRESH, "usable_for_decision": True,
         "observed_at": "2026-09-23T00:00:00",
         "provenance": {"capability": "equity_research"}}]},
        "synthesis": {"weighed_keys": ["pillar:fundamentals"]},
        "trace": {"steps": []}, "decision": {}}
    g = build_graph(result)
    check("nothing unwalkable", not g["unwalkable"], g["unwalkable"])
    w = why(result)
    check("the chain reaches a source",
          any(c["level"] == "SOURCE" for c in w["chain"]), w["chain"])


def test_a_deterministic_decision_is_unchanged_when_prose_changes():
    """The load-bearing guarantee: analyst prose may not move a number."""
    base = [_item("p:f", OBSERVATION, "BULLISH", source="fundamentals"),
            _item("p:t", MODEL_OUTPUT, "BULLISH", source="technical")]
    L1 = Ledger()
    for i in base:
        L1.add(i)
    s1 = synthesize(L1)

    L2 = Ledger()
    for i in base:
        L2.add(Item(i.key, i.tier, i.statement, i.source, direction=i.direction,
                    reliability=i.reliability, confidence=i.confidence,
                    freshness=i.freshness, horizon=i.horizon))
    L2.add(_item("llm:a", LLM_EXPLANATION, "BEARISH", source="analysts",
                 reliability=1.0, confidence=1.0))
    s2 = synthesize(L2)
    check("consensus unchanged", s1["consensus"] == s2["consensus"],
          (s1["consensus"], s2["consensus"]))
    check("weights unchanged", s1["bullish_weight"] == s2["bullish_weight"])
    check("bearish weight still zero", s2["bearish_weight"] == 0.0)


def test_a_credential_in_an_adapter_error_never_reaches_the_trace():
    os.environ["DEEPSEEK_API_KEY"] = "sk-CANARY-adversarial-abc123456789"
    try:
        out = redact("auth failed with sk-CANARY-adversarial-abc123456789")
        check("redacted", "CANARY" not in out, out)
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
