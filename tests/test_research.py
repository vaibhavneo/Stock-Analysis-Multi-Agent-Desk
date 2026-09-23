#!/usr/bin/env python3
"""The research pipeline: plan, tools, execution, evidence, synthesis, checks.

The baseline audit found a chat question about ADDING to a position returning
a composite score and a volatility number, while 41 decision-engine fields —
add_analysis, entry_plan, mind_changers, scenarios among them — sat
unreachable. These tests hold the architecture that closed that, and they are
written to the project's standing rule: do not test that a function exists,
test that removing its output changes what happens downstream.
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas.research import intent_corpus as CORPUS
from mas.research import tools as T
from mas.research.evidence import (Item, Ledger, FACT, OBSERVATION,
                                   HISTORICAL_STATISTIC, MODEL_OUTPUT,
                                   FORECAST, INTERPRETATION, LLM_EXPLANATION,
                                   TIERS, TIER_RANK, DECISION_TIERS,
                                   FRESH, AGEING, STALE, UNKNOWN_FRESHNESS)
from mas.research.execute import execute, Trace, Step, run_stage
from mas.research.intent import classify, ALL_INTENTS, POSITION_INTENTS
from mas.research.plan import (build_plan, SPECS, EVIDENCE_CAPABILITY,
                               detect_horizon, FAST, DEEP, SHORT, MEDIUM, LONG)
from mas.research.position import extract as pextract, merge as pmerge, to_engine
from mas.research.synthesis import (synthesize, classify_conflict,
                                    MIN_SOURCES_FOR_AGREEMENT,
                                    STRONG_AGREEMENT, CONFLICTED, INSUFFICIENT)
from mas.research.validate import validate, SATISFIED_BY
from mas.converse.symbols import extract as xsym

PASS = 0
FAIL = 0

# Set AT the achieved rate. A threshold below what the code does is a ratchet
# that permits silent decay — the exact mistake made once already in this
# project, where a gate of 0.98 let a mutation through at an achieved 0.99.
MIN_INTENT_ACCURACY = 1.00
MIN_POSITION_ACCURACY = 1.00


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def _grade():
    ok = pos_ok = 0
    misses = []
    for c in CORPUS.CASES:
        syms = xsym(c["q"], context_symbol=c.get("ctx"))["symbols"]
        r = classify(c["q"], context_symbol=c.get("ctx"),
                     has_symbol=bool(syms), n_symbols=len(syms))
        if r["intent"] == c["intent"]:
            ok += 1
        else:
            misses.append((c["q"], c["intent"], r["intent"]))
        if r["is_position_question"] == c["position"]:
            pos_ok += 1
    n = len(CORPUS.CASES)
    return ok / n, pos_ok / n, misses


# ── Corpus and the ratchet ────────────────────────────────────────────────

def test_intent_corpus_covers_every_intent():
    covered = {c["intent"] for c in CORPUS.CASES}
    check("every intent has cases", covered == set(ALL_INTENTS),
          sorted(set(ALL_INTENTS) - covered))
    check("corpus is substantial", CORPUS.size() >= 70, CORPUS.size())


def test_intent_accuracy_meets_the_ratchet():
    acc, _, misses = _grade()
    check(f"intent >= {MIN_INTENT_ACCURACY:.0%}", acc >= MIN_INTENT_ACCURACY,
          f"got {acc:.1%}; misses {misses[:5]}")


def test_position_detection_meets_the_ratchet():
    _, pos, _ = _grade()
    check(f"position >= {MIN_POSITION_ACCURACY:.0%}",
          pos >= MIN_POSITION_ACCURACY, f"got {pos:.1%}")


def test_the_ratchet_itself_cannot_be_weakened_silently():
    """Phase 41's rule, made enforceable: the gate must sit at the achieved
    rate, so lowering it is a visible edit that fails this test."""
    acc, pos, _ = _grade()
    check("intent gate is not below achieved",
          MIN_INTENT_ACCURACY >= min(acc, 1.0) - 1e-9,
          f"gate {MIN_INTENT_ACCURACY} vs achieved {acc}")
    check("position gate is not below achieved",
          MIN_POSITION_ACCURACY >= min(pos, 1.0) - 1e-9)


# ── Intent ────────────────────────────────────────────────────────────────

def test_exactly_one_intent_per_utterance():
    """Capabilities are plural; the decision being asked for is singular.
    Two would mean the plan had not decided anything."""
    for c in CORPUS.CASES[:20]:
        r = classify(c["q"], has_symbol=True, n_symbols=1)
        check("intent is a single string", isinstance(r["intent"], str), r)


def test_position_intents_are_distinguished_from_general_research():
    """The baseline failure: 'Should I add to my IONQ position?' and
    'Tell me about IONQ' produced identical work."""
    add = classify("Should I add to my IONQ position?", has_symbol=True, n_symbols=1)
    gen = classify("Tell me about IONQ", has_symbol=True, n_symbols=1)
    check("add is a position question", add["intent"] == "ADD_TO_POSITION")
    check("general is not", gen["intent"] == "GENERAL_RESEARCH")
    check("they differ", add["intent"] != gen["intent"])


def test_averaging_down_is_an_add_question_not_a_buy():
    r = classify("IONQ dropped 10%. Should I average down?", has_symbol=True, n_symbols=1)
    check("routes to add", r["intent"] == "ADD_TO_POSITION", r["intent"])
    check("and is a position question", r["is_position_question"])


def test_invalidation_is_answerable_at_all():
    """It was UNROUTABLE in the baseline, while mind_changers already existed."""
    r = classify("What could invalidate the thesis?", context_symbol="NVDA",
                 has_symbol=False, n_symbols=0)
    check("has an intent", r["intent"] == "INVALIDATION", r["intent"])


def test_ownership_is_read_from_the_utterance_not_only_the_intent():
    """'How do I hedge my NVDA position' is an options question ABOUT a
    holding. Deriving ownership from intent alone misses it."""
    r = classify("how do i hedge my NVDA position", has_symbol=True, n_symbols=1)
    check("options intent", r["intent"] == "OPTIONS_ANALYSIS")
    check("but still a position question", r["is_position_question"])


def test_traps_do_not_become_position_actions():
    for q in ("I want to add a watchlist", "reduce the noise in this analysis",
              "should i buy more time before deciding"):
        r = classify(q, has_symbol=False, n_symbols=0)
        check(f"{q!r} is not an action", r["intent"] == "NONE", r["intent"])


# ── Position context ──────────────────────────────────────────────────────

def test_shares_and_cost_are_taken_from_the_sentence():
    c = pextract("I own 44 shares at $197.80. Should I add?")
    check("shares", c["shares"] == 44.0, c)
    check("cost", c["avg_cost"] == 197.8, c)
    check("owns", c["owns"] is True)


def test_nothing_is_inferred_when_nothing_is_stated():
    c = pextract("Should I buy NVDA now?")
    check("no ownership", c["owns"] is None, c)
    check("no numbers", c["shares"] is None and c["avg_cost"] is None)


def test_an_opinion_is_not_a_holding():
    c = pextract("my position on this is that it's overvalued")
    check("not read as ownership", c["owns"] is None, c)
    check("and it says why", any("opinion" in n for n in c["notes"]), c["notes"])


def test_partial_context_names_what_is_missing_rather_than_guessing():
    c = pextract("I hold 200 shares of AMD")
    check("shares captured", c["shares"] == 200.0)
    check("cost NOT invented", c["avg_cost"] is None)
    check("and the gap is named", "average cost" in c["missing"], c["missing"])


def test_the_engine_shape_refuses_a_holding_without_a_basis():
    """decision/ treats avg_cost as the gate for a real position. A partial
    dict would be read as an assumed holding."""
    check("no basis -> None", to_engine(pextract("I hold 200 shares of AMD")) is None)
    full = to_engine(pextract("I own 44 shares at $197.80"))
    check("with a basis -> a dict", isinstance(full, dict) and full["avg_cost"] == 197.8)


def test_application_context_fills_gaps_but_the_message_wins():
    merged = pmerge(pextract("I own 44 shares at $197.80"),
                    {"avg_cost": 1.0, "portfolio_value": 250_000})
    check("the message's cost wins", merged["avg_cost"] == 197.8, merged)
    check("the application fills the gap", merged["portfolio_value"] == 250_000)


# ── Plan ──────────────────────────────────────────────────────────────────

def test_a_plan_states_what_the_decision_needs_not_which_module_runs():
    p = build_plan("Should I add to my NVDA position?", symbols=["NVDA"])
    check("required evidence stated", "position_context" in p.required_evidence)
    check("thesis required", "thesis" in p.required_evidence)
    check("statistical edge required", "statistical_edge" in p.required_evidence)
    check("capabilities are derived, with a reason",
          all(c.get("why") for c in p.capabilities), p.capabilities)


def test_different_intents_require_different_evidence():
    """If every plan asked for the same things, the planner would not be
    planning."""
    add = set(build_plan("Should I add to my NVDA position?",
                         symbols=["NVDA"]).required_evidence)
    short = set(build_plan("is NVDA a good trade for the next few days",
                           symbols=["NVDA"]).required_evidence)
    longt = set(build_plan("is NVDA a decade holding",
                           symbols=["NVDA"]).required_evidence)
    check("add needs position context", "position_context" in add)
    check("a swing question does not", "position_context" not in short)
    check("a decade question does not require chart structure",
          "technical_structure" not in longt, longt)
    check("the three differ", len({frozenset(add), frozenset(short),
                                   frozenset(longt)}) == 3)


def test_a_position_question_without_a_position_is_not_answerable():
    p = build_plan("Should I add?", symbols=["NVDA"])
    check("not answerable", p.answerable is False)
    check("and it says what is missing",
          any("position" in m["what"] for m in p.missing), p.missing)


def test_a_stated_position_makes_it_answerable():
    p = build_plan("I own 44 shares at $197.80. Should I add to NVDA?",
                   symbols=["NVDA"])
    check("answerable", p.answerable is True, p.missing)
    check("position propagated", p.position["shares"] == 44.0)


def test_horizon_is_detected_and_otherwise_defaulted_with_a_reason():
    stated = detect_horizon("over the next few days")
    check("stated short", stated["horizon"] == SHORT and stated["stated"])
    default = detect_horizon("", MEDIUM)
    check("default medium", default["horizon"] == MEDIUM)
    check("and explains itself", "no timeframe" in default["why"])


def test_the_plan_does_no_market_io():
    """It must be assertable offline and showable before anything runs.
    Tool DISCOVERY probes imports and env, which is not market I/O."""
    import socket
    real = socket.socket

    class Boom(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("build_plan opened a socket")

    socket.socket = Boom
    try:
        for q in ("Should I add to my NVDA position?", "NVDA",
                  "What could invalidate the thesis?", "hi"):
            build_plan(q, symbols=["NVDA"])
        check("planning is offline", True)
    finally:
        socket.socket = real


# ── Tool discovery ────────────────────────────────────────────────────────

def test_availability_is_probed_not_declared():
    d = T.discover("EQUITY")
    check("every tool reports a state",
          all(r.get("state") for r in d["tools"]), d["tools"][:2])
    unavailable = [r for r in d["tools"] if not r["available"]]
    check("unavailable tools name their reason",
          all(r["reason"] for r in unavailable), unavailable)


def test_a_missing_key_is_reported_as_unavailable_with_the_variable_named():
    tools = {t["id"]: t for t in T.discover("EQUITY")["tools"]}
    op = tools.get("optionspilot_chain")
    if op and not op["available"]:
        check("the variable is named",
              "OPTIONSPILOT_ACCESS_CODE" in op["reason"], op["reason"])


def test_asset_class_narrows_what_is_reachable():
    eq = T.discover("EQUITY")
    cr = T.discover("CRYPTO")
    check("crypto reaches fewer tools", cr["n_available"] < eq["n_available"],
          (cr["n_available"], eq["n_available"]))
    check("and names what it cannot supply",
          "fundamentals" in cr["evidence_unreachable"], cr["evidence_unreachable"])


def test_an_unreachable_requirement_becomes_a_stated_gap_not_silence():
    p = build_plan("show me options on BTC-USD", symbols=["BTC-USD"],
                   asset_class="CRYPTO")
    unreachable = [t for t in p.tools if t["tool"] is None]
    check("at least one tool is unreachable", unreachable, p.tools)
    check("each names why", all(t["why"] for t in unreachable))


# ── Execution ─────────────────────────────────────────────────────────────

def _plan_with(caps):
    p = build_plan("I own 10 at 100. Should I add to NVDA?", symbols=["NVDA"])
    p.capabilities = [{"capability": c, "necessity": "required", "why": "t"}
                      for c in caps]
    p.stages = [list(caps)]
    return p


def test_independent_steps_run_in_parallel():
    plan = _plan_with(["a", "b", "c", "d"])

    def runner(cap):
        time.sleep(0.4)
        return {"status": "OK", "data": {"x": 1}}

    t0 = time.time()
    execute(plan, runner, timeout_sec=5)
    elapsed = time.time() - t0
    check("4 x 0.4s finished well under serial time", elapsed < 1.0,
          f"{elapsed:.2f}s")


def test_the_nine_outcomes_are_distinguished():
    plan = _plan_with(["ok", "boom", "declined", "empty", "slow"])

    def runner(cap):
        if cap == "boom":
            raise RuntimeError("exploded")
        if cap == "declined":
            return {"status": "UNAVAILABLE", "reason": "no data for this name"}
        if cap == "empty":
            return {"status": "OK", "data": {}}
        if cap == "slow":
            time.sleep(3)
        return {"status": "OK", "data": {"x": 1}}

    out = execute(plan, runner, timeout_sec=0.6)
    got = {c: s.outcome for c, s in out["steps"].items()}
    check("success", got["ok"] == T.SUCCESS, got)
    check("error is not unavailable", got["boom"] == T.ERROR, got)
    check("unavailable is not error", got["declined"] == T.UNAVAILABLE, got)
    check("empty is its own outcome", got["empty"] == T.EMPTY, got)
    check("timeout is its own outcome", got["slow"] == T.TIMEOUT, got)


def test_a_raising_step_does_not_take_the_others_down():
    plan = _plan_with(["good", "bad"])

    def runner(cap):
        if cap == "bad":
            raise RuntimeError("boom")
        return {"status": "OK", "data": {"x": 1}}

    out = execute(plan, runner, timeout_sec=3)
    check("the good one still succeeded",
          out["steps"]["good"].outcome == T.SUCCESS)
    check("and the failure is recorded with its message",
          "boom" in out["steps"]["bad"].reason)


def test_the_trace_separates_produced_from_consumed():
    """A tool called often that never changes a decision is not obviously
    useful, and that cannot be measured without recording consumption."""
    tr = Trace(budget_ms=1000)
    s1 = tr.add(Step(capability="a", outcome=T.SUCCESS, elapsed_ms=10))
    tr.add(Step(capability="b", outcome=T.SUCCESS, elapsed_ms=10))
    s1.consumed = True
    summary = tr.summary()
    check("both produced", summary["n_success"] == 2)
    check("one consumed", summary["n_consumed"] == 1, summary)
    check("the unconsumed one is named", summary["unconsumed"] == ["b"], summary)


# ── Evidence tiers ────────────────────────────────────────────────────────

def test_tiers_are_ordered_and_an_llm_sentence_cannot_decide():
    check("tier order is fixed", TIERS[0] == FACT and TIERS[-1] == LLM_EXPLANATION)
    check("llm is not a decision tier", LLM_EXPLANATION not in DECISION_TIERS)
    check("interpretation is not either", INTERPRETATION not in DECISION_TIERS)
    check("observation is", OBSERVATION in DECISION_TIERS)


def test_a_confident_interpretation_weighs_less_than_a_hedged_measurement():
    """Tier leads, so wording cannot buy authority."""
    prose = Item("a", LLM_EXPLANATION, "very bullish", "llm",
                 reliability=1.0, confidence=1.0, freshness=FRESH)
    measured = Item("b", OBSERVATION, "margins fell", "fundamentals",
                    reliability=0.6, confidence=0.4, freshness=FRESH)
    check("measurement outweighs prose", measured.weight > prose.weight,
          (measured.weight, prose.weight))
    check("and prose cannot decide", not prose.usable_for_decision)


def test_stale_evidence_is_down_weighted_and_barred_from_decisions():
    fresh = Item("a", OBSERVATION, "x", "s", reliability=0.8, confidence=0.8,
                 freshness=FRESH)
    stale = Item("b", OBSERVATION, "x", "s", reliability=0.8, confidence=0.8,
                 freshness=STALE)
    check("stale weighs less", stale.weight < fresh.weight)
    check("and cannot decide", not stale.usable_for_decision)


def test_an_unknown_tier_is_rejected_at_construction():
    try:
        Item("a", "VIBES", "x", "s")
        check("unknown tier raises", False)
    except ValueError:
        check("unknown tier raises", True)


def test_unavailable_evidence_is_not_decision_usable():
    i = Item("a", OBSERVATION, "options unavailable", "options",
             reliability=1.0, confidence=1.0, freshness=FRESH,
             flags=["unavailable"])
    check("absence is not a neutral reading", not i.usable_for_decision)


# ── Synthesis ─────────────────────────────────────────────────────────────

def _led(*specs):
    L = Ledger()
    for key, tier, src, direction, rel, conf, hz in specs:
        L.add(Item(key, tier, f"{src} says {direction}", src,
                   direction=direction, reliability=rel, confidence=conf,
                   freshness=FRESH, horizon=hz))
    return L


def test_synthesis_names_what_agrees_rather_than_averaging():
    L = _led(("f:q", OBSERVATION, "fundamentals", "BEARISH", 0.9, 0.8, "MEDIUM"),
             ("a:t", MODEL_OUTPUT, "algo", "BEARISH", 0.8, 0.75, "MEDIUM"),
             ("t:s", MODEL_OUTPUT, "technical", "NEUTRAL", 0.8, 0.7, "MEDIUM"))
    s = synthesize(L, horizon="MEDIUM")
    check("names both agreeing sources",
          "fundamentals" in s["statement"] and "algo" in s["statement"],
          s["statement"])
    check("names the neutral one", "technical" in s["statement"])
    check("no bare average appears", "0.4" not in s["statement"])


def test_one_directional_source_is_not_agreement():
    L = _led(("r:c", MODEL_OUTPUT, "research", "BULLISH", 0.8, 0.8, "MEDIUM"))
    s = synthesize(L)
    check("not reported as agreement", s["consensus"] == INSUFFICIENT,
          s["consensus"])
    check("and it says why", "not agreement" in s["statement"], s["statement"])
    check("the threshold is explicit", MIN_SOURCES_FOR_AGREEMENT == 2)


def test_directional_evidence_below_the_decision_tier_is_named_not_dropped():
    L = _led(("f:q", OBSERVATION, "fundamentals", "BEARISH", 0.9, 0.8, "MEDIUM"),
             ("a:t", MODEL_OUTPUT, "algo", "BEARISH", 0.8, 0.75, "MEDIUM"),
             ("s:s", INTERPRETATION, "social", "BULLISH", 0.3, 0.5, "SHORT"))
    s = synthesize(L, horizon="MEDIUM")
    check("social is mentioned", "social" in s["statement"], s["statement"])
    check("and excluded from the weighing",
          "social" in s["directional_but_not_weighed"], s)
    check("it does not count as a bullish source",
          "social" not in s["agrees_bullish"], s["agrees_bullish"])


def test_a_horizon_mismatch_is_not_a_real_conflict():
    a = Item("x:1", OBSERVATION, "s", "f", direction="BULLISH", horizon="SHORT",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    b = Item("x:1", OBSERVATION, "s", "g", direction="BEARISH", horizon="LONG",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    c = classify_conflict(a, b)
    check("not a real conflict", c["real"] is False, c)
    check("and it says both can hold",
          any("both can hold" in r for r in c["reasons"]), c["reasons"])


def test_a_genuine_conflict_is_reported_as_decision_changing():
    a = Item("x:1", OBSERVATION, "s", "f", direction="BULLISH", horizon="MEDIUM",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    b = Item("x:1", OBSERVATION, "s", "g", direction="BEARISH", horizon="MEDIUM",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    c = classify_conflict(a, b)
    check("real", c["real"] is True, c)
    check("decision changing", c["decision_changing"] is True)


def test_an_evenly_weighted_conflict_is_left_unresolved():
    a = Item("x:1", OBSERVATION, "s", "f", direction="BULLISH", horizon="MEDIUM",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    b = Item("x:1", OBSERVATION, "s", "g", direction="BEARISH", horizon="MEDIUM",
             reliability=0.8, confidence=0.8, freshness=FRESH)
    c = classify_conflict(a, b)
    check("unresolved is a finding", "unresolved" in c["resolution"],
          c["resolution"])


# ── Validator ─────────────────────────────────────────────────────────────

def _answer(**over):
    base = {"answerable": True, "plan": {"intent": "GENERAL_RESEARCH",
                                         "required_evidence": [],
                                         "horizon": {"days": 126},
                                         "position": {"owns": True}},
            "synthesis": {"consensus": "MIXED", "n_directional_sources": 2},
            "evidence": {"items": []}, "decision": {}}
    base.update(over)
    return base


def test_agreement_of_one_is_an_error():
    v = validate(_answer(synthesis={"consensus": "STRONG_AGREEMENT",
                                    "n_directional_sources": 1}))
    check("flagged", any(i["code"] == "AGREEMENT_OF_ONE" for i in v["issues"]))
    check("as an error", not v["passed"])


def test_an_uncalibrated_probability_is_an_error():
    v = validate(_answer(scenarios={"any_probability_stated": True,
                                    "calibrated": False}))
    check("flagged", any(i["code"] == "UNCALIBRATED_PROBABILITY"
                         for i in v["issues"]))


def test_a_catalyst_beyond_the_horizon_cannot_be_cited_as_bearing_on_it():
    v = validate(_answer(
        plan={"intent": "GENERAL_RESEARCH", "required_evidence": [],
              "horizon": {"days": 21}, "position": {"owns": True}},
        evidence={"items": [{"key": "catalyst:next", "value": 56, "flags": []}]},
        decision={"state": "HOLD", "why": "the catalyst supports it"}))
    check("flagged", any(i["code"] == "CATALYST_OUTSIDE_HORIZON"
                         for i in v["issues"]), v["issues"])


def test_prose_admitted_to_the_weighing_is_an_error():
    v = validate(_answer(evidence={"items": [
        {"key": "llm:take", "tier": "LLM_EXPLANATION",
         "usable_for_decision": True, "flags": []}]}))
    check("flagged", any(i["code"] == "PROSE_AS_DECISION_EVIDENCE"
                         for i in v["issues"]))


def test_an_unavailable_source_counted_as_evidence_is_an_error():
    v = validate(_answer(evidence={"items": [
        {"key": "options:x", "tier": "OBSERVATION",
         "usable_for_decision": True, "flags": ["unavailable"]}]}))
    check("flagged", any(i["code"] == "UNAVAILABLE_AS_EVIDENCE"
                         for i in v["issues"]))


def test_a_position_question_answered_without_a_position_is_an_error():
    v = validate(_answer(plan={"intent": "ADD_TO_POSITION",
                               "required_evidence": [],
                               "horizon": {"days": 126},
                               "position": {"owns": False}}))
    check("flagged", any(i["code"] == "POSITION_QUESTION_WITHOUT_POSITION"
                         for i in v["issues"]))


def test_required_evidence_is_checked_through_an_explicit_map():
    """The first version compared evidence KINDS against item key PREFIXES —
    different namespaces that never matched, so the warning fired on every
    answer and therefore meant nothing."""
    for kind in ("thesis", "levels", "risk", "statistical_edge", "catalysts"):
        check(f"{kind} has a satisfying item", kind in SATISFIED_BY, kind)
    v = validate(_answer(
        plan={"intent": "NEW_ENTRY", "required_evidence": ["thesis", "levels"],
              "horizon": {"days": 126}, "position": {"owns": True}},
        evidence={"items": [{"key": "research:composite", "flags": []},
                            {"key": "research:levels", "flags": []}]}))
    check("satisfied requirements do not warn",
          not any(i["code"] == "REQUIRED_EVIDENCE_MISSING" for i in v["issues"]),
          v["issues"])
    v2 = validate(_answer(
        plan={"intent": "NEW_ENTRY", "required_evidence": ["thesis", "levels"],
              "horizon": {"days": 126}, "position": {"owns": True}},
        evidence={"items": [{"key": "research:composite", "flags": []}]}))
    check("a genuine gap does warn",
          any(i["code"] == "REQUIRED_EVIDENCE_MISSING" for i in v2["issues"]),
          v2["issues"])


def test_a_clean_answer_passes():
    v = validate(_answer())
    check("no errors", v["n_errors"] == 0, v["issues"])


# ── Escalation ────────────────────────────────────────────────────────────

def _res(consensus, ratio, sources, intent, changing=0, missing=None, items=None):
    return {"synthesis": {"consensus": consensus, "agreement_ratio": ratio,
                          "n_directional_sources": sources,
                          "n_decision_changing": changing},
            "plan": {"intent": intent, "missing": missing or []},
            "evidence": {"items": items or []}}


def test_a_coherent_fast_pass_is_not_escalated():
    from mas.research.escalate import decide_depth
    d = decide_depth(_res("AGREEMENT", 0.7, 3, "GENERAL_RESEARCH"))
    check("stays fast", d["to"] == FAST, d)
    check("and says going deeper would change nothing",
          "change nothing" in d["statement"], d["statement"])


def test_conflict_escalates_to_deep():
    from mas.research.escalate import decide_depth
    d = decide_depth(_res("CONFLICTED", 0.5, 4, "NEW_ENTRY", changing=2))
    check("goes deep", d["to"] == DEEP, d)
    check("naming the disagreements", "decision-changing" in d["statement"])


def test_strong_agreement_on_a_consequential_decision_earns_a_challenge():
    from mas.research.escalate import decide_depth
    from mas.research.plan import ADVERSARIAL
    d = decide_depth(_res("STRONG_AGREEMENT", 1.0, 3, "ADD_TO_POSITION"))
    check("goes adversarial", d["to"] == ADVERSARIAL, d)
    check("because nobody looks for the counter-case then",
          "counter-case" in d["statement"], d["statement"])


def test_one_source_at_100_percent_is_not_strong_agreement():
    """A ratio of 1.0 from a single source is one measurement dividing by
    itself. Challenging it as consensus would dress a thin reading as a
    robust one."""
    from mas.research.escalate import decide_depth
    from mas.research.plan import ADVERSARIAL
    d = decide_depth(_res("INSUFFICIENT_EVIDENCE", 1.0, 1, "ADD_TO_POSITION"))
    check("not adversarial", d["to"] != ADVERSARIAL, d)


def test_the_escalation_reason_matches_the_target_it_escalated_to():
    """A flat reason list reported an intermediate escalation's reason beside
    the final target — 'escalating to adversarial' followed by why it went
    deep."""
    from mas.research.escalate import decide_depth
    d = decide_depth(_res("STRONG_AGREEMENT", 1.0, 3, "ADD_TO_POSITION"))
    check("the stated reason belongs to the final target",
          "counter-case" in d["reasons"][0], d["reasons"])


# ── Adversarial ───────────────────────────────────────────────────────────

def test_the_challenge_is_built_from_evidence_not_invented():
    from mas.research.adversarial import challenge
    items = [{"key": "research:composite", "source": "research",
              "direction": "BULLISH", "weight": 0.1, "flags": [],
              "statement": "composite reads ACCUMULATE"},
             {"key": "stats:backtest", "source": "backtester",
              "direction": "NOT_DIRECTIONAL", "weight": 0.2,
              "flags": ["no_demonstrated_edge"],
              "statement": "Nothing beat holding"}]
    c = challenge({"synthesis": {"bullish_weight": 0.1, "bearish_weight": 0.0,
                                 "n_directional_sources": 1},
                   "evidence": {"items": items},
                   "plan": {"required_evidence": [], "horizon": {"days": 126},
                            "missing": []}})
    check("it identifies a false-positive risk", c["false_positive_risks"], c)
    check("tied to the flagged item",
          any(r["what"] == "stats:backtest" for r in c["false_positive_risks"]))
    check("and names the single-source weakness",
          any("single measurement" in r["why"] or "one source" in r["why"]
              for r in c["false_positive_risks"]), c["false_positive_risks"])


def test_no_opposing_evidence_is_stated_as_such_not_fabricated():
    from mas.research.adversarial import challenge
    c = challenge({"synthesis": {"bullish_weight": 0.3, "bearish_weight": 0.0,
                                 "n_directional_sources": 2},
                   "evidence": {"items": [
                       {"key": "a", "source": "research", "direction": "BULLISH",
                        "weight": 0.3, "flags": [], "statement": "x"}]},
                   "plan": {"required_evidence": [], "horizon": {"days": 126},
                            "missing": []}})
    check("it says nothing argues the other way",
          c["challenges"][0]["kind"] == "NO_OPPOSING_EVIDENCE", c["challenges"])
    check("and flags that as suspicious rather than reassuring",
          "not looked for" in c["challenges"][0]["statement"])


def test_a_catalyst_outside_the_horizon_cannot_falsify_within_it():
    from mas.research.adversarial import challenge
    c = challenge({"synthesis": {"bullish_weight": 0.3, "bearish_weight": 0.0,
                                 "n_directional_sources": 2},
                   "evidence": {"items": [
                       {"key": "catalyst:next", "source": "catalysts",
                        "direction": "NOT_DIRECTIONAL", "weight": 0.5,
                        "flags": [], "value": 56,
                        "statement": "Earnings in 56 days"}]},
                   "plan": {"required_evidence": [], "horizon": {"days": 21},
                            "missing": []}})
    f = [x for x in c["falsifiers"] if x["from"] == "catalyst:next"]
    check("the falsifier is qualified", f, c["falsifiers"])
    check("because it falls outside the horizon",
          "outside this horizon" in f[0]["condition"], f)


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    a, p, _ = _grade()
    print(f"\nintent {a:.1%} · position {p:.1%} · {PASS} passed, {FAIL} failed")
