#!/usr/bin/env python3
"""The multi-agent layer: registry, contract, planner, executor, synthesis.

Routing is the part of a multi-agent system most likely to be wrong and least
likely to be noticed, because a mis-routed request still returns a confident
answer — just from the wrong specialist. So the planner is built to be
testable WITHOUT a network: every routing assertion below runs offline, for
every asset class, with no HTTP mocking at all.

The other thing these tests hold is the failure vocabulary. UNAVAILABLE (the
agent cannot answer this request), SKIPPED (nobody asked it) and ERROR (it
raised) are three different facts about an answer, and collapsing them is how
a broken specialist becomes indistinguishable from an absent one.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas import registry
from mas.contract import (AgentRequest, AgentResult, OK, UNAVAILABLE, SKIPPED,
                          ERROR, ok, unavailable, skipped, error)
from mas.plan import build_plan, describe, DEFAULT_CAPABILITIES
from mas.run import execute
from mas.synthesis import synthesize, to_evidence, BENCHMARK_DOMINANCE_R2

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


# ── Registry ──────────────────────────────────────────────────────────────

def test_registry_validates_at_load():
    reg = registry.load()
    check("agents present", len(reg["agents"]) >= 3)
    for aid, a in reg["agents"].items():
        check(f"{aid} declares a writes map", isinstance(a.get("writes"), dict))
        for c in a["capabilities"]:
            check(f"{aid}.{c} is a known capability", c in registry.CAPABILITIES)


def test_real_chains_outrank_model_pricing():
    """The ordering that keeps a model from ever shadowing a quote."""
    order = [a["id"] for a in registry.agents("option_structures", "EQUITY")]
    check("options_pilot is offered first", order[0] == "options_pilot", order)
    check("derivatives is the fallback", "derivatives" in order[1:], order)


def test_crypto_has_no_chain_agent_at_all():
    order = [a["id"] for a in registry.agents("option_structures", "CRYPTO")]
    check("only the model engine serves crypto", order == ["derivatives"], order)


def test_write_capabilities_are_declared():
    check("run_pipeline is declared as a write",
          registry.writes_for("options_pilot", "run_pipeline"))
    check("reading structures writes nothing",
          registry.writes_for("options_pilot", "option_structures") is None)
    check("the model engine writes nothing anywhere",
          registry.get("derivatives")["writes"] == {})


# ── Contract ──────────────────────────────────────────────────────────────

def test_a_decline_without_a_reason_is_rejected():
    """A SKIPPED or UNAVAILABLE with no reason is indistinguishable from a
    step that never ran — the exact bug the trace exists to prevent."""
    for status in (UNAVAILABLE, SKIPPED, ERROR):
        try:
            AgentResult(agent_id="x", capability="option_pricing", status=status)
            check(f"{status} without reason raises", False)
        except ValueError:
            check(f"{status} without reason raises", True)
    r = AgentResult(agent_id="x", capability="option_pricing", status=OK)
    check("OK needs no reason", r.ok)


def test_unknown_status_is_rejected():
    try:
        AgentResult(agent_id="x", capability="option_pricing", status="MAYBE")
        check("unknown status raises", False)
    except ValueError:
        check("unknown status raises", True)


# ── Planning, entirely offline ────────────────────────────────────────────

def test_crypto_never_routes_to_optionspilot():
    """The routing error with the worst consequence: OptionsPilot would spend
    a 45s timeout discovering what the symbol's shape already said."""
    p = build_plan("BTC-USD")
    for s in p["steps"]:
        for a in s["attempts"]:
            check("no optionspilot step for a coin",
                  a["agent_id"] != "options_pilot", a)
    check("classified as crypto", p["asset_class"] == "CRYPTO")


def test_equity_plans_optionspilot_first_then_the_model():
    p = build_plan("AAPL", params={"_test": True})
    os.environ.setdefault("OPTIONSPILOT_ACCESS_CODE", "")
    step = [s for s in p["steps"] if s["capability"] == "option_structures"]
    declined = [d for d in p["declined"] if d["capability"] == "option_structures"]
    check("the capability is either planned or explicitly declined",
          bool(step or declined))
    if step:
        ids = [a["agent_id"] for a in step[0]["attempts"]]
        if "options_pilot" in ids:
            check("optionspilot comes first", ids[0] == "options_pilot", ids)


def test_declining_always_carries_a_reason():
    for sym in ("BTC-USD", "EURUSD=X", "^GSPC", "GC=F", "AAPL"):
        p = build_plan(sym)
        for d in p["declined"]:
            check(f"{sym}: decline has a reason", bool(d.get("reason")), d)


def test_unknown_symbol_plans_nothing_but_explains_itself():
    p = build_plan("")
    check("nothing is planned", p["steps"] == [], p["steps"])
    check("and every requested capability is accounted for",
          {d["capability"] for d in p["declined"]}
          >= set(DEFAULT_CAPABILITIES),
          p["declined"])


def test_writes_are_never_planned():
    """Triggering OptionsPilot's pipeline freezes ideas into ITS journal. A
    planner must never decide to mutate another service's record."""
    p = build_plan("AAPL", capabilities=["option_structures"])
    for s in p["steps"]:
        for a in s["attempts"]:
            check("no planned step writes",
                  not registry.writes_for(a["agent_id"], s["capability"]), a)


def test_plan_is_describable_in_plain_sentences():
    lines = describe(build_plan("BTC-USD", view="BULLISH"))
    check("describes the classification", any("crypto" in l for l in lines), lines)
    check("names an agent it will ask",
          any("derivatives" in l for l in lines), lines)


def test_planning_does_no_network_io():
    """If planning touched the network, none of the assertions above would be
    trustworthy offline — and the UI could not show a plan before running it."""
    import socket
    real = socket.socket

    class Boom(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("build_plan opened a socket")

    socket.socket = Boom
    try:
        for sym in ("AAPL", "BTC-USD", "EURUSD=X", "^GSPC", "ZZZZ-USD", ""):
            build_plan(sym, view="BULLISH")
        check("planning is fully offline", True)
    finally:
        socket.socket = real


# ── Execution and isolation ───────────────────────────────────────────────

def _fake_plan(agent_id="derivatives", capability="option_structures"):
    return {"symbol": "TEST", "asset_class": "EQUITY", "classification": {},
            "capabilities_requested": [capability], "view": None,
            "steps": [{"capability": capability, "params": {},
                       "attempts": [{"agent_id": agent_id, "priority": 10,
                                     "timeout_sec": 5, "price_basis": "MODEL",
                                     "why": "test"}]}],
            "declined": []}


def test_an_agent_that_raises_becomes_an_error_not_a_crash():
    import mas.run as run_mod
    class Boom:
        def run(self, req):
            raise RuntimeError("specialist exploded")
    orig = registry.load_adapter
    registry.load_adapter = lambda aid: Boom()
    try:
        out = execute(_fake_plan())
        step = out["by_capability"]["option_structures"]
        check("no exception escaped", isinstance(out, dict))
        check("recorded as ERROR", step["attempts"][0]["status"] == ERROR,
              step["attempts"][0])
        check("the exception text is kept",
              "specialist exploded" in step["attempts"][0]["reason"])
        check("and the capability is unanswered", step["answered_by"] is None)
    finally:
        registry.load_adapter = orig


def test_an_adapter_returning_junk_is_an_error_not_a_silent_pass():
    class Junk:
        def run(self, req):
            return {"looks": "plausible"}
    orig = registry.load_adapter
    registry.load_adapter = lambda aid: Junk()
    try:
        out = execute(_fake_plan())
        a = out["by_capability"]["option_structures"]["attempts"][0]
        check("non-AgentResult is an ERROR", a["status"] == ERROR, a)
        check("and names the type", "AgentResult" in a["reason"], a["reason"])
    finally:
        registry.load_adapter = orig


def test_a_slow_agent_is_bounded_not_waited_on():
    import time
    class Slow:
        def run(self, req):
            time.sleep(5)
            return ok("slow", req.capability, {})
    orig = registry.load_adapter
    registry.load_adapter = lambda aid: Slow()
    try:
        plan = _fake_plan()
        plan["steps"][0]["attempts"][0]["timeout_sec"] = 0.2
        t0 = time.time()
        out = execute(plan)
        elapsed = time.time() - t0
        check("returned well before the agent would have", elapsed < 3.0,
              f"{elapsed:.2f}s")
        a = out["by_capability"]["option_structures"]["attempts"][0]
        check("recorded as UNAVAILABLE", a["status"] == UNAVAILABLE, a)
        check("and names the timeout", "0.2" in a["reason"], a["reason"])
    finally:
        registry.load_adapter = orig


def test_fallback_is_recorded_as_a_fallback():
    """An answer from the model engine because the chain was down is weaker
    than one from the chain. The reader has to be able to tell which."""
    class TwoStep:
        def __init__(self, aid): self.aid = aid
        def run(self, req):
            if self.aid == "options_pilot":
                return unavailable(self.aid, req.capability, "out of universe")
            return ok(self.aid, req.capability, {"candidates": []},
                      price_basis="MODEL")
    orig = registry.load_adapter
    registry.load_adapter = lambda aid: TwoStep(aid)
    try:
        plan = _fake_plan()
        plan["steps"][0]["attempts"] = [
            {"agent_id": "options_pilot", "priority": 10, "timeout_sec": 5,
             "price_basis": "TRADED_PRICE", "why": "t"},
            {"agent_id": "derivatives", "priority": 50, "timeout_sec": 5,
             "price_basis": "MODEL", "why": "t"}]
        step = execute(plan)["by_capability"]["option_structures"]
        check("the fallback answered", step["answered_by"] == "derivatives")
        check("and it is flagged as a fallback", step["fell_back"] is True)
        check("the first attempt's decline is preserved",
              step["attempts"][0]["status"] == UNAVAILABLE)
        check("basis reflects who actually answered",
              step["price_basis"] == "MODEL")
    finally:
        registry.load_adapter = orig


def test_later_attempts_are_skipped_not_run_once_one_answers():
    calls = []
    class Counting:
        def __init__(self, aid): self.aid = aid
        def run(self, req):
            calls.append(self.aid)
            return ok(self.aid, req.capability, {}, price_basis="TRADED_PRICE")
    orig = registry.load_adapter
    registry.load_adapter = lambda aid: Counting(aid)
    try:
        plan = _fake_plan()
        plan["steps"][0]["attempts"] = [
            {"agent_id": "options_pilot", "priority": 10, "timeout_sec": 5,
             "price_basis": "TRADED_PRICE", "why": "t"},
            {"agent_id": "derivatives", "priority": 50, "timeout_sec": 5,
             "price_basis": "MODEL", "why": "t"}]
        step = execute(plan)["by_capability"]["option_structures"]
        check("only the winner ran", calls == ["options_pilot"], calls)
        check("the other is SKIPPED, not absent",
              step["attempts"][1]["status"] == SKIPPED, step["attempts"][1])
        check("and says why it was skipped",
              "already answered" in step["attempts"][1]["reason"])
    finally:
        registry.load_adapter = orig


# ── Synthesis and the return path ─────────────────────────────────────────

def _bench_execution(r2, corr=0.9, action=None):
    return {
        "symbol": "ETH-USD", "asset_class": "CRYPTO", "answered": ["benchmark_relation"],
        "unanswered": [], "elapsed_ms": 1, "trace": [],
        "by_capability": {"benchmark_relation": {
            "capability": "benchmark_relation", "answered_by": "cross_asset",
            "price_basis": "PRICE_HISTORY_DERIVED", "fell_back": False,
            "attempts": [], "unanswered_reason": None,
            "result": {"agent_id": "cross_asset", "capability": "benchmark_relation",
                       "status": OK, "price_basis": "PRICE_HISTORY_DERIVED",
                       "reason": "", "provenance": {}, "elapsed_ms": 1,
                       "data": {"benchmark": "BTC-USD", "benchmark_label": "bitcoin",
                                "r_squared": r2, "correlation": corr,
                                "overlapping_returns": 300, "period": "1y",
                                "verdict": "bitcoin explains most of it"}}}}}


def test_a_dominant_benchmark_becomes_decisive_evidence():
    ev = to_evidence(_bench_execution(0.82))
    check("one evidence item", len(ev) == 1, ev)
    e = ev[0]
    check("category RELATIVE", e["category"] == "RELATIVE")
    check("marked DECISIVE", e["decision_relevance"] == "DECISIVE", e)
    check("flagged", "thesis_is_largely_a_benchmark_bet" in e["flags"])


def test_a_weak_benchmark_relation_is_context_not_decisive():
    e = to_evidence(_bench_execution(0.12, corr=0.35))[0]
    check("CONTEXT not DECISIVE", e["decision_relevance"] == "CONTEXT", e)
    check("no flag", e["flags"] == [], e)


def test_benchmark_evidence_is_not_directional():
    """'This moves with bitcoin' is not a claim that it goes up — the same
    reason the risk pillar is NOT_DIRECTIONAL."""
    for r2 in (0.1, 0.5, 0.95):
        e = to_evidence(_bench_execution(r2))[0]
        check(f"r2={r2} is NOT_DIRECTIONAL", e["direction"] == "NOT_DIRECTIONAL")


def test_evidence_items_satisfy_the_decision_layer_contract():
    """They are emitted as dicts so mas/ does not depend on decision/, but
    they must still construct a real DecisionEvidence — otherwise the return
    path silently drops them at the boundary."""
    from decision.evidence import DecisionEvidence
    for r2 in (0.05, 0.62, 0.99):
        for e in to_evidence(_bench_execution(r2)):
            obj = DecisionEvidence(**e)
            check("constructs", obj.source == "cross_asset")
            check("weight is reliability x magnitude",
                  abs(obj.weight - round(obj.reliability * obj.magnitude, 4)) < 1e-9)


def test_model_priced_volatility_is_never_emitted_as_market_expectation():
    """The single most misleading thing this layer could do: realized vol
    relabelled as what the market charges. It would look exactly like signal."""
    ex = {"symbol": "BTC-USD", "asset_class": "CRYPTO", "answered": [],
          "unanswered": [], "elapsed_ms": 1, "trace": [],
          "by_capability": {"implied_volatility": {
              "capability": "implied_volatility", "answered_by": "derivatives",
              "price_basis": "MODEL", "fell_back": True, "attempts": [],
              "unanswered_reason": None,
              "result": {"agent_id": "derivatives", "capability": "implied_volatility",
                         "status": OK, "price_basis": "MODEL", "reason": "",
                         "provenance": {}, "elapsed_ms": 1,
                         "data": {"surface": {"iv_rank": 0.9}}}}}}
    check("a MODEL basis emits no IV evidence", to_evidence(ex) == [], to_evidence(ex))


def test_dominant_benchmark_raises_a_tension_against_a_directional_core():
    s = synthesize(_bench_execution(0.82), core={"action": "ACCUMULATE"})
    check("a refinement is stated", len(s["refinements"]) >= 1, s["refinements"])
    check("a tension is raised", len(s["tensions"]) == 1, s["tensions"])
    check("it names the double-up risk",
          "double up" in s["tensions"][0]["tension"], s["tensions"])
    check("article agrees", "an ACCUMULATE" in s["tensions"][0]["tension"],
          s["tensions"][0]["tension"])


def test_no_tension_is_invented_when_the_core_has_no_view():
    s = synthesize(_bench_execution(0.82), core={"action": "HOLD"})
    check("a HOLD raises no tension", s["tensions"] == [], s["tensions"])
    s2 = synthesize(_bench_execution(0.82), core=None)
    check("no core raises no tension", s2["tensions"] == [], s2["tensions"])
    check("but the refinement still stands", len(s2["refinements"]) >= 1)


def test_synthesis_works_without_a_core_read_at_all():
    """A coin has no seven-pillar recommendation. The specialists still have
    something useful to say, and requiring a core would block the whole class."""
    s = synthesize(_bench_execution(0.4))
    check("synthesizes", s["symbol"] == "ETH-USD")
    check("class caveats are present", len(s["class_caveats"]) >= 1)
    check("crypto is told it has no accessible options venue",
          any("venue" in c for c in s["class_caveats"]), s["class_caveats"])


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
