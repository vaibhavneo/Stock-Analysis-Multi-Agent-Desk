"""Request -> an inspectable execution plan.

THE PLAN IS DATA, AND IT IS BUILT WITHOUT I/O
---------------------------------------------
Two reasons, both load-bearing.

First, testability. Routing is the part of a multi-agent system most likely to
be wrong and least likely to be noticed, because a mis-routed request still
returns a confident answer from the wrong specialist. A plan that can be built
with no network is a plan that can be asserted against in a unit test, for
every asset class, without mocking a single HTTP call.

Second, honesty to the user. The plan can be shown BEFORE it runs — "here is
who I am about to ask, and why" — and it names the agents it decided NOT to
ask, with reasons. A specialist that was never consulted and a specialist that
declined are completely different facts about an answer, and a system that
shows only its successes cannot distinguish them.

WRITES ARE NEVER PLANNED
------------------------
`registry.writes_for()` names every capability that mutates another service.
Those are excluded here unconditionally. Triggering OptionsPilot's pipeline
freezes ideas into its journal — a side effect on someone else's record — and
it must stay a thing a user presses, not a thing a planner decides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import registry
from .asset_class import classify, spec_for

# What a general "analyse this symbol" request asks for.
DEFAULT_CAPABILITIES = ("benchmark_relation", "option_structures")


def _adapter_precheck(agent_id: str, symbol: str, asset_class: str):
    """The adapter's own cheap, local availability test. Never does I/O —
    adapters that need a network call to know report it at run time instead."""
    try:
        mod = registry.load_adapter(agent_id)
        fn = getattr(mod, "available", None)
        if fn is None:
            return True, ""
        return fn(symbol, asset_class)
    except Exception as e:
        return False, f"the adapter could not be loaded ({type(e).__name__}: {e})"


def build_plan(symbol: str,
               metadata: Optional[Dict[str, Any]] = None,
               capabilities: Optional[Sequence[str]] = None,
               view: Optional[str] = None,
               params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Plan the specialist calls for one symbol. Pure.

    `view` is THIS engine's directional read, handed down explicitly. The
    orchestrator holds a thesis before it asks anyone about options, so it
    tells the specialists which direction to express rather than letting each
    one guess — two agents guessing independently is how a bullish structure
    and a bearish one end up in the same answer.
    """
    cls = classify(symbol, metadata)
    asset_class = cls["asset_class"]
    spec = spec_for(asset_class)
    wanted = tuple(capabilities or DEFAULT_CAPABILITIES)
    base_params = dict(params or {})
    if view:
        base_params["view"] = view

    steps: List[Dict[str, Any]] = []
    declined: List[Dict[str, Any]] = []

    for cap in wanted:
        if cap not in registry.CAPABILITIES:
            declined.append({"agent_id": None, "capability": cap,
                             "reason": f"{cap!r} is not a known capability"})
            continue

        offering = registry.agents(capability=cap, asset_class=asset_class)
        if not offering:
            all_offering = registry.agents(capability=cap)
            names = ", ".join(a["id"] for a in all_offering) or "no agent"
            declined.append({
                "agent_id": None, "capability": cap,
                "reason": (f"no agent offers {cap} for {asset_class}. "
                           f"{names} offer{'s' if len(all_offering) == 1 else ''} it, "
                           f"but not for this asset class.")})
            continue

        # Priority order: a real chain before a model. Candidates are ATTEMPTS,
        # tried until one answers — not parallel competitors.
        attempts: List[Dict[str, Any]] = []
        for a in offering:
            writes = registry.writes_for(a["id"], cap)
            if writes:
                declined.append({"agent_id": a["id"], "capability": cap,
                                 "reason": f"excluded from planning: it writes — {writes}"})
                continue
            ready, why = _adapter_precheck(a["id"], symbol, asset_class)
            if not ready:
                declined.append({"agent_id": a["id"], "capability": cap,
                                 "reason": why or "the adapter declined"})
                continue
            attempts.append({
                "agent_id": a["id"],
                "price_basis": a.get("price_basis"),
                "priority": a.get("priority"),
                "timeout_sec": a.get("timeout_sec", 30),
                "why": (f"{a['name']} offers {cap} for {asset_class} "
                        f"({a.get('price_basis')}, priority {a.get('priority')})"),
            })

        if attempts:
            steps.append({"capability": cap, "params": dict(base_params),
                          "attempts": attempts})
        elif not any(d["capability"] == cap for d in declined):
            declined.append({"agent_id": None, "capability": cap,
                             "reason": "every agent offering it declined"})

    return {
        "symbol": cls["symbol"],
        "classification": cls,
        "asset_class": asset_class,
        "capabilities_requested": list(wanted),
        "view": view,
        "steps": steps,
        "declined": declined,
        "notes": [n for n in [spec.note] if n],
    }


def describe(plan: Dict[str, Any]) -> List[str]:
    """The plan in plain sentences, for showing before it runs."""
    out = [f"{plan['symbol']} is {plan['classification']['label']} "
           f"({plan['classification']['reason']})."]
    for s in plan["steps"]:
        first = s["attempts"][0]
        line = f"Ask {first['agent_id']} for {s['capability'].replace('_', ' ')}"
        if len(s["attempts"]) > 1:
            line += (", falling back to "
                     + " then ".join(a["agent_id"] for a in s["attempts"][1:]))
        out.append(line + ".")
    for d in plan["declined"]:
        who = d["agent_id"] or "nothing"
        out.append(f"Not asking {who} for "
                   f"{d['capability'].replace('_', ' ')}: {d['reason']}")
    return out
