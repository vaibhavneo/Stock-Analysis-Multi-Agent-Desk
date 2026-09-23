"""The loop: question -> plan -> tools -> specialists -> evidence -> synthesis.

This is the piece the baseline audit said was missing. Previously an intent
mapped onto one capability and that capability ran a fixed computation; there
was no representation of what the question needed, so nothing could reason
about sufficiency, freshness, cost or gaps.

WHAT RUNS HERE, AND WHAT DOES NOT
---------------------------------
Everything numeric is produced by the existing deterministic engines. This
module plans which of them to call, runs them under a budget, normalises what
they return into tiered evidence, and synthesises. It computes no price, no
level and no verdict of its own — a planner that also decided would make the
decision unreproducible, and reproducibility is the property the whole desk
is built around.

The LLM's place is strictly downstream of everything in this file.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..asset_class import classify as classify_symbol
from . import tools as T
from .evidence import (Item, Ledger, FACT, OBSERVATION, MODEL_OUTPUT,
                       HISTORICAL_STATISTIC, FORECAST, INTERPRETATION,
                       FRESH, AGEING, STALE, UNKNOWN_FRESHNESS)
from .budget import Accounting, BARS
from .execute import execute
from .plan import build_plan, FAST, DEEP
from .position import to_engine as position_to_engine
from .synthesis import synthesize

# How each capability's payload becomes tiered evidence. A capability that is
# not listed contributes nothing to the weighing rather than contributing an
# untyped item — an unclassified claim is exactly what the tier system exists
# to prevent.
# What kind of claim each pillar actually is.
PILLAR_TIER = {
    "fundamentals": OBSERVATION,       # reported financials
    "technical": MODEL_OUTPUT,         # a voting meter over indicators
    "algo": MODEL_OUTPUT,              # quantitative signals
    "risk": OBSERVATION,               # measured volatility and drawdown
    "research": INTERPRETATION,        # analyst consensus, an opinion poll
    "social": INTERPRETATION,          # sentiment from a self-selected crowd
}

# Pillars that measure something real but take no side. Risk is the canonical
# one: "less risky" is not the same claim as "goes up".
NON_DIRECTIONAL_PILLARS = frozenset({"risk"})

DIRECTION_FROM_ACTION = {
    "BUY": "BULLISH", "ACCUMULATE": "BULLISH",
    "REDUCE": "BEARISH", "SELL": "BEARISH", "HOLD": "NEUTRAL",
}


def _reliability(capability: str) -> float:
    from .. import registry
    try:
        agents = registry.agents(capability=capability)
        return max((a.get("reliability") or 0.5) for a in agents)
    except Exception:
        return 0.5


def _items_from(capability: str, data: Any, horizon: str) -> List[Item]:
    """Normalise one capability's output into evidence."""
    out: List[Item] = []
    if not isinstance(data, dict):
        return out
    rel = _reliability(capability)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    if capability == "equity_research":
        core = data.get("core") or data
        action = (core.get("action") or "").upper()
        if action:
            out.append(Item(
                key="research:composite", tier=MODEL_OUTPUT,
                statement=f"Composite {core.get('composite')} reads {action}",
                # NOT "research": there is also a `research` PILLAR (analyst
                # consensus), and calling both by one name made the synthesis
                # list a single source twice.
                source="composite", direction=DIRECTION_FROM_ACTION.get(action, "NEUTRAL"),
                horizon=horizon, reliability=rel, confidence=0.8,
                freshness=FRESH, observed_at=now, value=core.get("composite"),
                decision_relevance="DECISIVE",
                provenance={"capability": capability,
                            "weights": core.get("weights")}))
        # EACH PILLAR IS ITS OWN SOURCE. Emitting only the composite meant one
        # directional item per request, so every synthesis reported
        # INSUFFICIENT_EVIDENCE — technically true and practically useless,
        # because the pillars are separate measurements of separate things
        # and their agreement or disagreement is the whole question.
        #
        # Tiers differ by what each pillar actually is: reported financials
        # are an observation, a voting meter over indicators is a model
        # output, and a sentiment ratio from a self-selected population is an
        # interpretation that must not weigh on a decision.
        pillars = core.get("pillars") or {}
        for name, meta in (pillars.items() if isinstance(pillars, dict) else []):
            if not isinstance(meta, dict) or meta.get("applicable") is False:
                continue
            score = meta.get("score")
            if score is None:
                continue
            tier = PILLAR_TIER.get(name, MODEL_OUTPUT)
            direction = ("NOT_DIRECTIONAL" if name in NON_DIRECTIONAL_PILLARS
                         else "BULLISH" if score > 55
                         else "BEARISH" if score < 45 else "NEUTRAL")
            out.append(Item(
                key=f"pillar:{name}", tier=tier,
                statement=(f"{name.capitalize()} scores {score:g}"
                           + ("" if direction == "NOT_DIRECTIONAL"
                              else f" ({direction.lower()})")),
                source=name, direction=direction, horizon=horizon,
                reliability=rel,
                confidence=float(meta.get("confidence") or 0.5),
                freshness=FRESH, observed_at=now, value=score,
                flags=list(meta.get("flags") or []),
                decision_relevance=("CONTEXT" if tier == INTERPRETATION
                                    else "SUPPORTING"),
                provenance={"capability": capability, "pillar": name,
                            "backtestable": meta.get("backtestable")}))

        # Levels and scenarios are REQUIRED evidence for several intents, so
        # they have to leave the capability as items. Emitting only the
        # composite made every plan that needed them look unmet.
        levels = core.get("levels") or data.get("level_map") or {}
        n_levels = len((levels.get("supports") or [])) + \
            len((levels.get("resistances") or []))
        if n_levels:
            traded = sum(1 for r in ((levels.get("supports") or [])
                                     + (levels.get("resistances") or []))
                         if r.get("basis") == "TRADED_PRICE")
            out.append(Item(
                key="research:levels", tier=OBSERVATION,
                statement=(f"{n_levels} price levels, {traded} of them prices "
                           f"actually traded at"),
                source="research", direction="NOT_DIRECTIONAL", horizon=horizon,
                reliability=rel, confidence=0.8 if traded else 0.4,
                freshness=FRESH, observed_at=now, value=n_levels,
                flags=[] if traded else ["derived_only"],
                provenance={"capability": capability, "traded": traded}))

        scen = data.get("scenarios") or {}
        rows = scen.get("scenarios") or []
        if rows:
            out.append(Item(
                key="research:scenarios", tier=FORECAST,
                statement=(f"{len(rows)} scenarios"
                           + ("" if scen.get("any_probability_stated")
                              else ", none carrying a calibrated probability")),
                source="research", direction="NOT_DIRECTIONAL", horizon=horizon,
                reliability=rel, confidence=0.5, freshness=FRESH,
                observed_at=now, value=len(rows),
                provenance={"capability": capability}))

        vol = (core.get("volatility") or {})
        if vol.get("historical_volatility_20d") is not None:
            out.append(Item(
                key="risk:volatility", tier=OBSERVATION,
                statement=(f"{vol['historical_volatility_20d']}% annualized "
                           f"volatility, {str(vol.get('regime','')).lower()} "
                           f"for this class"),
                source="research", direction="NOT_DIRECTIONAL", horizon=horizon,
                reliability=rel, confidence=0.9, freshness=FRESH,
                observed_at=now, value=vol["historical_volatility_20d"],
                provenance={"capability": capability,
                            "annualization_days": vol.get("annualization_days")}))

    elif capability == "strategy_backtest":
        rows = data.get("rows") or []
        bh = data.get("buy_hold") or {}
        if rows:
            best = max(rows, key=lambda r: r.get("sharpe") or -99)
            beat = (bh.get("sharpe") is not None
                    and (best.get("sharpe") or -99) > bh["sharpe"])
            out.append(Item(
                key="stats:backtest", tier=HISTORICAL_STATISTIC,
                statement=("The best strategy beat buy-and-hold on this name"
                           if beat else
                           "Nothing here beat simply holding the stock"),
                source="backtester", direction="NOT_DIRECTIONAL",
                horizon=horizon, reliability=rel, confidence=0.7,
                freshness=FRESH, observed_at=now,
                value={"best_sharpe": best.get("sharpe"),
                       "buy_hold_sharpe": bh.get("sharpe")},
                decision_relevance="DECISIVE",
                flags=[] if beat else ["no_demonstrated_edge"],
                provenance={"capability": capability, "n_strategies": len(rows)}))

    elif capability == "event_calendar":
        nxt = data.get("next_event") or data.get("next_scheduled_event")
        if nxt and nxt.get("days_away") is not None:
            out.append(Item(
                key="catalyst:next", tier=FACT,
                statement=(f"{nxt.get('event')} on {nxt.get('date')}, "
                           f"{nxt['days_away']} days away"),
                source="catalysts", direction="NOT_DIRECTIONAL", horizon=horizon,
                reliability=rel, confidence=0.9, freshness=FRESH,
                observed_at=now, data_timestamp=nxt.get("date"),
                value=nxt.get("days_away"),
                provenance={"capability": capability, "category": nxt.get("category")}))

    elif capability == "benchmark_relation":
        r2 = data.get("r_squared")
        if r2 is not None:
            out.append(Item(
                key="relative:benchmark", tier=OBSERVATION,
                statement=data.get("verdict") or "",
                source="cross_asset", direction="NOT_DIRECTIONAL", horizon="ALL",
                reliability=rel,
                confidence=0.8 if (data.get("overlapping_returns") or 0) >= 120 else 0.5,
                freshness=FRESH, observed_at=now, value=r2,
                decision_relevance="DECISIVE" if r2 >= 0.6 else "CONTEXT",
                provenance={"capability": capability,
                            "benchmark": data.get("benchmark"),
                            "overlap": data.get("overlapping_returns")}))

    elif capability == "market_regime":
        out.append(Item(
            key="macro:regime", tier=OBSERVATION,
            statement=(f"Market trend {str(data.get('trend','')).lower()}, "
                       f"{str(data.get('risk_stance','')).replace('_','-').lower()}"),
            source="regime", direction="NOT_DIRECTIONAL", horizon="ALL",
            reliability=rel, confidence=float(data.get("confidence") or 0.6),
            freshness=FRESH, observed_at=now, value=data.get("vix_level"),
            provenance={"capability": capability}))

    elif capability == "forward_record":
        lr = data.get("live_record") or {}
        rows = [r for r in (lr.get("rows") or []) if r.get("readable")]
        out.append(Item(
            key="stats:forward_record", tier=HISTORICAL_STATISTIC,
            statement=(lr.get("statement") or
                       "No horizon has enough matured predictions to quote"),
            source="track_record", direction="NOT_DIRECTIONAL", horizon="ALL",
            reliability=rel, confidence=0.9 if rows else 0.2,
            freshness=FRESH, observed_at=now,
            value=lr.get("total_matured"),
            flags=[] if rows else ["insufficient_sample"],
            provenance={"capability": capability}))

    elif capability == "option_structures":
        if data.get("status") == "OK":
            out.append(Item(
                key="options:structures", tier=MODEL_OUTPUT,
                statement=data.get("headline") or "",
                source="options", direction="NOT_DIRECTIONAL", horizon=horizon,
                reliability=0.9 if data.get("basis") == "TRADED_PRICE" else 0.55,
                confidence=0.7, freshness=FRESH, observed_at=now,
                value=data.get("n_candidates"),
                provenance={"capability": capability,
                            "basis": data.get("basis"),
                            "answered_by": data.get("answered_by")}))
    return out


def research(question: str,
             symbols: Optional[List[str]] = None,
             context_symbol: Optional[str] = None,
             supplied_position: Optional[Dict[str, Any]] = None,
             depth: str = FAST,
             runner=None,
             timeout_sec: float = 30.0,
             on_progress=None) -> Dict[str, Any]:
    """One research request, end to end. Never raises.

    `on_progress(stage, detail)` is called as each stage completes. Research
    that blocks a UI for seconds with no signal is indistinguishable from
    research that has hung — the previous production timeout proved that.
    """
    t0 = time.time()

    def _emit(stage, **detail):
        if on_progress:
            try:
                on_progress(stage, detail)
            except Exception:
                pass
    syms = [s.upper() for s in (symbols or [])]
    primary = syms[0] if syms else (context_symbol or "")
    cls = classify_symbol(primary) if primary else {"asset_class": "UNKNOWN"}
    asset_class = cls.get("asset_class", "UNKNOWN")

    _emit("planning", question=question, symbols=syms,
          asset_class=asset_class)
    plan = build_plan(question, symbols=syms, asset_class=asset_class,
                      context_symbol=context_symbol,
                      supplied_position=supplied_position, depth=depth)
    _emit("planned", intent=plan.intent, answerable=plan.answerable,
          required=plan.required_evidence,
          capabilities=[c["capability"] for c in plan.capabilities])

    out: Dict[str, Any] = {
        "question": question, "plan": plan.to_dict(),
        "plan_description": plan.describe(),
        "tool_discovery": T.discover(asset_class),
        "asset_class": asset_class, "symbols": syms,
    }

    if not plan.answerable:
        out.update({"answerable": False, "evidence": {"items": [], "summary": {}},
                    "synthesis": None, "trace": {},
                    "elapsed_ms": int((time.time() - t0) * 1000)})
        return out

    if runner is None:
        runner = _default_runner(primary, asset_class, plan)

    acct = Accounting(budget_ms=plan.budget_ms)
    ex = execute(plan, runner, timeout_sec=timeout_sec, on_progress=on_progress)
    for cap, step in (ex["steps"] or {}).items():
        acct.record(cap, llm=(cap == "analyst_narrative"),
                    external=(cap != "forward_record"))
    ledger = Ledger()
    horizon = (plan.horizon or {}).get("horizon", "ALL")

    for cap, step in (ex["steps"] or {}).items():
        if step.outcome != T.SUCCESS:
            ledger.add(Item(
                key=f"{cap}:unavailable", tier=OBSERVATION,
                statement=f"{cap.replace('_', ' ')} produced nothing: {step.reason}",
                source=cap, direction="NOT_DIRECTIONAL", horizon="ALL",
                reliability=1.0, confidence=1.0, freshness=UNKNOWN_FRESHNESS,
                flags=["unavailable"],
                provenance={"outcome": step.outcome, "reason": step.reason}))
            continue
        produced = _items_from(cap, step.data, horizon)
        for it in produced:
            ledger.add(it)
        step.records = len(produced)

    _emit("evidence", n_items=len(ledger.items),
          n_usable=len(ledger.decision_usable()))
    syn = synthesize(ledger, horizon=horizon)
    _emit("synthesis", consensus=syn.get("consensus"),
          sources=syn.get("n_directional_sources"))

    # Consumption is recorded on the STEP, so tool value can be measured
    # later: a capability that produced evidence nothing used is visible.
    weighed_keys = set(syn.get("weighed_keys") or [])
    consumed_keys = {i.key for i in ledger.consumed()}
    for cap, step in (ex["steps"] or {}).items():
        mine = [i for i in ledger.items
                if i.provenance.get("capability") == cap]
        step.consumed = any(i.key in consumed_keys for i in mine)
        step.changed_synthesis = any(i.key in weighed_keys for i in mine)

    # Recomputed AFTER consumption is marked. The first version reported the
    # summary built during execution, which was necessarily "0 used" because
    # nothing had been consumed yet — a trace that contradicted the evidence
    # ledger sitting beside it.
    out.update({
        "answerable": True,
        "evidence": ledger.to_dict(),
        "synthesis": syn,
        "trace": {"steps": [s.to_dict() for s in ex["trace"].steps],
                  "summary": ex["trace"].summary()},
        "elapsed_ms": int((time.time() - t0) * 1000),
    })
    acct.elapsed_ms = out["elapsed_ms"]
    acct.cache_hits = BARS.hits
    acct.cache_misses = BARS.misses
    out["accounting"] = acct.to_dict()
    out["cache"] = BARS.stats()
    # The decision engine's own fields, selected by what was asked. It
    # produced a state, an add verdict, an entry plan, invalidation triggers
    # and a monitoring schedule all along; the research answer reached them
    # and rendered none, so an ADD question got evidence and synthesis while
    # `add_analysis.verdict` sat one dict away and unread.
    from .decision import build as _build_decision
    decision_obj = None
    for cap, step in (ex["steps"] or {}).items():
        if cap == "equity_research" and isinstance(step.data, dict):
            decision_obj = step.data.get("decision")
            break
    out["decision"] = _build_decision(decision_obj, plan.intent)
    # Kept for the brief, which builds the full section set from it.
    out["_decision_object"] = decision_obj
    _emit("decision", n_sections=(out["decision"] or {}).get("n_produced"))

    # Change detection runs BEFORE the contract check, because
    # `changed_since_previous` is a contract field and stamping it afterwards
    # would report every specialist as non-compliant on a field the pipeline
    # had simply not filled in yet. The prior is read before this result is
    # stored, or the comparison would be against itself.
    try:
        from .change import compare, UNCHANGED
        from . import snapshots
        prior = snapshots.latest(primary) if primary else None
        out["change"] = compare(prior, out)
        by_key = {c["key"]: c for c in (out["change"].get("changes") or [])}
        for i in ledger.items:
            c = by_key.get(i.key)
            i.changed_since_previous = (
                c["kind"] if c else
                ("NO_PRIOR" if not out["change"].get("has_previous")
                 else UNCHANGED))
        out["evidence"] = ledger.to_dict()
        if primary:
            snapshots.save(primary, out)
    except Exception as e:
        out["change"] = {"has_previous": False, "changes": [],
                         "statement": f"Change detection was unavailable "
                                      f"({type(e).__name__})."}
        for i in ledger.items:
            i.changed_since_previous = "UNKNOWN"

    # Did each specialist meet the contract it was given? Reported, never
    # enforced by rejection: discarding a useful answer over a missing key
    # would be a worse failure than the one being checked for.
    from .delegation import (validate_response as _contract,
                             REQUIRED_FIELDS as _REQUIRED)
    compliance = []
    for cap, step in (ex["steps"] or {}).items():
        if step.outcome != T.SUCCESS:
            continue
        mine = [i for i in ledger.items
                if i.provenance.get("capability") == cap]
        if not mine:
            compliance.append({"capability": cap, "complete": False,
                               "present": [], "missing": list(_REQUIRED),
                               "statement": ("ran, but produced no normalised "
                                             "evidence, so nothing entered "
                                             "the comparison")})
            continue
        # Checked on the NORMALISED evidence — that is the layer the contract
        # lives at and the layer synthesis compares. Checking the raw adapter
        # payload measured the wrong thing and reported 0 of 4.
        worst = min((_contract(cap, i.contract()) for i in mine),
                    key=lambda c: len(c["present"]))
        worst["n_items"] = len(mine)
        compliance.append(worst)
    out["specialist_contract"] = {
        "checked": compliance,
        "n_complete": sum(1 for c in compliance if c["complete"]),
        "n_checked": len(compliance),
        "statement": (
            (lambda n, t: (
                f"All {t} specialists returned the full structured contract."
                if n == t and t else
                f"{n} of {t} specialists returned the full structured "
                f"contract; the remaining {t - n} answered and had their gaps "
                f"recorded rather than their evidence discarded."
                if t else "No specialist produced evidence to check."))(
                    sum(1 for c in compliance if c["complete"]), len(compliance))),
    }

    # Depth is decided AFTER the first pass, from what it actually found —
    # not from the wording of the question. An ADVERSARIAL pass runs only
    # when the evidence is strong enough that nobody would otherwise look for
    # the counter-case.
    from .escalate import decide_depth
    from .adversarial import challenge as _challenge
    esc = decide_depth(out, current=depth)
    out["escalation"] = esc

    if esc["to"] == "ADVERSARIAL":
        out["adversarial"] = _challenge(out)

    # Freeze the research itself, not only the brief it produced. Without
    # this a decision can be re-read later with no way to reconstruct which
    # plan was made, which tools were reachable, or what was missing.
    try:
        from .journal import journal as _journal_research
        out["journal_id"] = _journal_research(out)
    except Exception:
        out["journal_id"] = None

    # Accumulate what each capability contributed, so "is this tool worth its
    # latency" becomes a question about a distribution rather than a hunch.
    try:
        from .toolvalue import record as _record_value
        _record_value(out)
    except Exception:
        pass

    try:
        from .brief import build as _build_brief
        out["brief"] = _build_brief(out)
    except Exception as e:
        out["brief"] = {"sections": [],
                        "statement": f"The brief could not be assembled "
                                     f"({type(e).__name__})."}

    _emit("done", elapsed_ms=out.get("elapsed_ms"))
    out.pop("_decision_object", None)
    out["_ledger"] = ledger
    return out


# Evidence kinds only the FULL decision build produces. A plan that requires
# any of them must reach the decision engine — the fast pillar read cannot
# supply a level ladder or a scenario set, and answering a position question
# without them is what the baseline audit found: a composite score standing in
# for an add decision.
NEEDS_FULL_BUILD = frozenset({"levels", "scenarios"})


def _default_runner(symbol: str, asset_class: str, plan):
    """Calls the REAL registered capabilities. Nothing is simulated."""
    from ..contract import AgentRequest
    from .. import registry

    position = position_to_engine(plan.position)

    def runner(capability: str):
        if capability == "option_structures":
            from ..options_brief import build as build_options
            return {"status": "OK", "data": build_options(symbol)}
        agents = registry.agents(capability=capability,
                                 asset_class=asset_class,
                                 have_symbol=bool(symbol))
        if not agents:
            return {"status": "UNAVAILABLE",
                    "reason": f"no agent offers {capability} for {asset_class}"}
        for a in agents:
            mod = registry.load_adapter(a["id"])
            params: Dict[str, Any] = {}
            brief = next((c.get("brief") for c in (plan.capabilities or [])
                          if c["capability"] == capability), None)
            if brief:
                # The specialist receives the specific question it must
                # settle. Adapters that ignore it are unchanged; the one that
                # reads it (the LLM analysts) stops being asked "analyze this
                # stock".
                params["research_question"] = brief["question"]
                params["expects"] = brief["expects"]
            if capability == "equity_research":
                if position:
                    params["position"] = position
                if NEEDS_FULL_BUILD & set(plan.required_evidence or []):
                    params["full"] = True
            res = mod.run(AgentRequest(symbol=symbol, asset_class=asset_class,
                                       capability=capability, params=params))
            if getattr(res, "status", None) == "OK":
                return {"status": "OK", "data": res.data}
            last = res
        return {"status": "UNAVAILABLE",
                "reason": getattr(last, "reason", "every agent declined")}

    return runner
