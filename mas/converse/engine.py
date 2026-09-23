"""One conversational turn: parse, plan, run the specialists, say it back.

The orchestrator's loop, made conversational. Nothing about the specialists
changes — the same roster, the same contract, the same trace. What changes is
the front door: the subject and the intent are read from a sentence instead of
from a ticker box and a button.

Two properties are deliberate:

  - Every turn RE-RUNS the specialists. Caching a verdict across turns would
    serve yesterday's price as today's advice with nothing on screen to say so.
  - The trace survives into the reply. A conversational interface is the
    easiest place in a system to lose the distinction between "the specialist
    said no" and "nobody asked it", and that distinction is the whole basis on
    which a user should trust any of this.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import reply as reply_mod
from . import session as session_mod
from .intent import parse

# Capabilities that need no symbol; they get one plan, not one per name.
GLOBAL_CAPS = frozenset({"market_regime", "forward_record", "portfolio_review",
                         "strategy_backtest"})

# Intents whose answer is a DECISION rather than a lookup. These are the ones
# the flat capability path answered badly — a position question got a
# composite score, and "what would invalidate this" got nothing at all.
RESEARCH_INTENTS = frozenset({
    "ADD_TO_POSITION", "REDUCE_POSITION", "EXIT_POSITION",
    "EXISTING_POSITION", "NEW_ENTRY", "INVALIDATION", "RISK_ANALYSIS",
    "EARNINGS_PREVIEW", "SHORT_TERM_SETUP", "MEDIUM_TERM_SETUP",
    "LONG_TERM_THESIS",
})


def _run_capability(cap: str, symbol: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """One capability for one subject, through the roster."""
    from ..plan import build_plan
    from ..run import execute
    plan = build_plan(symbol or "", capabilities=[cap],
                      view=params.get("view"), params=params)
    ex = execute(plan)
    step = (ex.get("by_capability") or {}).get(cap) or {}
    step["symbol"] = symbol
    step["declined_at_planning"] = [
        d for d in (plan.get("declined") or []) if d.get("capability") == cap]
    if not step.get("result") and step["declined_at_planning"]:
        step["unanswered_reason"] = "; ".join(
            d["reason"] for d in step["declined_at_planning"] if d.get("reason"))
    return step


def _options_for(symbol: str, view: Optional[str]) -> Dict[str, Any]:
    """Options go through options_brief so the conversational answer and the
    written brief produce the same structures and the same plain sentences."""
    from ..options_brief import build as build_options
    section = build_options(symbol, view=view)
    return {
        "capability": "option_structures", "symbol": symbol,
        "answered_by": section.get("answered_by"),
        "unanswered_reason": section.get("reason"),
        "result": {"status": "OK" if section.get("status") == "OK" else "UNAVAILABLE",
                   "data": section, "price_basis": section.get("basis")},
        "trace": section.get("trace") or [],
    }


def turn(text: str, session_id: Optional[str] = None,
         holdings: Optional[List[Dict[str, Any]]] = None,
         full_research: bool = False,
         supplied_position: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Answer one message. Never raises."""
    t0 = time.time()
    sess = session_mod.get(session_id)
    parsed = parse(text, context_symbol=sess.get("subject"))

    results: Dict[str, Any] = {}
    symbols = parsed.get("symbols") or []
    subject = symbols[0] if symbols else sess.get("subject")
    policy_out: Dict[str, Any] = {}

    if parsed.get("kind") == "QUERY":
        # Depth CHAT implies NOTHING. A question about a share price is a
        # question about a share price: the router found what was asked for,
        # and the orchestrator adds no specialists on top of it. Running the
        # decisions through the policy anyway keeps one vocabulary for why a
        # capability ran, wherever the request came from.
        from ..asset_class import classify as _classify
        from ..policy import decide as _decide, CHAT as _CHAT
        _cls = _classify(subject or "")
        _pol = _decide(parsed["capabilities"], _cls["asset_class"],
                       depth=_CHAT, have_symbol=bool(subject),
                       have_holdings=bool(holdings))
        policy_out = _pol.to_dict()
        parsed = dict(parsed)
        parsed["capabilities"] = _pol.run

    if parsed.get("kind") == "QUERY":
        view = None
        # Research first when it is wanted: its direction is what the options
        # specialist should express, so deriving one independently would let a
        # single answer hold a bullish structure and a bearish verdict.
        if "equity_research" in parsed["capabilities"] and subject:
            step = _run_capability("equity_research", subject,
                                   {"full": full_research})
            results["equity_research"] = step
            res = step.get("result") or {}
            if res.get("status") == "OK":
                view = ((res.get("data") or {}).get("core") or {}).get("view")

        # A follow-up that asks only for options still needs a direction. The
        # session deliberately does NOT cache the verdict — serving yesterday's
        # read as today's would be invisible to the user — so the direction is
        # RE-DERIVED here rather than remembered. Without this, "and the
        # options?" after an ACCUMULATE returned range structures, which is a
        # different trade than the one the desk had just argued for.
        if (view is None and "option_structures" in parsed["capabilities"]
                and subject):
            quiet = _run_capability("equity_research", subject, {})
            qres = quiet.get("result") or {}
            if qres.get("status") == "OK":
                view = ((qres.get("data") or {}).get("core") or {}).get("view")
                results["_direction_source"] = quiet

        for cap in parsed["capabilities"]:
            if cap in results:
                continue
            if cap == "option_structures":
                if subject:
                    results[cap] = _options_for(subject, view)
                else:
                    results[cap] = {"capability": cap, "result": None,
                                    "unanswered_reason": "no symbol to price"}
                continue
            params: Dict[str, Any] = {}
            if cap == "portfolio_review":
                params["holdings"] = holdings or []
            sym = "" if cap in GLOBAL_CAPS and not subject else (subject or "")
            if cap == "strategy_backtest":
                sym = subject or ""
            step = _run_capability(cap, sym, params)
            # A bounded-out analyst pass is not a failure, it is a job too long
            # for this surface. Say where it does run rather than reporting a
            # timeout the user can do nothing with.
            if (cap == "analyst_narrative"
                    and not (step.get("result") or {}).get("status") == "OK"
                    and "within" in (step.get("unanswered_reason") or "")):
                from ..agents.desk_capabilities import ANALYST_STREAMING_ROUTE
                step["unanswered_reason"] = ANALYST_STREAMING_ROUTE
            results[cap] = step

    # Decision-shaped questions go through the RESEARCH pipeline: plan,
    # discovery, parallel execution, tiered evidence, synthesis, validation.
    # The flat capability dispatch above answered "should I add?" with a
    # composite score while add_analysis, the entry plan and the invalidation
    # sat unreachable — the defect the baseline audit named.
    research_out = None
    if parsed.get("kind") == "QUERY" and subject:
        try:
            from ..research.orchestrator import research as _research
            from ..research.plan import SPECS as _SPECS
            from ..research.intent import classify as _rclassify
            from ..research.intent_corpus import NONE as _RNONE
            from ..research import reply as _rreply
            from ..research.validate import validate as _validate

            _ri = _rclassify(text, context_symbol=sess.get("subject"),
                             has_symbol=bool(symbols), n_symbols=len(symbols))
            if _ri["intent"] in RESEARCH_INTENTS:
                research_out = _research(
                    text, symbols=symbols or [subject],
                    context_symbol=sess.get("subject"),
                    supplied_position=supplied_position, timeout_sec=25.0)
                research_out["validation"] = _validate(research_out)
                answer = _rreply.compose(research_out)
        except Exception as e:
            research_out = {"error": f"{type(e).__name__}: {e}"}

    if research_out is None or research_out.get("error"):
        answer = reply_mod.compose(parsed, results, subject)
    session_mod.record(sess, text, parsed, answer)

    trace: List[Dict[str, Any]] = []
    for cap, step in results.items():
        for a in (step.get("trace") or step.get("attempts") or []):
            trace.append({"capability": cap, "agent_id": a.get("agent_id"),
                          "status": a.get("status"),
                          "elapsed_ms": a.get("elapsed_ms"),
                          "reason": a.get("reason")})
        for d in (step.get("declined_at_planning") or []):
            trace.append({"capability": cap, "agent_id": d.get("agent_id"),
                          "status": "SKIPPED", "elapsed_ms": None,
                          "reason": d.get("reason")})

    return {
        "session": session_mod.public(sess),
        "understood": {
            "kind": parsed.get("kind"),
            "symbols": parsed.get("symbols"),
            "symbol_basis": parsed.get("symbol_basis"),
            "from_context": parsed.get("symbols_from_context"),
            "capabilities": parsed.get("capabilities"),
            "why": parsed.get("why"),
            "scores": parsed.get("scores"),
        },
        "reply": answer,
        "research": ({k: v for k, v in research_out.items()
                      if k not in ("_ledger",)} if research_out else None),
        "policy": policy_out,
        "trace": trace,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
