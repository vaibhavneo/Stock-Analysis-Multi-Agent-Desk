"""The research answer's own decision fields, selected by what was asked.

WHY THIS EXISTS
---------------
The decision engine already produces a state, an entry plan, an add verdict,
a playbook, sizing, invalidation triggers and a monitoring schedule. The
research pipeline reached all of it and rendered none of it: a question about
adding got evidence and synthesis, while `add_analysis.verdict` — the field
that answers that exact question, and carries the guard against averaging down
— sat one dict away and unread.

WHAT THIS IS NOT
----------------
It does not decide anything. Every value here is lifted from
`decision/`'s deterministic output; this module chooses WHICH of those fields
the question asked for and states them in the question's own terms. An ADD
question gets the add verdict and its blockers; an EXIT question gets the
playbook's exit conditions; a RISK question gets the budget and the
invalidation distance. Returning all of them for every question would be the
same failure as returning none.

NO ACTION IS FORCED
-------------------
`INSUFFICIENT_DATA` and `CONFLICTED` are real answers and appear whenever the
engine produced them. A decision layer that always names an action is not a
decision layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .intent_corpus import (ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                            EXISTING_POSITION, NEW_ENTRY, GENERAL_RESEARCH,
                            RISK_ANALYSIS, INVALIDATION, EARNINGS_PREVIEW,
                            SHORT_TERM_SETUP, MEDIUM_TERM_SETUP,
                            LONG_TERM_THESIS, EVENT_ANALYSIS, COMPARISON,
                            PORTFOLIO_CONTEXT, OPTIONS_ANALYSIS)

# Which decision sections each question actually asks for. The point of the
# table is what it LEAVES OUT: an entry plan on an exit question is noise.
SECTIONS_FOR_INTENT: Dict[str, tuple] = {
    ADD_TO_POSITION: ("state", "add", "risk", "invalidation", "monitoring"),
    REDUCE_POSITION: ("state", "reduce", "risk", "invalidation", "monitoring"),
    EXIT_POSITION: ("state", "exit", "risk", "invalidation"),
    EXISTING_POSITION: ("state", "playbook", "risk", "invalidation", "monitoring"),
    NEW_ENTRY: ("state", "entry", "sizing", "risk", "invalidation"),
    RISK_ANALYSIS: ("risk", "scenarios", "invalidation"),
    INVALIDATION: ("invalidation", "scenarios", "state"),
    EARNINGS_PREVIEW: ("state", "risk", "invalidation", "monitoring"),
    EVENT_ANALYSIS: ("monitoring", "risk"),
    SHORT_TERM_SETUP: ("state", "entry", "risk", "invalidation"),
    MEDIUM_TERM_SETUP: ("state", "entry", "risk", "invalidation", "scenarios"),
    LONG_TERM_THESIS: ("state", "scenarios", "risk"),
    GENERAL_RESEARCH: ("state", "entry", "risk", "scenarios"),
    COMPARISON: ("state", "risk"),
    OPTIONS_ANALYSIS: ("state", "risk"),
    PORTFOLIO_CONTEXT: ("risk",),
}

NOT_PRODUCED = "NOT_PRODUCED"


def _text(x: Any) -> str:
    """Render a condition without leaking its dict repr.

    The engine returns conditions as objects — {trigger, measurable_as, ...} —
    and str()ing one put a Python literal in front of the reader. The trigger
    is the sentence; how it is measured is the useful qualifier after it.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        head = (x.get("trigger") or x.get("condition") or x.get("what")
                or x.get("text") or x.get("statement") or "")
        tail = x.get("measurable_as") or x.get("threshold") or ""
        met = x.get("met")
        out = str(head)
        if tail and str(tail) != str(head):
            out += f" (measured as {tail})"
        if met is not None:
            out += f" [{'already true' if met else 'not yet'}]"
        return out.strip()
    return str(x)


def _sec(title: str, status: str, headline: str,
         detail: Optional[List[str]] = None,
         source: str = "") -> Dict[str, Any]:
    return {"title": title, "status": status, "headline": headline,
            "detail": [str(d).strip() for d in (detail or [])
                       if d and str(d).strip()],
            "source": source}


def _state(d: Dict[str, Any]) -> Dict[str, Any]:
    st = d.get("decision_state") or {}
    if not st.get("headline_state"):
        return _sec("Where this stands", NOT_PRODUCED,
                    "The engine produced no decision state.", source="decision_state")
    detail = [st.get("headline_reason"), st.get("headline_meaning")]
    if st.get("ownership_note"):
        detail.append(st["ownership_note"])
    return _sec("Where this stands", st["headline_state"],
                st.get("headline_reason") or "", detail, "decision_state")


def _add(d: Dict[str, Any]) -> Dict[str, Any]:
    a = d.get("add_analysis") or {}
    if not a or a.get("status") == "POSITION_CONTEXT_NOT_PROVIDED":
        return _sec("Should you add?", "POSITION_CONTEXT_NOT_PROVIDED",
                    (a.get("message") or
                     "Adding can only be assessed against a position you hold, "
                     "and none was supplied."), source="add_analysis")
    detail: List[str] = []
    if a.get("rule"):
        detail.append(a["rule"])
    for b in (a.get("blockers") or []):
        detail.append(f"Blocker: {_text(b) or (b.get('reason') if isinstance(b, dict) else b)}")
    for c in (a.get("conditions") or [])[:4]:
        detail.append(_text(c))
    for chk in (a.get("checks") or [])[:5]:
        if isinstance(chk, dict) and chk.get("question"):
            detail.append(f"{chk['question']} — {chk.get('answer')}")
    return _sec("Should you add?", a.get("verdict") or a.get("status") or "UNKNOWN",
                a.get("headline") or a.get("message") or "", detail, "add_analysis")


def _from_playbook(d: Dict[str, Any], which: str, title: str) -> Dict[str, Any]:
    pb = d.get("playbook") or {}
    key = {"reduce": "reduce_conditions", "exit": "exit_conditions",
           "hold": "hold_conditions", "add": "add_conditions"}[which]
    rows = pb.get(key) or []
    if not rows:
        return _sec(title, NOT_PRODUCED,
                    f"The playbook produced no {which} conditions.",
                    source="playbook")
    detail = []
    for r in rows[:6]:
        detail.append(_text(r))
    status = pb.get("position_status") or "CONDITIONS_STATED"
    return _sec(title, status,
                pb.get("position_status_meaning") or pb.get("why") or "",
                detail, "playbook")


def _entry(d: Dict[str, Any]) -> Dict[str, Any]:
    ep = d.get("entry_plan") or {}
    en = d.get("entry") or {}
    if not ep.get("supported") and not en.get("status"):
        return _sec("Where to enter", NOT_PRODUCED,
                    "No entry plan was produced.", source="entry_plan")
    detail = []
    if ep.get("entry_low") is not None and ep.get("entry_high") is not None:
        detail.append(f"Entry area {ep['entry_low']} to {ep['entry_high']}.")
    if ep.get("instruction"):
        detail.append(ep["instruction"])
    for c in (ep.get("conditions") or en.get("conditions") or [])[:4]:
        detail.append(_text(c))
    for r in (en.get("reasons") or [])[:3]:
        detail.append(_text(r))
    return _sec("Where to enter", en.get("status") or ep.get("status") or "UNKNOWN",
                ep.get("instruction") or en.get("reason") or "", detail, "entry_plan")


def _sizing(d: Dict[str, Any]) -> Dict[str, Any]:
    sz = d.get("sizing") or {}
    if not sz:
        return _sec("How much", NOT_PRODUCED, "No sizing was produced.",
                    source="sizing")
    return _sec("How much",
                "GATED" if sz.get("gated") else "SIZED",
                sz.get("statement") or "",
                [sz.get("basis")], "sizing")


def _risk(d: Dict[str, Any]) -> Dict[str, Any]:
    rb = d.get("risk_budget") or {}
    if not rb:
        return _sec("Risk", NOT_PRODUCED, "No risk budget was produced.",
                    source="risk_budget")
    vol = rb.get("volatility") or {}
    detail = []
    if rb.get("invalidation_level") is not None:
        detail.append(
            f"Invalidation at {rb['invalidation_level']} "
            f"({rb.get('risk_to_invalidation_pct')}% away), basis "
            f"{rb.get('invalidation_basis')}.")
    if vol.get("historical_volatility_20d") is not None:
        detail.append(f"20-day volatility {vol['historical_volatility_20d']}%, "
                      f"regime {vol.get('vol_regime')}.")
    dd = rb.get("drawdown") or {}
    if dd.get("max_drawdown_pct") is not None:
        detail.append(f"Worst drawdown {dd['max_drawdown_pct']}% "
                      f"({dd.get('window')}).")
    for m in (rb.get("inputs_missing") or []):
        detail.append(f"Not computable without {m.replace('_', ' ')}.")
    return _sec("Risk", rb.get("status") or "MEASURED",
                detail[0] if detail else "", detail[1:], "risk_budget")


def _invalidation(d: Dict[str, Any]) -> Dict[str, Any]:
    mc = d.get("mind_changers") or {}
    inv = mc.get("invalidation") or []
    conf = mc.get("confirmation") or []
    if not inv and not conf:
        return _sec("What would change this", NOT_PRODUCED,
                    "No invalidation conditions were produced.",
                    source="mind_changers")
    detail = []
    for x in inv[:4]:
        detail.append(f"Would falsify it: {_text(x)}")
    for x in conf[:3]:
        detail.append(f"Would confirm it: {_text(x)}")
    if mc.get("note"):
        detail.append(mc["note"])
    return _sec("What would change this", "STATED",
                f"{len(inv)} condition(s) would falsify this reading, "
                f"{len(conf)} would confirm it.", detail, "mind_changers")


def _monitoring(d: Dict[str, Any]) -> Dict[str, Any]:
    mon = d.get("monitoring") or {}
    if not mon:
        return _sec("What to watch", NOT_PRODUCED,
                    "No monitoring plan was produced.", source="monitoring")
    detail = []
    for c in (mon.get("checks") or [])[:5]:
        detail.append(_text(c))
    return _sec("What to watch", "SCHEDULED",
                f"Review {mon.get('review_date')}, every "
                f"{mon.get('cadence_days')} days. {mon.get('review_reason') or ''}",
                detail, "monitoring")


def _scenarios(d: Dict[str, Any]) -> Dict[str, Any]:
    sc = d.get("scenarios") or {}
    rows = sc.get("scenarios") or []
    if not rows:
        return _sec("Bull, base, bear", NOT_PRODUCED,
                    "No scenarios were produced.", source="scenarios")
    detail = []
    for r in rows[:5]:
        name = r.get("name") or r.get("label") or "scenario"
        detail.append(f"{name}: {r.get('thesis') or r.get('summary') or ''}"
                      + (f" Needs: {r.get('needs')}" if r.get("needs") else ""))
    if not sc.get("any_probability_stated"):
        detail.append("PROBABILITY NOT CALIBRATED — no likelihood is attached "
                      "to any of these, because none has been measured.")
    return _sec("Bull, base, bear",
                "CALIBRATED" if sc.get("any_probability_stated")
                else "PROBABILITY_NOT_CALIBRATED",
                f"{len(rows)} scenarios.", detail, "scenarios")


BUILDERS = {
    "state": _state,
    "add": _add,
    "reduce": lambda d: _from_playbook(d, "reduce", "What would justify reducing"),
    "exit": lambda d: _from_playbook(d, "exit", "What would justify exiting"),
    "playbook": lambda d: _from_playbook(d, "hold", "What holding requires"),
    "entry": _entry,
    "sizing": _sizing,
    "risk": _risk,
    "invalidation": _invalidation,
    "monitoring": _monitoring,
    "scenarios": _scenarios,
}


def build(decision_obj: Optional[Dict[str, Any]],
          intent: str, all_sections: bool = False) -> Dict[str, Any]:
    """The decision sections THIS question asked for. Never raises.

    `all_sections` is for the full brief, which is a document rather than an
    answer: it carries every section that can be built regardless of what was
    asked. The intent-scoped set stays the default, because returning an entry
    plan on an exit question is the noise this table exists to remove.
    """
    wanted = (tuple(BUILDERS) if all_sections else
              SECTIONS_FOR_INTENT.get(intent, SECTIONS_FOR_INTENT[GENERAL_RESEARCH]))
    if not decision_obj:
        return {"available": False, "intent": intent, "sections": [],
                "asked_for": list(wanted),
                "statement": ("The full decision build did not run for this "
                              "question, so no decision fields are stated. "
                              "Nothing here is estimated in its place.")}
    sections = []
    for name in wanted:
        fn = BUILDERS.get(name)
        if not fn:
            continue
        try:
            sec = fn(decision_obj)
        except Exception as e:
            sec = _sec(name, NOT_PRODUCED,
                       f"This section could not be built ({type(e).__name__}).")
        sec["key"] = name
        sections.append(sec)

    produced = [s for s in sections if s["status"] != NOT_PRODUCED]
    return {
        "available": True, "intent": intent,
        "asked_for": list(wanted),
        "sections": sections,
        "n_produced": len(produced),
        "statement": (
            f"All {len(sections)} decision sections this question asks for "
            f"were produced."
            if len(produced) == len(sections) else
            f"{len(produced)} of {len(sections)} decision sections this "
            f"question asks for were produced; the other "
            f"{len(sections) - len(produced)} are named rather than filled "
            f"in."),
    }
