"""What each specialist is actually asked, and what it must return.

THE DIFFERENCE THIS MAKES
-------------------------
"Analyze this stock" produces five essays that overlap, contradict each other
in prose, and have to be read in full before they can be compared. A
specialist that is asked

    "Does the recent fundamental change materially alter the medium-term
     thesis for someone already holding at 197.80?"

answers a question the orchestrator can act on, and answers it in a shape the
orchestrator can compare against the other four without reading any of them.

So delegation carries three things per specialist, all derived from the plan:

    QUESTION   the specific thing this specialist must settle
    RECEIVES   what it is given, so it is not re-deriving shared work
    EXPECTS    the fields it must return

TEMPLATED, NOT GENERATED
------------------------
The question is built from intent, horizon and position by a template, not by
an LLM. A generated question can drift off what the plan needs, and then the
specialist answers something adjacent while appearing to comply — which is
harder to notice than a missing answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .intent_corpus import (ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                            EXISTING_POSITION, NEW_ENTRY, GENERAL_RESEARCH,
                            RISK_ANALYSIS, INVALIDATION, EARNINGS_PREVIEW,
                            SHORT_TERM_SETUP, MEDIUM_TERM_SETUP,
                            LONG_TERM_THESIS, EVENT_ANALYSIS, COMPARISON,
                            OPTIONS_ANALYSIS, PORTFOLIO_CONTEXT)

# The output every specialist owes, whatever it was asked. These are the
# fields the synthesis layer compares; anything else is prose it may add but
# nothing reads structurally.
REQUIRED_FIELDS = ("finding", "direction", "horizon", "confidence",
                   "data_quality", "provenance", "decision_relevance",
                   "uncertainty", "changed_since_previous")

# Per-capability: what it is for, and what its finding must be about.
CAPABILITY_FOCUS = {
    "equity_research": ("the security's own read",
                        "whether the pillar evidence supports the action asked about"),
    "strategy_backtest": ("demonstrated edge",
                          "whether following this has historically beaten holding it, "
                          "net of costs"),
    "event_calendar": ("scheduled events",
                       "whether a dated event falls inside the horizon and what it "
                       "would resolve"),
    "benchmark_relation": ("what is really being bought",
                           "how much of this name's movement is its benchmark's"),
    "market_regime": ("conditions",
                      "what kind of market this decision is being made in"),
    "forward_record": ("this desk's own accuracy",
                       "whether enough predictions have matured to quote a rate"),
    "option_structures": ("expression",
                          "how the stated view would be expressed with a defined "
                          "worst case"),
    "market_intelligence": ("historical context",
                            "how this name has behaved in setups like this one"),
    "analyst_narrative": ("written reasoning",
                          "the case in prose, which changes no number"),
}

_INTENT_CLAUSE = {
    ADD_TO_POSITION: "adding to a position already held",
    REDUCE_POSITION: "reducing a position already held",
    EXIT_POSITION: "exiting a position already held",
    EXISTING_POSITION: "continuing to hold",
    NEW_ENTRY: "starting a position",
    RISK_ANALYSIS: "how much can go wrong",
    INVALIDATION: "what would falsify the current thesis",
    EARNINGS_PREVIEW: "holding through the coming earnings event",
    EVENT_ANALYSIS: "which scheduled events matter",
    SHORT_TERM_SETUP: "a trade over days to weeks",
    MEDIUM_TERM_SETUP: "a position over one to six months",
    LONG_TERM_THESIS: "a multi-year holding",
    COMPARISON: "choosing between these names",
    OPTIONS_ANALYSIS: "expressing this view in options",
    PORTFOLIO_CONTEXT: "the book as a whole",
    GENERAL_RESEARCH: "the overall read",
}


def question_for(capability: str, plan) -> str:
    """The specific thing this specialist must settle for THIS plan."""
    focus = CAPABILITY_FOCUS.get(capability, (capability, "its own measure"))
    what = focus[1]
    subject = ", ".join(plan.symbols) or "this security"
    horizon = (plan.horizon or {}).get("horizon", "").lower() or "unspecified"
    days = (plan.horizon or {}).get("days")
    clause = _INTENT_CLAUSE.get(plan.intent, "this decision")

    pos = plan.position or {}
    holding = ""
    if pos.get("owns"):
        bits = []
        if pos.get("shares") is not None:
            bits.append(f"{pos['shares']:g} shares")
        if pos.get("avg_cost") is not None:
            bits.append(f"at ${pos['avg_cost']:,.2f}")
        holding = (" for someone already holding "
                   + (" ".join(bits) if bits else "a position"))

    return (f"Determine {what}, for {subject}, over a {horizon} horizon "
            f"(~{days} days), in the context of {clause}{holding}. "
            f"Answer that question specifically; do not restate the general "
            f"picture.")


def receives_for(capability: str, plan) -> List[str]:
    """What this specialist is handed, so it does not re-derive shared work."""
    base = ["the resolved symbol and asset class",
            "the horizon this decision is being made over"]
    if (plan.position or {}).get("owns"):
        base.append("the position context, exactly as the user stated it")
    if capability == "option_structures":
        base.append("this desk's own directional read, so it expresses that "
                    "view rather than deriving a second one")
    if capability in ("equity_research", "strategy_backtest"):
        base.append("the settled bar series, not a live tick")
    return base


def brief_for(capability: str, plan) -> Dict[str, Any]:
    """The full delegation brief for one specialist."""
    focus = CAPABILITY_FOCUS.get(capability, (capability, "its own measure"))
    return {
        "capability": capability,
        "focus": focus[0],
        "question": question_for(capability, plan),
        "receives": receives_for(capability, plan),
        "expects": list(REQUIRED_FIELDS),
        "must_not": [
            "invent a price level, probability, catalyst or source",
            "answer a broader question than the one asked",
            "return a direction it cannot attribute to measured data",
        ],
    }


def attach(plan) -> None:
    """Attach a brief to every capability on the plan, in place."""
    for entry in (plan.capabilities or []):
        entry["brief"] = brief_for(entry["capability"], plan)


def validate_response(capability: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Does a specialist's answer meet the contract?

    Checked against the NORMALISED evidence, not the raw adapter payload.
    Adapters return whatever their engine produces — a decision object, a
    backtest table, a calendar — and normalisation into `evidence.Item` is
    what makes five specialists comparable. Checking the raw payload measured
    the wrong layer and reported 0 of 4 compliant while the evidence carried
    every field.

    Reported rather than enforced by rejection: a specialist that answered
    usefully but omitted a field should have its answer used and the omission
    recorded, because dropping real evidence over a missing key would be a
    worse failure than the one being checked for.
    """
    present = [f for f in REQUIRED_FIELDS if payload.get(f) is not None]
    missing = [f for f in REQUIRED_FIELDS if f not in present]
    return {
        "capability": capability,
        "complete": not missing,
        "present": present,
        "missing": missing,
        "statement": ("meets the specialist contract" if not missing else
                      f"answered, but omitted {', '.join(missing)} — used "
                      f"anyway, with the gap recorded rather than the evidence "
                      f"discarded"),
    }
