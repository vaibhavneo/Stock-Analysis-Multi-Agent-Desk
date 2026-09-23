"""A research result, said back to a person.

Everything here describes structure that already exists. No number is computed
in this file, and no sentence asserts anything the evidence ledger does not
contain — the ordering is deliberate: plan, then what was found, then what it
means together, then what is missing. A reader who stops after two paragraphs
should still have the honest version.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .intent_corpus import (ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                            INVALIDATION, NEW_ENTRY, RISK_ANALYSIS,
                            EARNINGS_PREVIEW, PORTFOLIO_CONTEXT)

# Templates take the symbol, so the sentence reads as English rather than as
# a label with a ticker appended: "Whether to add to NVDA", not
# "Whether to add NVDA".
_INTENT_OPENER = {
    ADD_TO_POSITION: "Whether to add to {s}",
    REDUCE_POSITION: "Whether to reduce {s}",
    EXIT_POSITION: "Whether to exit {s}",
    NEW_ENTRY: "Whether to start a position in {s}",
    INVALIDATION: "What would falsify the {s} thesis",
    RISK_ANALYSIS: "What can go wrong in {s}",
    EARNINGS_PREVIEW: "{s} into its earnings event",
    PORTFOLIO_CONTEXT: "Across the book",
}


def compose(result: Dict[str, Any]) -> Dict[str, Any]:
    """Research result -> chat blocks."""
    plan = result.get("plan") or {}
    syn = result.get("synthesis") or {}
    ev = (result.get("evidence") or {}).get("items") or []
    trace = (result.get("trace") or {}).get("summary") or {}
    val = result.get("validation") or {}
    intent = plan.get("intent", "")
    syms = ", ".join(plan.get("symbols") or []) or "this"

    blocks: List[Dict[str, Any]] = []

    if not result.get("answerable", True):
        opener = _INTENT_OPENER.get(intent, "That question").format(s=syms)
        lines = [f"**{opener} — I cannot answer this as asked.**"]
        for m in plan.get("missing") or []:
            lines.append(f"- {m['why']}")
        blocks.append({"capability": None, "lines": lines})
        return {"headline": "I need one more thing", "blocks": blocks,
                "kind": "NEEDS_CONTEXT"}

    # 1. What was researched, and why that and not everything.
    hz = plan.get("horizon") or {}
    opener = _INTENT_OPENER.get(intent, "On {s}").format(s=syms)
    head = [f"**{opener}** — "
            f"{plan.get('objective', '').lower()}, over "
            f"{str(hz.get('horizon', '')).lower()} (~{hz.get('days')} days)."]
    pos = plan.get("position") or {}
    if pos.get("owns"):
        bits = []
        if pos.get("shares") is not None:
            bits.append(f"{pos['shares']:g} shares")
        if pos.get("avg_cost") is not None:
            bits.append(f"at ${pos['avg_cost']:,.2f}")
        head.append("Using the position you gave me"
                    + (": " + " ".join(bits) if bits else "") + ".")
        if pos.get("missing"):
            head.append(f"Still unknown: {', '.join(pos['missing'])} — asked "
                        f"once, never assumed.")
    blocks.append({"capability": None, "lines": head})

    # 2. The evidence, strongest tier first.
    usable = [i for i in ev if i.get("usable_for_decision")]
    if usable:
        lines = ["**What the evidence says**"]
        for i in sorted(usable, key=lambda x: -x.get("weight", 0))[:6]:
            lines.append(f"- {i['statement']} "
                         f"*({i['source']}, {i['tier'].replace('_', ' ').lower()})*")
        blocks.append({"capability": None, "lines": lines})

    # 3. Synthesis — the part that is not a list.
    if syn.get("statement"):
        blocks.append({"capability": None,
                       "lines": [f"**Together:** {syn['statement']}"]})

    # 4. What was not reachable. Absence is reported, never rendered neutral.
    gaps = [i for i in ev if "unavailable" in (i.get("flags") or [])]
    missing_tools = [m for m in (plan.get("missing") or [])]
    if gaps or missing_tools:
        lines = ["**What I could not get**"]
        for g in gaps:
            lines.append(f"- {g['statement']}")
        for m in missing_tools:
            lines.append(f"- {m['what']}: {m['why']}")
        lines.append("None of this is treated as a neutral reading.")
        blocks.append({"capability": None, "declined": True, "lines": lines})

    # 5. The counter-case, when the evidence was strong enough to earn one.
    adv = result.get("adversarial") or {}
    if adv.get("statement"):
        blocks.append({"capability": None, "lines": [
            "**The case against this**", adv["statement"],
            f"*Raised because {(result.get('escalation') or {}).get('reasons', [''])[0]}*"
        ]})

    # 6. Contradictions, if the validator found any.
    if val.get("issues"):
        lines = ["**Internal checks**"]
        for i in val["issues"]:
            lines.append(f"- [{i['severity']}] {i['message']}")
        blocks.append({"capability": None, "declined": True, "lines": lines})

    return {"headline": head[0].replace("**", "")[:120], "blocks": blocks,
            "kind": "RESEARCH",
            "trace_statement": trace.get("statement", "")}
