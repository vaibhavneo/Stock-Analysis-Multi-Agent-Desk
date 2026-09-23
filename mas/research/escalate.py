"""How much research is enough for THIS question?

Always running the deepest pass wastes latency on questions the fast pass
already answered coherently. Never running it means a conflicted or
consequential decision gets the same three seconds as "how does AAPL look".

So depth is decided AFTER the first pass, from what that pass actually found.
The trigger is not the question's wording — it is whether the evidence is
coherent, complete, and sufficient for the weight of the decision.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .plan import FAST, DEEP, ADVERSARIAL
from .synthesis import CONFLICTED, MIXED, INSUFFICIENT
from .intent_corpus import (ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                            NEW_ENTRY, INVALIDATION)

# Decisions that move money if acted on. These earn a deeper look when the
# evidence is anything other than clean.
CONSEQUENTIAL = frozenset({ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                           NEW_ENTRY})

# A directional read this strong deserves to be argued against. Strong
# agreement is exactly when nobody looks for the counter-case.
STRONG_ENOUGH_TO_CHALLENGE = 0.85


def decide_depth(result: Dict[str, Any],
                 current: str = FAST) -> Dict[str, Any]:
    """Whether to go deeper, and the reason. Never escalates silently."""
    syn = result.get("synthesis") or {}
    plan = result.get("plan") or {}
    intent = plan.get("intent", "")
    ev = (result.get("evidence") or {}).get("items") or []

    # Reasons are kept PER TARGET. A flat list reported the reason for an
    # intermediate escalation next to the final one — "escalating to
    # adversarial" followed by the reason for going deep.
    reasons_by_target: Dict[str, List[str]] = {}
    target = current

    def _note(to: str, why: str):
        reasons_by_target.setdefault(to, []).append(why)

    unmet = [m["what"] for m in (plan.get("missing") or [])]
    unavailable = [i["key"] for i in ev if "unavailable" in (i.get("flags") or [])]
    consensus = syn.get("consensus")
    ratio = syn.get("agreement_ratio")
    n_changing = syn.get("n_decision_changing") or 0

    if current == FAST:
        if consensus in (CONFLICTED, MIXED) or n_changing:
            target = DEEP
            _note(DEEP,
                f"the evidence is {str(consensus or 'conflicted').lower()}"
                + (f" with {n_changing} decision-changing disagreement(s)"
                   if n_changing else "")
                + " — a fast pass is not enough to settle it")
        elif consensus == INSUFFICIENT and intent in CONSEQUENTIAL:
            target = DEEP
            _note(DEEP,
                "this decision would move money and the first pass did not "
                "find enough directional evidence to support one")
        elif unavailable and intent in CONSEQUENTIAL:
            target = DEEP
            _note(DEEP,
                f"{len(unavailable)} source(s) were unreachable on a decision "
                f"that would move money, so the gap is worth a second attempt")

    if current in (FAST, DEEP):
        # A 100% ratio from ONE source is not strong agreement — it is one
        # measurement dividing by itself. Challenging that as though it were
        # a consensus would dress a thin reading as a robust one.
        n_sources = syn.get("n_directional_sources") or 0
        if (intent in CONSEQUENTIAL or intent == INVALIDATION) \
                and ratio is not None and n_sources >= 2 \
                and ratio >= STRONG_ENOUGH_TO_CHALLENGE:
            target = ADVERSARIAL
            _note(ADVERSARIAL,
                f"{n_sources} sources agree {ratio:.0%} one way on a "
                f"consequential decision — strong agreement is exactly when "
                f"nobody looks for the counter-case, so it gets argued against")

    if target == current:
        _note(current,
              "the first pass was coherent and complete enough for what was "
              "asked; going deeper would cost latency and change nothing")

    final_reasons = reasons_by_target.get(target, [])
    all_reasons = [r for rs in reasons_by_target.values() for r in rs]
    return {"from": current, "to": target, "escalated": target != current,
            "reasons": final_reasons, "all_reasons": all_reasons,
            "statement": (
                f"Escalating to {target.lower()}: {final_reasons[0]}"
                if target != current else
                f"Staying {current.lower()}: {final_reasons[0]}")}
