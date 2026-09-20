"""
Decision state (Phase 4).

An existing position and a new position are two different problems, and the
previous brief answered only one of them twice — `if_owned` / `if_not_owned`
were the same tier re-worded. Here they are computed separately, because the
inputs genuinely differ: the not-owned branch turns on entry location and the
edge gate, while the owned branch turns additionally on cost basis, risk budget
and what the thesis has done since the position was opened.

The state vocabulary is deliberately wider than a verb list, and includes three
states that refuse to act:

    NO_TRADE          there is nothing here worth doing either way
    INSUFFICIENT_DATA the inputs needed to decide are absent
    CONFLICTED        the evidence is genuinely split and forcing a direction
                      would be manufacturing a decision the evidence does not
                      support

Forcing a directional action when the evidence does not support one is the
failure this vocabulary exists to prevent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

OWNED_STATES = ("HOLD", "ADD_CONDITIONALLY", "REDUCE", "EXIT_CONDITIONALLY",
                "CONFLICTED", "INSUFFICIENT_DATA")
UNOWNED_STATES = ("WATCH", "WAIT_FOR_ENTRY", "AVOID_NEW_POSITION", "NO_TRADE",
                  "CONFLICTED", "INSUFFICIENT_DATA")

STATE_MEANING = {
    "WATCH": "Keep this on the list; there is no entry condition worth acting on yet.",
    "WAIT_FOR_ENTRY": "The case is workable but the current price is not the place to start — wait for a named condition below.",
    "HOLD": "Keep the existing position; do not add and do not trim on what is currently known.",
    "ADD_CONDITIONALLY": "An addition is defensible, but only if every listed condition holds.",
    "REDUCE": "Trim exposure — the reasons for the current size have weakened.",
    "EXIT_CONDITIONALLY": "Exit if the named invalidation conditions trigger; they have not all triggered yet.",
    "AVOID_NEW_POSITION": "Do not start a position here.",
    "NO_TRADE": "Nothing to do in either direction — this is a legitimate answer, not an omission.",
    "CONFLICTED": "The evidence is genuinely split. No directional action is supported until it resolves.",
    "INSUFFICIENT_DATA": "The inputs required to make this call are missing.",
}

# Agreement ratio below which the evidence is called split rather than leaning.
CONFLICTED_AGREEMENT_MAX = 0.55


def _insufficient(reason: str) -> Dict[str, Any]:
    return {"state": "INSUFFICIENT_DATA", "reason": reason,
            "meaning": STATE_MEANING["INSUFFICIENT_DATA"], "drivers": []}


def decide_if_not_owned(thesis: Dict[str, Any], edge: Dict[str, Any],
                        entry: Dict[str, Any], conflict: Dict[str, Any],
                        quality_blockers: Optional[List[str]] = None) -> Dict[str, Any]:
    """The new-position problem: should one be started, and where?"""
    drivers: List[str] = []

    if entry.get("status") == "INSUFFICIENT_DATA":
        return _insufficient(entry.get("reason", "Entry location cannot be assessed."))
    if quality_blockers:
        return _insufficient("; ".join(quality_blockers))

    if conflict.get("consensus") == "NO_DIRECTIONAL_EVIDENCE":
        return {"state": "NO_TRADE",
                "reason": ("No directional evidence carries meaningful weight. There is "
                           "nothing here to act on in either direction."),
                "meaning": STATE_MEANING["NO_TRADE"], "drivers": ["no_directional_evidence"]}

    ratio = conflict.get("agreement_ratio")
    if ratio is not None and ratio <= CONFLICTED_AGREEMENT_MAX:
        return {"state": "CONFLICTED",
                "reason": (f"Evidence is split — only {ratio:.0%} of directional weight sits "
                           f"on the larger side. Forcing a direction here would be "
                           f"manufacturing a decision the evidence does not support."),
                "meaning": STATE_MEANING["CONFLICTED"],
                "drivers": ["agreement_ratio_low"]}

    if thesis["direction"] == "BEARISH":
        drivers.append("bearish_thesis")
        return {"state": "AVOID_NEW_POSITION",
                "reason": (f"{thesis['strength'].title()} bearish thesis. Note this is not a "
                           f"short recommendation — this system evaluates long exposure only, "
                           f"and a bearish read means 'do not start', not 'bet against'."),
                "meaning": STATE_MEANING["AVOID_NEW_POSITION"], "drivers": drivers}

    if entry.get("status") == "HIGH_RISK":
        return {"state": "AVOID_NEW_POSITION",
                "reason": entry.get("reason", "Risk conditions block a new position."),
                "meaning": STATE_MEANING["AVOID_NEW_POSITION"], "drivers": ["risk_veto"]}

    if thesis["direction"] == "NEUTRAL":
        return {"state": "WATCH",
                "reason": ("No directional thesis. Nothing to enter; worth monitoring for the "
                           "confirmation triggers listed below."),
                "meaning": STATE_MEANING["WATCH"], "drivers": ["no_thesis"]}

    # Bullish from here on.
    drivers.append("bullish_thesis")
    if not edge.get("demonstrated"):
        drivers.append("no_demonstrated_edge")
    if entry.get("status") in ("ATTRACTIVE", "ACCEPTABLE"):
        if edge.get("demonstrated"):
            return {"state": "WAIT_FOR_ENTRY" if entry["status"] == "ACCEPTABLE" else "WAIT_FOR_ENTRY",
                    "reason": (f"Bullish thesis, demonstrated edge, and entry location reads "
                               f"{entry['status']}. Sizing is unlocked; the conditions below "
                               f"state where to start."),
                    "meaning": STATE_MEANING["WAIT_FOR_ENTRY"], "drivers": drivers}
        return {"state": "WATCH",
                "reason": (f"Bullish thesis and a workable entry location "
                           f"({entry['status']}), but the statistical gate has not cleared, so "
                           f"position size remains 0%. A good location for an unproven signal "
                           f"is still an unproven signal."),
                "meaning": STATE_MEANING["WATCH"], "drivers": drivers}

    if entry.get("status") in ("WAIT", "OVEREXTENDED"):
        return {"state": "WAIT_FOR_ENTRY",
                "reason": (f"Bullish thesis, but entry location reads {entry['status']}: "
                           f"{entry.get('reason', '')}"),
                "meaning": STATE_MEANING["WAIT_FOR_ENTRY"], "drivers": drivers + ["poor_entry_location"]}

    return {"state": "WATCH",
            "reason": "Bullish thesis without a supported entry condition.",
            "meaning": STATE_MEANING["WATCH"], "drivers": drivers}


def decide_if_owned(thesis: Dict[str, Any], edge: Dict[str, Any],
                    entry: Dict[str, Any], conflict: Dict[str, Any],
                    position_context: Dict[str, Any], risk_budget: Dict[str, Any],
                    add_analysis: Dict[str, Any],
                    quality_blockers: Optional[List[str]] = None) -> Dict[str, Any]:
    """The existing-position problem. Note what this does NOT do: it does not
    read unrealized P&L as a signal. A gain is not a reason to sell and a loss
    is not a reason to hold; both are facts about the holder, and they enter
    only through the risk budget and through explicitly-labelled observations.
    """
    drivers: List[str] = []
    if quality_blockers:
        return _insufficient("; ".join(quality_blockers))

    ratio = conflict.get("agreement_ratio")
    if conflict.get("consensus") == "NO_DIRECTIONAL_EVIDENCE":
        return {"state": "HOLD",
                "reason": ("No directional evidence carries weight either way. Holding is the "
                           "default when nothing argues for a change — this is inertia by "
                           "decision, not by omission."),
                "meaning": STATE_MEANING["HOLD"], "drivers": ["no_directional_evidence"]}

    if thesis["direction"] == "BEARISH":
        drivers.append("bearish_thesis")
        strong = thesis["strength"] in ("STRONG", "MODERATE")
        return {"state": "REDUCE" if strong else "EXIT_CONDITIONALLY",
                "reason": (f"{thesis['strength'].title()} bearish thesis on a position that is "
                           f"held. The reasons for the current exposure have turned against it."),
                "meaning": STATE_MEANING["REDUCE" if strong else "EXIT_CONDITIONALLY"],
                "drivers": drivers}

    if ratio is not None and ratio <= CONFLICTED_AGREEMENT_MAX:
        return {"state": "CONFLICTED",
                "reason": (f"Evidence is split ({ratio:.0%} of directional weight on the larger "
                           f"side). Holding through a genuine split is defensible; adding to it "
                           f"is not."),
                "meaning": STATE_MEANING["CONFLICTED"], "drivers": ["agreement_ratio_low"]}

    over_budget = (risk_budget.get("position_risk") or {}).get("within_risk_budget") is False
    if over_budget:
        drivers.append("over_risk_budget")
        return {"state": "REDUCE",
                "reason": ((risk_budget.get("position_risk") or {}).get("budget_note") or
                           "The position exceeds the stated risk budget."),
                "meaning": STATE_MEANING["REDUCE"], "drivers": drivers}

    if entry.get("status") == "HIGH_RISK":
        drivers.append("risk_veto")
        return {"state": "EXIT_CONDITIONALLY",
                "reason": ("A risk veto is active. This does not force an exit, but it does "
                           "mean the named invalidation conditions should be treated as live "
                           "triggers rather than distant possibilities."),
                "meaning": STATE_MEANING["EXIT_CONDITIONALLY"], "drivers": drivers}

    if add_analysis.get("verdict") == "ADD_CONDITIONALLY":
        drivers.append("add_case_established")
        return {"state": "ADD_CONDITIONALLY",
                "reason": add_analysis.get("headline", ""),
                "meaning": STATE_MEANING["ADD_CONDITIONALLY"], "drivers": drivers}

    drivers.append("thesis_" + thesis["direction"].lower())
    if not edge.get("demonstrated"):
        drivers.append("no_demonstrated_edge")
    return {"state": "HOLD",
            "reason": (f"{thesis['strength'].title() if thesis['strength'] != 'NONE' else 'No'} "
                       f"{thesis['direction'].lower()} thesis, "
                       + ("no demonstrated edge to justify adding, "
                          if not edge.get("demonstrated") else "")
                       + "and nothing that argues for reducing."),
            "meaning": STATE_MEANING["HOLD"], "drivers": drivers}


def build_decision_state(thesis: Dict[str, Any], edge: Dict[str, Any],
                         entry: Dict[str, Any], conflict: Dict[str, Any],
                         position_context: Dict[str, Any],
                         risk_budget: Dict[str, Any],
                         add_analysis: Dict[str, Any],
                         quality_blockers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Both branches, plus the headline state for whichever branch the caller's
    position context actually supports.

    When ownership is unknown, `headline_state` is the NOT-owned branch and
    `ownership` says so explicitly — the reader is told the assumption rather
    than having it made silently.
    """
    not_owned = decide_if_not_owned(thesis, edge, entry, conflict, quality_blockers)

    if position_context.get("status") == "PROVIDED":
        owned = decide_if_owned(thesis, edge, entry, conflict, position_context,
                                risk_budget, add_analysis, quality_blockers)
        headline = owned
        ownership = "OWNED"
        ownership_note = (f"You hold this position (average cost "
                          f"${position_context['avg_cost']}), so the owned branch is the "
                          f"headline.")
    else:
        owned = decide_if_owned(thesis, edge, entry, conflict, position_context,
                                risk_budget, add_analysis, quality_blockers)
        headline = not_owned
        ownership = "UNKNOWN"
        ownership_note = (
            "POSITION CONTEXT NOT PROVIDED — ownership is unknown. The not-owned branch is "
            "shown as the headline because starting a position is the decision that can be "
            "made without knowing anything about an existing one. The owned branch is shown "
            "beside it and assumes a position exists without knowing its cost basis or size.")

    return {
        "headline_state": headline["state"],
        "headline_reason": headline["reason"],
        "headline_meaning": headline["meaning"],
        "ownership": ownership,
        "ownership_note": ownership_note,
        "if_owned": owned,
        "if_not_owned": not_owned,
        "state_vocabulary": STATE_MEANING,
        "separation_note": ("An existing position and a new position are evaluated as "
                            "different problems, not as one verdict worded two ways."),
    }
