"""Cross-agent synthesis: what the evidence agrees on, and where it does not.

WHY NOT AVERAGE
---------------
Averaging four directional reads produces a number with no referent. If
fundamentals and the algo layer both read deterioration, the technical layer
is neutral, social is mildly positive and the statistical layer establishes no
edge, the useful output is not "slightly bearish, 0.42". It is:

    The fundamental and algorithmic layers agree on deterioration. Technical
    evidence is currently neutral. Social evidence mildly offsets the negative
    signal but has lower demonstrated reliability. The statistical layer does
    not establish a validated edge.

That names what agrees, what disagrees, and how much each source's agreement
is worth — which is what a reader needs in order to disagree with it.

THE SEVEN QUESTIONS
-------------------
A disagreement is only worth resolving if it is real. Before weighing sides,
each conflict is tested for whether the two items are even talking about the
same thing: different horizons, different variables, one stale, one
statistically validated, one descriptive rather than predictive, one built on
too little history — and finally whether the disagreement changes the answer
at all. Most do not, and saying so is information.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .evidence import (Item, Ledger, TIER_RANK, DECISION_TIERS, STALE,
                       HISTORICAL_STATISTIC, FORECAST, INTERPRETATION,
                       LLM_EXPLANATION, MODEL_OUTPUT, OBSERVATION, FACT)

BULLISH, BEARISH, NEUTRAL, NOT_DIRECTIONAL = (
    "BULLISH", "BEARISH", "NEUTRAL", "NOT_DIRECTIONAL")

# Consensus labels.
STRONG_AGREEMENT = "STRONG_AGREEMENT"
AGREEMENT = "AGREEMENT"
MIXED = "MIXED"
CONFLICTED = "CONFLICTED"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# One source agreeing with itself is not agreement.
MIN_SOURCES_FOR_AGREEMENT = 2


def _directional(items: List[Item]) -> List[Item]:
    return [i for i in items if i.direction in (BULLISH, BEARISH)]


def _side_weight(items: List[Item], side: str) -> float:
    return round(sum(i.weight for i in items if i.direction == side), 4)


def classify_conflict(a: Item, b: Item) -> Dict[str, Any]:
    """The seven questions, in the order that makes most conflicts dissolve."""
    reasons: List[str] = []
    real = True

    if a.horizon != b.horizon and "ALL" not in (a.horizon, b.horizon):
        reasons.append(
            f"they measure different horizons ({a.horizon.lower()} vs "
            f"{b.horizon.lower()}) — both can hold at once")
        real = False
    if a.key.split(":")[0] != b.key.split(":")[0] and a.horizon == b.horizon:
        reasons.append(f"they measure different variables "
                       f"({a.key.split(':')[0]} vs {b.key.split(':')[0]})")
    if STALE in (a.freshness, b.freshness):
        older = a if a.freshness == STALE else b
        reasons.append(f"{older.source} is working from stale data, so its "
                       f"disagreement carries little")
        real = False
    if a.tier != b.tier:
        stronger, weaker = ((a, b) if TIER_RANK[a.tier] < TIER_RANK[b.tier]
                            else (b, a))
        reasons.append(
            f"{stronger.source} is a {stronger.tier.replace('_', ' ').lower()} "
            f"while {weaker.source} is a "
            f"{weaker.tier.replace('_', ' ').lower()} — the first is the "
            f"stronger kind of claim")
    for it in (a, b):
        if it.tier in (FORECAST, INTERPRETATION, LLM_EXPLANATION):
            reasons.append(f"{it.source} is descriptive rather than "
                           f"predictive, so it cannot settle a directional "
                           f"question")
            real = False
        if "insufficient_sample" in it.flags:
            reasons.append(f"{it.source} rests on too little history to "
                           f"adjudicate anything")
            real = False

    decides = real and a.usable_for_decision and b.usable_for_decision
    return {
        "between": [a.key, b.key],
        "sources": [a.source, b.source],
        "real": real,
        "decision_changing": decides,
        "reasons": reasons or ["they genuinely disagree on the same variable, "
                               "over the same horizon, on comparable evidence"],
        "resolution": _resolve(a, b, real),
    }


def _resolve(a: Item, b: Item, real: bool) -> str:
    if not real:
        return ("Not a real conflict once the two are compared like for like.")
    if a.weight == b.weight:
        return ("Evenly weighted, so this is unresolved — and an unresolved "
                "conflict is a finding, not something to break with a "
                "tiebreak.")
    winner, loser = (a, b) if a.weight > b.weight else (b, a)
    return (f"{winner.source} carries more weight ({winner.weight} vs "
            f"{loser.weight}) because it is a stronger kind of claim from a "
            f"more reliable source — but the disagreement stands and is "
            f"reported rather than removed.")


def synthesize(ledger: Ledger,
               horizon: Optional[str] = None) -> Dict[str, Any]:
    """What the evidence says, together."""
    items = ledger.items
    directional = _directional(items)
    usable = [i for i in directional if i.usable_for_decision]

    bull_w = _side_weight(usable, BULLISH)
    bear_w = _side_weight(usable, BEARISH)
    total = bull_w + bear_w

    # Agreement needs at least two INDEPENDENT sources. One directional item
    # carries 100% of the weight by arithmetic, and reporting that as
    # STRONG_AGREEMENT describes a consensus of one — the most flattering
    # possible reading of the thinnest possible evidence.
    n_sources = len({i.source for i in usable})
    if not usable or total == 0:
        consensus = INSUFFICIENT
    elif n_sources < MIN_SOURCES_FOR_AGREEMENT:
        consensus = INSUFFICIENT
    else:
        share = max(bull_w, bear_w) / total
        consensus = (STRONG_AGREEMENT if share >= 0.85 else
                     AGREEMENT if share >= 0.65 else
                     MIXED if share >= 0.55 else CONFLICTED)

    conflicts: List[Dict[str, Any]] = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if a.direction != b.direction:
                conflicts.append(classify_conflict(a, b))
    decision_changing = [c for c in conflicts if c["decision_changing"]]

    # What agrees, named by source rather than counted.
    agree_bull = sorted({i.source for i in usable if i.direction == BULLISH})
    agree_bear = sorted({i.source for i in usable if i.direction == BEARISH})
    neutral = sorted({i.source for i in items if i.direction == NEUTRAL})
    non_dir = sorted({i.source for i in items
                      if i.direction == NOT_DIRECTIONAL})

    missing = [i.key for i in items if "unavailable" in i.flags]
    stale = [i.key for i in items if i.freshness == STALE]
    low_conf = [i.key for i in items
                if i.usable_for_decision and i.confidence < 0.35]

    for i in usable:
        ledger.mark_consumed(i.key)

    lines: List[str] = []
    if agree_bull and agree_bear:
        lines.append(
            f"{_join(agree_bull)} read constructively; {_join(agree_bear)} "
            f"read the other way.")
    elif agree_bull or agree_bear:
        side = agree_bull or agree_bear
        word = "constructively" if agree_bull else "negatively"
        if len(side) < MIN_SOURCES_FOR_AGREEMENT:
            lines.append(
                f"Only {_join(side)} takes a direction here, reading {word}. "
                f"One source is not agreement, so this is reported as "
                f"insufficient rather than as consensus.")
        else:
            lines.append(f"{_join(side)} read {word}, and nothing usable "
                         f"reads the other way.")
    else:
        lines.append("No usable evidence takes a direction.")
    if neutral:
        lines.append(f"{_join(neutral)} {'is' if len(neutral) == 1 else 'are'} "
                     f"neutral — present and measured, taking no side.")

    # Directional evidence that is NOT decision-usable still gets named. Being
    # excluded from the weighing is not the same as being absent, and dropping
    # it silently would hide a real reading that happens to sit in a weaker
    # tier — which a reader is entitled to weigh for themselves.
    offset = [i for i in directional if not i.usable_for_decision]
    if offset:
        for side, label in ((BULLISH, "offsets"), (BEARISH, "adds to")):
            group = sorted({i.source for i in offset if i.direction == side})
            if not group:
                continue
            against = "negative" if side == BULLISH else "positive"
            why = ("it is interpretation rather than measurement"
                   if all(i.tier in (INTERPRETATION, LLM_EXPLANATION)
                          for i in offset if i.direction == side)
                   else "it is stale or carries no demonstrated reliability")
            lines.append(
                f"{_join(group)} {label} the {against} reading, but does not "
                f"enter the weighing: {why}.")
    if not decision_changing and conflicts:
        lines.append(
            f"{len(conflicts)} disagreement{'s' if len(conflicts) != 1 else ''} "
            f"appear{'' if len(conflicts) != 1 else 's'} in the evidence, none "
            f"of which would change the decision once horizon, staleness and "
            f"kind of claim are accounted for.")
    elif decision_changing:
        lines.append(
            f"{len(decision_changing)} of {len(conflicts)} disagreements are "
            f"decision-changing.")
    if missing:
        lines.append(f"Unavailable and therefore absent rather than neutral: "
                     f"{_join(missing)}.")
    if stale:
        lines.append(f"Stale and down-weighted: {_join(stale)}.")

    return {
        "consensus": consensus,
        "bullish_weight": bull_w,
        "bearish_weight": bear_w,
        "agreement_ratio": round(max(bull_w, bear_w) / total, 3) if total else None,
        "agrees_bullish": agree_bull,
        "agrees_bearish": agree_bear,
        "neutral": neutral,
        "not_directional": non_dir,
        "conflicts": conflicts,
        "n_conflicts": len(conflicts),
        "n_decision_changing": len(decision_changing),
        "missing_evidence": missing,
        "stale_evidence": stale,
        "low_confidence_evidence": low_conf,
        "n_usable": len(usable),
        "n_directional_sources": len({i.source for i in usable}),
        "directional_but_not_weighed": sorted(
            {i.source for i in directional if not i.usable_for_decision}),
        "statement": " ".join(lines),
        "horizon": horizon,
        "method": ("Weighted by evidence tier first, then source reliability, "
                   "then the producer's own confidence — never averaged. An "
                   "average of four directional reads is a number with no "
                   "referent."),
    }


def _join(names: List[str]) -> str:
    names = [n.replace("_", " ") for n in names]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f" and {names[-1]}"
