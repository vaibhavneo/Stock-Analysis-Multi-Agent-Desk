"""Which specialists are worth calling for THIS request.

WHY THE ORCHESTRATOR NEEDS JUDGEMENT
------------------------------------
Before this, two callers asked for a fixed set every time: `mas/core.ask()`
always ran benchmark-relation and options, and the decision brief always built
options. So someone asking "how does AAPL look" paid for an options analysis
they did not want, and — worse — got range structures volunteered on a name the
desk had no directional view on. Offering a premium-selling trade to someone
who asked about a share price is not thoroughness; it is the system answering a
question nobody posed, which is the same failure as routing to the wrong
specialist.

THE ONE RULE THAT OUTRANKS EVERYTHING
-------------------------------------
**Explicit always runs.** If the user asked for options, options run — on any
asset class, whatever the view, whatever the budget. A layer that decides it
knows better than a stated request is not autonomy, it is refusal. Autonomy
here applies ONLY to capabilities that were IMPLIED by the shape of the
request, never to ones that were named.

WHAT "ADDS INFORMATION" MEANS
-----------------------------
Borrowed from `intelligence/orchestration.py`, which already applies this idea
one level down: spend computation where it changes what the reader would do.
An implied capability is skipped when its answer is knowable in advance, is
inapplicable, or would express a view the desk does not hold.

EVERY SKIP IS RECORDED WITH ITS REASON. A capability that silently did not run
is indistinguishable from one that ran and found nothing, and that ambiguity is
exactly what the trace exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

from . import registry
from .asset_class import spec_for

# How the request arrived, which decides what may be inferred from it.
CHAT = "CHAT"      # a specific question; infer nothing the router did not find
BRIEF = "BRIEF"    # "analyse this name" — an end-to-end read is wanted
FULL = "FULL"      # the multi-asset panel — everything that applies

# Capabilities a request of each depth IMPLIES, on top of what was asked for.
IMPLIED_BY_DEPTH: Dict[str, tuple] = {
    CHAT: (),
    BRIEF: ("option_structures",),
    FULL: ("benchmark_relation", "option_structures"),
}

# Default latency budget for implied work, per depth. Explicit work is never
# budgeted — the user is waiting for it on purpose.
BUDGET_MS: Dict[str, int] = {CHAT: 1500, BRIEF: 4000, FULL: 6000}

EXPLICIT = "EXPLICIT"
IMPLIED = "IMPLIED"


@dataclass
class Decision:
    capability: str
    run: bool
    basis: str
    reason: str
    est_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyResult:
    run: List[str] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    depth: str = CHAT
    budget_ms: int = 0
    spent_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"run": list(self.run), "depth": self.depth,
                "budget_ms": self.budget_ms, "spent_ms": self.spent_ms,
                "decisions": [d.to_dict() for d in self.decisions]}

    def skipped(self) -> List[Decision]:
        return [d for d in self.decisions if not d.run]


def _est_ms(capability: str, asset_class: str, have_symbol: bool = True) -> int:
    """Cheapest agent that could serve this capability. The cheapest, not the
    preferred one: a budget built on the expensive path would skip work the
    system would in fact have done quickly via a fallback."""
    offers = registry.agents(capability=capability, asset_class=asset_class,
                             have_symbol=have_symbol)
    if not offers:
        return 0
    return min(int(a.get("est_ms") or 0) for a in offers)


def _implied_value(capability: str, asset_class: str,
                   direction: Optional[str],
                   have_holdings: bool) -> tuple:
    """(worth_running, reason). Only consulted for IMPLIED capabilities."""
    spec = spec_for(asset_class)

    if capability == "option_structures":
        if not direction:
            return False, ("the desk holds no directional view on this name, so "
                           "the only structures it could offer are range trades "
                           "it has not argued for — volunteering one would be "
                           "suggesting a bet the analysis does not support")
        if spec.options == "NO_ACCESSIBLE_VENUE":
            return False, (f"no venue this system reads lists options on a "
                           f"{spec.label}, so anything shown would be a model "
                           f"shape rather than something tradeable. Ask for it "
                           f"explicitly and it will still be priced.")
        return True, (f"the desk has a {direction.lower()} view, and options are "
                      f"how that view is expressed with a defined worst case")

    if capability == "benchmark_relation":
        return True, ("cheap, and it answers whether a view on this name is "
                      "really a view on its benchmark")

    if capability == "event_calendar":
        if not spec.has_earnings:
            return False, (f"a {spec.label} has no scheduled company events to "
                           f"list")
        return True, "a scheduled event inside the horizon changes the entry"

    if capability == "portfolio_review":
        if not have_holdings:
            return False, ("no holdings were supplied, and positions are never "
                           "inferred")
        return True, "holdings are available to review"

    return True, "no rule excludes it"


def decide(requested: Optional[Sequence[str]] = None,
           asset_class: str = "EQUITY",
           *,
           depth: str = CHAT,
           direction: Optional[str] = None,
           have_symbol: bool = True,
           have_holdings: bool = False,
           budget_ms: Optional[int] = None) -> PolicyResult:
    """Choose the specialists for one request.

    `requested` is what the user actually asked for — those always run.
    `direction` is the desk's own view when it already holds one; it is what
    makes an implied options call informative rather than noise.
    """
    depth = depth if depth in IMPLIED_BY_DEPTH else CHAT
    budget = int(budget_ms if budget_ms is not None else BUDGET_MS[depth])
    explicit = [c for c in (requested or []) if c in registry.CAPABILITIES]
    unknown = [c for c in (requested or []) if c not in registry.CAPABILITIES]

    result = PolicyResult(depth=depth, budget_ms=budget)

    for cap in unknown:
        result.decisions.append(Decision(
            capability=cap, run=False, basis=EXPLICIT,
            reason=f"{cap!r} is not a registered capability"))

    # 1. Everything explicitly asked for. No value test, no budget.
    for cap in explicit:
        est = _est_ms(cap, asset_class, have_symbol)
        result.run.append(cap)
        result.spent_ms += est
        result.decisions.append(Decision(
            capability=cap, run=True, basis=EXPLICIT, est_ms=est,
            reason="you asked for it"))

    # 2. Implied work, cheapest first, until the budget is spent. Cheapest
    #    first rather than most-valuable first because these are all already
    #    judged worth running; ordering by cost maximises how many fit.
    implied = [c for c in IMPLIED_BY_DEPTH[depth] if c not in explicit]
    implied.sort(key=lambda c: _est_ms(c, asset_class, have_symbol))

    for cap in implied:
        est = _est_ms(cap, asset_class, have_symbol)
        offers = registry.agents(capability=cap, asset_class=asset_class,
                                 have_symbol=have_symbol)
        if not offers:
            result.decisions.append(Decision(
                capability=cap, run=False, basis=IMPLIED, est_ms=est,
                reason=(f"no agent offers {cap.replace('_', ' ')} for "
                        f"{asset_class}")))
            continue

        worth, why = _implied_value(cap, asset_class, direction, have_holdings)
        if not worth:
            result.decisions.append(Decision(
                capability=cap, run=False, basis=IMPLIED, est_ms=est,
                reason=f"not run because {why}. You did not ask for it."))
            continue

        if result.spent_ms + est > budget:
            result.decisions.append(Decision(
                capability=cap, run=False, basis=IMPLIED, est_ms=est,
                reason=(f"skipped to stay inside the {budget}ms budget for "
                        f"work you did not explicitly ask for "
                        f"({result.spent_ms}ms already committed)")))
            continue

        result.run.append(cap)
        result.spent_ms += est
        result.decisions.append(Decision(
            capability=cap, run=True, basis=IMPLIED, est_ms=est, reason=why))

    return result


def explain(result: PolicyResult) -> List[str]:
    """The decisions in plain sentences, for a trace a person reads."""
    out = []
    for d in result.decisions:
        verb = "Running" if d.run else "Not running"
        out.append(f"{verb} {d.capability.replace('_', ' ')} — {d.reason}")
    return out
