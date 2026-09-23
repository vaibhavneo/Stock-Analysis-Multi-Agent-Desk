"""Evidence tiers — what kind of claim is this, really?

The recurring way an analysis system misleads is not by stating something
false. It is by letting one kind of claim wear another kind's authority: a
model output presented as an observation, a backtested statistic presented as
a demonstrated edge, an LLM sentence presented as a fact.

So every item carries a TIER, and the tiers are ordered. Nothing in the
pipeline may promote an item to a higher tier, and the synthesis layer weighs
by tier before it weighs by anything else — because a strongly-worded
interpretation should never outrank a weakly-worded measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

FACT = "FACT"                              # observed and checkable
OBSERVATION = "OBSERVATION"                # measured from data
HISTORICAL_STATISTIC = "HISTORICAL_STATISTIC"
MODEL_OUTPUT = "MODEL_OUTPUT"
FORECAST = "FORECAST"
INTERPRETATION = "INTERPRETATION"
LLM_EXPLANATION = "LLM_EXPLANATION"

# Ordered strongest to weakest. Position in this tuple IS the authority.
TIERS = (FACT, OBSERVATION, HISTORICAL_STATISTIC, MODEL_OUTPUT, FORECAST,
         INTERPRETATION, LLM_EXPLANATION)
TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Tiers that may influence a decision field. An LLM sentence may explain a
# decision; it may never be a reason for one.
DECISION_TIERS = frozenset({FACT, OBSERVATION, HISTORICAL_STATISTIC,
                            MODEL_OUTPUT})

# The promotions that would be lies. Checked explicitly, because each one has
# a natural-sounding phrasing that makes it easy to commit by accident.
FORBIDDEN_PROMOTIONS = {
    (LLM_EXPLANATION, FACT): "an explanation is not an observation",
    (INTERPRETATION, FACT): "an interpretation is not an observation",
    (FORECAST, HISTORICAL_STATISTIC): "a forecast is not a measured frequency",
    (FORECAST, FACT): "a forecast is not a fact",
    (MODEL_OUTPUT, OBSERVATION): "a model output is not a measurement",
    (HISTORICAL_STATISTIC, FACT): "a past statistic is not a future fact",
}

FRESH = "FRESH"
AGEING = "AGEING"
STALE = "STALE"
UNKNOWN_FRESHNESS = "UNKNOWN"


@dataclass
class Item:
    """One piece of evidence, with everything needed to weigh it."""
    key: str
    tier: str
    statement: str
    source: str                      # tool or capability id
    direction: str = "NOT_DIRECTIONAL"   # BULLISH/BEARISH/NEUTRAL/NOT_DIRECTIONAL
    horizon: str = "ALL"
    confidence: float = 0.5          # 0..1, the producer's own
    reliability: float = 0.5         # 0..1, the SOURCE's, from the registry
    freshness: str = UNKNOWN_FRESHNESS
    observed_at: Optional[str] = None
    data_timestamp: Optional[str] = None
    decision_relevance: str = "SUPPORTING"
    value: Any = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    # The specialist contract's remaining two fields. `uncertainty` is stated
    # rather than inferred from confidence, because "I am 60% confident" and
    # "the sample is too small to say" are different admissions and only the
    # second names what would fix it.
    uncertainty: str = ""
    changed_since_previous: Optional[str] = None

    def __post_init__(self):
        if self.tier not in TIER_RANK:
            raise ValueError(f"unknown evidence tier {self.tier!r}")
        self.confidence = max(0.0, min(1.0, float(self.confidence or 0.0)))
        self.reliability = max(0.0, min(1.0, float(self.reliability or 0.0)))

    def contract(self) -> Dict[str, Any]:
        """This item in the specialist-contract shape, which is what the
        synthesis layer compares. The contract lives HERE rather than on the
        raw adapter payload: adapters return whatever their engine produces,
        and normalisation is what makes five specialists comparable."""
        return {
            "finding": self.statement,
            "direction": self.direction,
            "horizon": self.horizon,
            "confidence": self.confidence,
            "data_quality": self.freshness,
            "provenance": self.provenance,
            "decision_relevance": self.decision_relevance,
            "uncertainty": self.uncertainty or self._default_uncertainty(),
            "changed_since_previous": self.changed_since_previous,
        }

    def _default_uncertainty(self) -> str:
        if "insufficient_sample" in self.flags:
            return "the sample is too small to distinguish skill from luck"
        if "derived_only" in self.flags:
            return "derived from arithmetic rather than observed trading"
        if "unavailable" in self.flags:
            return "the source could not be reached, so nothing is claimed"
        if self.freshness == STALE:
            return "the underlying data is stale"
        if self.confidence < 0.4:
            return "the producer's own confidence is low"
        return "within the normal error of this measurement"

    @property
    def usable_for_decision(self) -> bool:
        return (self.tier in DECISION_TIERS
                and self.freshness != STALE
                and "unavailable" not in self.flags)

    @property
    def weight(self) -> float:
        """Tier first, then the source's reliability, then the producer's own
        confidence.

        The tier factor HALVES per rank rather than declining linearly. Under
        the linear version a maximally confident LLM sentence scored 0.25
        against a hedged observation's 0.21 — so tier did not actually lead,
        and wording bought authority, which is the one thing this ordering
        exists to prevent. Halving puts roughly 32x between an observation and
        an explanation: enough that no amount of stated confidence can cross
        the boundary, while still letting a strong model output outrank a weak
        observation, which is a judgement that should stay available.
        """
        tier_factor = 0.5 ** TIER_RANK[self.tier]
        fresh_factor = {FRESH: 1.0, AGEING: 0.85, STALE: 0.3,
                        UNKNOWN_FRESHNESS: 0.7}[self.freshness]
        return round(tier_factor * self.reliability * self.confidence
                     * fresh_factor, 4)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["weight"] = self.weight
        d["usable_for_decision"] = self.usable_for_decision
        return d


class Ledger:
    """Every item produced for one request, and what became of it."""

    def __init__(self):
        self.items: List[Item] = []
        self._consumed: set = set()

    def add(self, item: Item) -> Item:
        self.items.append(item)
        return item

    def mark_consumed(self, key: str) -> None:
        """Recorded so 'produced' and 'used' can be told apart afterwards —
        the whole point of tracking tool value."""
        self._consumed.add(key)

    def consumed(self) -> List[Item]:
        return [i for i in self.items if i.key in self._consumed]

    def unconsumed(self) -> List[Item]:
        return [i for i in self.items if i.key not in self._consumed]

    def decision_usable(self) -> List[Item]:
        return [i for i in self.items if i.usable_for_decision]

    def by_direction(self) -> Dict[str, List[Item]]:
        out: Dict[str, List[Item]] = {}
        for i in self.items:
            out.setdefault(i.direction, []).append(i)
        return out

    def check_promotions(self) -> List[Dict[str, str]]:
        """Any item whose stated tier is stronger than its source allows."""
        bad = []
        for i in self.items:
            claimed = i.provenance.get("claimed_tier")
            if claimed and (i.tier, claimed) in FORBIDDEN_PROMOTIONS:
                bad.append({"key": i.key, "from": i.tier, "to": claimed,
                            "why": FORBIDDEN_PROMOTIONS[(i.tier, claimed)]})
        return bad

    def summary(self) -> Dict[str, Any]:
        by_tier: Dict[str, int] = {}
        for i in self.items:
            by_tier[i.tier] = by_tier.get(i.tier, 0) + 1
        usable = self.decision_usable()
        return {
            "n_items": len(self.items),
            "by_tier": by_tier,
            "n_decision_usable": len(usable),
            "n_consumed": len(self.consumed()),
            "n_unconsumed": len(self.unconsumed()),
            "unconsumed_keys": [i.key for i in self.unconsumed()],
            "statement": (
                f"{len(self.items)} evidence items, {len(usable)} of which may "
                f"influence a decision; the rest are interpretation or "
                f"explanation and inform the wording only."),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items],
                "summary": self.summary(),
                "tier_order": list(TIERS)}
