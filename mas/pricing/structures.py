"""Multi-leg structures, in OptionsPilot's vocabulary.

The names, families and leg conventions here deliberately mirror
`options/structures.py` in the OptionsPilot repo. Two agents that both answer
"what structure fits this view" must answer in one language, or the synthesis
layer is comparing a `bull_call_spread` against a `BULL_CALL_SPREAD` against a
"call debit spread" and quietly treating them as three different ideas.

ADMISSIBILITY
-------------
A structure whose loss is unbounded is rejected at construction, exactly as
OptionsPilot rejects it. This is not risk advice — it is a property of the
payoff function, checkable from the legs alone, and the one place where a
model-priced suggestion could do real harm if it were emitted casually. Note
that the downside is always bounded because the underlying cannot go below
zero; only the call side can run away.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

CALL = "call"
PUT = "put"

DIRECTIONAL = "DIRECTIONAL"   # needs the underlying to move a particular way
VOLATILITY = "VOLATILITY"     # needs movement, either way
RANGE = "RANGE"               # needs the underlying to stay put


class UnboundedLoss(ValueError):
    """Raised when a proposed leg set has no floor on the call side."""


@dataclass(frozen=True)
class Leg:
    kind: str            # "call" | "put"
    strike: float
    qty: int             # +1 long, -1 short

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def side(self) -> str:
        return "long" if self.qty > 0 else "short"


@dataclass(frozen=True)
class Structure:
    name: str
    family: str
    view: Optional[str]          # "BULLISH" | "BEARISH" | None
    legs: Tuple[Leg, ...]
    expiry_years: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "family": self.family, "view": self.view,
            "expiry_years": round(self.expiry_years, 6),
            "expiry_days": round(self.expiry_years * 365.0),
            "legs": [l.to_dict() for l in self.legs],
            "rationale": self.rationale,
        }


def unbounded_reason(legs) -> Optional[str]:
    """The call-side slope above every strike. Negative means loss grows
    without limit as the underlying rises."""
    call_slope = sum(l.qty for l in legs if l.kind == CALL)
    if call_slope < 0:
        return ("net short calls above the highest strike: loss grows without "
                "limit as the underlying rises")
    return None


def is_admissible(legs) -> bool:
    return unbounded_reason(legs) is None


def _mk(name, family, view, legs, expiry_years, rationale) -> Structure:
    reason = unbounded_reason(legs)
    if reason:
        raise UnboundedLoss(f"{name}: {reason}")
    return Structure(name=name, family=family, view=view, legs=tuple(legs),
                     expiry_years=float(expiry_years), rationale=rationale)


# ── Builders ──────────────────────────────────────────────────────────────

def long_call(strike, T):
    return _mk("long_call", DIRECTIONAL, "BULLISH", [Leg(CALL, strike, 1)], T,
               "Simplest long-side expression. Loss is capped at the premium; "
               "the whole premium is at risk if the move does not arrive.")


def long_put(strike, T):
    return _mk("long_put", DIRECTIONAL, "BEARISH", [Leg(PUT, strike, 1)], T,
               "Simplest short-side expression without borrowing the shares.")


def bull_call_spread(long_strike, short_strike, T):
    return _mk("bull_call_spread", DIRECTIONAL, "BULLISH",
               [Leg(CALL, long_strike, 1), Leg(CALL, short_strike, -1)], T,
               "Caps the upside to cut the premium. Pays for a MEASURED move "
               "to the short strike, not an open-ended one.")


def bear_put_spread(long_strike, short_strike, T):
    return _mk("bear_put_spread", DIRECTIONAL, "BEARISH",
               [Leg(PUT, long_strike, 1), Leg(PUT, short_strike, -1)], T,
               "The downside mirror of the bull call spread.")


def bull_put_spread(short_strike, long_strike, T):
    return _mk("bull_put_spread", DIRECTIONAL, "BULLISH",
               [Leg(PUT, short_strike, -1), Leg(PUT, long_strike, 1)], T,
               "Collects premium for the view that the underlying stays above "
               "the short strike. Wins on time as well as direction.")


def bear_call_spread(short_strike, long_strike, T):
    return _mk("bear_call_spread", DIRECTIONAL, "BEARISH",
               [Leg(CALL, short_strike, -1), Leg(CALL, long_strike, 1)], T,
               "Collects premium for the view that the underlying stays below "
               "the short strike.")


def long_straddle(strike, T):
    return _mk("long_straddle", VOLATILITY, None,
               [Leg(CALL, strike, 1), Leg(PUT, strike, 1)], T,
               "Pays for MOVEMENT, not direction. Needs a move larger than the "
               "combined premium to break even, either way.")


def long_strangle(put_strike, call_strike, T):
    return _mk("long_strangle", VOLATILITY, None,
               [Leg(PUT, put_strike, 1), Leg(CALL, call_strike, 1)], T,
               "Cheaper than a straddle and needs a larger move to pay.")


def iron_condor(put_long, put_short, call_short, call_long, T):
    return _mk("iron_condor", RANGE, None,
               [Leg(PUT, put_long, 1), Leg(PUT, put_short, -1),
                Leg(CALL, call_short, -1), Leg(CALL, call_long, 1)], T,
               "Collects premium for the underlying staying inside a band. "
               "The long wings are what make the loss finite.")


def iron_butterfly(put_long, body, call_long, T):
    return _mk("iron_butterfly", RANGE, None,
               [Leg(PUT, put_long, 1), Leg(PUT, body, -1),
                Leg(CALL, body, -1), Leg(CALL, call_long, 1)], T,
               "A tighter, higher-premium condor: the short strikes coincide.")


def call_butterfly(lower, body, upper, T):
    return _mk("call_butterfly", RANGE, None,
               [Leg(CALL, lower, 1), Leg(CALL, body, -2), Leg(CALL, upper, 1)], T,
               "Pays most if the underlying pins the body strike at expiry.")


def put_butterfly(lower, body, upper, T):
    return _mk("put_butterfly", RANGE, None,
               [Leg(PUT, lower, 1), Leg(PUT, body, -2), Leg(PUT, upper, 1)], T,
               "The put-side mirror of the call butterfly.")


STRUCTURES = {
    "long_call": long_call, "long_put": long_put,
    "bull_call_spread": bull_call_spread, "bear_put_spread": bear_put_spread,
    "bull_put_spread": bull_put_spread, "bear_call_spread": bear_call_spread,
    "long_straddle": long_straddle, "long_strangle": long_strangle,
    "iron_condor": iron_condor, "iron_butterfly": iron_butterfly,
    "call_butterfly": call_butterfly, "put_butterfly": put_butterfly,
}

FAMILY = {name: fn for name, fn in STRUCTURES.items()}


def build_structure(name: str, *strikes, expiry_years: float) -> Structure:
    """Build by name. Unknown names raise rather than falling back to a
    default structure — silently substituting a long call for a misspelled
    iron condor would be a different trade entirely."""
    key = (name or "").strip().lower()
    if key not in STRUCTURES:
        raise KeyError(f"unknown structure {name!r}; known: {sorted(STRUCTURES)}")
    return STRUCTURES[key](*strikes, expiry_years)
