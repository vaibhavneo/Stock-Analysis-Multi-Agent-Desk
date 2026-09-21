"""What a structure costs, what it can make, where it breaks even, and how
likely each of those is.

The payoff of any option structure at expiry is PIECEWISE LINEAR with kinks
only at the strikes. That is what makes the numbers here exact rather than
sampled: breakevens are solved by linear interpolation between adjacent kinks,
not found by scanning a grid, so a breakeven is the true root and not the
nearest gridpoint to it.

The probabilities are the part to read carefully. They are RISK-NEUTRAL: the
probability measure under which the model prices, not a forecast of what the
asset will do. Every such field is named `..._risk_neutral` so it cannot be
quoted as "a 68% chance of profit" without the qualifier travelling with it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .black_scholes import (CALL, PUT, MODEL_BASIS, greeks, price, _ncdf, forward)
from .structures import Structure, Leg

# Multiple of spot used as the "far" evaluation point. Anything beyond this
# carries negligible lognormal mass at sane volatilities.
_FAR = 4.0


def _leg_payoff(leg: Leg, S: float) -> float:
    if leg.kind == CALL:
        return leg.qty * max(0.0, S - leg.strike)
    return leg.qty * max(0.0, leg.strike - S)


def payoff_at(structure: Structure, S: float) -> float:
    """Gross payoff at expiry, before the cost of entry."""
    return sum(_leg_payoff(l, S) for l in structure.legs)


def net_cost(structure: Structure, S: float, r: float, sigma: float,
             q: float = 0.0) -> float:
    """Model cost to open. Positive = net debit paid, negative = net credit."""
    T = structure.expiry_years
    return sum(l.qty * price(l.kind, S, l.strike, T, r, sigma, q)
               for l in structure.legs)


def _kinks(structure: Structure, S: float) -> List[float]:
    pts = sorted({l.strike for l in structure.legs})
    return [0.0] + pts + [max(_FAR * S, (pts[-1] if pts else S) * 2.0)]


def _breakevens(structure: Structure, cost: float, S: float) -> List[float]:
    """Exact roots of (payoff - cost) on a piecewise-linear function."""
    xs = _kinks(structure, S)
    ys = [payoff_at(structure, x) - cost for x in xs]
    roots: List[float] = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 == 0.0:
            roots.append(xs[i])
        elif y0 * y1 < 0:
            # linear between kinks -> exact root
            roots.append(xs[i] + (xs[i + 1] - xs[i]) * (-y0) / (y1 - y0))
    if ys and ys[-1] == 0.0:
        roots.append(xs[-1])
    return sorted({round(r_, 10) for r_ in roots if r_ >= 0})


def _slopes(structure: Structure):
    """Payoff slope above every strike, and below every strike."""
    upper = sum(l.qty for l in structure.legs if l.kind == CALL)
    lower = -sum(l.qty for l in structure.legs if l.kind == PUT)
    return lower, upper


def _prob_above(x: float, S: float, T: float, r: float, sigma: float,
                q: float) -> float:
    """Risk-neutral P(S_T > x) under the lognormal pricing measure."""
    if x <= 0:
        return 1.0
    if T <= 0 or sigma <= 0:
        return 1.0 if S > x else 0.0
    d2 = (math.log(S / x) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d2)


def evaluate(structure: Structure, S: float, r: float, sigma: float,
             q: float = 0.0, contract_multiplier: float = 100.0) -> Dict[str, Any]:
    """Full economics of one structure at one spot/vol.

    `contract_multiplier` is 100 for US listed equity options and 1 for a
    per-unit quote (crypto and FX conventions vary by venue). It scales the
    dollar figures only — never the probabilities or the breakevens.
    """
    T = structure.expiry_years
    cost = net_cost(structure, S, r, sigma, q)
    lower_slope, upper_slope = _slopes(structure)

    xs = _kinks(structure, S)
    pls = [payoff_at(structure, x) - cost for x in xs]

    # Extremes live at kinks unless a tail runs away. The downside is always
    # bounded because the underlying cannot go below zero; only the call side
    # can be unbounded, and `structures` refuses to build those.
    max_profit: Optional[float] = max(pls)
    max_loss: Optional[float] = min(pls)
    profit_unbounded = upper_slope > 0
    if profit_unbounded:
        max_profit = None

    breakevens = _breakevens(structure, cost, S)

    # Probability of profit: sum the lognormal mass over the intervals where
    # P&L > 0. Derived from the breakevens rather than sampled, so it is exact
    # for the model.
    edges = [0.0] + breakevens + [float("inf")]
    pop = 0.0
    profitable: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        probe = (lo + hi) / 2.0 if hi != float("inf") else max(hi_probe := lo * 1.5,
                                                              lo + S * 0.5)
        if payoff_at(structure, probe) - cost > 0:
            p_lo = _prob_above(lo, S, T, r, sigma, q)
            p_hi = _prob_above(hi, S, T, r, sigma, q) if hi != float("inf") else 0.0
            mass = max(0.0, p_lo - p_hi)
            pop += mass
            profitable.append({
                "from": round(lo, 6),
                "to": None if hi == float("inf") else round(hi, 6),
                "probability_risk_neutral": round(mass, 6),
            })

    agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for l in structure.legs:
        g = greeks(l.kind, S, l.strike, T, r, sigma, q)
        for k in agg:
            agg[k] += l.qty * getattr(g, k)

    return {
        "structure": structure.to_dict(),
        "basis": MODEL_BASIS,
        "spot": round(S, 6),
        "forward": round(forward(S, T, r, q), 6),
        "volatility_used": round(sigma, 6),
        "rate_used": round(r, 6),
        "yield_used": round(q, 6),
        "net_cost_per_unit": round(cost, 6),
        "net_cost": round(cost * contract_multiplier, 2),
        "direction_of_cash": "DEBIT" if cost > 0 else "CREDIT",
        "max_profit": None if max_profit is None else round(max_profit * contract_multiplier, 2),
        "max_profit_unbounded": profit_unbounded,
        "max_loss": round(max_loss * contract_multiplier, 2),
        # 8dp, not 2-4: a sub-penny underlying (many altcoins, some FX
        # crosses) would have its breakevens rounded to zero at display
        # precision, and a breakeven of 0 is not a degraded answer, it is
        # a wrong one.
        "breakevens": [round(b, 8) for b in breakevens],
        "probability_of_profit_risk_neutral": round(min(1.0, pop), 6),
        "profitable_regions": profitable,
        "greeks": {k: round(v, 6) for k, v in agg.items()},
        "contract_multiplier": contract_multiplier,
        "caveats": [
            "Priced from a model on REALIZED volatility, not from live quotes. "
            "Implied volatility routinely differs from realized by 10-20 points, "
            "and that gap is where an option's edge actually lives.",
            "Probabilities are RISK-NEUTRAL — the measure the model prices "
            "under, not a forecast of the asset.",
            "European exercise and constant volatility are assumed. American "
            "early exercise and the volatility smile are not modelled.",
        ],
    }
