"""The local derivatives sub-agent.

Answers "what option structure expresses this view, and what would it cost"
for ANY underlying with a price history — which is the whole point: it covers
the crypto, FX, index and commodity underlyings that OptionsPilot does not,
and the thousands of equities outside its universe.

It is deliberately ranked below OptionsPilot (priority 50 vs 10). Where a real
chain exists, a real chain wins, and nothing here should ever be read as a
quote. Two things keep that honest:

  - every number carries `basis: MODEL`
  - the volatility used is REALIZED, stated in the output, on the asset's own
    calendar (365 days for a coin — see mas/asset_class.py for why that is not
    a detail)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..asset_class import spec_for, classify
from ..contract import AgentRequest, AgentResult, ok, unavailable, error
from ..pricing.black_scholes import MODEL_BASIS
from ..pricing.payoff import evaluate
from ..pricing.structures import STRUCTURES, build_structure, UnboundedLoss
from .rates import risk_free_rate

AGENT_ID = "derivatives"

# Structures offered per directional view. Each view gets a debit expression
# (pays more, wins less often) and a credit one (wins more often, pays less),
# because offering only one hides the trade-off that decides between them.
BY_VIEW = {
    "BULLISH": ["long_call", "bull_call_spread", "bull_put_spread"],
    "BEARISH": ["long_put", "bear_put_spread", "bear_call_spread"],
    "NEUTRAL": ["iron_condor", "iron_butterfly"],
    "VOLATILE": ["long_straddle", "long_strangle"],
}

MIN_BARS = 40          # below this a realized-vol estimate is noise
DEFAULT_DAYS = 45


def _strike_step(price: float) -> float:
    """A sane strike increment for this price magnitude.

    Listed chains do not quote continuous strikes, and proposing a $81,437.22
    strike on BTC would be proposing something that does not exist anywhere.
    """
    if price <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(price))
    for frac in (0.025, 0.05, 0.1):
        step = mag * frac
        if price / step <= 60:
            return step
    return mag * 0.25


def _round_strike(x: float, step: float) -> float:
    r = round(x / step) * step
    return round(r, 10)


def realized_volatility(closes, days_per_year: int, window: int = 30) -> Optional[float]:
    """Annualized realized volatility on the asset's OWN calendar."""
    if closes is None or len(closes) < MIN_BARS:
        return None
    rets = []
    tail = list(closes)[-(window + 1):]
    for a, b in zip(tail, tail[1:]):
        if a and a > 0 and b and b > 0:
            rets.append(math.log(b / a))
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(days_per_year)


def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
    spec = spec_for(asset_class)
    if spec.asset_class == "UNKNOWN":
        return False, ("the symbol could not be classified, so neither its "
                       "volatility calendar nor its option conventions are known")
    return True, ""


def _candidates(view: str, S: float, sigma: float, T: float,
                step: float) -> List[Tuple[str, tuple]]:
    """Strikes placed at multiples of the expected move, not at round guesses.

    One standard deviation over the life of the option is `S * sigma * sqrt(T)`.
    Anchoring the short strike there is what makes the structure express the
    view at the scale the asset actually moves — a 10%-wide spread is a
    different trade on a 12%-vol index than on an 80%-vol altcoin.
    """
    sd = S * sigma * math.sqrt(T)
    atm = _round_strike(S, step)
    up1, up2 = _round_strike(S + sd, step), _round_strike(S + 2 * sd, step)
    dn1, dn2 = _round_strike(S - sd, step), _round_strike(S - 2 * sd, step)

    if view == "BULLISH":
        out = [("long_call", (atm,)),
               ("bull_call_spread", (atm, up1)),
               ("bull_put_spread", (dn1, dn2))]
    elif view == "BEARISH":
        out = [("long_put", (atm,)),
               ("bear_put_spread", (atm, dn1)),
               ("bear_call_spread", (up1, up2))]
    elif view == "VOLATILE":
        out = [("long_straddle", (atm,)), ("long_strangle", (dn1, up1))]
    else:
        out = [("iron_condor", (dn2, dn1, up1, up2)),
               ("iron_butterfly", (dn1, atm, up1))]
    # Degenerate strikes (all collapsing to the same level on a low-vol, short
    # -dated underlying) would build structures with zero width.
    return [(n, a) for n, a in out if len(set(a)) == len(a)]


def run(request: AgentRequest) -> AgentResult:
    cap = request.capability
    try:
        if cap == "option_structures":
            return _structures(request)
        if cap == "option_pricing":
            return _price_one(request)
        if cap == "implied_volatility":
            return _implied(request)
        return unavailable(AGENT_ID, cap, f"this agent does not offer {cap!r}")
    except Exception as e:                      # never raises past here
        return error(AGENT_ID, cap, e)


def _bars(request: AgentRequest):
    closes = request.params.get("closes")
    if closes:
        return list(closes)
    from financial_data import get_bars_df
    df = get_bars_df(request.symbol, period=request.params.get("period", "1y"))
    return [float(v) for v in df["Close"].tolist()]


def _structures(request: AgentRequest) -> AgentResult:
    spec = spec_for(request.asset_class)
    closes = _bars(request)
    if len(closes) < MIN_BARS:
        return unavailable(AGENT_ID, request.capability,
                           f"only {len(closes)} price bars are available; "
                           f"{MIN_BARS} are needed for a realized-volatility "
                           "estimate that is not mostly noise")

    sigma = realized_volatility(closes, spec.days_per_year)
    if sigma is None or sigma <= 0:
        return unavailable(AGENT_ID, request.capability,
                           "realized volatility could not be estimated from "
                           "this price history")

    S = float(closes[-1])
    view = (request.params.get("view") or "NEUTRAL").upper()
    if view not in BY_VIEW:
        view = "NEUTRAL"
    days = int(request.params.get("days") or DEFAULT_DAYS)
    T = max(days, 1) / 365.0
    mult = float(request.params.get("contract_multiplier")
                 or (100.0 if spec.options == "LISTED" else 1.0))

    rate = risk_free_rate()
    step = _strike_step(S)

    priced: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for name, strikes in _candidates(view, S, sigma, T, step):
        try:
            s = build_structure(name, *strikes, expiry_years=T)
            priced.append(evaluate(s, S=S, r=rate["rate"], sigma=sigma,
                                   contract_multiplier=mult))
        except UnboundedLoss as e:
            rejected.append({"structure": name, "reason": str(e)})
        except Exception as e:
            rejected.append({"structure": name,
                             "reason": f"{type(e).__name__}: {e}"})

    if not priced:
        return unavailable(AGENT_ID, request.capability,
                           "no admissible structure could be built at this "
                           "spot and volatility")

    honesty = [
        f"Priced from a MODEL, not from quotes. No option chain for "
        f"{request.symbol} was read.",
        f"Volatility is REALIZED ({sigma * 100:.1f}% annualized over 30 bars, "
        f"{spec.days_per_year}-day calendar), not implied. The market's own "
        f"price for volatility is usually different, and that difference is "
        f"where an option's edge lives.",
        rate["note"],
    ]
    if spec.options == "NO_ACCESSIBLE_VENUE":
        honesty.insert(0, (
            f"No venue this system can read lists options on {request.symbol}. "
            f"These structures describe the SHAPE of an exposure and its fair "
            f"model value; they are not orders anyone could place from here."))

    return ok(AGENT_ID, request.capability,
              {"symbol": request.symbol, "asset_class": spec.asset_class,
               "view": view, "spot": round(S, 8),
               "expiry_days": days,
               "volatility_annualized_pct": round(sigma * 100, 2),
               "volatility_basis": "REALIZED_30_BAR",
               "annualization_days": spec.days_per_year,
               "strike_step": step,
               "risk_free_rate": rate,
               "contract_multiplier": mult,
               "candidates": priced,
               "rejected": rejected,
               "honesty": honesty},
              price_basis=MODEL_BASIS,
              engine="black_scholes_merton", n_candidates=len(priced))


def _price_one(request: AgentRequest) -> AgentResult:
    """Price a structure the caller specifies, rather than one this agent chose."""
    p = request.params
    name = p.get("structure")
    strikes = p.get("strikes") or []
    if not name or not strikes:
        return unavailable(AGENT_ID, request.capability,
                           "option_pricing needs `structure` and `strikes`")
    spec = spec_for(request.asset_class)
    closes = _bars(request)
    sigma = p.get("sigma") or realized_volatility(closes, spec.days_per_year)
    if not sigma:
        return unavailable(AGENT_ID, request.capability,
                           "no volatility estimate is available")
    S = float(p.get("spot") or closes[-1])
    T = max(int(p.get("days") or DEFAULT_DAYS), 1) / 365.0
    mult = float(p.get("contract_multiplier")
                 or (100.0 if spec.options == "LISTED" else 1.0))
    rate = risk_free_rate()
    s = build_structure(name, *strikes, expiry_years=T)
    return ok(AGENT_ID, request.capability,
              {"symbol": request.symbol,
               "evaluation": evaluate(s, S=S, r=rate["rate"], sigma=float(sigma),
                                      contract_multiplier=mult),
               "risk_free_rate": rate},
              price_basis=MODEL_BASIS)


def _implied(request: AgentRequest) -> AgentResult:
    """This agent has no chain, so it cannot report implied volatility.

    Declining explicitly matters: `implied_volatility` is in its registry
    capability list so the planner knows to ASK, and OptionsPilot answering it
    for an equity while this declines for a coin is exactly the routing
    difference the trace should show. Returning realized vol relabelled as
    implied would be the worst available answer.
    """
    spec = spec_for(request.asset_class)
    closes = _bars(request)
    sigma = realized_volatility(closes, spec.days_per_year)
    return unavailable(
        AGENT_ID, request.capability,
        "this engine reads no option chain, so it has no IMPLIED volatility. "
        + (f"Realized volatility is {sigma * 100:.1f}% annualized on a "
           f"{spec.days_per_year}-day calendar, which is a different quantity "
           "and is reported under option_structures."
           if sigma else "No realized estimate is available either."))
