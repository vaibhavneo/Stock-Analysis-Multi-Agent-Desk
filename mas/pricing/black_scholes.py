"""Black-Scholes-Merton, with the caveats attached to the numbers.

WHAT A MODEL PRICE IS NOT
-------------------------
A model price is what an option WOULD be worth if the assumptions held. They
do not. Returns are not lognormal, volatility is not constant, and — the one
that bites hardest here — the volatility fed in is REALIZED (what the
underlying just did) rather than IMPLIED (what the market charges). Realized
and implied routinely differ by 10-20 volatility points, and the whole edge in
options trading lives in that gap. So a model price computed from realized vol
is a REFERENCE, not a quote, and is labelled MODEL_BASIS everywhere it appears.

The risk-neutral probabilities below carry the same caveat and it is a sharper
one: N(d2) is the probability of finishing in the money UNDER THE PRICING
MEASURE, which is not the real-world probability and is not a forecast. It is
reported because it is the correct input to a breakeven comparison, and it is
named `risk_neutral` in every field so it cannot be quietly read as a forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

CALL = "call"
PUT = "put"

MODEL_BASIS = "MODEL"      # matches decision/levels.py LEVEL_BASIS vocabulary

# Below this, time value is numerically meaningless and the option is its
# intrinsic value. Roughly 1 minute of a trading year.
_MIN_T = 1e-6
_MIN_SIGMA = 1e-6


def _ncdf(x: float) -> float:
    """Standard normal CDF via erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class BSResult:
    kind: str
    price: float
    intrinsic: float
    time_value: float
    delta: float
    gamma: float
    vega: float           # per 1 volatility POINT (1%), the trading convention
    theta: float          # per CALENDAR DAY, the trading convention
    rho: float            # per 1 rate POINT (1%)
    d1: float
    d2: float
    prob_itm_risk_neutral: float
    basis: str = MODEL_BASIS

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def price(kind: str, S: float, K: float, T: float, r: float, sigma: float,
          q: float = 0.0) -> float:
    """Black-Scholes-Merton price.

    S spot, K strike, T years to expiry, r risk-free rate, sigma annualized
    volatility (decimal, not percent), q continuous yield.

    `q` carries the dividend yield for an equity and the FOREIGN interest rate
    for a currency pair (Garman-Kohlhagen is Black-Scholes with q = r_foreign).
    For crypto it is the staking/funding yield and defaults to 0, which is the
    honest default only because this system has no funding-rate source; a
    perpetual's funding can run several percent annualized and would move a
    long-dated price.
    """
    kind = kind.lower()
    if kind not in (CALL, PUT):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if S <= 0 or K <= 0:
        raise ValueError("spot and strike must be positive")
    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        # At expiry (or with no volatility) an option IS its intrinsic value.
        fwd = S * math.exp(-q * max(T, 0.0))
        disc_k = K * math.exp(-r * max(T, 0.0))
        return max(0.0, fwd - disc_k) if kind == CALL else max(0.0, disc_k - fwd)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if kind == CALL:
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def greeks(kind: str, S: float, K: float, T: float, r: float, sigma: float,
           q: float = 0.0) -> BSResult:
    """Price plus the five greeks, in TRADING conventions.

    Vega is per one volatility point and theta is per calendar day, because
    those are the units a position is actually reasoned about in. The raw
    per-unit-vol and per-year forms are an easy source of 100x and 365x
    errors that still look like plausible numbers.
    """
    kind = kind.lower()
    p = price(kind, S, K, T, r, sigma, q)
    intrinsic = max(0.0, S - K) if kind == CALL else max(0.0, K - S)

    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        delta = (1.0 if S > K else 0.0) if kind == CALL else (-1.0 if S < K else 0.0)
        return BSResult(kind=kind, price=p, intrinsic=intrinsic,
                        time_value=max(0.0, p - intrinsic), delta=delta,
                        gamma=0.0, vega=0.0, theta=0.0, rho=0.0,
                        d1=float("nan"), d2=float("nan"),
                        prob_itm_risk_neutral=(1.0 if delta else 0.0))

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrtT = math.sqrt(T)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    nd1 = _npdf(d1)

    gamma = disc_q * nd1 / (S * sigma * sqrtT)
    vega_per_unit = S * disc_q * nd1 * sqrtT

    if kind == CALL:
        delta = disc_q * _ncdf(d1)
        theta_per_year = (-S * nd1 * sigma * disc_q / (2 * sqrtT)
                          - r * K * disc_r * _ncdf(d2)
                          + q * S * disc_q * _ncdf(d1))
        rho_per_unit = K * T * disc_r * _ncdf(d2)
        prob_itm = _ncdf(d2)
    else:
        delta = disc_q * (_ncdf(d1) - 1.0)
        theta_per_year = (-S * nd1 * sigma * disc_q / (2 * sqrtT)
                          + r * K * disc_r * _ncdf(-d2)
                          - q * S * disc_q * _ncdf(-d1))
        rho_per_unit = -K * T * disc_r * _ncdf(-d2)
        prob_itm = _ncdf(-d2)

    return BSResult(
        kind=kind, price=p, intrinsic=intrinsic,
        time_value=max(0.0, p - intrinsic),
        delta=delta, gamma=gamma,
        vega=vega_per_unit / 100.0,        # per 1 vol POINT
        theta=theta_per_year / 365.0,      # per CALENDAR day
        rho=rho_per_unit / 100.0,          # per 1 rate POINT
        d1=d1, d2=d2, prob_itm_risk_neutral=prob_itm)


def implied_volatility(kind: str, market_price: float, S: float, K: float,
                       T: float, r: float, q: float = 0.0,
                       tol: float = 1e-8, max_iter: int = 100) -> Optional[float]:
    """Solve for the volatility that reproduces `market_price`.

    Returns None rather than a number when no volatility can produce that
    price — which happens whenever the quote is below intrinsic or above the
    no-arbitrage bound, i.e. whenever the input is bad. Returning a clamped
    boundary value instead would turn a data error into a confident 500% vol.

    Newton-Raphson on vega, with bisection as the fallback: Newton is fast but
    vega collapses toward zero for deep in- or out-of-the-money options, where
    it diverges. Bisection cannot diverge.
    """
    kind = kind.lower()
    if market_price is None or market_price <= 0 or T <= _MIN_T or S <= 0 or K <= 0:
        return None

    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    if kind == CALL:
        lo_bound, hi_bound = max(0.0, S * disc_q - K * disc_r), S * disc_q
    else:
        lo_bound, hi_bound = max(0.0, K * disc_r - S * disc_q), K * disc_r
    # Strict: a price AT the bound implies infinite or zero vol.
    if market_price <= lo_bound + 1e-12 or market_price >= hi_bound - 1e-12:
        return None

    sigma = 0.25
    for _ in range(max_iter):
        diff = price(kind, S, K, T, r, sigma, q) - market_price
        if abs(diff) < tol:
            return sigma
        v = greeks(kind, S, K, T, r, sigma, q).vega * 100.0    # back to per-unit
        if v < 1e-10:
            break
        step = diff / v
        sigma -= step
        if sigma <= _MIN_SIGMA or sigma > 50.0:
            break
    else:
        return None

    lo, hi = _MIN_SIGMA, 50.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if price(kind, S, K, T, r, mid, q) > market_price:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return 0.5 * (lo + hi)
    return None


def forward(S: float, T: float, r: float, q: float = 0.0) -> float:
    """The forward price. What the model expects, and the correct centre for
    a scenario band — using spot instead biases every long-dated band low by
    the cost of carry."""
    return S * math.exp((r - q) * T)
