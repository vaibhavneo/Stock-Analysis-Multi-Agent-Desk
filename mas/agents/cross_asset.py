"""How much of this move is just the benchmark?

THE ERROR THIS REPLACES
-----------------------
`fetch_fundamentals` returns a `beta` field for almost anything, and the risk
pillar consumes it. For a US equity that beta is measured against the equity
market and means what it says. For a coin, the provider's beta — when it
exists at all — is a number computed against an index the asset has no
structural relationship to, and the risk pillar was scoring it as if it did.
`asset_class.has_equity_beta` now gates that. This agent supplies the
replacement: a relation to the benchmark the asset is ACTUALLY driven by.

THE ALIGNMENT TRAP
------------------
BTC returns 184 bars over six months; the S&P returns 127. They trade on
different calendars. Zipping two unaligned series produces a correlation
between Monday's bitcoin and the previous Thursday's equities — a number that
looks perfectly reasonable and means nothing. Returns are therefore joined on
DATE, and the overlap count is reported so a thin join is visible rather than
silently confident.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..asset_class import spec_for
from ..contract import AgentRequest, AgentResult, ok, unavailable, error

AGENT_ID = "cross_asset"

# The benchmark each class is actually driven by.
BENCHMARK = {
    "EQUITY": ("^GSPC", "the S&P 500"),
    "ETF": ("^GSPC", "the S&P 500"),
    "INDEX": ("^GSPC", "the S&P 500"),
    "CRYPTO": ("BTC-USD", "bitcoin"),
    "FX": ("DX-Y.NYB", "the dollar index"),
    "COMMODITY": ("^GSPC", "the S&P 500"),
}

# Bitcoin has no crypto benchmark above it, so it is measured against risk
# assets instead — which is the question actually being asked of it.
SELF_BENCHMARK_FALLBACK = ("^GSPC", "the S&P 500")

MIN_OVERLAP = 40


def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
    spec = spec_for(asset_class)
    if spec.asset_class == "UNKNOWN":
        return False, "the symbol could not be classified, so no benchmark applies"
    if asset_class not in BENCHMARK:
        return False, f"no benchmark is defined for {asset_class}"
    return True, ""


def benchmark_for(symbol: str, asset_class: str) -> Tuple[str, str]:
    sym = (symbol or "").upper().strip()
    bench, label = BENCHMARK.get(asset_class, SELF_BENCHMARK_FALLBACK)
    if bench == sym:
        return SELF_BENCHMARK_FALLBACK
    return bench, label


def _aligned_returns(a_close: Dict[str, float],
                     b_close: Dict[str, float]) -> Tuple[List[float], List[float], List[str]]:
    """Log returns on the DATES BOTH SERIES SHARE.

    Consecutive shared dates only: a return computed across a gap in one
    series (a crypto weekend the equity market did not trade) spans a
    different amount of time in each, which is not a comparable observation.
    """
    dates = sorted(set(a_close) & set(b_close))
    ra: List[float] = []
    rb: List[float] = []
    used: List[str] = []
    for prev, cur in zip(dates, dates[1:]):
        pa, ca = a_close.get(prev), a_close.get(cur)
        pb, cb = b_close.get(prev), b_close.get(cur)
        if all(v and v > 0 for v in (pa, ca, pb, cb)):
            ra.append(math.log(ca / pa))
            rb.append(math.log(cb / pb))
            used.append(cur)
    return ra, rb, used


def _regress(y: List[float], x: List[float]) -> Dict[str, float]:
    n = len(y)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    beta = sxy / sxx if sxx > 0 else float("nan")
    corr = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    return {"beta": beta, "correlation": corr,
            "r_squared": corr * corr if corr == corr else float("nan")}


def _closes_by_date(symbol: str, period: str) -> Dict[str, float]:
    from financial_data import get_bars_df
    df = get_bars_df(symbol, period=period)
    return {str(idx)[:10]: float(v) for idx, v in df["Close"].items()}


def run(request: AgentRequest) -> AgentResult:
    if request.capability != "benchmark_relation":
        return unavailable(AGENT_ID, request.capability,
                           f"this agent does not offer {request.capability!r}")
    try:
        period = request.params.get("period", "1y")
        bench, label = benchmark_for(request.symbol, request.asset_class)

        try:
            mine = _closes_by_date(request.symbol, period)
        except Exception as e:
            return unavailable(AGENT_ID, request.capability,
                               f"no price history for {request.symbol} "
                               f"({type(e).__name__})")
        try:
            theirs = _closes_by_date(bench, period)
        except Exception as e:
            return unavailable(AGENT_ID, request.capability,
                               f"the benchmark {bench} could not be fetched "
                               f"({type(e).__name__})")

        y, x, used = _aligned_returns(mine, theirs)
        if len(y) < MIN_OVERLAP:
            return unavailable(
                AGENT_ID, request.capability,
                f"{request.symbol} and {bench} share only {len(y)} consecutive "
                f"trading dates over {period}; {MIN_OVERLAP} are needed before a "
                "beta is worth reporting")

        reg = _regress(y, x)
        r2 = reg["r_squared"]
        idio = 1.0 - r2 if r2 == r2 else None

        if idio is None:
            verdict = "the relationship could not be measured"
        elif idio > 0.85:
            verdict = (f"almost none of {request.symbol}'s movement is "
                       f"explained by {label}; it is trading on its own story")
        elif idio > 0.5:
            verdict = (f"most of {request.symbol}'s movement is its own; "
                       f"{label} explains a minority of it")
        else:
            verdict = (f"{label} explains most of {request.symbol}'s movement — "
                       f"a view on it is largely a view on {label}")

        return ok(AGENT_ID, request.capability, {
            "symbol": request.symbol,
            "benchmark": bench,
            "benchmark_label": label,
            "period": period,
            "overlapping_returns": len(y),
            "first_date": used[0] if used else None,
            "last_date": used[-1] if used else None,
            "beta": round(reg["beta"], 4) if reg["beta"] == reg["beta"] else None,
            "correlation": round(reg["correlation"], 4) if reg["correlation"] == reg["correlation"] else None,
            "r_squared": round(r2, 4) if r2 == r2 else None,
            "idiosyncratic_share": round(idio, 4) if idio is not None else None,
            "verdict": verdict,
            "honesty": [
                (f"Returns joined on the {len(y)} dates {request.symbol} and "
                 f"{bench} BOTH traded, out of {len(mine)} and {len(theirs)} "
                 "bars respectively. Unaligned series would correlate one "
                 "asset's Monday with the other's previous Thursday."),
                ("Beta is a description of the past window, not a forecast, "
                 "and it is unstable across regimes."),
            ],
        }, price_basis="PRICE_HISTORY_DERIVED", benchmark=bench)
    except Exception as e:
        return error(AGENT_ID, request.capability, e)
