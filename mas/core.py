"""The orchestrator's OWN read — the primary agent's answer, class-aware.

This is Stock Agent's seat at its own table. It runs the deterministic
pillar stack for any asset class, with the applicability mask from
`asset_class.py` doing the work: a coin is scored on technical and algo
renormalized to 1.0, not on a fundamentals pillar filled with a neutral 50.

It is deliberately the FAST path — indicators, algo signals and pillars, no
LLM agents and no deep backtest. The orchestrator needs a direction to hand
to the options specialist before it can ask anything useful, and making that
hand-off wait on a 70-second ranking query would make the whole flow unusable.
Callers that want the full Decision Intelligence object still go to
`/api/decision-intelligence`; this exists so a BTC or EURUSD request, which
has no seven-agent pipeline behind it at all, still produces a real read.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .asset_class import classify, spec_for

# Composite -> the direction handed to an options specialist. HOLD maps to
# None, not to NEUTRAL: "I have no directional view" and "I expect it to sit
# still" are different claims, and only the second justifies selling premium.
ACTION_TO_VIEW = {
    "BUY": "BULLISH", "ACCUMULATE": "BULLISH",
    "REDUCE": "BEARISH", "SELL": "BEARISH",
    "HOLD": None,
}


def view_for_action(action: Optional[str]) -> Optional[str]:
    return ACTION_TO_VIEW.get((action or "").upper())


def core_read(symbol: str, period: str = "1y",
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The primary agent's own read. Degrades rather than raising."""
    cls = classify(symbol, metadata)

    out: Dict[str, Any] = {"symbol": cls["symbol"], "status": "OK"}

    try:
        from financial_data import get_bars_df
        df = get_bars_df(cls["symbol"], period=period)
    except Exception as e:
        out.update({"asset_class": cls["asset_class"], "classification": cls,
                    "status": "NO_DATA",
                    "reason": (f"no price history for {cls['symbol']} "
                               f"({type(e).__name__})")})
        return out

    # A bare alphabetic ticker cannot be separated from an ETF by shape, so
    # the first pass returns ASSUMED. Resolve it before anything is scored:
    # an ETF carries no income statement, and scoring one on the equity mask
    # weights a fundamentals pillar built from PORTFOLIO aggregates as if it
    # measured company quality. SPY was scoring 80.5 there, at 20% of its
    # composite, for a fund that has no margins, no ROE and no filings.
    prefetched: Dict[str, Any] = {}
    if cls.get("basis") == "ASSUMED":
        try:
            from tools.market_data import fetch_fundamentals as _ff
            prefetched = _ff(cls["symbol"]) or {}
            if prefetched:
                cls = classify(symbol, prefetched)
        except Exception:
            prefetched = {}

    asset_class = cls["asset_class"]
    spec = spec_for(asset_class)
    out["asset_class"] = asset_class
    out["classification"] = cls
    # The identity fields ONLY, handed back so the planner can classify the
    # same way this did. Without it the two classify independently and can
    # disagree: SPY was scored on the ETF mask while the answer reported
    # EQUITY, because build_plan had no metadata and fell back to ASSUMED.
    out["identity"] = {k: prefetched[k] for k in
                       ("quoteType", "fundFamily", "navPrice", "sector",
                        "trailingEps")
                       if prefetched.get(k) is not None}

    try:
        from tools.market_data import (compute_indicators, compute_algo_signals,
                                       compute_signal_summary, fetch_fundamentals)
        from backtest.pillars import compute_pillar_scores

        ind = compute_indicators(df)
        algo = compute_algo_signals(df, ind, asset_class=asset_class)
        sig = compute_signal_summary(ind)

        # Fundamentals are fetched only where they can exist. For a coin the
        # provider returns a handful of price fields dressed as fundamentals,
        # and the mask already excludes the pillar — so the call is skipped
        # rather than made and discarded.
        fundamentals: Dict[str, Any] = dict(prefetched)
        if not fundamentals and (spec.has_fundamentals or spec.has_equity_beta):
            try:
                fundamentals = fetch_fundamentals(cls["symbol"]) or {}
            except Exception:
                fundamentals = {}

        pillars = compute_pillar_scores(cls["symbol"], ind, sig, algo,
                                        fundamentals, asset_class=asset_class)
        action = pillars.get("action")
        out.update({
            "current_price": ind.get("current_price"),
            "composite": pillars.get("composite"),
            "action": action,
            "view": view_for_action(action),
            "pillars": pillars.get("pillars"),
            "weights": pillars.get("weights"),
            "nominal_weights": pillars.get("nominal_weights"),
            "inapplicable_pillars": pillars.get("inapplicable_pillars"),
            "composite_scorable": pillars.get("composite_scorable"),
            "risk_veto": pillars.get("risk_veto"),
            "volatility": {
                "historical_volatility_20d": algo.get("historical_volatility_20d"),
                "regime": algo.get("vol_regime"),
                "annualization_days": algo.get("annualization_days"),
                "bands": algo.get("vol_bands"),
            },
            "bars": len(df),
            "period": period,
        })
        if spec.inapplicable_pillars:
            out["scoring_note"] = (
                f"A {spec.label} has no "
                + ", ".join(sorted(spec.inapplicable_pillars))
                + " input. Those pillars are excluded and the remaining "
                  "weights renormalized, rather than scored a neutral 50 that "
                  "the composite would still weight.")
    except Exception as e:
        out.update({"status": "ERROR",
                    "reason": f"{type(e).__name__}: {e}"})
    return out


def ask(symbol: str, period: str = "1y", capabilities=None,
        params: Optional[Dict[str, Any]] = None,
        include_core: bool = True) -> Dict[str, Any]:
    """The whole flow: read, plan, run the specialists, synthesize.

    The orchestrator's own direction is handed DOWN to the specialists rather
    than letting each derive its own — two agents guessing a direction
    independently is how a bullish structure and a bearish one end up in the
    same answer.
    """
    from .plan import build_plan, describe
    from .run import execute
    from .synthesis import synthesize

    core = core_read(symbol, period=period) if include_core else None
    view = (core or {}).get("view")

    # The orchestrator chooses what to ask for. Capabilities the CALLER named
    # are explicit and always run; the rest are implied by the depth of the
    # request and are run only where they would add something.
    from .policy import decide as _decide, FULL as _FULL
    pol = _decide(capabilities or [], (core or {}).get("asset_class")
                  or classify(symbol)["asset_class"],
                  depth=_FULL, direction=view)

    # One classification, shared. The planner must see what the read saw.
    plan = build_plan(symbol, metadata=(core or {}).get("identity"),
                      capabilities=pol.run, view=view, params=params)
    execution = execute(plan)
    answer = synthesize(execution, core=core)

    answer["policy"] = pol.to_dict()
    answer["classification"] = (core or {}).get("classification") or \
        plan.get("classification")
    answer["core"] = core
    answer["plan"] = plan
    answer["plan_description"] = describe(plan)
    return answer
