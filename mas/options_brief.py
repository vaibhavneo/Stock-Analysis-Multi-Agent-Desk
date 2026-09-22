"""Options as part of the ANSWER, not a separate app you have to reach.

WHAT WAS WRONG
--------------
The decision brief called `optionspilot/client.py` directly. When OptionsPilot
was unreachable — or, far more commonly, when its access code was absent or
rotated — the user got a yellow error where an analysis should have been. The
orchestration layer existed by then and could have answered locally, but the
brief never asked it.

So the error was not "OptionsPilot is down". The error was that one specialist
had become a hard dependency of the primary agent. A sub-agent that can take
the orchestrator's answer down with it is not a sub-agent.

This module is the seam. It asks the ROSTER for option structures, and the
roster tries a live chain first and a local model second. Both sources are
normalized to one shape, so the brief renders whichever answered and the
reader is told, prominently, which one they got — because a model price and a
quote are different kinds of claim and must never look alike.

The one thing that never degrades: no path here can place an order.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from .asset_class import classify

DEFAULT_DAYS = 45

# Which way the cash moves at entry, by structure. Needed because the chain
# specialist reports max_loss and max_gain but not the net premium, and for a
# vertical those are the same number seen from opposite sides: a debit
# spread's worst case IS what it cost, and a credit spread's best case IS what
# it collected. Inferring the side from the structure name recovers the cost
# without asking the other service to change its payload.
CASH_AT_ENTRY = {
    "long_call": "DEBIT", "long_put": "DEBIT",
    "bull_call_spread": "DEBIT", "bear_put_spread": "DEBIT",
    "long_straddle": "DEBIT", "long_strangle": "DEBIT",
    "call_butterfly": "DEBIT", "put_butterfly": "DEBIT",
    "bull_put_spread": "CREDIT", "bear_call_spread": "CREDIT",
    "short_put_spread": "CREDIT", "short_call_spread": "CREDIT",
    "iron_condor": "CREDIT", "iron_butterfly": "CREDIT",
}

# Names for the shapes, in the words someone would actually use.
PLAIN_NAME = {
    "long_call": "buy a call",
    "long_put": "buy a put",
    "bull_call_spread": "buy a call spread",
    "bear_put_spread": "buy a put spread",
    "bull_put_spread": "sell a put spread",
    "bear_call_spread": "sell a call spread",
    "long_straddle": "buy a straddle",
    "long_strangle": "buy a strangle",
    "iron_condor": "sell an iron condor",
    "iron_butterfly": "sell an iron butterfly",
    "call_butterfly": "buy a call butterfly",
    "put_butterfly": "buy a put butterfly",
    # OptionsPilot's own keys, so both sources read the same.
    "short_put_spread": "sell a put spread",
    "short_call_spread": "sell a call spread",
}


def _money(v: Optional[float]) -> str:
    """Money keeps both decimals. Stripping trailing zeros turned $1,254.80
    into "$1,254.8", which reads as a truncation bug in a cost figure."""
    if v is None:
        return "—"
    v = float(v)
    sign = "-$" if v < 0 else "$"
    a = abs(v)
    if 0 < a < 0.01:          # sub-cent premiums exist on low-priced underlyings
        return sign + f"{a:,.6f}".rstrip("0")
    return sign + f"{a:,.2f}"


def _price(v: Optional[float]) -> str:
    if v is None:
        return "—"
    v = float(v)
    # Sub-dollar underlyings need their decimals; index levels do not.
    if abs(v) >= 100:
        return f"${v:,.0f}"
    if abs(v) >= 1:
        return f"${v:,.2f}"
    return f"${v:,.6f}".rstrip("0")


def _easter(year: int) -> dt.date:
    """Anonymous Gregorian algorithm. Needed only to locate Good Friday."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 432
    month = (h + l - 7 * m + 90) // 25
    day = (h + l - 7 * m + 33 * month + 19) % 32
    return dt.date(year, month, day)


def _market_closed(d: dt.date) -> bool:
    """US market holidays that can land on a Friday.

    Good Friday is the one that matters most here: it is ALWAYS a Friday, so a
    rule that snaps forward to Friday will collide with it every year. The
    fixed-date holidays are included because the same rule produced 25 Dec as
    an expiry — Christmas Day, when nothing trades.
    """
    if (d.month, d.day) in ((1, 1), (7, 4), (12, 25)):
        return True
    return d == _easter(d.year) - dt.timedelta(days=2)


def expiry_date(days: int, today: Optional[dt.date] = None) -> dt.date:
    """The Friday at or after `days` out, stepped back off a market holiday.

    Listed options expire on Fridays, so a Tuesday expiry would name a
    contract that does not exist — the same class of error as an off-grid
    strike. When that Friday is a holiday the exchange moves expiry to the
    preceding trading day, and so does this.
    """
    d = (today or dt.date.today()) + dt.timedelta(days=max(1, int(days)))
    d = d + dt.timedelta(days=(4 - d.weekday()) % 7)
    while _market_closed(d):
        d -= dt.timedelta(days=1)
        while d.weekday() >= 5:        # never land on a weekend
            d -= dt.timedelta(days=1)
    return d


def _plain_for(c: Dict[str, Any], ticker: str, spot: Optional[float],
               expiry: dt.date) -> str:
    """One sentence a person can act on, per structure."""
    name = (c.get("structure") or "").lower()
    cost = c.get("net_cost")
    gain = c.get("max_gain")
    loss = c.get("max_loss")
    bes = c.get("breakevens") or []
    credit = (c.get("cash_direction") == "CREDIT")
    by = expiry.strftime("%-d %b %Y") if hasattr(expiry, "strftime") else str(expiry)

    def pct(level):
        if not spot or not level:
            return ""
        return f" ({(level / spot - 1) * 100:+.1f}%)"

    strikes = [l.get("strike") for l in (c.get("legs") or []) if l.get("strike") is not None]
    lo, hi = (min(strikes), max(strikes)) if strikes else (None, None)

    if credit:
        keep = _money(abs(cost) if cost is not None else None)
        head = f"Collect {keep} now."
        if name in ("bull_put_spread", "short_put_spread"):
            return (f"{head} You keep it all if {ticker} holds above "
                    f"{_price(hi)}{pct(hi)} through {by}. Worst case you lose "
                    f"{_money(abs(loss) if loss is not None else None)} below "
                    f"{_price(lo)}.")
        if name in ("bear_call_spread", "short_call_spread"):
            return (f"{head} You keep it all if {ticker} stays below "
                    f"{_price(lo)}{pct(lo)} through {by}. Worst case you lose "
                    f"{_money(abs(loss) if loss is not None else None)} above "
                    f"{_price(hi)}.")
        if name in ("iron_condor", "iron_butterfly"):
            # An iron butterfly's two short strikes are the SAME strike — that
            # is what distinguishes it from a condor. Describing it as a band
            # produced "stays between $85,000 and $85,000", which reads as a
            # bug and understates how demanding the structure is.
            band = sorted(set(sorted(strikes)[1:3])) if len(strikes) >= 4 else \
                sorted(set(strikes))
            worst = _money(abs(loss) if loss is not None else None)
            if len(band) >= 2:
                return (f"{head} You keep it if {ticker} stays between "
                        f"{_price(band[0])} and {_price(band[-1])} through "
                        f"{by}. Worst case you lose {worst}.")
            if band:
                # The date belongs with the condition, before the warning —
                # otherwise the sentence reads "...costs the maximum through
                # 6 Nov", attaching the expiry to the wrong clause.
                return (f"{head} You keep it only if {ticker} finishes very "
                        f"close to {_price(band[0])}{pct(band[0])} on {by}. "
                        f"That is a pin, not a range: any real move either way "
                        f"costs the full {worst}.")
            return (f"{head} You keep it if {ticker} stays in a narrow range "
                    f"through {by}. Worst case you lose {worst}.")
        return (f"{head} Worst case you lose "
                f"{_money(abs(loss) if loss is not None else None)} by {by}.")

    # Debit structures
    paid = _money(cost)
    if name == "long_call":
        return (f"Pay {paid}. Above {_price(bes[0]) if bes else '—'}"
                f"{pct(bes[0]) if bes else ''} by {by} you profit, with no ceiling. "
                f"Below {_price(lo)} the whole {paid} is gone.")
    if name == "long_put":
        return (f"Pay {paid}. Below {_price(bes[0]) if bes else '—'}"
                f"{pct(bes[0]) if bes else ''} by {by} you profit. "
                f"Above {_price(hi)} the whole {paid} is gone.")
    if name in ("bull_call_spread", "bear_put_spread"):
        target = hi if name == "bull_call_spread" else lo
        direction = "reaches" if name == "bull_call_spread" else "falls to"
        return (f"Pay {paid}. Pays up to {_money(gain)} if {ticker} {direction} "
                f"{_price(target)}{pct(target)} by {by}. Break-even "
                f"{_price(bes[0]) if bes else '—'}; most you can lose is {paid}.")
    if name in ("long_straddle", "long_strangle"):
        if len(bes) >= 2:
            return (f"Pay {paid}. This needs MOVEMENT, not direction: it pays "
                    f"beyond {_price(bes[0])}{pct(bes[0])} or "
                    f"{_price(bes[1])}{pct(bes[1])} by {by}. Anything quieter "
                    f"than that and you lose, up to the full {paid}.")
        return (f"Pay {paid}. Needs a large move either way by {by}.")
    if "butterfly" in name:
        body = sorted(strikes)[len(strikes) // 2] if strikes else None
        return (f"Pay {paid}. Pays most if {ticker} finishes near "
                f"{_price(body)}{pct(body)} on {by}. Most you can lose is {paid}.")
    return (f"Pay {paid}. Most you can lose is {paid}; most you can make is "
            f"{_money(gain) if gain is not None else 'uncapped'}, by {by}.")


def _from_model(data: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
    """The local engine's evaluations, in the brief's candidate shape."""
    out: List[Dict[str, Any]] = []
    spot = data.get("spot")
    days = data.get("expiry_days") or DEFAULT_DAYS
    exp = expiry_date(days)
    for i, ev in enumerate(data.get("candidates") or [], start=1):
        s = ev.get("structure") or {}
        legs = s.get("legs") or []
        cand = {
            "rank": i,
            "structure": s.get("name"),
            "label": (PLAIN_NAME.get(s.get("name"), (s.get("name") or "")
                                     .replace("_", " ")).capitalize()),
            "structure_text": PLAIN_NAME.get(s.get("name")),
            "legs": legs,
            "legs_text": "; ".join(
                f"{'buy' if (l.get('qty') or 0) > 0 else 'sell'} "
                f"{abs(l.get('qty') or 0)} {l.get('kind')}"
                f"{'s' if abs(l.get('qty') or 0) != 1 else ''} at "
                f"{_price(l.get('strike'))}" for l in legs).capitalize() + ".",
            "note": s.get("rationale"),
            "net_cost": ev.get("net_cost"),
            "cash_direction": ev.get("direction_of_cash"),
            "max_loss": ev.get("max_loss"),
            "max_gain": (None if ev.get("max_profit_unbounded")
                         else ev.get("max_profit")),
            "max_gain_unbounded": bool(ev.get("max_profit_unbounded")),
            "breakevens": ev.get("breakevens") or [],
            "probability_of_profit": ev.get("probability_of_profit_risk_neutral"),
            "probability_basis": ("risk-neutral under the pricing model, from "
                                  "REALIZED volatility — not a measured frequency "
                                  "and not the market's own number"),
            "greeks": ev.get("greeks"),
        }
        mg, ml = cand["max_gain"], cand["max_loss"]
        cand["risk_reward_ratio"] = (round(mg / abs(ml), 2)
                                     if mg is not None and ml not in (None, 0)
                                     else None)
        be = cand["breakevens"]
        if be and spot:
            cand["breakeven_text"] = (
                f"Breaks even at {_price(be[0])}, {(be[0] / spot - 1) * 100:+.1f}% "
                f"from here" + (f"; and again at {_price(be[1])}."
                                if len(be) > 1 else "."))
        cand["plain"] = _plain_for(cand, ticker, spot, exp)
        out.append(cand)
    return out


def _from_chain(overlay: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
    """OptionsPilot's own overlay, already close to the shape. Kept as a
    separate branch rather than merged, because its numbers come from quotes
    and its probability comes from IMPLIED volatility — the same field name
    carrying a materially stronger claim."""
    days = overlay.get("days_to_expiry") or DEFAULT_DAYS
    exp = expiry_date(days)
    spot = overlay.get("spot")
    out = []
    for c in overlay.get("candidates") or []:
        c = dict(c)
        name = (c.get("structure") or "").lower()
        side = c.get("cash_direction") or CASH_AT_ENTRY.get(name)
        c["cash_direction"] = side
        if c.get("net_cost") is None:
            if side == "CREDIT" and c.get("max_gain") is not None:
                c["net_cost"] = -abs(float(c["max_gain"]))
            elif side == "DEBIT" and c.get("max_loss") is not None:
                c["net_cost"] = abs(float(c["max_loss"]))
        c["max_gain_unbounded"] = c.get("max_gain") is None
        c["probability_basis"] = ("implied by the option chain's own volatility "
                                  "— the market's number, model-derived, not a "
                                  "measured frequency")
        c["plain"] = _plain_for(c, ticker, spot, exp)
        out.append(c)
    return out


def build(ticker: str,
          view: Optional[str] = None,
          closes: Optional[List[float]] = None,
          days: Optional[int] = None,
          metadata: Optional[Dict[str, Any]] = None,
          spot: Optional[float] = None) -> Dict[str, Any]:
    """The options section of a brief. Never raises; always returns a section.

    `closes` lets a caller that already holds the price history hand it over,
    so folding options into the brief costs no second fetch. `view` is the
    orchestrator's own direction — handed down, never re-derived, so one
    answer cannot contain a bullish structure and a bearish one.
    """
    from .plan import build_plan
    from .run import execute

    cls = classify(ticker, metadata)
    days = int(days or DEFAULT_DAYS)
    params: Dict[str, Any] = {"days": days}
    if closes:
        params["closes"] = list(closes)

    section: Dict[str, Any] = {
        "ticker": cls["symbol"],
        "asset_class": cls["asset_class"],
        "view": view,
        "days_to_expiry": days,
        "expiry": expiry_date(days).isoformat(),
        "expiry_note": ("The Friday at or after this horizon — listed options "
                        "expire on Fridays, so a mid-week date would name a "
                        "contract that does not exist. Stepped back a day when "
                        "that Friday is a market holiday."),
        "no_execution": ("Nothing in this system can place an order. These are "
                         "structures to study."),
    }

    try:
        plan = build_plan(cls["symbol"], metadata=metadata,
                          capabilities=["option_structures"], view=view,
                          params=params)
        execution = execute(plan)
        step = (execution.get("by_capability") or {}).get("option_structures") or {}
        declined = [d for d in (plan.get("declined") or [])
                    if d.get("capability") == "option_structures"
                    and d.get("agent_id")]
    except Exception as e:
        section.update({"status": "UNAVAILABLE",
                        "reason": f"the options layer failed ({type(e).__name__})",
                        "candidates": [], "honesty": [], "trace": []})
        return section

    # The trace, compacted. `closes` is stripped: it is a few hundred floats
    # and would dominate the payload for no reader's benefit.
    # Agents that were DECLINED at planning time are listed alongside the ones
    # that ran. A specialist skipped before any call and one that was called
    # and failed are different facts, and only showing the latter makes a
    # missing chain look like an absence of information rather than a
    # configuration gap the reader can close.
    section["trace"] = [
        {"agent_id": d["agent_id"], "status": "SKIPPED",
         "elapsed_ms": None, "reason": d.get("reason")}
        for d in declined
    ] + [
        {"agent_id": a.get("agent_id"), "status": a.get("status"),
         "elapsed_ms": a.get("elapsed_ms"), "reason": a.get("reason")}
        for a in (step.get("attempts") or [])]

    result = step.get("result") or {}
    if result.get("status") != "OK":
        section.update({
            "status": "UNAVAILABLE",
            "reason": (step.get("unanswered_reason")
                       or "no options specialist could answer"),
            "candidates": [], "honesty": [],
            "headline": "No option structures could be built for this symbol.",
        })
        return section

    basis = result.get("price_basis")
    data = result.get("data") or {}
    answered_by = step.get("answered_by")

    if basis == "TRADED_PRICE":
        candidates = _from_chain(data, cls["symbol"])
        honesty = list(data.get("honesty") or [])
        section["spot"] = data.get("spot")
        section["volatility_basis"] = "IMPLIED_BY_CHAIN"
    else:
        candidates = _from_model(data, cls["symbol"])
        honesty = list(data.get("honesty") or [])
        section["spot"] = data.get("spot") or spot
        section["volatility_annualized_pct"] = data.get("volatility_annualized_pct")
        section["volatility_basis"] = data.get("volatility_basis")
        section["annualization_days"] = data.get("annualization_days")
        if data.get("view") == "NONE_GIVEN":
            section["view"] = "NONE_GIVEN"

    n = len(candidates)
    vw = (view or "").lower()
    if section.get("view") == "NONE_GIVEN":
        section["headline"] = (
            f"{n} range structure{'s' if n != 1 else ''} — the desk holds no "
            f"directional view, so none of these express one.")
    else:
        section["headline"] = (
            f"{n} way{'s' if n != 1 else ''} to express the desk's "
            f"{vw or 'current'} view in options, expiring "
            f"{expiry_date(days).strftime('%-d %b %Y')}.")

    section.update({
        "status": "OK",
        "basis": basis,
        "answered_by": answered_by,
        "fell_back": bool(step.get("fell_back")),
        "is_model_priced": basis != "TRADED_PRICE",
        "candidates": candidates,
        "n_candidates": n,
        "honesty": honesty,
        "direction_source": {
            "view": section.get("view"),
            "source": "THIS_ENGINE" if view else "NONE",
            "statement": (
                f"Built for the {vw} view this engine's own evidence produced. "
                "The direction was handed down, not derived again here."
                if view else
                "The desk produced no directional view, so nothing directional "
                "was requested."),
        },
        "source": {
            "service": answered_by,
            "relationship": (
                "A separate service reading live option chains. Strictly better "
                "information than a model, and preferred whenever it answers."
                if basis == "TRADED_PRICE" else
                "This desk's own pricing engine. It runs in-process, needs no "
                "credential and no other service, and covers underlyings no "
                "chain here reaches — at the cost of pricing from realized "
                "volatility rather than from quotes."),
        },
    })

    if basis != "TRADED_PRICE":
        from . import registry
        chain_agents = [a["id"] for a in registry.agents("option_structures",
                                                          cls["asset_class"])
                        if a.get("price_basis") == "TRADED_PRICE"]
        if chain_agents:
            why = "; ".join(t["reason"] for t in section["trace"]
                            if t["agent_id"] in chain_agents and t.get("reason"))
            section["upgrade"] = (
                "These are MODEL prices. A live option chain would be better, "
                "and one is registered for this asset class but did not answer"
                + (f": {why}" if why else ".")
                + " Nothing above depends on it — the equity decision and "
                  "these structures were both produced without it.")
        else:
            section["upgrade"] = (
                f"No live option chain is reachable for a "
                f"{cls['spec']['label']} from this system, so a model price is "
                f"the best available answer rather than a fallback from one.")
    return section
