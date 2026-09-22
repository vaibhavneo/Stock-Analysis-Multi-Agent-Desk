"""Specialist results -> something a person would say back.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
Every number in a reply comes from a specialist that ran. Nothing here
computes, rounds up, softens or fills a gap. When a specialist declined, the
reply says it declined and why — a conversation is exactly where a missing
result is easiest to paper over, because prose flows whether or not the number
behind it exists.

The tone target is a colleague at the next desk: direct, numerate, and willing
to say "I don't know" or "this hasn't worked".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

ACTION_WORD = {
    "BUY": "buy", "ACCUMULATE": "add to", "HOLD": "hold",
    "REDUCE": "trim", "SELL": "sell out of",
}


def _upper_first(text: str) -> str:
    """Capitalise the first letter only.

    str.capitalize() lower-cases everything after it, which turned
    "Most of BTC-USD's movement is its own" into "btc-usd" — a ticker
    destroyed by a formatting call."""
    text = (text or "").strip()
    return text[:1].upper() + text[1:] if text else text


def _pct(v, nd=1):
    return "—" if v is None else f"{float(v):.{nd}f}%"


def _money(v):
    if v is None:
        return "—"
    v = float(v)
    return ("-$" if v < 0 else "$") + f"{abs(v):,.2f}"


# ── One composer per capability ───────────────────────────────────────────

def _say_research(d: Dict[str, Any], symbol: str) -> List[str]:
    core = d.get("core") or {}
    if d.get("depth") == "FULL":
        dec = d.get("decision") or {}
        st = (dec.get("decision_state") or {})
        lines = [f"**{symbol} — {st.get('headline_state', 'no state')}**"]
        if st.get("headline_reason"):
            lines.append(st["headline_reason"])
        return lines
    action = core.get("action")
    vol = core.get("volatility") or {}
    lines = [
        f"**{symbol} — {action}** (composite {core.get('composite')}). "
        f"On the numbers I'd {ACTION_WORD.get(action, 'wait on')} it."
    ]
    if vol.get("historical_volatility_20d") is not None:
        lines.append(
            f"Volatility is {_pct(vol['historical_volatility_20d'])} annualized, "
            f"which is {str(vol.get('regime', '')).lower()} for "
            f"{core.get('classification', {}).get('label', 'this asset')} — "
            f"measured on its own {vol.get('annualization_days')}-day calendar.")
    if core.get("scoring_note"):
        lines.append(core["scoring_note"])
    return lines


def _say_options(d: Dict[str, Any], symbol: str) -> List[str]:
    if d.get("status") != "OK":
        return [f"No option structures for {symbol}: {d.get('reason', 'unavailable')}."]
    lead = None
    for name in ("bull_call_spread", "bear_put_spread", "iron_condor",
                 "long_call", "long_put", "iron_butterfly"):
        lead = next((c for c in d.get("candidates", []) if c.get("structure") == name), None)
        if lead:
            break
    lead = lead or (d.get("candidates") or [None])[0]
    basis = ("model-priced by this desk, not from live quotes"
             if d.get("is_model_priced") else "priced from live chain quotes")
    view = d.get("view")
    # Say which direction these express. A structure set without its view is
    # unreadable — an iron condor and a call spread are opposite bets, and
    # which one is right depends entirely on a read stated somewhere else.
    if view and view != "NONE_GIVEN":
        stance = f"expressing the desk's {str(view).lower()} read"
    else:
        stance = "the desk has no directional read, so these are range structures"
    lines = [f"**Options on {symbol}** — {stance} ({basis}):"]
    if lead:
        lines.append(f"{lead.get('label')} — {lead.get('plain')}")
    others = [c for c in d.get("candidates", []) if c is not lead]
    if others:
        # A credit structure's cost is negative in the data because cash moves
        # the other way. Printing "-$1,959.27" next to "sell an iron butterfly"
        # reads as a loss rather than as premium received.
        def _side(c):
            v = c.get("net_cost")
            if v is None:
                return "—"
            return (f"collect {_money(abs(v))}" if c.get("cash_direction") == "CREDIT"
                    else f"pay {_money(abs(v))}")
        lines.append("Also on the table: "
                     + ", ".join(f"{c.get('label')} ({_side(c)})" for c in others)
                     + ".")
    if d.get("upgrade"):
        lines.append(d["upgrade"])
    return lines


def _say_backtest(d: Dict[str, Any], symbol: str) -> List[str]:
    rows = d.get("rows") or []
    bh = d.get("buy_hold") or {}
    if not rows:
        return [f"No backtest result came back for {symbol}."]
    best = max(rows, key=lambda r: r.get("sharpe") or -99)
    bh_sharpe = bh.get("sharpe")
    bh_ret = bh.get("annualized_return_pct")
    lines = [
        f"**Backtest — {symbol}.** Best of {len(rows)} strategies net of costs "
        f"is *{best.get('strategy', best.get('name', 'unnamed'))}*: "
        f"{_pct(best.get('annualized_return_pct'))} a year, Sharpe "
        f"{best.get('sharpe')}. Buy-and-hold did {_pct(bh_ret)}, Sharpe "
        f"{bh_sharpe}."
    ]
    if bh_sharpe is not None and best.get("sharpe") is not None:
        if best["sharpe"] <= bh_sharpe:
            lines.append(
                "So on this name nothing here beat simply holding it. That is "
                "the honest read, and it is the one I'd act on.")
        else:
            lines.append(
                "It beat holding on this name — but one symbol is one sample, "
                "and a strategy picked as the best of many has been selected "
                "for luck as well as skill.")
    return lines


def _say_record(d: Dict[str, Any], symbol: str) -> List[str]:
    lr = d.get("live_record") or {}
    all_rows = lr.get("rows") or []
    rows = [r for r in all_rows if r.get("readable")]
    if not rows:
        # "Nothing has matured" and "some has matured but not enough to quote"
        # are different facts, and the second is the more useful one: it tells
        # the user the record is accumulating and what it is waiting for.
        # Production reported the first while holding 20 matured predictions.
        from decision.track_record import LIVE_SAMPLE_READABLE
        total = lr.get("total_matured") or 0
        best = max((r.get("n") or 0) for r in all_rows) if all_rows else 0
        if total:
            return [
                f"Not yet — and I would rather say so than round up. "
                f"{total:,} prediction{'s' if total != 1 else ''} of mine "
                f"{'have' if total != 1 else 'has'} matured, but the largest "
                f"sample at any single horizon is {best:,}, and I do not quote "
                f"a win rate below {LIVE_SAMPLE_READABLE}. Under that, the "
                f"number moves more with luck than with skill.",
                "Ask me again as it fills in. Nothing here is a backtest "
                "dressed up as a record."]
        return ["No frozen prediction has matured yet, so I have no measured "
                "record to show you. I'd rather say that than quote a backtest "
                "as if it were a track record."]
    lines = ["**My measured record** — predictions frozen before the outcome "
             "was known, then graded:"]
    for r in rows:
        lines.append(
            f"- {r['horizon_days']}-day: {r['win_rate']*100:.1f}% win rate over "
            f"{r['n']:,} calls, average excess return "
            f"{_pct(r.get('avg_excess_return_pct'), 2)}.")
    near = [r for r in rows if r["horizon_days"] <= 5]
    if near and all(abs(r["win_rate"] - 0.5) < 0.03 for r in near):
        lines.append(
            "At the short horizons that is a coin flip, and the average excess "
            "return is negative. I would not trade the 1- and 5-day calls.")
    if lr.get("caveat"):
        lines.append(lr["caveat"])
    return lines


def _say_regime(d: Dict[str, Any], symbol: str) -> List[str]:
    lines = [
        f"**The market right now:** trend {str(d.get('trend', '')).lower()}, "
        f"{str(d.get('risk_stance', '')).replace('_', '-').lower()}, volatility "
        f"{str(d.get('volatility_regime', '')).lower()} with VIX at "
        f"{d.get('vix_level')}."
    ]
    if d.get("growth_vs_value"):
        lines.append(_upper_first(str(d["growth_vs_value"]).replace("_", " ").lower()) + ".")
    lines.append(
        "That is a description of conditions, not a forecast. I have no model "
        "that predicts crashes and would not offer one if I did.")
    return lines


def _say_events(d: Dict[str, Any], symbol: str) -> List[str]:
    nxt = d.get("next_event") or d.get("next_scheduled_event")
    if not nxt:
        return [f"Nothing scheduled that I can see for {symbol}. "
                + (d.get("waiting_for") or "")]
    lines = [f"**{symbol} — next event:** {nxt.get('event')} on "
             f"{nxt.get('date')}, {nxt.get('days_away')} days out."]
    # Compared on DATE, not identity: `next_event` is often a different dict
    # object holding the same event, so an identity filter left the headline
    # event repeated in the "after that" list.
    rest = [e for e in (d.get("upcoming") or [])
            if e.get("date") != nxt.get("date")
            or e.get("event") != nxt.get("event")][:3]
    if rest:
        lines.append("After that: "
                     + "; ".join(f"{e.get('event')} ({e.get('date')})" for e in rest) + ".")
    if d.get("disclaimer"):
        lines.append(d["disclaimer"])
    return lines


def _say_portfolio(d: Dict[str, Any], symbol: str) -> List[str]:
    summ = d.get("summary") or {}
    stance = d.get("stance") or summ.get("stance") or {}
    lines = ["**Your portfolio.**"]
    if isinstance(stance, dict) and stance.get("statement"):
        lines.append(stance["statement"])
    elif isinstance(stance, str):
        lines.append(stance)
    acts = [h for h in (d.get("holdings") or [])
            if (h.get("action") or "").upper() not in ("", "HOLD")]
    if acts:
        lines.append("Worth acting on: "
                     + "; ".join(f"{h.get('ticker')} — {h.get('action')}"
                                 for h in acts[:6]) + ".")
    else:
        lines.append("Nothing in it is asking to be acted on today.")
    return lines


def _say_benchmark(d: Dict[str, Any], symbol: str) -> List[str]:
    lines = [f"**{symbol} vs {d.get('benchmark_label', d.get('benchmark'))}:** "
             f"beta {d.get('beta')}, r² {d.get('r_squared')} over "
             f"{d.get('overlapping_returns')} shared trading days."]
    if d.get("verdict"):
        lines.append(_upper_first(d["verdict"]) + ".")
    return lines


COMPOSERS = {
    "equity_research": _say_research,
    "option_structures": _say_options,
    "strategy_backtest": _say_backtest,
    "forward_record": _say_record,
    "market_regime": _say_regime,
    "event_calendar": _say_events,
    "portfolio_review": _say_portfolio,
    "benchmark_relation": _say_benchmark,
}

ROSTER_BLURB = [
    "Here is what I can actually do, and each one is a specialist that runs "
    "and reports rather than a phrase I can improvise:",
    "- **Read a name** — any stock, ETF, index, currency, commodity or coin. "
    "Just say the symbol.",
    "- **Options** — calls, puts and spreads, with what each costs and what it "
    "pays. Works on things no option chain here covers, priced from a model "
    "and labelled as such.",
    "- **Backtest** — has following this actually beaten simply holding it?",
    "- **My record** — what I got right, graded after the fact, not a backtest.",
    "- **The market** — regime, volatility, risk stance.",
    "- **Catalysts** — what is scheduled that could move a name.",
    "- **Your portfolio** — weights, concentration, what to trim.",
    "- **What is it really tracking** — beta and correlation to the benchmark "
    "that actually drives it.",
    "I cannot place orders, and I will not pick a stock for you out of "
    "nowhere — name one and I'll do the work.",
]


def compose(parsed: Dict[str, Any], results: Dict[str, Any],
            subject: Optional[str]) -> Dict[str, Any]:
    """Build the reply. Prose only ever describes results that exist."""
    kind = parsed.get("kind")

    if kind == "SMALL_TALK":
        return {"headline": "Hello.",
                "blocks": [{"capability": None, "lines": [
                    "Ask me about a stock, the market, your portfolio, or how "
                    "this desk has actually done. A bare ticker works."]}],
                "kind": kind}

    if kind == "CAPABILITY_QUESTION":
        return {"headline": "What this desk can do",
                "blocks": [{"capability": None, "lines": ROSTER_BLURB}],
                "kind": kind}

    if kind in ("NEEDS_SUBJECT", "UNROUTABLE"):
        return {"headline": "I need one more thing",
                "blocks": [{"capability": None,
                            "lines": [parsed.get("clarification") or
                                      "I could not tell what you were asking."]}],
                "kind": kind}

    blocks: List[Dict[str, Any]] = []
    declined: List[str] = []
    for cap in parsed.get("capabilities") or []:
        entry = results.get(cap) or {}
        res = entry.get("result") or {}
        if res.get("status") == "OK":
            fn = COMPOSERS.get(cap)
            data = res.get("data") or {}
            sym = entry.get("symbol") or subject or ""
            blocks.append({
                "capability": cap,
                "answered_by": entry.get("answered_by"),
                "lines": fn(data, sym) if fn else [f"{cap}: answered."],
            })
        else:
            declined.append(
                f"- *{cap.replace('_', ' ')}* — "
                + (entry.get("unanswered_reason") or "no specialist answered."))

    if declined:
        blocks.append({"capability": None, "declined": True,
                       "lines": ["Could not answer all of that:"] + declined})

    if not blocks:
        return {"headline": "Nothing came back",
                "blocks": [{"capability": None, "lines": [
                    "Every specialist I asked declined. That is a failure on "
                    "my side, not an answer about the market."]}],
                "kind": "EMPTY"}

    first = blocks[0]["lines"][0] if blocks[0].get("lines") else ""
    return {"headline": first.replace("**", "").split(".")[0][:120],
            "blocks": blocks, "kind": kind}
