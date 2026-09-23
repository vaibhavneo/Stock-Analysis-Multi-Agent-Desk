"""How old is the number you are looking at, and has the price left it behind?

THE BUG THIS EXISTS FOR
-----------------------
The brief showed "PRICE NOW $227.38" while the last trade was $228.87. The
daily bar series had not yet published today's session — 390 one-minute bars
had already printed and closed — so the headline price, and every
distance-from-here percentage derived from it, were a day behind and labelled
as current. A 0.65% error is small; labelling stale data "now" is not, because
nothing on the page let the reader discover it.

WHAT THIS DOES AND DELIBERATELY DOES NOT DO
-------------------------------------------
It does NOT re-run the analysis on a live tick. Indicators, pillars and levels
stay computed on the settled bar series, because that is the series the
backtest measured and swapping in a mid-session price would mean the displayed
levels were derived from a different number than the one shown next to them —
trading one inconsistency for a subtler one.

What it does instead is the part that is actually decision-relevant:

  - report the live price with its age and source, next to the price the
    analysis was computed on, and the drift between them
  - say whether that drift has CROSSED anything — an entry trigger, an
    invalidation, a support — because a level crossed since the close is the
    one piece of real-time information that changes what to do

A price that has quietly crossed the invalidation since the close is the
single most useful thing a freshness layer can tell anyone.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:                                   # pragma: no cover
    _ET = None

OPEN = "OPEN"
CLOSED = "CLOSED"
PRE = "PRE_MARKET"
POST = "AFTER_HOURS"
ALWAYS_OPEN = "ALWAYS_OPEN"

# US equity regular session, Eastern.
_OPEN_T = dt.time(9, 30)
_CLOSE_T = dt.time(16, 0)
_PRE_T = dt.time(4, 0)
_POST_T = dt.time(20, 0)

# Beyond this a quote is not "live" in any useful sense.
STALE_QUOTE_SEC = 30 * 60


def _easter(year: int) -> dt.date:
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


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1) if month < 12 \
        else dt.date(year, 12, 31)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    """A fixed-date holiday on a weekend is observed on the adjacent weekday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def market_holidays(year: int) -> List[dt.date]:
    """NYSE full-day closures. Enumerated rather than hard-coded per year so
    this does not silently expire — a holiday calendar that runs out turns
    into "the market is open" on Christmas."""
    return [
        _observed(dt.date(year, 1, 1)),                       # New Year's Day
        _nth_weekday(year, 1, 0, 3),                          # MLK
        _nth_weekday(year, 2, 0, 3),                          # Presidents
        _easter(year) - dt.timedelta(days=2),                 # Good Friday
        _last_weekday(year, 5, 0),                            # Memorial
        _observed(dt.date(year, 6, 19)),                      # Juneteenth
        _observed(dt.date(year, 7, 4)),                       # Independence
        _nth_weekday(year, 9, 0, 1),                          # Labor
        _nth_weekday(year, 11, 3, 4),                         # Thanksgiving
        _observed(dt.date(year, 12, 25)),                     # Christmas
    ]


def market_session(asset_class: str = "EQUITY",
                   now: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Which session the asset is in right now, in its own venue's terms."""
    from .asset_class import spec_for
    spec = spec_for(asset_class)

    if spec.sessions == "24/7":
        return {"state": ALWAYS_OPEN, "venue": "continuous",
                "statement": "Trades continuously; there is no close to be behind."}

    if _ET is None:                                  # pragma: no cover
        return {"state": "UNKNOWN", "venue": "unknown",
                "statement": "Local timezone data is unavailable, so the "
                             "session could not be determined."}

    now_et = (now or dt.datetime.now(dt.timezone.utc)).astimezone(_ET)
    today = now_et.date()
    t = now_et.time()

    if spec.sessions == "24/5":
        weekend = (today.weekday() == 5
                   or (today.weekday() == 6 and t < dt.time(17, 0))
                   or (today.weekday() == 4 and t >= dt.time(17, 0)))
        return {"state": CLOSED if weekend else ALWAYS_OPEN, "venue": "24/5",
                "statement": ("Currency markets are shut for the weekend."
                              if weekend else
                              "Currency markets trade around the clock on "
                              "weekdays.")}

    if today.weekday() >= 5:
        return {"state": CLOSED, "venue": "NYSE",
                "statement": "The US market is closed for the weekend."}
    if today in market_holidays(today.year):
        return {"state": CLOSED, "venue": "NYSE",
                "statement": "The US market is closed for a holiday."}

    if _OPEN_T <= t < _CLOSE_T:
        state, msg = OPEN, "The US market is open."
    elif _PRE_T <= t < _OPEN_T:
        state, msg = PRE, "Pre-market. Quotes are thin and move on little size."
    elif _CLOSE_T <= t < _POST_T:
        state, msg = POST, ("After hours. Quotes are thin and move on little "
                            "size.")
    else:
        state, msg = CLOSED, "The US market is closed for the day."
    return {"state": state, "venue": "NYSE", "statement": msg,
            "as_of_et": now_et.isoformat(timespec="seconds")}


def live_quote(symbol: str) -> Dict[str, Any]:
    """The most recent trade this system can see. Never raises.

    Returns `available: False` with a reason rather than a price of None that
    a caller might format as a number.
    """
    out: Dict[str, Any] = {"symbol": (symbol or "").upper(), "available": False,
                           "price": None, "source": None, "as_of": None,
                           "age_sec": None, "reason": ""}
    try:
        import yfinance as yf
        t = yf.Ticker(out["symbol"])
        price = None
        try:
            fi = t.fast_info
            price = fi.get("lastPrice") or fi.get("last_price")
        except Exception:
            price = None
        if price:
            out.update({"available": True, "price": float(price),
                        "source": "last trade",
                        "as_of": dt.datetime.now().isoformat(timespec="seconds"),
                        "age_sec": 0})
            return out
        out["reason"] = "the provider returned no last trade for this symbol"
    except Exception as e:
        out["reason"] = f"the quote could not be fetched ({type(e).__name__})"
    return out


def _crossings(basis: float, live: float,
               levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Levels the price has moved THROUGH since the analysis was computed.

    This is the whole point of the layer. A level crossed since the close is
    the one piece of real-time information that changes what to do, and it is
    invisible if the page only ever shows the settled close.
    """
    out = []
    lo, hi = (basis, live) if live >= basis else (live, basis)
    for lv in levels:
        p = lv.get("price")
        if p is None:
            continue
        p = float(p)
        if lo < p <= hi or hi >= p > lo:
            if (basis < p <= live) or (live <= p < basis):
                out.append({
                    "label": lv.get("label") or lv.get("name") or "level",
                    "price": p,
                    "direction": "upward through" if live > basis else "downward through",
                })
    return out


def build_freshness(symbol: str,
                    asset_class: str,
                    computed_on_price: Optional[float],
                    computed_on_date: Optional[str] = None,
                    levels: Optional[List[Dict[str, Any]]] = None,
                    quote: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The freshness block for a decision. Never raises."""
    session = market_session(asset_class)
    q = quote if quote is not None else live_quote(symbol)

    out: Dict[str, Any] = {
        "symbol": (symbol or "").upper(),
        "session": session,
        "quote": q,
        "computed_on": {"price": computed_on_price, "date": computed_on_date,
                        "basis": "the settled bar series the analysis and its "
                                 "backtest both use"},
        "drift_pct": None,
        "crossed": [],
        "statement": "",
        "stale": False,
    }

    if not q.get("available") or computed_on_price in (None, 0):
        out["statement"] = (
            f"No live quote was available ({q.get('reason') or 'no reason given'}), "
            f"so every price here is the one the analysis was computed on.")
        return out

    live = float(q["price"])
    basis = float(computed_on_price)
    drift = (live / basis - 1.0) * 100.0
    out["drift_pct"] = round(drift, 3)
    out["crossed"] = _crossings(basis, live, levels or [])

    moved = "unchanged" if abs(drift) < 0.05 else \
        f"{'up' if drift > 0 else 'down'} {abs(drift):.2f}%"
    parts = [f"Computed on {basis:,.2f}"
             + (f" ({computed_on_date})" if computed_on_date else "")
             + f"; last trade {live:,.2f}, {moved}."]
    if session.get("state") in (OPEN, PRE, POST):
        parts.append(session["statement"])
    if out["crossed"]:
        names = "; ".join(
            f"{c['label']} at {c['price']:,.2f}" for c in out["crossed"])
        parts.append(f"Since then the price has moved {out['crossed'][0]['direction']} "
                     f"{names} — that level is behind the analysis above.")
        out["stale"] = True
    elif abs(drift) >= 2.0:
        parts.append("That is a large enough move that the levels above are "
                     "worth re-running.")
        out["stale"] = True
    out["statement"] = " ".join(parts)
    return out
