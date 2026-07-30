"""
FIL provider — Finnhub earnings calendar (free API key: finnhub.io).

Key-gated (FINNHUB_API_KEY): serves the `events` kind — upcoming and recent
earnings dates with consensus estimates. The single highest-value event a
holder of a stock needs to know about: vol regimes, options premia, and the
coach's "earnings upcoming in book" trigger all key off it.

Datum semantics: a FUTURE earnings date is a forecastable calendar fact —
status="estimate", available_at = retrieval time (the calendar entry is what
was knowable now, not the future event itself). A PAST event with epsActual is
status="actual". Nothing here is PIT-reconstructable (calendars shift), so
pit_capable=[] — honest.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from .. import cache
from ..keys import NotConfiguredError, get_key
from ..schemas import make_datum, make_source

BASE = ("https://finnhub.io/api/v1/calendar/earnings"
        "?from={frm}&to={to}&symbol={sym}&token={token}")
KINDS = ("events",)


class ProviderError(RuntimeError):
    pass


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          reliability: float = 0.8, **kwargs: Any) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ProviderError(f"finnhub does not serve kind {kind!r}")
    token = get_key("FINNHUB_API_KEY", "finnhub")

    frm = start or (date.today() - timedelta(days=180)).isoformat()
    to = end or (date.today() + timedelta(days=120)).isoformat()
    now = datetime.now()

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []

    for sym in symbols:
        sym = sym.upper().strip()
        key = f"{sym}:{frm}:{to}"
        payload = cache.get("finnhub", "events", key, max_age_sec=6 * 3600)
        if payload is None:
            url = BASE.format(frm=frm, to=to, sym=sym, token=token)
            req = urllib.request.Request(url, headers={"User-Agent": "AIOS-StockAgent/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    payload = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise NotConfiguredError(
                        f"Finnhub rejected FINNHUB_API_KEY (HTTP {e.code}) — check the key") from e
                unavailable.append({"symbol": sym, "reason": f"finnhub HTTP {e.code}"})
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                unavailable.append({"symbol": sym, "reason": f"finnhub unreachable: {e}"})
                continue
            cache.put("finnhub", "events", key, payload)

        events = (payload or {}).get("earningsCalendar") or []
        if not events:
            unavailable.append({"symbol": sym, "reason": "no_earnings_events_in_window"})
            continue
        for e in events:
            ev_date = e.get("date")
            if not ev_date:
                continue
            actual = e.get("epsActual")
            est = e.get("epsEstimate")
            value = actual if actual is not None else (est if est is not None else 0.0)
            data.append(make_datum(
                kind="events", value=float(value),
                # A calendar entry is knowable at retrieval, not at the event.
                available_at=now if ev_date > now.date().isoformat() else ev_date,
                source=make_source(provider="finnhub", document=f"earnings:{sym}:{ev_date}",
                                   ref="calendar/earnings"),
                symbol=sym, concept="earnings_eps", unit="USD_per_share",
                period_end=ev_date, confidence=reliability,
                status="actual" if actual is not None else "estimate",
                extra={"event_date": ev_date, "eps_estimate": est, "eps_actual": actual,
                       "revenue_estimate": e.get("revenueEstimate"),
                       "revenue_actual": e.get("revenueActual"),
                       "hour": e.get("hour"), "quarter": e.get("quarter"),
                       "year": e.get("year")},
            ))
    return {"data": data, "unavailable": unavailable, "warnings": []}
