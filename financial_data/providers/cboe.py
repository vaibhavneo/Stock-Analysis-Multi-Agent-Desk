"""
FIL provider — CBOE VIX daily history, KEYLESS.

CBOE publishes the full VIX OHLC history as a plain CSV (1990 -> today, updated
daily). The VIX is the market's own 30-day implied-volatility gauge — the
regime input a risk pillar wants that realized volatility lags on.

PIT note: a VIX bar dated D was published at D's close and the history file is
not restated (index methodology changes are rare and announced), so bar-date
availability is honest — pit_capable for macro, with day-granularity boundary
semantics (visible_at's inclusive rule).
"""
from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import _http
from .. import cache
from ..schemas import make_datum, make_source

URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
KINDS = ("macro",)


class ProviderError(RuntimeError):
    pass


def _fetch_history(max_age_sec: int = 6 * 3600) -> str:
    payload = cache.get("cboe", "macro", "VIX_History", max_age_sec=max_age_sec)
    if payload is not None:
        return payload
    try:
        text = _http.get_text(URL, timeout=30)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise ProviderError(f"CBOE unreachable: {e}") from e
    if not text.upper().startswith("DATE"):
        raise ProviderError("CBOE returned unexpected payload")
    cache.put("cboe", "macro", "VIX_History", text)
    return text


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          reliability: float = 0.95, **kwargs: Any) -> Dict[str, Any]:
    """Serves the symbol VIX (aliases ^VIX / VIXCLS accepted). Other symbols are
    honestly unavailable — CBOE's free file covers the VIX only."""
    if kind not in KINDS:
        raise ProviderError(f"cboe does not serve kind {kind!r}")

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []

    wanted = [s.upper().strip() for s in symbols]
    vix_asked = [s for s in wanted if s in ("VIX", "^VIX", "VIXCLS")]
    for s in wanted:
        if s not in ("VIX", "^VIX", "VIXCLS"):
            unavailable.append({"symbol": s, "reason": "cboe_free_file_serves_vix_only"})
    if not vix_asked:
        return {"data": [], "unavailable": unavailable, "warnings": []}

    try:
        text = _fetch_history()
    except ProviderError as e:
        unavailable.append({"symbol": "VIX", "reason": str(e)})
        return {"data": [], "unavailable": unavailable, "warnings": []}

    for row in csv.DictReader(io.StringIO(text)):
        try:
            date = datetime.strptime(row["DATE"], "%m/%d/%Y").date().isoformat()
        except (KeyError, ValueError):
            continue
        if start and date < start:
            continue
        if end and date > end:
            continue
        try:
            data.append(make_datum(
                kind="macro", value=float(row["CLOSE"]), available_at=date,
                source=make_source(provider="cboe", document=f"VIX_History:{date}",
                                   ref="VIX.CLOSE", url=URL),
                symbol="VIX", concept="vix_close", unit="index",
                period_end=date, confidence=reliability, status="actual",
                extra={"open": float(row["OPEN"]), "high": float(row["HIGH"]),
                       "low": float(row["LOW"])},
            ))
        except Exception:
            continue
    return {"data": data, "unavailable": unavailable, "warnings": []}
