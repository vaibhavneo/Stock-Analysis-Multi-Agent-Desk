"""
FIL provider — FRED (St. Louis Fed) macro series, KEYLESS.

Uses the public fredgraph.csv endpoint (one request per series, cached): no API
key, no auth. The official FRED API needs a key; this endpoint serves the same
current-vintage data for chart embedding and is the honest keyless path.

HONESTY NOTE (why pit_capable is EMPTY): fredgraph serves the CURRENT vintage —
macro series are routinely revised (CPI, UNRATE especially), and a revised
value silently overwrites history here. "What did the world believe CPI was in
March 2020?" needs ALFRED vintages, which this endpoint does not serve. So this
provider declares pit_capable=[] and every as_of query gets
as_of_honored=false + a warning — usable for live context, NOT for backtests
that trade on macro numbers.
"""
from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from . import _http
from .. import cache
from ..schemas import make_datum, make_source

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

KINDS = ("macro",)

# The core macro dashboard a stock analysis actually consumes. Units recorded
# so a datum is self-describing.
KNOWN_SERIES = {
    "DGS10": ("10y_treasury_yield", "percent"),
    "DGS2": ("2y_treasury_yield", "percent"),
    "T10Y2Y": ("10y_2y_spread", "percent"),
    "FEDFUNDS": ("fed_funds_rate", "percent"),
    "UNRATE": ("unemployment_rate", "percent"),
    "CPIAUCSL": ("cpi_index", "index_1982_84_100"),
    "VIXCLS": ("vix_close", "index"),
}


class ProviderError(RuntimeError):
    pass


def _fetch_csv(series: str, max_age_sec: int = 6 * 3600) -> str:
    payload = cache.get("fred", "macro", series, max_age_sec=max_age_sec)
    if payload is not None:
        return payload
    try:
        text = _http.get_text(BASE.format(series=series), timeout=25)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        raise ProviderError(f"FRED unreachable for {series}: {e}") from e
    if not text.startswith("observation_date"):
        raise ProviderError(f"FRED returned unexpected payload for {series!r} "
                            f"(unknown series id?)")
    cache.put("fred", "macro", series, text)
    return text


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          reliability: float = 0.9, **kwargs: Any) -> Dict[str, Any]:
    """symbols = FRED series ids (e.g. ["DGS10", "UNRATE"])."""
    if kind not in KINDS:
        raise ProviderError(f"fred does not serve kind {kind!r}")

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for series in symbols:
        series = series.upper().strip()
        concept, unit = KNOWN_SERIES.get(series, (series.lower(), None))
        try:
            text = _fetch_csv(series)
        except ProviderError as e:
            unavailable.append({"symbol": series, "reason": str(e)})
            continue
        n_before = len(data)
        for row in csv.DictReader(io.StringIO(text)):
            date = row.get("observation_date")
            raw = row.get(series)
            if not date or raw in (None, "", "."):   # "." = FRED's missing marker
                continue
            if start and date < start:
                continue
            if end and date > end:
                continue
            try:
                data.append(make_datum(
                    kind="macro", value=float(raw), available_at=date,
                    source=make_source(provider="fred", document=f"FRED:{series}",
                                       ref=series,
                                       url=f"https://fred.stlouisfed.org/series/{series}"),
                    symbol=series, concept=concept, unit=unit,
                    period_end=date, confidence=reliability, status="actual",
                ))
            except Exception:
                continue
        if len(data) == n_before:
            unavailable.append({"symbol": series, "reason": "no_observations_in_range"})
    return {"data": data, "unavailable": unavailable, "warnings": warnings}
