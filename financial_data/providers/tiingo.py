"""
FIL provider — Tiingo daily bars (free API key: tiingo.com, generous free tier).

Key-gated: activates the moment TIINGO_API_KEY lands in stock_agent/.env;
absent, it raises NotConfiguredError so the gateway falls through to yfinance
with a warning naming the exact variable — never a silent downgrade.

Why it earns a registry slot above yfinance once configured: a documented API
with real SLAs and split/dividend-adjusted OHLC (adjOpen/adjHigh/adjLow/
adjClose/adjVolume), vs yfinance's unannounced schema drift. Same retroactive-
adjustment caveat as yfinance: availability is PIT-correct, value stability is
not (declared in registry caveats).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .. import cache
from ..keys import NotConfiguredError, get_key
from ..schemas import make_datum, make_source

BASE = ("https://api.tiingo.com/tiingo/daily/{sym}/prices"
        "?format=json{range}&token={token}")
KINDS = ("bars",)


class ProviderError(RuntimeError):
    pass


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          period: str = "1y", reliability: float = 0.85,
          **kwargs: Any) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ProviderError(f"tiingo does not serve kind {kind!r}")
    token = get_key("TIINGO_API_KEY", "tiingo")   # raises NotConfiguredError

    # period -> startDate when no explicit range given (Tiingo wants dates).
    rng = ""
    if start:
        rng += f"&startDate={start}"
    if end:
        rng += f"&endDate={end}"
    if not rng:
        years = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "6mo": 1, "3mo": 1}.get(period, 5)
        from datetime import date, timedelta
        rng = f"&startDate={(date.today() - timedelta(days=365 * years)).isoformat()}"

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []

    for sym in symbols:
        sym = sym.upper().strip()
        key = f"{sym}:{rng}"
        rows = cache.get("tiingo", "bars", key, max_age_sec=6 * 3600)
        if rows is None:
            url = BASE.format(sym=sym.lower(), range=rng, token=token)
            req = urllib.request.Request(url, headers={"User-Agent": "AIOS-StockAgent/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    rows = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise NotConfiguredError(
                        f"Tiingo rejected TIINGO_API_KEY (HTTP {e.code}) — check the key") from e
                unavailable.append({"symbol": sym, "reason": f"tiingo HTTP {e.code}"})
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                unavailable.append({"symbol": sym, "reason": f"tiingo unreachable: {e}"})
                continue
            cache.put("tiingo", "bars", key, rows)

        if not isinstance(rows, list) or not rows:
            unavailable.append({"symbol": sym, "reason": "no_bars_returned"})
            continue
        for r in rows:
            close = r.get("adjClose", r.get("close"))
            if close is None or not r.get("date"):
                continue
            data.append(make_datum(
                kind="bars", value=float(close), available_at=r["date"],
                source=make_source(provider="tiingo",
                                   document=f"{sym}:{str(r['date'])[:10]}",
                                   ref="daily.adjClose"),
                symbol=sym, concept="close", unit="USD",
                period_end=r["date"], confidence=reliability, status="actual",
                extra={"open": r.get("adjOpen"), "high": r.get("adjHigh"),
                       "low": r.get("adjLow"), "volume": r.get("adjVolume")},
            ))
    return {"data": data, "unavailable": unavailable, "warnings": []}
