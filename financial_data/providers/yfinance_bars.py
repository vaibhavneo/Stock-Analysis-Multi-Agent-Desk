"""
FIL provider — price bars via yfinance.

The incumbent price source, DEMOTED behind the gateway rather than deleted: the
7-agent pipeline still reads it directly through tools/market_data.py (a tracked
legacy exception, see FIL.md §Known gaps), but every NEW consumer goes through
the gateway so the eventual swap to a paid provider is a registry edit.

Honesty note that matters for backtests: a bar's availability is point-in-time
correct — the close for date D was knowable at D's close, and we stamp
available_at accordingly. But yfinance serves SPLIT/DIVIDEND-ADJUSTED prices,
and those adjustments are applied retroactively, so the value you read today for
an old bar is not the value a trader saw then. Availability guarantee: holds.
Value stability: does not. That caveat is declared in registry.json and the
gateway surfaces it as a warning rather than leaving the caller to discover it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .. import cache
from ..schemas import make_datum, make_source

KINDS = ("bars",)


class ProviderError(RuntimeError):
    pass


def _cache_key(sym: str, period: str, start: Optional[str], end: Optional[str]) -> str:
    return f"{sym}_{start or ''}_{end or ''}_{period}"


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          period: str = "1y", reliability: float = 0.6,
          **kwargs: Any) -> Dict[str, Any]:
    if kind not in KINDS:
        raise ProviderError(f"yfinance provider does not serve kind {kind!r}")
    try:
        import yfinance as yf
    except ImportError as e:
        raise ProviderError("yfinance not installed") from e

    # Was the only uncached provider in the gateway - every analysis re-hit
    # yfinance fresh, and every new SPY/QQQ/regime fetch this session adds
    # would have doubled/tripled that. Short TTL for near-live windows (an
    # open "1d"/"5d" request wants fresh data), long TTL for anything longer
    # (a 5y history fetched twice in the same afternoon is still the same
    # history), matching cboe.py's own 6h VIX cache TTL.
    max_age_sec = 900 if period in ("1d", "5d") else 21600

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for sym in symbols:
        cache_key = _cache_key(sym, period, start, end)
        bars = cache.get("yfinance", "bars", cache_key, max_age_sec=max_age_sec)

        if bars is None:
            try:
                t = yf.Ticker(sym)
                # auto_adjust=True EXPLICITLY: the pre-gateway fetch_price_history
                # used it, and the whole point of routing that path through here
                # is to preserve its behavior exactly. Leaving it to yfinance's
                # shifting default would silently change adjusted prices across a
                # version bump.
                hist = (t.history(start=start, end=end, auto_adjust=True) if (start or end)
                        else t.history(period=period, auto_adjust=True))
            except Exception as e:
                unavailable.append({"symbol": sym, "reason": f"fetch_failed: {e}"})
                continue
            if hist is None or hist.empty:
                unavailable.append({"symbol": sym, "reason": "no_bars_returned"})
                continue

            bars = []
            for ts, row in hist.iterrows():
                bar = {k: (float(row[k]) if k in row and row[k] == row[k] else None)
                       for k in ("Open", "High", "Low", "Close", "Volume")}
                if bar["Close"] is None:
                    continue
                bar["_ts"] = ts.isoformat()
                bars.append(bar)
            # Only cache a genuine, non-empty result - a fetch failure or an
            # empty response must cost a re-fetch next time, never get stuck.
            cache.put("yfinance", "bars", cache_key, bars)

        for bar in bars:
            ts = datetime.fromisoformat(bar["_ts"])
            # available_at = the bar's own timestamp: a daily bar is knowable at
            # that session's close, which is precisely when it may be acted on.
            data.append(make_datum(
                kind="bars",
                value=bar["Close"],
                available_at=ts,
                source=make_source(provider="yfinance", document=f"{sym}:{ts.date()}",
                                   ref="history.Close"),
                symbol=sym,
                concept="close",
                unit="USD",
                period_end=ts,
                confidence=reliability,
                status="actual",
                extra={"open": bar["Open"], "high": bar["High"],
                       "low": bar["Low"], "volume": bar["Volume"]},
            ))
    return {"data": data, "unavailable": unavailable, "warnings": warnings}
