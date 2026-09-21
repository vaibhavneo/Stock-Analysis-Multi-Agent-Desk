"""The risk-free rate, fetched once and cached, with an honest fallback.

An option price is not very sensitive to `r` at short tenors — but it is not
insensitive either, and more importantly a hard-coded rate that drifts years
out of date is the kind of stale constant nobody re-reads. So: fetch the
10-year from FRED through the existing gateway, cache it for the process, and
if that fails say plainly that a default was used and what it was.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict

FALLBACK_RATE = 0.04
_TTL_SEC = 3600.0
_lock = threading.Lock()
_cache: Dict[str, Any] = {"at": 0.0, "value": None}


def risk_free_rate() -> Dict[str, Any]:
    """{rate, source, as_of, is_fallback}. Never raises."""
    with _lock:
        if _cache["value"] is not None and (time.time() - _cache["at"]) < _TTL_SEC:
            return dict(_cache["value"])
    out = {"rate": FALLBACK_RATE, "source": "default",
           "as_of": None, "is_fallback": True,
           "note": (f"FRED was not reachable; priced at a default "
                    f"{FALLBACK_RATE:.1%} risk-free rate")}
    try:
        from datetime import date, timedelta
        from financial_data import get
        start = (date.today() - timedelta(days=45)).isoformat()
        r = get("macro", ["DGS10"], start=start)
        rows = [d for d in r.get("data", []) if d.get("value") is not None]
        if rows:
            latest = max(rows, key=lambda d: d["available_at"])
            out = {"rate": float(latest["value"]) / 100.0,
                   "source": r.get("provider", "fred"),
                   "as_of": latest["available_at"],
                   "is_fallback": False,
                   "note": "10-year Treasury yield, current vintage"}
    except Exception as e:
        out["note"] = (f"FRED lookup failed ({type(e).__name__}); priced at a "
                       f"default {FALLBACK_RATE:.1%} risk-free rate")
    with _lock:
        _cache.update({"at": time.time(), "value": out})
    return dict(out)
