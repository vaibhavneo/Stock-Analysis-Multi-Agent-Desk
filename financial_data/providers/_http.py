"""
Shared HTTP fetch for FIL providers.

Exists because of an empirically-found landmine (2026-07-18): FRED's CDN
silently HANGS (no 403, no error — a clean 18s timeout) on requests whose
User-Agent doesn't look like a browser and that lack Accept headers, while the
same URL answers in 0.1s with browser-ish headers. A per-provider headers
snowflake would reintroduce that class of bug one provider at a time, so the
header set lives here once. Handles gzip transparently.

(SEC EDGAR is the deliberate exception — it REQUIRES a contact-email UA and has
its own fetcher; see providers/edgar.py.)
"""
from __future__ import annotations

import gzip
import urllib.error
import urllib.request

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AIOS-StockAgent/1.0"),
    "Accept": "text/csv,application/json,text/plain,*/*",
    "Accept-Encoding": "gzip, deflate",
}


def get_text(url: str, timeout: int = 25) -> str:
    """GET a URL, return decoded text. Raises urllib errors to the caller —
    each provider owns its own honest failure reporting."""
    req = urllib.request.Request(url, headers=dict(HEADERS))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode()
