"""
Sharadar production UniverseProvider — integration-ready, survivorship-safe.

Sharadar (Nasdaq Data Link / Quandl) is the reference licensed dataset for
point-in-time US equity research because it ships the three things free data
never does:

  - SHARADAR/TICKERS — one row per (permaticker, ticker). `permaticker` is a
    PERMANENT integer identity that survives ticker changes; `firstpricedate`/
    `lastpricedate` bound tradability; `isdelisted` marks removal. This is what
    makes a historical universe survivorship-safe: delisted names are present
    with their real membership windows, not silently dropped.
  - SHARADAR/SEP — daily equity prices with `closeadj` (split+dividend adjusted),
    i.e. corporate-action-adjusted closes for genuine total-return momentum.
  - SHARADAR/SF1 — fundamentals whose `datekey` is the FILED/available date (the
    anti-look-ahead field): keep rows with datekey <= as_of and a filing filed
    after the ranking date can never leak into an earlier rank.
  - SHARADAR/ACTIONS — corporate actions (delisted, bankruptcy, acquisition…),
    used to assign the conservative delisting return (DELISTING_POLICY.md).

**Integration-ready, not a stub.** The parse layer (`parse_tickers`,
`membership_as_of`, `parse_sf1_records`, `delisting_return_from_action`) is pure,
fully unit-tested offline against Sharadar-shaped rows, and is what actually
turns raw dataset rows into the engine's member/feature/identity shapes. The
only thing gated on the license is the *network fetch* (`_client`/`_fetch`):
without `NASDAQ_DATA_LINK_API_KEY` every entry point raises `UniverseIncomplete`
(UNIVERSE_INCOMPLETE) and NOTHING is fabricated — no current-constituent
fallback, no synthetic substitution. Configure the key and the same code path
produces a real survivorship-safe universe. See PRODUCTION_DATA_ACTIVATION_REPORT.md.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from xsection.universe import (UNIVERSE_INCOMPLETE, SecurityMaster,
                               UniverseIncomplete, UniverseProvider)

KEY_VAR = "NASDAQ_DATA_LINK_API_KEY"
_MIN_INTERVAL_SEC = 0.30            # be a good citizen; Sharadar allows more but we throttle
_last_call = [0.0]


def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < _MIN_INTERVAL_SEC:
        time.sleep(_MIN_INTERVAL_SEC - dt)
    _last_call[0] = time.time()


# ── Pure parse layer (offline-testable; the real adapter logic) ─────────────

def parse_tickers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SHARADAR/TICKERS rows -> per-permaticker security records with ticker
    history. permaticker is the permanent identity; multiple rows for one
    permaticker (a ticker rename) collapse into one security with several ticker
    windows. A ticker string reused by a DIFFERENT permaticker stays a different
    security — identity is never keyed on the ticker."""
    by_perma: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if str(r.get("table", "SEP")) not in ("SEP", "SFP"):   # equities/funds only
            continue
        perma = str(r.get("permaticker"))
        if not perma or perma == "None":
            continue
        sid = f"SHARADAR:{perma}"
        rec = by_perma.setdefault(sid, {
            "security_id": sid, "permaticker": perma, "name": r.get("name"),
            "cik": _digits(r.get("secfilings")) or r.get("cik"),
            "exchange": r.get("exchange"), "sector": r.get("sector"),
            "industry": r.get("industry"), "tickers": [], "membership": [],
            "listing_status": "active", "delisting_date": None,
            "first_tradable": None, "last_tradable": None, "provenance": "sharadar",
        })
        first, last = r.get("firstpricedate"), r.get("lastpricedate")
        rec["tickers"].append({"ticker": (r.get("ticker") or "").upper(),
                               "start": first, "end": last})
        # membership window == tradable window from the licensed dataset
        rec["membership"].append({"start": first, "end": last})
        # earliest firstpricedate / latest lastpricedate across ticker rows
        rec["first_tradable"] = _min_date(rec["first_tradable"], first)
        rec["last_tradable"] = _max_date(rec["last_tradable"], last)
        if str(r.get("isdelisted", "N")).upper() == "Y":
            rec["listing_status"] = "delisted"
            rec["delisting_date"] = _max_date(rec["delisting_date"], last)
    # Sort ticker windows so ticker_as_of resolution is deterministic.
    for rec in by_perma.values():
        rec["tickers"].sort(key=lambda t: (t.get("start") or ""))
        rec["membership"].sort(key=lambda m: (m.get("start") or ""))
    return list(by_perma.values())


def membership_as_of(securities: List[Dict[str, Any]], as_of: str,
                     master: SecurityMaster) -> List[Dict[str, Any]]:
    """Point-in-time constituents: a security is a member iff as_of falls inside
    a tradable window (firstpricedate <= as_of <= lastpricedate, or open-ended for
    still-active names). Delisted names ARE returned for dates they were members."""
    out = []
    for s in securities:
        member = False
        for m in s.get("membership", []):
            start = m.get("start")
            end = m.get("end")
            if start and as_of < start:
                continue
            if end and as_of > end:
                continue
            member = True
            break
        if not member:
            continue
        out.append({
            "security_id": s["security_id"],
            "ticker_as_of": master.ticker_as_of(s["security_id"], as_of),
            "name": s.get("name"), "cik": s.get("cik"), "exchange": s.get("exchange"),
            "sector": s.get("sector"), "industry": s.get("industry"),
            "listing_status": s.get("listing_status"),
            "delisting_date": s.get("delisting_date"),
            "first_tradable": s.get("first_tradable"),
            "last_tradable": s.get("last_tradable"),
            "provenance": "sharadar",
        })
    return out


# Map SF1 columns -> the feature engine's fundamentals record shape.
_SF1_MAP = {"revenue": "revenue", "gp": "gross_profit", "opinc": "operating_income",
            "netinc": "net_income", "fcf": "free_cash_flow", "debt": "total_debt",
            "equity": "equity", "sharesbas": "shares"}


def parse_sf1_records(rows: List[Dict[str, Any]], as_of: str) -> List[Dict[str, Any]]:
    """SHARADAR/SF1 (ARQ dimension) rows -> quarterly records the feature engine
    consumes, keeping ONLY rows whose `datekey` (filed/available date) <= as_of.
    This is the point-in-time guarantee at the source-record level."""
    recs = []
    for r in rows:
        datekey = r.get("datekey")            # the filed/available date
        if not datekey or datekey > as_of:
            continue                          # future filing — invisible at as_of
        rec = {"period_end": r.get("reportperiod") or r.get("calendardate"),
               "filed": datekey, "source": "sharadar:SF1",
               "accessn": f"SHARADAR-SF1-{r.get('ticker')}-{r.get('reportperiod')}"}
        for col, key in _SF1_MAP.items():
            v = r.get(col)
            rec[key] = float(v) if v not in (None, "") else None
        recs.append(rec)
    recs.sort(key=lambda x: (x.get("period_end") or "", x.get("filed") or ""))
    return recs


def delisting_return_from_action(action: Optional[str]) -> Optional[float]:
    """Conservative delisting return by Sharadar ACTIONS.action (DELISTING_POLICY.md)."""
    if not action:
        return None
    a = action.lower()
    if "bankrupt" in a or "liquidat" in a:
        return -100.0
    if "delist" in a:
        return -35.0                          # hard delist / going dark: conservative haircut
    if "acqui" in a or "merg" in a:
        return None                           # buyout value belongs in the last price, not here
    return -100.0                             # unknown removal: assume total loss


def _digits(v):
    if not v:
        return None
    d = "".join(ch for ch in str(v) if ch.isdigit())
    return d or None


def _min_date(a, b):
    xs = [x for x in (a, b) if x]
    return min(xs) if xs else None


def _max_date(a, b):
    xs = [x for x in (a, b) if x]
    return max(xs) if xs else None


# ── Provider (network gated on the license) ─────────────────────────────────

class SharadarUniverseProvider(UniverseProvider):
    """Survivorship-safe production provider. BLOCKED (raises UniverseIncomplete)
    until NASDAQ_DATA_LINK_API_KEY is configured — never fabricates membership."""
    universe_id = "sharadar"
    survivorship_safe = True                  # provably, once the licensed data flows

    def __init__(self, table_universe: str = "SEP"):
        self.table_universe = table_universe
        self._sm: Optional[SecurityMaster] = None
        self._secs: Optional[List[Dict[str, Any]]] = None
        self._actions: Dict[str, str] = {}

    # -- license gate ---------------------------------------------------------
    def _require_key(self) -> str:
        from financial_data.keys import has_key
        if not has_key(KEY_VAR):
            raise UniverseIncomplete(
                f"{UNIVERSE_INCOMPLETE}: production survivorship safety is BLOCKED — "
                f"the Sharadar historical-constituent dataset is not configured "
                f"(set {KEY_VAR}). No membership is fabricated. "
                f"See SURVIVORSHIP_POLICY.md / PRODUCTION_DATA_ACTIVATION_REPORT.md.")
        return KEY_VAR

    def _client(self):
        self._require_key()
        from financial_data.keys import get_key
        try:
            import nasdaqdatalink as ndl        # the maintained Quandl successor
        except ImportError as e:                 # pragma: no cover - env dependent
            raise UniverseIncomplete(
                f"{UNIVERSE_INCOMPLETE}: Sharadar client library not installed "
                f"(`pip install nasdaq-data-link`). {e}")
        ndl.ApiConfig.api_key = get_key(KEY_VAR, "Sharadar universe")
        return ndl

    def _fetch(self, table: str, **params) -> List[Dict[str, Any]]:  # pragma: no cover - network
        ndl = self._client()
        _throttle()
        df = ndl.get_table(f"SHARADAR/{table}", paginate=True, **params)
        return df.to_dict("records")

    # -- ingestion ------------------------------------------------------------
    def _ensure_loaded(self):
        if self._secs is not None:
            return
        rows = self._fetch("TICKERS", table=self.table_universe)   # pragma: no cover
        self._secs = parse_tickers(rows)                            # pragma: no cover
        self._sm = SecurityMaster(self._secs)                      # pragma: no cover
        for a in self._fetch("ACTIONS", **{}):                      # pragma: no cover
            self._actions[str(a.get("ticker", "")).upper()] = a.get("action")

    # -- UniverseProvider API -------------------------------------------------
    def security_master(self) -> SecurityMaster:
        self._ensure_loaded()                                       # pragma: no cover
        return self._sm                                            # pragma: no cover

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        self._require_key()
        self._ensure_loaded()                                       # pragma: no cover
        return membership_as_of(self._secs, as_of, self._sm)       # pragma: no cover

    def prices(self, security_id: str, start: str, end: str):       # pragma: no cover
        self._require_key()
        import pandas as pd
        self._ensure_loaded()
        tk = self._sm.ticker_as_of(security_id, end) or self._sm.ticker_as_of(security_id, start)
        if not tk:
            return pd.Series(dtype=float)
        rows = self._fetch("SEP", ticker=tk, **{"date.gte": start, "date.lte": end})
        if not rows:
            return pd.Series(dtype=float)
        s = pd.Series({r["date"]: float(r["closeadj"]) for r in rows})  # corp-action adjusted
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    def fundamentals_fn(self):
        """Return an EDGAR-independent, Sharadar-SF1-backed fundamentals function."""
        self._require_key()
        def _fn(security_raw: Dict[str, Any], as_of: str):          # pragma: no cover
            self._ensure_loaded()
            tk = self._sm.ticker_as_of(security_raw["security_id"], as_of)
            if not tk:
                return []
            rows = self._fetch("SF1", ticker=tk, dimension="ARQ")
            return parse_sf1_records(rows, as_of)
        return _fn

    def delisting_return_pct(self, security_id: str) -> Optional[float]:  # pragma: no cover
        self._ensure_loaded()
        s = self._sm.get(security_id) or {}
        if not s.get("delisting_date"):
            return None
        for t in s.get("tickers", []):
            act = self._actions.get(t["ticker"].upper())
            if act:
                return delisting_return_from_action(act)
        return -100.0                          # delisted, reason unknown: conservative

    def coverage(self) -> Dict[str, str]:
        return {"start": "1998-01-01", "end": "present", "status": "requires_license",
                "note": "Sharadar SEP/SF1 coverage; live once NASDAQ_DATA_LINK_API_KEY is set."}

    def benchmark_id(self) -> Optional[str]:
        return None                            # operator maps a benchmark (e.g. IWM permaticker)
