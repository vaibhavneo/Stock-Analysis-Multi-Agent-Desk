"""
Real point-in-time fundamentals from SEC EDGAR, shaped for the feature engine.

This is the free-data half of production activation: it turns EDGAR companyfacts
(via the FIL gateway, `available_at` = the FILED date) into the same quarterly
record shape `features.compute_features` already consumes — but with REAL,
filed-date-governed numbers instead of synthetic ones. A filing filed after the
ranking date is invisible; a restatement filed later never overwrites what a
filer actually knew on `as_of`.

Honest limits (surfaced as explicit missingness, never faked):
  - Only quarterly-length flow periods (≈80–100 day spans) feed growth/TTM, so
    annual 10-K figures don't corrupt a trailing-4-quarter sum.
  - Concepts a company simply never tags (some small caps omit GrossProfit) come
    back as None -> the feature is PARTIAL_DATA, not a guessed value.
  - Analyst estimates / revisions / sentiment / macro vintages remain UNAVAILABLE
    (features.py already marks these) — there is no free PIT history for them.

The parse layer (`assemble_records`) is pure and unit-tested offline; only the
gateway fetch touches the network. WatchlistUniverseProvider wires real prices +
real fundamentals together, but is explicitly NOT survivorship-safe: a fixed
ticker list is a watchlist, not a point-in-time index reconstruction. It exists
to validate the real feature pipeline, and every output says so.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from xsection.universe import SecurityMaster, UniverseProvider

# EDGAR concept -> our record key. Flows are quarterly-length; balance items are
# instantaneous (period_start is None).
_FLOW = {"revenue": "revenue", "gross_profit": "gross_profit",
         "operating_income": "operating_income", "net_income": "net_income",
         "operating_cash_flow": "operating_cash_flow", "capex": "capex"}
_INSTANT = {"equity": "equity", "long_term_debt": "total_debt",
            "shares_outstanding": "shares"}
FETCH_CONCEPTS = list(_FLOW) + list(_INSTANT)


def _days(a: Optional[str], b: Optional[str]) -> Optional[int]:
    if not a or not b:
        return None
    import datetime as dt
    try:
        return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days
    except Exception:
        return None


def _discretize_flows(flow_chosen: Dict[tuple, Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    """Recover discrete quarterly flows from EDGAR's mix of reporting styles.

    Many filers tag flow concepts (revenue, income, cash flow) as YEAR-TO-DATE
    cumulative — e.g. a Q2 10-Q reports a 181-day figure spanning from fiscal-year
    start. Summing those as if they were quarters would double-count. The standard
    fix: within a fiscal-year chain (datums sharing a period_start), difference
    consecutive cumulative values (Q_n = YTD_n - YTD_{n-1}). Filers that already
    report discrete ~90-day quarters have single-element chains and pass through
    unchanged. When both styles exist for a period_end, the as-reported discrete
    quarter wins over the differenced one.

    flow_chosen: {(concept, period_start, period_end): datum}. Returns
    {(concept, period_end): {"value","filed","accessn"}}."""
    from collections import defaultdict
    out: Dict[tuple, Dict[str, Any]] = {}
    by_concept: Dict[str, list] = defaultdict(list)
    for (concept, ps, pe), d in flow_chosen.items():
        by_concept[concept].append((ps, pe, float(d["value"]), d.get("available_at"),
                                    (d.get("source") or {}).get("document")))
    for concept, items in by_concept.items():
        chains: Dict[str, list] = defaultdict(list)     # period_start -> [(pe,val,filed,doc)]
        for ps, pe, val, filed, doc in items:
            chains[ps].append((pe, val, filed, doc))
        for ps, seq in chains.items():
            seq.sort(key=lambda x: x[0])                 # by period_end
            prev = 0.0
            for pe, val, filed, doc in seq:
                span = _days(ps, pe)
                discrete = val if (span is not None and span <= 100) else val - prev
                prev = val
                key = (concept, pe)
                # prefer an as-reported discrete quarter over a differenced one
                is_discrete = span is not None and 80 <= span <= 100
                if key not in out or (is_discrete and not out[key].get("_discrete")):
                    out[key] = {"value": discrete, "filed": filed, "accessn": doc,
                                "_discrete": is_discrete}
    return out


def assemble_records(datums: List[Dict[str, Any]], as_of: str) -> List[Dict[str, Any]]:
    """Pure: EDGAR datums -> discrete quarterly fundamentals records (filed <= as_of).

    Per (concept, period) keep the value with the LATEST filed date <= as_of — the
    most recent figure a filer could have known, honoring restatements point-in-
    time. Flow concepts are differenced from YTD to discrete quarters
    (`_discretize_flows`); balance-sheet concepts are attached instantaneously.
    free_cash_flow = operating_cash_flow - capex when both are present; anything
    a filer never tagged stays None (explicit missingness, never guessed)."""
    flow_chosen: Dict[tuple, Dict[str, Any]] = {}       # (concept, period_start, period_end)
    inst_chosen: Dict[tuple, Dict[str, Any]] = {}       # (concept, period_end)
    for d in datums:
        filed = d.get("available_at")
        pe = d.get("period_end")
        concept = d.get("concept")
        if not filed or not pe or concept is None or d.get("value") is None:
            continue
        if filed > as_of:                       # future filing — invisible now
            continue
        if concept in _FLOW:
            ps = d.get("period_start")
            if not ps:
                continue                        # a flow with no start has no measurable span
            key = (concept, ps, pe)
            if key not in flow_chosen or filed > flow_chosen[key].get("available_at", ""):
                flow_chosen[key] = d
        elif concept in _INSTANT:
            key = (concept, pe)
            if key not in inst_chosen or filed > inst_chosen[key].get("available_at", ""):
                inst_chosen[key] = d

    by_pe: Dict[str, Dict[str, Any]] = {}
    # discrete quarterly flows
    for (concept, pe), v in _discretize_flows(flow_chosen).items():
        rec = by_pe.setdefault(pe, {"period_end": pe, "filed": None,
                                    "source": "edgar:companyfacts", "accessn": None})
        rec[_FLOW[concept]] = v["value"]
        if v["filed"] and (rec["filed"] is None or v["filed"] > rec["filed"]):
            rec["filed"], rec["accessn"] = v["filed"], v["accessn"] or f"EDGAR-{pe}"
    # instantaneous balance-sheet items
    for (concept, pe), d in inst_chosen.items():
        rec = by_pe.setdefault(pe, {"period_end": pe, "filed": None,
                                    "source": "edgar:companyfacts", "accessn": None})
        rec[_INSTANT[concept]] = float(d["value"])
        f = d.get("available_at")
        if f and (rec["filed"] is None or f > rec["filed"]):
            rec["filed"] = f
            rec["accessn"] = (d.get("source") or {}).get("document") or f"EDGAR-{pe}"

    out = []
    for pe, rec in by_pe.items():
        # derive free cash flow; leave None (explicit missingness) if inputs absent
        ocf, capex = rec.pop("operating_cash_flow", None), rec.pop("capex", None)
        rec["free_cash_flow"] = (ocf - capex) if (ocf is not None and capex is not None) else None
        rec.setdefault("gross_profit", None)
        rec.setdefault("operating_income", None)
        rec.setdefault("net_income", None)
        rec.setdefault("total_debt", None)
        rec.setdefault("equity", None)
        rec.setdefault("shares", None)
        rec.setdefault("revenue", None)
        out.append(rec)
    out.sort(key=lambda x: (x.get("period_end") or "", x.get("filed") or ""))
    return out


def edgar_fundamentals(security_raw: Dict[str, Any], as_of: str) -> List[Dict[str, Any]]:
    """Fetch real PIT fundamentals via the gateway and assemble quarterly records.
    `security_raw` must carry a resolvable ticker (`ticker_as_of` or `ticker`)."""
    ticker = security_raw.get("ticker_as_of") or security_raw.get("ticker")
    if not ticker and security_raw.get("tickers"):
        ticker = security_raw["tickers"][-1].get("ticker")
    if not ticker:
        return []
    from financial_data.gateway import get
    try:
        res = get("fundamentals_pit", ticker, as_of=as_of, concepts=FETCH_CONCEPTS)
    except Exception:
        return []                               # network/provider failure -> missingness
    return assemble_records(res.get("data") or [], as_of)


# ── Watchlist provider (real data, explicitly NOT survivorship-safe) ────────

class WatchlistUniverseProvider(UniverseProvider):
    """A fixed, operator-supplied ticker list scored with REAL prices (gateway,
    corporate-action adjusted) and REAL EDGAR fundamentals. `survivorship_safe`
    is False and every member carries a NOT_SURVIVORSHIP_SAFE flag: a watchlist
    is not a point-in-time index, so it must never be read as a production
    universe. Purpose: validate the real feature pipeline end to end."""
    survivorship_safe = False

    def __init__(self, tickers: List[str], universe_id: str = "watchlist-demo",
                 benchmark: Optional[str] = "SPY", start: str = "2015-01-01",
                 sectors: Optional[Dict[str, str]] = None):
        self.universe_id = universe_id
        self._tickers = [t.upper() for t in tickers]
        self._benchmark = benchmark.upper() if benchmark else None
        self._start = start
        self._sectors = sectors or {}
        secs = [{"security_id": f"WL:{t}", "name": t,
                 "tickers": [{"ticker": t, "start": None, "end": None}],
                 "sector": self._sectors.get(t), "provenance": "watchlist"}
                for t in self._tickers]
        if self._benchmark:
            secs.append({"security_id": f"WL:{self._benchmark}", "name": self._benchmark,
                         "tickers": [{"ticker": self._benchmark, "start": None, "end": None}],
                         "sector": None, "provenance": "watchlist"})
        self._secs = secs
        self._sm = SecurityMaster(secs)

    def coverage(self) -> Dict[str, str]:
        import datetime as dt
        return {"start": self._start, "end": dt.date.today().isoformat(),
                "status": "not_survivorship_safe"}

    def security_master(self) -> SecurityMaster:
        return self._sm

    def benchmark_id(self) -> Optional[str]:
        return f"WL:{self._benchmark}" if self._benchmark else None

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        out = []
        for t in self._tickers:
            out.append({"security_id": f"WL:{t}", "ticker_as_of": t, "name": t,
                        "sector": self._sectors.get(t), "industry": None,
                        "listing_status": "active", "delisting_date": None,
                        "first_tradable": None, "last_tradable": None,
                        "provenance": "watchlist",
                        "flags": ["NOT_SURVIVORSHIP_SAFE"]})
        return out

    def prices(self, security_id: str, start: str, end: str):
        import pandas as pd
        tk = self._sm.ticker_as_of(security_id, end) or security_id.split(":")[-1]
        from financial_data.gateway import get_bars_df
        try:
            df = get_bars_df(tk, start=start, end=end)
        except Exception:
            return pd.Series(dtype=float)
        if df is None or not len(df):
            return pd.Series(dtype=float)
        return df["Close"].dropna()             # gateway yfinance bars are auto_adjust=True

    def delisting_return_pct(self, security_id: str) -> Optional[float]:
        return None                             # watchlist names are current listings

    def fundamentals_fn(self):
        return edgar_fundamentals
