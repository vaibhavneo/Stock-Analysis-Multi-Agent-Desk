"""
FIL provider — SEC EDGAR / XBRL companyfacts.

WHY THIS IS THE PRIMARY FUNDAMENTALS SOURCE
-------------------------------------------
Most retail fundamentals APIs serve the CURRENT view of history: ask for 2019
revenue and you get today's restated figure. Backtesting on that is look-ahead
— you are trading on a number nobody had in 2019.

EDGAR's companyfacts is structurally different. Every fact carries `filed`,
the date the document reached the SEC, and restatements arrive as ADDITIONAL
entries rather than overwrites. History is append-only, so "what did the world
know on date D?" is answerable exactly: keep entries with filed <= D, then take
the latest survivor per period. That is the whole point-in-time fix, and it is
free.

Endpoints (no key, no auth):
  company_tickers.json                      ticker -> CIK
  data.sec.gov/api/xbrl/companyfacts/CIK... every fact ever filed
  data.sec.gov/submissions/CIK...           filing index

SEC requires a descriptive User-Agent with contact info and asks for <10 req/s.
Both are honored below; being a bad citizen here would get the whole repo
blocked from a public good.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import cache
from ..schemas import make_datum, make_source

BASE_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
BASE_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

_MIN_INTERVAL = 1.0 / 8      # stay under SEC's 10 req/s guidance
_last_call = [0.0]


class NotConfiguredError(RuntimeError):
    """SEC_USER_AGENT is absent or non-compliant.

    Kept distinct from ProviderError on purpose: "you have not configured this"
    and "SEC is unreachable" demand opposite responses from a caller, and a test
    that lumps them together will happily skip past a config bug forever —
    which is precisely how this was nearly missed.
    """


def _load_user_agent() -> str:
    """SEC's fair-access policy REQUIRES a User-Agent naming a real contact
    ("Company Name admin@example.com"). Anything else — including a browser
    string — gets a hard 403, verified empirically 2026-07-16.

    Read from the environment, falling back to stock_agent/.env (the same
    convention web/app.py uses for DEEPSEEK_API_KEY). Deliberately NOT defaulted
    to a hardcoded address: an email in a committed file is both a privacy leak
    and a lie about who is making the requests.
    """
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("SEC_USER_AGENT="):
                    ua = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not ua:
        raise NotConfiguredError(
            "SEC_USER_AGENT is not set. SEC requires a User-Agent identifying a real "
            "contact, e.g. 'Your Name you@example.com'. Add to stock_agent/.env:\n"
            "    SEC_USER_AGENT=Your Name you@example.com\n"
            "See https://www.sec.gov/os/webmaster-faq#developers")
    if "@" not in ua:
        raise NotConfiguredError(
            f"SEC_USER_AGENT={ua!r} has no contact email; SEC will reject it with 403. "
            "Use the form 'Your Name you@example.com'.")
    return ua

# The ~20 core concepts. Each maps to an ORDERED list of us-gaap tags because
# companies legitimately tag the same economic quantity differently (and change
# tags between years). First tag that yields data wins; the tag actually used is
# recorded in the datum's source.ref, so a reader can always see which one.
CONCEPT_MAP: Dict[str, List[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "CommonStockSharesIssued"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "inventory": ["InventoryNet"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "sga_expense": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt"],
    "income_tax": ["IncomeTaxExpenseBenefit"],
}

KINDS = ("fundamentals_pit", "filings")


class ProviderError(RuntimeError):
    """Network/parse failure. Raised, never swallowed into an empty result —
    "SEC is down" and "this company reports nothing" must stay distinguishable."""


def _throttle() -> None:
    delta = time.time() - _last_call[0]
    if delta < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - delta)
    _last_call[0] = time.time()


def _fetch_json(url: str, timeout: int = 20) -> Any:
    _throttle()
    req = urllib.request.Request(url, headers={
        "User-Agent": _load_user_agent(),
        "Accept-Encoding": "gzip, deflate",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            return json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            # Near-always the User-Agent, not a real block. Say so, because a
            # generic "HTTP 403" sent me chasing the network for ten minutes.
            raise NotConfiguredError(
                f"EDGAR returned 403 for {url}. This is almost always a non-compliant "
                f"User-Agent — SEC requires 'Your Name you@example.com' in SEC_USER_AGENT.") from e
        raise ProviderError(f"EDGAR HTTP {e.code} for {url}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError(f"EDGAR unreachable: {e}") from e
    except json.JSONDecodeError as e:
        raise ProviderError(f"EDGAR returned non-JSON for {url}") from e


def resolve_cik(symbol: str, max_age_sec: int = 86400 * 7) -> Optional[int]:
    """Ticker -> CIK. Cached for a week; the mapping changes rarely."""
    payload = cache.get("sec-edgar", "_meta", "ticker_map", max_age_sec=max_age_sec)
    if payload is None:
        payload = _fetch_json(TICKER_MAP_URL)
        cache.put("sec-edgar", "_meta", "ticker_map", payload)
    sym = symbol.upper().strip()
    for row in payload.values():
        if str(row.get("ticker", "")).upper() == sym:
            return int(row["cik_str"])
    return None


def _companyfacts(cik: int, max_age_sec: int = 86400) -> Dict[str, Any]:
    key = f"CIK{cik:010d}"
    payload = cache.get("sec-edgar", "companyfacts", key, max_age_sec=max_age_sec)
    if payload is None:
        payload = _fetch_json(BASE_FACTS.format(cik=cik))
        cache.put("sec-edgar", "companyfacts", key, payload)
    return payload


def parse_companyfacts(payload: Dict[str, Any], symbol: str,
                       concepts: Optional[List[str]] = None,
                       reliability: float = 1.0) -> List[Dict[str, Any]]:
    """Turn a raw companyfacts payload into Datums.

    Pure function — no network. That is deliberate: it lets the point-in-time
    and restatement logic be tested exhaustively against fixtures, offline and
    deterministically, which is the only way this guarantee stays trustworthy.

    Every filed entry becomes its own Datum, INCLUDING superseded restatements.
    We do not resolve them here; the gateway resolves them relative to a caller's
    `as_of`. Discarding them at parse time would destroy the very history that
    makes point-in-time possible.
    """
    wanted = list(concepts) if concepts else list(CONCEPT_MAP)
    facts = (payload.get("facts") or {}).get("us-gaap") or {}
    out: List[Dict[str, Any]] = []

    for concept in wanted:
        for tag in CONCEPT_MAP.get(concept, [concept]):
            node = facts.get(tag)
            if not node:
                continue
            units = node.get("units") or {}
            for unit_name, entries in units.items():
                for e in entries:
                    filed, val = e.get("filed"), e.get("val")
                    if filed is None or val is None:
                        continue     # cannot place it in time -> unusable, skip
                    try:
                        out.append(make_datum(
                            kind="fundamentals_pit",
                            value=val,
                            available_at=filed,          # THE anti-look-ahead field
                            source=make_source(
                                provider="sec-edgar",
                                document=e.get("accn"),   # accession number: hand-checkable
                                ref=f"us-gaap:{tag}",
                                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                                    f"&CIK={payload.get('cik')}&type={e.get('form','')}",
                            ),
                            symbol=symbol,
                            concept=concept,
                            unit=unit_name,
                            period_start=e.get("start"),
                            period_end=e.get("end"),
                            confidence=reliability,
                            status="actual",             # EDGAR is as-reported by definition
                            extra={"form": e.get("form"), "fy": e.get("fy"), "fp": e.get("fp")},
                        ))
                    except Exception:
                        # One malformed entry must not sink the whole company's
                        # history; it is simply absent, and absence is visible
                        # because the caller sees fewer datums, never a fake one.
                        continue
            if any(d["concept"] == concept for d in out):
                break    # first tag that produced data wins
    return out


def fetch(kind: str, symbols: List[str], start: Optional[str] = None,
          end: Optional[str] = None, as_of: Optional[str] = None,
          concepts: Optional[List[str]] = None, reliability: float = 1.0,
          **kwargs: Any) -> Dict[str, Any]:
    """Gateway entrypoint. Returns {data, unavailable, warnings}.

    Note it does NOT apply `as_of` filtering — that is the gateway's job, in one
    place, for every provider. A provider that filtered on its own could quietly
    disagree with the gateway's guarantee.
    """
    if kind not in KINDS:
        raise ProviderError(f"sec-edgar does not serve kind {kind!r}")
    if kind == "filings":
        raise ProviderError("sec-edgar `filings` kind not implemented yet (M-F2 follow-up)")

    data: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for sym in symbols:
        try:
            cik = resolve_cik(sym)
        except ProviderError as e:
            unavailable.append({"symbol": sym, "reason": f"cik_lookup_failed: {e}"})
            continue
        if cik is None:
            # Honest: not "no data", but "this ticker is not an SEC filer" —
            # true for ADRs, most ETFs, crypto, and foreign issuers.
            unavailable.append({"symbol": sym, "reason": "no_sec_cik (not a US SEC filer?)"})
            continue
        try:
            payload = _companyfacts(cik)
        except ProviderError as e:
            unavailable.append({"symbol": sym, "reason": str(e)})
            continue

        datums = parse_companyfacts(payload, sym, concepts=concepts, reliability=reliability)
        if not datums:
            unavailable.append({"symbol": sym, "reason": "no_matching_us_gaap_concepts"})
            continue
        data.extend(datums)

    return {"data": data, "unavailable": unavailable, "warnings": warnings}
