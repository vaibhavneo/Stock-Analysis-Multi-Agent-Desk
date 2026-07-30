"""
FIL — the Datum: the atomic unit of financial evidence.

Every number that enters the Stock Agent is a Datum. A Datum is not just a
value; it is a value plus the answer to "where did this come from and WHEN
could I have known it?" — the second half is what makes point-in-time
backtesting possible at all.

The load-bearing field is `available_at`: the wall-clock moment the outside
world could first have seen this value (an SEC filing's `filed` date, a bar's
close, a news item's publication time). It is NOT `period_end` — Apple's
Q4-2023 revenue has period_end 2023-09-30 but was not knowable until the 10-K
was filed on 2023-11-03. Using period_end as the decision timestamp is exactly
the look-ahead bug this layer exists to make impossible.

`status` distinguishes actual (reported) from estimate (consensus/projected) —
mixing them silently is how a backtest accidentally trades on knowledge of the
future that only analysts had, or on a number nobody ever reported.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# The nine data kinds the gateway can serve. A provider declares which it
# serves in registry.json; asking for a kind nobody serves is an honest,
# typed failure — never an empty list that reads like "no data exists".
KINDS = (
    "bars",              # OHLCV price series
    "fundamentals_pit",  # point-in-time financial statement facts
    "corporate_actions", # splits, dividends
    "universe",          # index/exchange membership over time
    "macro",             # rates, CPI, unemployment
    "events",            # earnings dates, calendars
    "filings",           # raw filing documents/metadata
    "short_interest",    # exchange-reported short interest
    "sentiment",         # social/news sentiment records
    "derived_metric",    # a computed score/ratio whose provenance is its formula
                         # + input snapshot (e.g. pillar scores) — honest
                         # provenance, but snapshot-timed (as_of_honored=False)
)

STATUSES = ("actual", "estimate")

# Every Datum must carry these — the mission's non-negotiable provenance set.
REQUIRED_FIELDS = (
    "datum_id", "kind", "value", "available_at", "retrieved_at",
    "source", "confidence", "status",
)


class SchemaError(ValueError):
    """A datum failed its contract. Never silently coerced — a malformed datum
    is a bug in a provider, and hiding it would let bad evidence reach a claim."""


def _iso(v: Any) -> Optional[str]:
    """Normalize dates/datetimes/strings to a comparable ISO-8601 string.

    Everything downstream compares timestamps as strings, which is only safe
    because ISO-8601 is lexicographically ordered — hence normalizing here,
    once, rather than trusting each provider's format.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # Validate rather than trust: a provider handing us "Q4 2023" must fail
        # loudly here, not produce a datum that silently mis-sorts later.
        try:
            datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                datetime.strptime(s, "%Y-%m-%d")
            except ValueError as e:
                raise SchemaError(f"unparseable timestamp: {v!r}") from e
        return s
    raise SchemaError(f"unsupported timestamp type: {type(v).__name__}")


def make_source(provider: str, document: Optional[str] = None,
                ref: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """Build the provenance block.

    `document` is the citable artifact id (an EDGAR accession number, a URL,
    a post id) — the thing a human opens to check the number by hand. That
    hand-checkability is the whole point of the EvidenceLedger.
    """
    if not provider:
        raise SchemaError("source requires a provider id")
    return {"provider": provider, "document": document, "ref": ref, "url": url}


def make_datum(
    kind: str,
    value: Any,
    available_at: Any,
    source: Dict[str, Any],
    *,
    symbol: Optional[str] = None,
    concept: Optional[str] = None,
    unit: Optional[str] = None,
    period_start: Any = None,
    period_end: Any = None,
    confidence: float = 1.0,
    status: str = "actual",
    retrieved_at: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a validated Datum.

    `datum_id` is a content hash, not a counter, so the same fact fetched twice
    — in different sessions, by different code paths — collapses to one ledger
    row. That makes evidence links stable across re-runs, which is what lets a
    report be reproduced months later.
    """
    if kind not in KINDS:
        raise SchemaError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if status not in STATUSES:
        raise SchemaError(f"unknown status {status!r}; expected one of {STATUSES}")
    if value is None:
        # A missing value must be reported as `unavailable` by the gateway, not
        # smuggled through as a null datum that arithmetic will turn into 0.
        raise SchemaError("datum value cannot be None — report it as unavailable instead")
    if not isinstance(source, dict) or "provider" not in source:
        raise SchemaError("datum requires a source dict with a provider")
    if not 0.0 <= float(confidence) <= 1.0:
        raise SchemaError(f"confidence must be in [0,1], got {confidence}")

    avail = _iso(available_at)
    if avail is None:
        raise SchemaError(
            "available_at is mandatory — a datum with no knowable-at time cannot "
            "be used point-in-time, and would silently enable look-ahead")

    d: Dict[str, Any] = {
        "kind": kind,
        "symbol": symbol.upper() if isinstance(symbol, str) else symbol,
        "concept": concept,
        "value": value,
        "unit": unit,
        "period_start": _iso(period_start),
        "period_end": _iso(period_end),
        "available_at": avail,
        "retrieved_at": _iso(retrieved_at or datetime.now()),
        "source": source,
        "confidence": round(float(confidence), 4),
        "status": status,
    }
    if extra:
        d["extra"] = extra
    d["datum_id"] = datum_id(d)
    validate_datum(d)
    return d


def datum_id(d: Dict[str, Any]) -> str:
    """Deterministic content id.

    Deliberately excludes `retrieved_at` and `confidence`: re-fetching the same
    filed fact tomorrow, or re-weighting a provider's reliability, must not mint
    a new identity for the same underlying fact.
    """
    src = d.get("source") or {}
    payload = {
        "kind": d.get("kind"),
        "symbol": d.get("symbol"),
        "concept": d.get("concept"),
        "value": d.get("value"),
        "unit": d.get("unit"),
        "period_start": d.get("period_start"),
        "period_end": d.get("period_end"),
        "available_at": d.get("available_at"),
        "provider": src.get("provider"),
        "document": src.get("document"),
        "ref": src.get("ref"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def validate_datum(d: Dict[str, Any]) -> None:
    """Raise SchemaError unless the datum carries its full provenance set."""
    missing = [f for f in REQUIRED_FIELDS if d.get(f) is None]
    if missing:
        raise SchemaError(f"datum missing required field(s): {missing}")
    if d["kind"] not in KINDS:
        raise SchemaError(f"unknown kind {d['kind']!r}")
    if d["status"] not in STATUSES:
        raise SchemaError(f"unknown status {d['status']!r}")


def visible_at(d: Dict[str, Any], as_of: Optional[str]) -> bool:
    """Was this datum knowable at `as_of`?

    The single predicate the whole no-look-ahead guarantee rests on, so it is
    written to be checkable by eye rather than clever.

    Both sides are ISO-8601, which sorts lexicographically — but they may carry
    different granularity ("2024-02-01" vs "2024-02-01T16:00:00"). We compare at
    the COARSER of the two: with only a filing date in hand, the honest claim is
    "available that day", not "available at 00:00 that day". That makes the
    boundary inclusive — a filing made ON the decision date counts as available,
    which matches how a human reading the filing that morning would act.
    """
    if as_of is None:
        return True
    avail, cutoff = str(d["available_at"]), str(as_of)
    n = min(len(avail), len(cutoff))
    return avail[:n] <= cutoff[:n]


def filter_pit(data: List[Dict[str, Any]], as_of: Optional[str]) -> List[Dict[str, Any]]:
    """Drop everything that had not yet happened at `as_of`."""
    if as_of is None:
        return list(data)
    return [d for d in data if visible_at(d, as_of)]


def latest_by_period(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse restatements: keep the most recently FILED value per
    (symbol, concept, period_end).

    This is the restatement rule. A company can report Q3 revenue three times —
    original, revised, re-revised — each with a later `available_at`. After
    filter_pit() has removed everything the caller could not have seen, the
    latest survivor is precisely what they WOULD have seen. Applying this
    before filtering would leak the future; that ordering is the entire trick.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    for d in data:
        key = (d.get("symbol"), d.get("concept"), d.get("period_end"))
        cur = best.get(key)
        if cur is None or str(d["available_at"]) > str(cur["available_at"]):
            best[key] = d
    return sorted(best.values(), key=lambda x: (str(x.get("period_end") or ""),
                                                str(x.get("concept") or "")))
