"""
FinancialDataGateway — the single door between the Stock Agent and the outside
world's data.

Modeled on second_brain/gateway.py, and for the same reason: when every consumer
goes through one door, a guarantee can be enforced ONCE instead of trusted in
fifty call sites. There, the guarantee was "scope is never guessed". Here it is
"the future never leaks backwards".

Pipeline:
  1. RESOLVE   pick provider(s) for the kind from registry.json (by priority)
  2. FETCH     call the provider module's fetch() — providers never filter
  3. PIT       drop every datum with available_at > as_of        <- the guarantee
  4. RESTATE   collapse restatements to the latest SURVIVING filing
  5. STAMP     as_of_honored + caveats + unavailable, honestly
  6. RETURN    datums with full provenance, ready for the EvidenceLedger

as_of_honored is a PROMISE, not a courtesy flag. It is true only when all three
hold: the caller asked for a point-in-time view, the serving provider declares
`pit_capable` for that kind, and step 3 actually ran. Absent any of those, it is
false and the caller must treat the result as non-PIT. A backtester that ignores
this flag is choosing look-ahead with its eyes open.
"""
from __future__ import annotations

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schemas import KINDS, filter_pit, latest_by_period, validate_datum

REGISTRY_PATH = Path(__file__).parent / "registry.json"


class GatewayError(RuntimeError):
    """Typed failure. Like the retrieval gateway's NoScopeError, a request the
    gateway cannot honestly serve fails loudly rather than returning [] — an
    empty list reads as 'no such data exists', which is a different claim."""


class NoProviderError(GatewayError):
    pass


_registry_cache: Optional[Dict[str, Any]] = None
_module_cache: Dict[str, Any] = {}


def load_registry(refresh: bool = False) -> Dict[str, Any]:
    global _registry_cache
    if _registry_cache is None or refresh:
        with REGISTRY_PATH.open() as fh:
            _registry_cache = json.load(fh)
    return _registry_cache


def list_providers(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Providers serving `kind`, best first. Priority then reliability — a
    deliberate ordering: EDGAR outranks yfinance for fundamentals not because
    it is faster but because it is the only one that can tell the truth about
    what was knowable when."""
    provs = list(load_registry()["providers"].values())
    if kind:
        provs = [p for p in provs if kind in p.get("kinds", [])]
    return sorted(provs, key=lambda p: (p.get("priority", 999), -p.get("reliability", 0)))


def _load_module(provider: Dict[str, Any]):
    """Import a provider by the module path in its REGISTRY ENTRY.

    This indirection is the P8 seam: gateway code contains no vendor name, so
    swapping SEC for a paid feed is a JSON edit, not a code change.
    """
    mod_path = provider.get("module")
    if not mod_path:
        raise NoProviderError(f"provider {provider['id']!r} declares no module")
    if mod_path not in _module_cache:
        try:
            _module_cache[mod_path] = importlib.import_module(mod_path)
        except ImportError as e:
            raise NoProviderError(f"provider {provider['id']!r} module unimportable: {e}") from e
    return _module_cache[mod_path]


def get(kind: str,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        as_of: Optional[str] = None,
        provider: Optional[str] = None,
        concepts: Optional[List[str]] = None,
        collapse_restatements: bool = True,
        **kwargs: Any) -> Dict[str, Any]:
    """Fetch financial data with provenance and an explicit PIT guarantee.

    Args:
      kind:    one of schemas.KINDS
      symbols: ticker or list of tickers
      as_of:   decision date (ISO). Everything filed/published after it is
               removed. THIS is what makes a historical replay honest.
      provider: force a specific provider id (cross-checking; normally leave None)
      concepts: for fundamentals_pit, the subset of CONCEPT_MAP keys wanted
      collapse_restatements: keep only the latest surviving filing per period.
               False exposes the full restatement history — useful to SEE a
               restatement, which is exactly what the PIT test does.

    Returns a dict whose `as_of_honored` field is the contract; read it.
    """
    if kind not in KINDS:
        raise GatewayError(f"unknown kind {kind!r}; expected one of {KINDS}")
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    if not syms:
        raise GatewayError("at least one symbol is required")
    syms = [s.upper().strip() for s in syms]

    candidates = list_providers(kind)
    if provider:
        candidates = [p for p in candidates if p["id"] == provider]
        if not candidates:
            raise NoProviderError(f"provider {provider!r} does not serve kind {kind!r}")
    if not candidates:
        raise NoProviderError(
            f"no registered provider serves kind {kind!r} — add one to registry.json")

    warnings: List[str] = []
    unavailable: List[Dict[str, Any]] = []
    data: List[Dict[str, Any]] = []
    used: Optional[Dict[str, Any]] = None
    tried: List[str] = []

    # Try providers in order; fall through on failure. Fallback is a downgrade,
    # so it is recorded in `warnings` — a silent fallback to a 0.6-reliability
    # source would be exactly the kind of invisible quality loss P7 forbids.
    for prov in candidates:
        tried.append(prov["id"])
        try:
            mod = _load_module(prov)
            res = mod.fetch(kind=kind, symbols=syms, start=start, end=end, as_of=as_of,
                            concepts=concepts, reliability=prov.get("reliability", 1.0),
                            **kwargs)
        except Exception as e:
            warnings.append(f"provider {prov['id']} failed: {e}")
            continue
        data = res.get("data") or []
        unavailable = res.get("unavailable") or []
        warnings.extend(res.get("warnings") or [])
        used = prov
        if data:
            break
        warnings.append(f"provider {prov['id']} returned no data; trying next")

    if used is None:
        raise NoProviderError(
            f"every provider failed for kind={kind!r} symbols={syms}: {warnings}")

    for d in data:
        validate_datum(d)     # a provider bug must not reach the ledger

    # ---- THE GUARANTEE -------------------------------------------------
    pit_capable = kind in (used.get("pit_capable") or [])
    n_before = len(data)
    if as_of is not None and pit_capable:
        data = filter_pit(data, as_of)
        as_of_honored = True
    else:
        as_of_honored = False
        if as_of is not None and not pit_capable:
            warnings.append(
                f"as_of requested but provider {used['id']} is not pit_capable for "
                f"{kind!r}: result MAY contain look-ahead — as_of_honored=false")
    n_leaked = n_before - len(data)

    # Restatements are collapsed only AFTER the PIT cut, never before: the
    # latest filing that SURVIVES the cut is what the caller would have seen.
    # Reversing these two steps is the classic silent look-ahead bug.
    if collapse_restatements and kind == "fundamentals_pit":
        data = latest_by_period(data)

    caveat = (used.get("caveats") or {}).get(kind)
    if caveat:
        warnings.append(f"{used['id']}/{kind}: {caveat}")

    return {
        "kind": kind,
        "symbols": syms,
        "data": data,
        "n": len(data),
        "provider": used["id"],
        "providers_tried": tried,
        "reliability": used.get("reliability", 1.0),
        "as_of": as_of,
        "as_of_honored": as_of_honored,
        "excluded_future_datums": n_leaked,   # evidence the cut ran, not a claim
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "unavailable": unavailable,           # named, never silently dropped
        "warnings": warnings,
        "caveats": [caveat] if caveat else [],
    }


def get_bars_df(symbol: str, period: str = "6mo", start: Optional[str] = None,
                end: Optional[str] = None, as_of: Optional[str] = None,
                provider: Optional[str] = None, **kwargs: Any):
    """Bars as an OHLCV DataFrame — the shape the legacy indicator/backtest code
    consumes — while still flowing through the gateway (provenance + provider
    choice from the registry, not a hardcoded vendor call).

    Provenance lives in `df.attrs`, not per row: a price bar's citation is
    "which feed, fetched when", identical for every row, so per-row Datums would
    be noise. `df.attrs` carries provider / reliability / as_of_honored /
    caveats / retrieved_at — survives most pandas ops, and a caller that wants
    per-bar Datums can still call get('bars', ...) directly.

    Returns an empty DataFrame (not None, not a raise) when a symbol genuinely
    has no bars, so the live dashboard degrades to "no data" rather than a 500.
    """
    import pandas as pd

    res = get("bars", symbol, period=period, start=start, end=end,
              as_of=as_of, provider=provider, **kwargs)
    cols = ("Open", "High", "Low", "Close", "Volume")
    if not res["data"]:
        df = pd.DataFrame(columns=cols)
    else:
        rows, idx = [], []
        for d in res["data"]:
            ex = d.get("extra") or {}
            idx.append(d["available_at"])
            rows.append({"Open": ex.get("open"), "High": ex.get("high"),
                         "Low": ex.get("low"), "Close": d["value"],
                         "Volume": ex.get("volume")})
        # utc=True is mandatory, not cosmetic: daily bar timestamps arrive as ISO
        # strings whose UTC offset flips across DST (…-05:00 vs …-04:00), and
        # pandas refuses to build a single DatetimeIndex from mixed offsets
        # without it. tz_localize(None) then yields a consistent naive index —
        # downstream indicators use values and ordering, never the wall-clock tz.
        index = pd.to_datetime(idx, utc=True).tz_localize(None)
        df = pd.DataFrame(rows, index=index).sort_index()
    df.attrs.update({
        "provider": res["provider"], "reliability": res["reliability"],
        "as_of": res["as_of"], "as_of_honored": res["as_of_honored"],
        "caveats": res["caveats"], "retrieved_at": res["retrieved_at"],
        "unavailable": res["unavailable"],
    })
    return df


def get_concept(symbol: str, concept: str, as_of: Optional[str] = None,
                **kwargs: Any) -> Optional[Dict[str, Any]]:
    """Convenience: the single most recent value of one fundamental concept as
    of a date, or None if genuinely unavailable.

    Returns the Datum itself (not a bare float) so the caller physically cannot
    record a claim without also having the evidence in hand.
    """
    res = get("fundamentals_pit", symbol, as_of=as_of, concepts=[concept], **kwargs)
    hits = [d for d in res["data"] if d.get("concept") == concept]
    if not hits:
        return None
    return max(hits, key=lambda d: (str(d.get("period_end") or ""), str(d["available_at"])))
