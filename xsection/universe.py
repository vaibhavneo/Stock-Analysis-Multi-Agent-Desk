"""
Point-in-time universe + canonical security identity (survivorship-safe core).

The single most dangerous bias in cross-sectional research is SURVIVORSHIP:
building a historical universe from today's surviving tickers silently deletes
every company that went bankrupt, got acquired, or was delisted — exactly the
losers whose absence flatters every backtest. This module makes that mistake
structurally impossible: a historical universe comes ONLY from stored
point-in-time membership, and delisted securities remain present for every date
they were members.

Honest data reality (the flagged human decision): TRUE survivorship-free
membership for a real investable universe requires a PAID historical-constituent
dataset (Sharadar SF1/SEP/TICKERS, or EODHD). Free data (yfinance/EDGAR) gives
you today's listings only. So this module ships:

  - FixtureUniverseProvider — a committed REFERENCE universe (synthetic, clearly
    labelled) that DOES include delisted names, ticker changes, and reused
    tickers, so the survivorship-safe mechanics are real and testable end to end.
  - PaidUniverseProvider — the production interface, key-gated, that raises
    UniverseIncomplete('BLOCKED: needs a historical-constituent dataset') until
    configured. It never fabricates membership.

There is deliberately NO "today's constituents" provider — that is the banned
survivorship trap. A caller that wants live tickers must label them
CURRENT_CONSTITUENTS_ONLY and accept they are not survivorship safe.

SecurityMaster uses a PERMANENT internal security_id as the primary key, never
the ticker. A ticker change keeps one identity; a reused ticker resolves to
different identities by date — so history is never accidentally merged or split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

FIXTURE = Path(__file__).parent / "fixtures" / "reference_universe.json"

# Data-quality / membership status vocabulary (mission §4).
UNIVERSE_INCOMPLETE = "UNIVERSE_INCOMPLETE"


class UniverseIncomplete(RuntimeError):
    """Historical membership is unavailable/out-of-coverage. Never substitute
    current constituents — raise this and let the caller mark the run incomplete."""


# ── Security identity ───────────────────────────────────────────────────────

class SecurityMaster:
    """Canonical identity: permanent security_id <-> (ticker history, CIK,
    exchange, name) with effective-date ranges."""

    def __init__(self, securities: List[Dict[str, Any]]):
        self._by_id: Dict[str, Dict[str, Any]] = {s["security_id"]: s for s in securities}

    def get(self, security_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(security_id)

    def resolve(self, ticker: str, as_of: str) -> Optional[str]:
        """ticker + date -> security_id, using effective-date ranges. A reused
        ticker resolves to whichever company held it on `as_of` — never both."""
        tk = ticker.upper().strip()
        hits = []
        for sid, s in self._by_id.items():
            for t in s.get("tickers", []):
                if t["ticker"].upper() != tk:
                    continue
                start = t.get("start") or "0000-00-00"
                end = t.get("end") or "9999-12-31"
                if start <= as_of <= end:
                    hits.append(sid)
        # Non-overlapping ranges guarantee at most one hit for a valid ticker map.
        return hits[0] if len(hits) == 1 else (None if not hits else hits[0])

    def ticker_as_of(self, security_id: str, as_of: str) -> Optional[str]:
        s = self._by_id.get(security_id)
        if not s:
            return None
        for t in s.get("tickers", []):
            start = t.get("start") or "0000-00-00"
            end = t.get("end") or "9999-12-31"
            if start <= as_of <= end:
                return t["ticker"]
        return None

    def all_ids(self) -> List[str]:
        return list(self._by_id)


# ── Synthetic reference prices (deterministic, offline) ─────────────────────

def synthetic_prices(price_model: Dict[str, Any], start: str, end: str,
                     last_tradable: Optional[str] = None,
                     delisting_date: Optional[str] = None):
    """Deterministic adjusted-close series for a reference security. Delisted
    names truncate at last_tradable; a buyout steps to buyout_price on the
    delisting date. Labelled synthetic — NEVER real market data."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(int(price_model.get("seed", 1)))
    idx = pd.date_range(start, end, freq="B")
    steps = rng.normal(price_model.get("drift", 0.0), price_model.get("vol", 0.02), len(idx))
    close = pd.Series(float(price_model.get("start", 50.0)) * np.exp(np.cumsum(steps)), index=idx)
    if last_tradable:
        cut = pd.Timestamp(last_tradable)
        close = close[close.index <= cut]
        bp = price_model.get("buyout_price")
        if bp is not None and len(close):
            close.iloc[-1] = float(bp)     # merger buyout price on the last day
    return close


# ── Universe providers ──────────────────────────────────────────────────────

class UniverseProvider:
    """Interface. `members(as_of)` returns the point-in-time constituent list;
    `survivorship_safe` states whether the provider can PROVE it."""
    universe_id: str = "abstract"
    survivorship_safe: bool = False

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def security_master(self) -> SecurityMaster:
        raise NotImplementedError

    def prices(self, security_id: str, start: str, end: str):
        raise NotImplementedError

    def coverage(self) -> Dict[str, str]:
        raise NotImplementedError

    def benchmark_id(self) -> Optional[str]:
        return None

    def fundamentals_fn(self):
        """Return a `f(security_raw, as_of) -> [quarterly records]` for quality/
        growth/valuation features, or None to use the synthetic default. Real
        providers return an EDGAR-backed function; the fixture returns None."""
        return None


class FixtureUniverseProvider(UniverseProvider):
    """Survivorship-safe REFERENCE provider backed by the committed fixture.
    Includes delisted names / ticker changes / reused tickers. Genuinely PIT over
    its documented coverage; synthetic prices, honestly labelled."""

    def __init__(self, path: Path = FIXTURE):
        self._raw = json.loads(Path(path).read_text())
        self.universe_id = self._raw["universe_id"]
        self.survivorship_safe = bool(self._raw.get("survivorship_safe"))
        self._secs = self._raw["securities"]
        self._sm = SecurityMaster(self._secs)
        self._benchmark = self._raw.get("benchmark")

    def coverage(self) -> Dict[str, str]:
        return dict(self._raw["coverage"])

    def disclaimer(self) -> str:
        return self._raw.get("disclaimer", "")

    def security_master(self) -> SecurityMaster:
        return self._sm

    def benchmark_id(self) -> Optional[str]:
        for s in self._secs:
            if any(t["ticker"] == self._benchmark for t in s.get("tickers", [])):
                return s["security_id"]
        return None

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        """Constituents whose stored MEMBERSHIP window contains `as_of` — never
        today's survivors. Delisted names are included for dates they were
        members; added names are excluded before their entry date. Out of
        coverage raises UniverseIncomplete."""
        cov = self.coverage()
        if as_of < cov["start"] or as_of > cov["end"]:
            raise UniverseIncomplete(
                f"{UNIVERSE_INCOMPLETE}: {as_of} is outside fixture coverage "
                f"{cov['start']}..{cov['end']}")
        out = []
        for s in self._secs:
            mem = s.get("membership") or []
            if not mem:
                continue                     # e.g. the benchmark: not a constituent
            in_window = any((m["start"] <= as_of) and (m.get("end") is None or as_of <= m["end"])
                            for m in mem)
            if not in_window:
                continue
            if s.get("first_tradable") and as_of < s["first_tradable"]:
                continue
            out.append({
                "security_id": s["security_id"],
                "ticker_as_of": self._sm.ticker_as_of(s["security_id"], as_of),
                "name": s["name"], "cik": s.get("cik"), "exchange": s.get("exchange"),
                "sector": s.get("sector"), "industry": s.get("industry"),
                "listing_status": s.get("listing_status"),
                "delisting_date": s.get("delisting_date"),
                "delisting_reason": s.get("delisting_reason"),
                "first_tradable": s.get("first_tradable"),
                "last_tradable": s.get("last_tradable"),
                "provenance": self._raw.get("provider", "fixture_reference"),
            })
        return out

    def prices(self, security_id: str, start: str, end: str):
        s = self._sm.get(security_id)
        if not s:
            import pandas as pd
            return pd.Series(dtype=float)
        return synthetic_prices(s["price_model"], start, end,
                                last_tradable=s.get("last_tradable"),
                                delisting_date=s.get("delisting_date"))

    def delisting_return_pct(self, security_id: str) -> Optional[float]:
        """The documented conservative delisting return applied AFTER the last
        tradable price (bankruptcy -> -100%; going-private -> a haircut). See
        DELISTING_POLICY.md. None for names that never delisted."""
        s = self._sm.get(security_id)
        if not s or not s.get("delisting_date"):
            return None
        pm = s.get("price_model", {})
        if pm.get("delist_return_pct") is not None:
            return float(pm["delist_return_pct"])
        if pm.get("buyout_price") is not None:
            return None     # buyout value already in the last price; no extra return
        return -100.0        # unknown-reason delisting: assume total loss (conservative)


class PaidUniverseProvider(UniverseProvider):
    """Production interface for a paid historical-constituent dataset (Sharadar
    TICKERS/SEP or EODHD). Key-gated: BLOCKED until configured — it never
    fabricates membership, so production survivorship safety is provably gated
    on the human dataset decision (SURVIVORSHIP_POLICY.md)."""
    universe_id = "paid-provider"
    survivorship_safe = True     # ...if and only if configured

    def __init__(self, dataset: str = "sharadar"):
        self.dataset = dataset

    def _require_key(self):
        from financial_data.keys import NotConfiguredError, has_key
        var = {"sharadar": "NASDAQ_DATA_LINK_API_KEY", "eodhd": "EODHD_API_KEY"}.get(
            self.dataset, "HISTORICAL_CONSTITUENTS_API_KEY")
        if not has_key(var):
            raise UniverseIncomplete(
                f"{UNIVERSE_INCOMPLETE}: production survivorship safety is BLOCKED — "
                f"the {self.dataset} historical-constituent dataset is not configured "
                f"(set {var}). No membership is fabricated. See SURVIVORSHIP_POLICY.md.")
        return var

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        self._require_key()
        raise NotImplementedError(
            "paid provider ingestion not implemented in this milestone — interface only")

    def coverage(self) -> Dict[str, str]:
        return {"start": "unknown", "end": "unknown", "status": "blocked_needs_dataset"}


def get_provider(universe_id: str = "reference-smallcap-demo") -> UniverseProvider:
    """Factory. Only the reference fixture is available keyless; the Sharadar
    production provider is returned but BLOCKED until its dataset is configured.
    A watchlist provider (real data, NOT survivorship-safe) can be built directly
    via WatchlistUniverseProvider — it is deliberately not a named universe here."""
    if universe_id in ("reference-smallcap-demo", "fixture", "reference"):
        return FixtureUniverseProvider()
    if universe_id == "production-pilot":
        from xsection.providers.production_pilot import ProductionPilotProvider
        return ProductionPilotProvider()
    if universe_id in ("sharadar", "paid"):
        from xsection.providers.sharadar import SharadarUniverseProvider
        return SharadarUniverseProvider()
    if universe_id == "eodhd":
        return PaidUniverseProvider(dataset="eodhd")
    raise UniverseIncomplete(f"{UNIVERSE_INCOMPLETE}: unknown universe {universe_id!r}")


def list_universes() -> List[Dict[str, Any]]:
    fx = FixtureUniverseProvider()
    return [
        {"universe_id": fx.universe_id, "provider": "fixture_reference",
         "survivorship_safe": True, "coverage": fx.coverage(), "status": "available",
         "label": "REFERENCE (synthetic) — survivorship-safe mechanics demo",
         "disclaimer": fx.disclaimer()},
        {"universe_id": "sharadar", "provider": "paid_interface",
         "survivorship_safe": True, "coverage": {"status": "blocked_needs_dataset"},
         "status": "blocked", "label": "Sharadar (production) — BLOCKED: needs dataset key",
         "disclaimer": "Set NASDAQ_DATA_LINK_API_KEY to enable; no membership is fabricated."},
    ]
