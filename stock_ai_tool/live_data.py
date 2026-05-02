from __future__ import annotations

import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from io import StringIO
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .data import fill_missing_price_bars, load_fundamentals, load_price_bars, load_symbol_profiles
from .models import FundamentalSnapshot, PriceBar, SymbolProfile


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 5


def load_static_inputs(
    prices_csv: str,
    profiles_csv: str,
    fundamentals_csv: str,
) -> Tuple[Dict[str, List[PriceBar]], Dict[str, SymbolProfile], Dict[str, FundamentalSnapshot]]:
    bars = load_price_bars(prices_csv)
    profiles = load_symbol_profiles(profiles_csv)
    fundamentals = load_fundamentals(fundamentals_csv)
    bars = fill_missing_price_bars(bars, profiles, fundamentals)
    return bars, profiles, fundamentals


def load_market_inputs(
    prices_csv: str,
    profiles_csv: str,
    fundamentals_csv: str,
    mode: str = "sample",
    symbols: Set[str] | None = None,
) -> Tuple[
    Dict[str, List[PriceBar]],
    Dict[str, SymbolProfile],
    Dict[str, FundamentalSnapshot],
    dict,
]:
    bars, profiles, fundamentals = load_static_inputs(
        prices_csv, profiles_csv, fundamentals_csv
    )
    if symbols:
        bars = {s: v for s, v in bars.items() if s in symbols}
        profiles = {s: v for s, v in profiles.items() if s in symbols}
        fundamentals = {s: v for s, v in fundamentals.items() if s in symbols}

    meta = {
        "mode": "sample",
        "provider": "local sample CSV",
        "symbols": sorted(set(profiles.keys()) & set(bars.keys()) & set(fundamentals.keys())),
        "sources": {},
        "warnings": [],
    }
    if mode != "live":
        meta["sources"] = {symbol: "sample" for symbol in meta["symbols"]}
        return bars, profiles, fundamentals, meta

    live_bars, live_profiles, live_fundamentals, live_meta = _fetch_live_inputs(
        profiles.keys(), profiles, fundamentals
    )
    bars.update(live_bars)
    profiles.update(live_profiles)
    fundamentals.update(live_fundamentals)
    meta.update(
        {
            "mode": "live",
            "provider": "Yahoo/Stooq public feed with local fallback",
            "sources": live_meta["sources"],
            "warnings": live_meta["warnings"],
            "symbols": sorted(
                set(profiles.keys()) & set(bars.keys()) & set(fundamentals.keys())
            ),
        }
    )
    return bars, profiles, fundamentals, meta


def fetch_live_quote(symbol: str) -> dict:
    symbol = symbol.upper()
    warnings: List[str] = []
    chart = {"meta": {}, "bars": []}
    summary = {}

    try:
        chart["bars"] = _fetch_stooq_bars(symbol)
        chart["meta"] = {"source": "Stooq daily CSV"}
    except Exception as stooq_exc:
        warnings.append(f"Stooq price fallback failed: {stooq_exc}")
        try:
            chart = _fetch_chart(symbol)
        except Exception as yahoo_exc:
            warnings.append(f"Yahoo chart failed: {yahoo_exc}")

    try:
        summary = _fetch_quote_summary(symbol)
    except Exception as yahoo_exc:
        warnings.append(f"Yahoo quoteSummary unavailable: {yahoo_exc}")

    return {
        "symbol": symbol,
        "chart": chart.get("meta", {}),
        "fundamentals": summary,
        "source": "Yahoo/Stooq public feed",
        "warnings": warnings,
    }


def _fetch_live_inputs(
    symbols: Iterable[str],
    profiles: Dict[str, SymbolProfile],
    fundamentals: Dict[str, FundamentalSnapshot],
) -> Tuple[Dict[str, List[PriceBar]], Dict[str, SymbolProfile], Dict[str, FundamentalSnapshot], dict]:
    live_bars: Dict[str, List[PriceBar]] = {}
    live_profiles: Dict[str, SymbolProfile] = {}
    live_fundamentals: Dict[str, FundamentalSnapshot] = {}
    sources: Dict[str, str] = {}
    warnings: List[str] = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_one_symbol, symbol, profiles[symbol], fundamentals[symbol]): symbol
            for symbol in symbols
            if symbol in profiles and symbol in fundamentals
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                bars, profile, fundamental, source = future.result()
                if bars:
                    live_bars[symbol] = bars
                if profile:
                    live_profiles[symbol] = profile
                if fundamental:
                    live_fundamentals[symbol] = fundamental
                sources[symbol] = source
            except Exception as exc:
                sources[symbol] = "sample fallback"
                warnings.append(f"{symbol}: unexpected live-feed error; using local fallback ({exc})")

    return live_bars, live_profiles, live_fundamentals, {
        "sources": sources,
        "warnings": warnings,
    }


def _fetch_one_symbol(
    symbol: str,
    fallback_profile: SymbolProfile,
    fallback_fundamental: FundamentalSnapshot,
) -> Tuple[List[PriceBar], SymbolProfile, FundamentalSnapshot, str]:
    bars: List[PriceBar] = []
    source = "sample fallback"

    # Prefer Stooq for daily bars because Yahoo public chart endpoints frequently
    # throttle local apps with HTTP 429. Yahoo is kept as an opportunistic fallback.
    try:
        bars = _fetch_stooq_bars(symbol)
        source = "Stooq daily CSV + local fundamentals"
    except Exception:
        try:
            chart = _fetch_chart(symbol)
            if chart.get("bars"):
                bars = chart["bars"]
                source = "Yahoo chart + local fundamentals"
        except Exception:
            bars = []

    profile = fallback_profile
    fundamental = fallback_fundamental
    if bars:
        try:
            summary = _fetch_quote_summary(symbol)
            profile = _merge_profile(fallback_profile, summary)
            fundamental = _merge_fundamentals(fallback_fundamental, summary)
            source = f"{source} + Yahoo fundamentals"
        except Exception:
            source = f"{source} + local fundamentals"
    else:
        source = "local sample fallback"
    return bars, profile, fundamental, source


def _fetch_chart(symbol: str) -> dict:
    query = urlencode({"range": "6mo", "interval": "1d", "includePrePost": "false"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    payload = _get_json(url)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        error = payload.get("chart", {}).get("error") or "empty chart response"
        raise ValueError(f"Yahoo chart failed for {symbol}: {error}")

    meta = result.get("meta", {})
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    bars: List[PriceBar] = []
    for idx, ts in enumerate(timestamps):
        close = _list_get(quote.get("close"), idx)
        open_price = _list_get(quote.get("open"), idx)
        high = _list_get(quote.get("high"), idx)
        low = _list_get(quote.get("low"), idx)
        volume = _list_get(quote.get("volume"), idx) or 0
        if close is None or open_price is None or high is None or low is None:
            continue
        bars.append(
            PriceBar(
                symbol=symbol.upper(),
                day=datetime.fromtimestamp(ts).date(),
                open=round(float(open_price), 4),
                high=round(float(high), 4),
                low=round(float(low), 4),
                close=round(float(close), 4),
                volume=float(volume),
            )
        )
    return {"meta": meta, "bars": bars}


def _fetch_quote_summary(symbol: str) -> dict:
    modules = ",".join(
        [
            "price",
            "summaryDetail",
            "defaultKeyStatistics",
            "financialData",
            "assetProfile",
        ]
    )
    query = urlencode({"modules": modules})
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}?{query}"
    payload = _get_json(url)
    result = (payload.get("quoteSummary", {}).get("result") or [None])[0]
    if not result:
        error = payload.get("quoteSummary", {}).get("error") or "empty quoteSummary response"
        raise ValueError(f"Yahoo quoteSummary failed for {symbol}: {error}")
    return result


def _fetch_stooq_bars(symbol: str) -> List[PriceBar]:
    stooq_symbol = f"{symbol.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8")

    rows = list(csv.DictReader(StringIO(text)))
    bars: List[PriceBar] = []
    for row in rows[-140:]:
        if not row.get("Close") or row["Close"].lower() == "null":
            continue
        day = datetime.strptime(row["Date"], "%Y-%m-%d").date()
        bars.append(
            PriceBar(
                symbol=symbol.upper(),
                day=day,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume") or 0),
            )
        )
    if not bars:
        raise ValueError(f"Stooq returned no bars for {symbol}")
    return bars


def _merge_profile(profile: SymbolProfile, summary: dict) -> SymbolProfile:
    price = summary.get("price") or {}
    asset = summary.get("assetProfile") or {}
    company = _raw(price.get("longName")) or _raw(price.get("shortName")) or profile.company
    sector = asset.get("industry") or asset.get("sector") or profile.sector
    return replace(profile, company=company, sector=sector)


def _merge_fundamentals(
    fundamental: FundamentalSnapshot,
    summary: dict,
) -> FundamentalSnapshot:
    summary_detail = summary.get("summaryDetail") or {}
    stats = summary.get("defaultKeyStatistics") or {}
    financial = summary.get("financialData") or {}
    pe_ratio = (
        _raw(summary_detail.get("trailingPE"))
        or _raw(stats.get("trailingPE"))
        or fundamental.pe_ratio
    )
    sales_millions = (
        (_raw(financial.get("totalRevenue")) or 0) / 1_000_000
        if _raw(financial.get("totalRevenue")) is not None
        else fundamental.sales_millions
    )
    revenue_growth = _raw(financial.get("revenueGrowth"))
    gross_margin = _raw(financial.get("grossMargins"))
    operating_margin = _raw(financial.get("operatingMargins"))
    fcf_margin = _raw(financial.get("freeCashflow"))
    if fcf_margin is not None and sales_millions:
        fcf_margin = (fcf_margin / 1_000_000) / sales_millions

    return replace(
        fundamental,
        pe_ratio=float(pe_ratio or 0),
        sales_millions=float(sales_millions or fundamental.sales_millions),
        revenue_growth=float(
            revenue_growth
            if revenue_growth is not None
            else fundamental.revenue_growth
        ),
        gross_margin=float(
            gross_margin if gross_margin is not None else fundamental.gross_margin
        ),
        operating_margin=float(
            operating_margin
            if operating_margin is not None
            else fundamental.operating_margin
        ),
        fcf_margin=float(fcf_margin if fcf_margin is not None else fundamental.fcf_margin),
    )


def _get_json(url: str) -> dict:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _raw(value):
    if isinstance(value, dict):
        return value.get("raw")
    return value


def _list_get(values, index: int):
    if not values or index >= len(values):
        return None
    return values[index]
