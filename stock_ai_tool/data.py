from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from .models import FundamentalSnapshot, PriceBar, SymbolProfile


OPTIONAL_FUNDAMENTAL_DEFAULTS = {
    "sales_millions": 0.0,
    "gross_margin": 0.0,
    "operating_margin": 0.0,
    "industry_pe": 30.0,
    "industry_revenue_growth": 0.12,
    "industry_fcf_margin": 0.10,
}


def load_price_bars(csv_path: str) -> Dict[str, List[PriceBar]]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    by_symbol: Dict[str, List[PriceBar]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV must include columns: symbol,date,open,high,low,close,volume"
            )

        for row in reader:
            bar = PriceBar(
                symbol=row["symbol"].strip().upper(),
                day=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            by_symbol[bar.symbol].append(bar)

    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda x: x.day)

    return dict(by_symbol)


def load_symbol_profiles(csv_path: str) -> Dict[str, SymbolProfile]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile CSV not found: {csv_path}")

    profiles: Dict[str, SymbolProfile] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"symbol", "company", "sector", "theme"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Profile CSV must include: symbol,company,sector,theme")
        for row in reader:
            symbol = row["symbol"].strip().upper()
            theme = row["theme"].strip().upper()
            profiles[symbol] = SymbolProfile(
                symbol=symbol,
                company=row["company"].strip(),
                sector=row["sector"].strip(),
                theme=theme,
            )
    return profiles


def load_fundamentals(csv_path: str) -> Dict[str, FundamentalSnapshot]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Fundamentals CSV not found: {csv_path}")

    fundamentals: Dict[str, FundamentalSnapshot] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "symbol",
            "pe_ratio",
            "revenue_growth",
            "eps_growth",
            "debt_to_equity",
            "fcf_margin",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Fundamentals CSV must include: "
                "symbol,pe_ratio,revenue_growth,eps_growth,debt_to_equity,fcf_margin"
            )
        for row in reader:
            symbol = row["symbol"].strip().upper()
            fundamentals[symbol] = FundamentalSnapshot(
                symbol=symbol,
                pe_ratio=float(row["pe_ratio"]),
                sales_millions=_float_field(row, "sales_millions"),
                revenue_growth=float(row["revenue_growth"]),
                eps_growth=float(row["eps_growth"]),
                debt_to_equity=float(row["debt_to_equity"]),
                gross_margin=_float_field(row, "gross_margin"),
                operating_margin=_float_field(row, "operating_margin"),
                fcf_margin=float(row["fcf_margin"]),
                industry_pe=_float_field(row, "industry_pe"),
                industry_revenue_growth=_float_field(row, "industry_revenue_growth"),
                industry_fcf_margin=_float_field(row, "industry_fcf_margin"),
            )
    return fundamentals


def _float_field(row: dict, field: str) -> float:
    value = row.get(field)
    if value is None or value == "":
        return OPTIONAL_FUNDAMENTAL_DEFAULTS[field]
    return float(value)


def fill_missing_price_bars(
    bars_by_symbol: Dict[str, List[PriceBar]],
    profiles: Dict[str, SymbolProfile],
    fundamentals: Dict[str, FundamentalSnapshot],
    end_day: date | None = None,
) -> Dict[str, List[PriceBar]]:
    """Build deterministic sample history for symbols without CSV price rows."""
    output = {symbol: list(bars) for symbol, bars in bars_by_symbol.items()}
    end = end_day or date(2026, 4, 21)
    for symbol, profile in profiles.items():
        if symbol in output or symbol not in fundamentals:
            continue
        output[symbol] = _generated_price_bars(symbol, profile, fundamentals[symbol], end)
    return output


def _generated_price_bars(
    symbol: str,
    profile: SymbolProfile,
    fundamentals: FundamentalSnapshot,
    end_day: date,
) -> List[PriceBar]:
    seed = sum(ord(ch) for ch in symbol)
    base = 12 + (seed % 170)
    if profile.sector in {"Semiconductors", "Cloud Platforms"}:
        base += 80
    if profile.theme == "QUANTUM":
        base = max(2, base / 7)
    if profile.theme == "AI" and fundamentals.sales_millions > 50000:
        base += 120

    trend = (
        0.002
        + fundamentals.revenue_growth * 0.035
        + fundamentals.fcf_margin * 0.02
        - max(0.0, fundamentals.debt_to_equity - 0.7) * 0.008
    )
    if fundamentals.operating_margin < 0:
        trend += fundamentals.operating_margin * 0.015
    trend = max(-0.035, min(0.04, trend))

    bars: List[PriceBar] = []
    for index in range(24):
        day = end_day - timedelta(days=23 - index)
        close = max(0.25, base * (1 + trend) ** index)
        wobble = ((seed + index * 7) % 9 - 4) / 1000
        close = close * (1 + wobble)
        open_price = close * (1 - trend / 2)
        high = max(open_price, close) * 1.015
        low = min(open_price, close) * 0.985
        volume = 1_000_000 + (seed % 43) * 250_000 + index * 35_000
        bars.append(
            PriceBar(
                symbol=symbol,
                day=day,
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=float(volume),
            )
        )
    return bars
