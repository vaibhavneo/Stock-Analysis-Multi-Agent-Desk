from __future__ import annotations

from typing import Set


SYMBOL_ALIASES = {
    "NVD": "NVDA",
}


def parse_symbol_filter(symbols_filter: str | None) -> Set[str]:
    if not symbols_filter:
        return set()
    symbols = {s.strip().upper() for s in symbols_filter.split(",") if s.strip()}
    return {SYMBOL_ALIASES.get(symbol, symbol) for symbol in symbols}
