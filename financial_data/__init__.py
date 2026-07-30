"""
FIL — Financial Intelligence Layer (Milestone 1).

The public surface. Import from here, not from submodules:

    from financial_data import get, get_concept

    res = get("fundamentals_pit", "AAPL", as_of="2024-01-01")
    if not res["as_of_honored"]:
        ...        # the result carries NO point-in-time guarantee; do not backtest on it

Design in one line: every number arrives as a Datum carrying `available_at`, and
the gateway is the only thing allowed to decide what was knowable when.

Full contract: stock_agent/FIL.md
"""
from .gateway import (
    GatewayError,
    NoProviderError,
    get,
    get_bars_df,
    get_concept,
    list_providers,
    load_registry,
)
from .schemas import (
    KINDS,
    SchemaError,
    filter_pit,
    latest_by_period,
    make_datum,
    make_source,
    validate_datum,
    visible_at,
)

__all__ = [
    "get", "get_bars_df", "get_concept", "list_providers", "load_registry",
    "GatewayError", "NoProviderError",
    "make_datum", "make_source", "validate_datum", "visible_at",
    "filter_pit", "latest_by_period", "KINDS", "SchemaError",
]
