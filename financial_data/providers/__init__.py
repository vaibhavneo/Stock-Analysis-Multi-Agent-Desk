"""FIL providers. Each module exposes fetch(kind, symbols, start, end, as_of,
**kw) -> {data, unavailable, warnings} and NEVER applies as_of filtering itself
— the gateway owns that guarantee so it holds identically for every provider.

Vendor-specific code is confined to this package by design (the P8 seam); a test
in tests/test_financial_data.py enforces that no new code outside it imports a
provider SDK directly.
"""
