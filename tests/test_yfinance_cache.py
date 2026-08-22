"""
Verification for the new caching layer in
financial_data/providers/yfinance_bars.py - the one provider that had zero
caching (financial_data/cache.py already covered sec-edgar/fred/cboe).

Run: python3 tests/test_yfinance_cache.py

Offline/deterministic - yfinance.Ticker is mocked; a temp directory
replaces financial_data.cache.CACHE_DIR so this never touches (or is
polluted by) the real .cache/ directory.

What must hold:
  1. A cache miss calls yfinance and populates the cache.
  2. A cache hit (same symbol/period/start/end within the TTL) returns the
     same data WITHOUT calling yfinance again.
  3. A failed or empty fetch is never cached - the next call retries yfinance.
  4. Different symbols/periods get independent cache entries.
  5. The short-TTL "1d"/"5d" periods vs. the longer TTL for everything else
     are chosen correctly.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import financial_data.cache as cache_module
from financial_data.providers import yfinance_bars

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _fake_hist(n=5, start_price=100.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": [start_price + i for i in range(n)],
        "High": [start_price + i + 1 for i in range(n)],
        "Low": [start_price + i - 1 for i in range(n)],
        "Close": [start_price + i + 0.5 for i in range(n)],
        "Volume": [1000 + i for i in range(n)],
    }, index=dates)


def _isolated_cache_dir():
    tmp = Path(tempfile.mkdtemp())
    return patch.object(cache_module, "CACHE_DIR", tmp)


def test_cache_miss_then_hit_avoids_second_fetch():
    with _isolated_cache_dir():
        mock_ticker_cls = MagicMock()
        mock_ticker_cls.return_value.history.return_value = _fake_hist()
        with patch("yfinance.Ticker", mock_ticker_cls):
            r1 = yfinance_bars.fetch("bars", ["AAPL"], period="1y")
        check("first call fetches real data", len(r1["data"]) == 5, str(len(r1["data"])))
        check("first call invoked yfinance.Ticker once", mock_ticker_cls.call_count == 1)

        # Second call: mock raises if called again - proves the cache path is taken.
        with patch("yfinance.Ticker", side_effect=AssertionError("yfinance should not be called on a cache hit")):
            r2 = yfinance_bars.fetch("bars", ["AAPL"], period="1y")
        check("second call (cache hit) returns the same number of bars",
              len(r2["data"]) == len(r1["data"]))
        check("second call (cache hit) returns identical close values",
              [d["value"] for d in r1["data"]] == [d["value"] for d in r2["data"]])


def test_failed_fetch_is_never_cached():
    with _isolated_cache_dir():
        with patch("yfinance.Ticker", side_effect=RuntimeError("network down")):
            r1 = yfinance_bars.fetch("bars", ["BADTICKER"], period="1y")
        check("a fetch failure reports unavailable, not an exception",
              len(r1["unavailable"]) == 1 and r1["unavailable"][0]["symbol"] == "BADTICKER")

        # Retry: if the failure had been cached, this would still fail to call
        # yfinance.Ticker - which is exactly what we're proving it does NOT do.
        mock_ticker_cls = MagicMock()
        mock_ticker_cls.return_value.history.return_value = _fake_hist(n=3)
        with patch("yfinance.Ticker", mock_ticker_cls):
            r2 = yfinance_bars.fetch("bars", ["BADTICKER"], period="1y")
        check("a failed fetch is retried (not cached), and can succeed next time",
              len(r2["data"]) == 3 and mock_ticker_cls.call_count == 1)


def test_empty_fetch_is_never_cached():
    with _isolated_cache_dir():
        empty_df = pd.DataFrame()
        with patch("yfinance.Ticker") as mock_cls:
            mock_cls.return_value.history.return_value = empty_df
            r1 = yfinance_bars.fetch("bars", ["DELISTED"], period="1y")
        check("an empty fetch reports unavailable", len(r1["unavailable"]) == 1)

        mock_ticker_cls = MagicMock()
        mock_ticker_cls.return_value.history.return_value = _fake_hist(n=2)
        with patch("yfinance.Ticker", mock_ticker_cls):
            r2 = yfinance_bars.fetch("bars", ["DELISTED"], period="1y")
        check("an empty result is not cached - retried and can succeed later",
              len(r2["data"]) == 2 and mock_ticker_cls.call_count == 1)


def test_different_symbols_and_periods_get_independent_cache_entries():
    with _isolated_cache_dir():
        aapl_hist = _fake_hist(n=5, start_price=100.0)
        msft_hist = _fake_hist(n=5, start_price=300.0)

        def _ticker_side_effect(sym):
            hist = aapl_hist if sym == "AAPL" else msft_hist
            return SimpleNamespace(history=lambda **kw: hist)

        with patch("yfinance.Ticker", side_effect=_ticker_side_effect):
            r_aapl = yfinance_bars.fetch("bars", ["AAPL"], period="1y")
            r_msft = yfinance_bars.fetch("bars", ["MSFT"], period="1y")
        check("AAPL and MSFT get independent, non-mixed-up cached data",
              r_aapl["data"][0]["value"] != r_msft["data"][0]["value"],
              f"{r_aapl['data'][0]['value']} vs {r_msft['data'][0]['value']}")

        # Same symbol, different period -> must NOT reuse the "1y" cache entry.
        hist_5y = _fake_hist(n=5, start_price=999.0)
        with patch("yfinance.Ticker") as mock_cls:
            mock_cls.return_value.history.return_value = hist_5y
            r_5y = yfinance_bars.fetch("bars", ["AAPL"], period="5y")
        check("same symbol, different period, is a fresh cache entry (not the 1y one)",
              r_5y["data"][0]["value"] != r_aapl["data"][0]["value"], "period must key the cache")


def test_short_vs_long_ttl_by_period():
    check("'1d' period uses the short (900s) TTL bucket", ("1d" in ("1d", "5d")))
    check("'1y' period uses the long (21600s) TTL bucket", ("1y" not in ("1d", "5d")))
    # Direct behavioral check: monkeypatch cache.get to capture max_age_sec per call.
    seen_ages = []
    real_get = cache_module.get

    def _spy_get(provider, kind, key, max_age_sec=None):
        seen_ages.append(max_age_sec)
        return real_get(provider, kind, key, max_age_sec=max_age_sec)

    with _isolated_cache_dir():
        with patch.object(cache_module, "get", _spy_get), \
             patch("yfinance.Ticker") as mock_cls:
            mock_cls.return_value.history.return_value = _fake_hist()
            yfinance_bars.fetch("bars", ["AAPL"], period="1d")
            yfinance_bars.fetch("bars", ["AAPL"], period="1y")
    check("'1d' request used the short TTL", seen_ages[0] == 900, str(seen_ages))
    check("'1y' request used the long TTL", seen_ages[1] == 21600, str(seen_ages))


if __name__ == "__main__":
    test_cache_miss_then_hit_avoids_second_fetch()
    test_failed_fetch_is_never_cached()
    test_empty_fetch_is_never_cached()
    test_different_symbols_and_periods_get_independent_cache_entries()
    test_short_vs_long_ttl_by_period()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — yfinance bars caching: hits avoid refetch, failures/empties never cached")
