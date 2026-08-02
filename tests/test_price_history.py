import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import web.app as app_module


def _make_df():
    idx = pd.date_range("2026-01-02", periods=6, freq="B")
    return pd.DataFrame({
        "Open":   [100, 101, 99, 103, 104, 108],
        "High":   [102, 103, 101, 105, 107, 110],
        "Low":    [99,  100, 97,  102, 103, 106],
        "Close":  [101, 100, 103, 104, 106, 109],
        "Volume": [1000, 1100, 900, 1200, 1300, 1250],
    }, index=idx)


class TestPriceHistoryEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def test_missing_ticker(self):
        r = self.client.get("/api/price-history?period=3mo")
        self.assertEqual(r.status_code, 400)

    def test_unsupported_period(self):
        r = self.client.get("/api/price-history?ticker=AMD&period=17mo")
        self.assertEqual(r.status_code, 400)

    def test_invalid_ticker_returns_404(self):
        with patch("tools.market_data.fetch_price_history", side_effect=ValueError("no data")):
            r = self.client.get("/api/price-history?ticker=ZZZZZZ&period=3mo")
        self.assertEqual(r.status_code, 404)

    def test_provider_failure_returns_502(self):
        with patch("tools.market_data.fetch_price_history", side_effect=RuntimeError("network down")):
            r = self.client.get("/api/price-history?ticker=AMD&period=3mo")
        self.assertEqual(r.status_code, 502)

    def test_change_percent_formula_and_ordering(self):
        df = _make_df()
        with patch("tools.market_data.fetch_price_history", return_value=df):
            r = self.client.get("/api/price-history?ticker=AMD&period=3mo")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        pts = body["points"]
        timestamps = [p["timestamp"] for p in pts]
        self.assertEqual(timestamps, sorted(timestamps))
        expected_change_pct = (109 - 101) / 101 * 100
        self.assertAlmostEqual(body["summary"]["change_percent"], expected_change_pct, places=2)
        self.assertAlmostEqual(body["summary"]["start_price"], 101.0, places=2)
        self.assertAlmostEqual(body["summary"]["latest_price"], 109.0, places=2)
        self.assertAlmostEqual(body["summary"]["period_high"], 110.0, places=2)
        self.assertAlmostEqual(body["summary"]["period_low"], 97.0, places=2)

    def test_max_drawdown_is_negative_or_zero(self):
        df = _make_df()
        with patch("tools.market_data.fetch_price_history", return_value=df):
            r = self.client.get("/api/price-history?ticker=AMD&period=3mo")
        body = r.get_json()
        self.assertLessEqual(body["summary"]["maximum_drawdown_percent"], 0)


if __name__ == "__main__":
    unittest.main()
