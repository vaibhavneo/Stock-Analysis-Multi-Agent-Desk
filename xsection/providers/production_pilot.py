"""
Real-data production pilot for the cross-sectional ranking engine.

Runs the UNCHANGED ranking engine on ~100 real US equities with REAL prices
(yfinance, corp-action adjusted) and REAL fundamentals (SEC EDGAR, filed-date
governed). Every input is audited as REAL_PIT, REAL_REVISED, or UNAVAILABLE.

Survivorship safety: FALSE. This is a curated watchlist of currently-listed
tickers, NOT a point-in-time historical index. Delisted companies are absent.
The ranking metrics (IC, decile returns) are reported with this caveat —
survivorship bias inflates them, and the report says so. TRUE survivorship-safe
ranking requires the Sharadar licensed dataset (UNIVERSE_INCOMPLETE without it).
"""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from xsection.providers.edgar_features import edgar_fundamentals
from xsection.universe import SecurityMaster, UniverseProvider

PILOT_TICKERS = [
    # Technology (12)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ADBE", "CRM",
    "AMD", "INTC", "CSCO", "TXN",
    # Healthcare (10)
    "JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK", "TMO", "ABT", "AMGN", "BMY",
    # Financials (10)
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C", "USB", "PNC",
    # Consumer Discretionary (10)
    "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "GM", "F",
    # Consumer Staples (10)
    "PG", "KO", "PEP", "COST", "WMT", "CL", "MO", "PM", "GIS", "HSY",
    # Energy (8)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    # Industrials (10)
    "HON", "UNP", "RTX", "CAT", "DE", "GE", "BA", "LMT", "MMM", "ITW",
    # Utilities (8)
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
    # Materials (8)
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "PPG",
    # Communication (8)
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA",
]

SECTORS = {
    **{t: "Technology" for t in ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO",
                                  "ADBE", "CRM", "AMD", "INTC", "CSCO", "TXN"]},
    **{t: "Healthcare" for t in ["JNJ", "UNH", "PFE", "ABBV", "LLY", "MRK",
                                  "TMO", "ABT", "AMGN", "BMY"]},
    **{t: "Financials" for t in ["JPM", "BAC", "WFC", "GS", "MS", "BLK",
                                  "AXP", "C", "USB", "PNC"]},
    **{t: "Consumer Discretionary" for t in ["AMZN", "HD", "MCD", "NKE", "SBUX",
                                              "LOW", "TJX", "BKNG", "GM", "F"]},
    **{t: "Consumer Staples" for t in ["PG", "KO", "PEP", "COST", "WMT", "CL",
                                        "MO", "PM", "GIS", "HSY"]},
    **{t: "Energy" for t in ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"]},
    **{t: "Industrials" for t in ["HON", "UNP", "RTX", "CAT", "DE", "GE", "BA",
                                   "LMT", "MMM", "ITW"]},
    **{t: "Utilities" for t in ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL"]},
    **{t: "Materials" for t in ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "PPG"]},
    **{t: "Communication" for t in ["NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA"]},
}


def input_audit() -> List[Dict[str, str]]:
    return [
        {"input": "Equity prices", "classification": "REAL_PIT",
         "source": "yfinance (auto_adjust=True, corp-action adjusted)",
         "evidence": "Daily closes truncated at as_of; splits/dividends adjusted"},
        {"input": "Fundamental ratios", "classification": "REAL_PIT",
         "source": "SEC EDGAR companyfacts (XBRL)",
         "evidence": "filed-date governed (available_at <= as_of); YTD->quarter differenced"},
        {"input": "Benchmark (SPY)", "classification": "REAL_PIT",
         "source": "yfinance", "evidence": "Same as equity prices"},
        {"input": "Universe membership", "classification": "REAL_REVISED",
         "source": "Curated current-ticker watchlist (104 tickers)",
         "evidence": "NOT survivorship-safe; today's tickers applied historically; "
                     "delisted names ABSENT — survivorship bias inflates all metrics"},
        {"input": "Delisting returns", "classification": "UNAVAILABLE",
         "source": "No delisted securities in watchlist",
         "evidence": "Conservative returns not applicable; delisting bias unaddressed"},
        {"input": "Analyst estimates", "classification": "UNAVAILABLE",
         "source": "No free PIT estimate history",
         "evidence": "Marked UNAVAILABLE in features; never substituted"},
        {"input": "Revisions", "classification": "UNAVAILABLE",
         "source": "None", "evidence": "Same as estimates"},
        {"input": "Social sentiment", "classification": "UNAVAILABLE",
         "source": "Not in ranking path",
         "evidence": "Ranking is deterministic arithmetic (no LLM)"},
    ]


class ProductionPilotProvider(UniverseProvider):
    """Real-data pilot provider: real prices + real EDGAR fundamentals on a
    curated watchlist. Explicitly NOT survivorship-safe — every output says so."""
    universe_id = "production-pilot"
    survivorship_safe = False

    def __init__(self, tickers: Optional[List[str]] = None,
                 sectors: Optional[Dict[str, str]] = None,
                 benchmark: str = "SPY", start: str = "2018-01-01"):
        self._tickers = [t.upper() for t in (tickers or PILOT_TICKERS)]
        self._sectors = sectors or SECTORS
        self._benchmark = benchmark.upper()
        self._start = start
        secs = [{"security_id": f"PP:{t}", "name": t,
                 "tickers": [{"ticker": t, "start": None, "end": None}],
                 "sector": self._sectors.get(t), "provenance": "production_pilot"}
                for t in self._tickers]
        secs.append({"security_id": f"PP:{self._benchmark}", "name": self._benchmark,
                     "tickers": [{"ticker": self._benchmark, "start": None, "end": None}],
                     "sector": None, "provenance": "production_pilot"})
        self._sm = SecurityMaster(secs)
        self._price_cache: Dict[str, pd.Series] = {}
        self._load_failures: List[str] = []

    def coverage(self) -> Dict[str, str]:
        import datetime as dt
        return {"start": self._start, "end": dt.date.today().isoformat(),
                "status": "not_survivorship_safe"}

    def security_master(self) -> SecurityMaster:
        return self._sm

    def benchmark_id(self) -> Optional[str]:
        return f"PP:{self._benchmark}"

    def _ensure_prices(self, ticker: str) -> pd.Series:
        if ticker not in self._price_cache:
            from financial_data.gateway import get_bars_df
            try:
                df = get_bars_df(ticker, start=self._start)
                if df is not None and len(df):
                    self._price_cache[ticker] = df["Close"].dropna()
                else:
                    self._price_cache[ticker] = pd.Series(dtype=float)
                    self._load_failures.append(ticker)
            except Exception:
                self._price_cache[ticker] = pd.Series(dtype=float)
                self._load_failures.append(ticker)
        return self._price_cache[ticker]

    def preload_prices(self, progress: bool = True) -> Dict[str, Any]:
        all_tickers = self._tickers + [self._benchmark]
        n = len(all_tickers)
        t0 = time.time()
        for i, tk in enumerate(all_tickers):
            self._ensure_prices(tk)
            if progress and (i + 1) % 10 == 0:
                print(f"  prices: {i+1}/{n} loaded ({len(self._load_failures)} failures)", flush=True)
        elapsed = time.time() - t0
        if progress:
            print(f"  prices: {n}/{n} done in {elapsed:.1f}s "
                  f"({len(self._load_failures)} failures: {self._load_failures[:10]})", flush=True)
        return {"loaded": n - len(self._load_failures), "failed": len(self._load_failures),
                "failures": self._load_failures, "elapsed_sec": round(elapsed, 1)}

    def members(self, as_of: str) -> List[Dict[str, Any]]:
        out = []
        for t in self._tickers:
            out.append({"security_id": f"PP:{t}", "ticker_as_of": t, "name": t,
                        "sector": self._sectors.get(t), "industry": None,
                        "listing_status": "active", "delisting_date": None,
                        "first_tradable": None, "last_tradable": None,
                        "provenance": "production_pilot",
                        "flags": ["NOT_SURVIVORSHIP_SAFE"]})
        return out

    def prices(self, security_id: str, start: str, end: str) -> pd.Series:
        tk = security_id.split(":")[-1]
        full = self._ensure_prices(tk)
        if not len(full):
            return full
        mask = (full.index >= pd.Timestamp(start)) & (full.index <= pd.Timestamp(end))
        return full[mask]

    def delisting_return_pct(self, security_id: str) -> Optional[float]:
        return None

    def fundamentals_fn(self):
        return edgar_fundamentals


def run_pilot(start: str = "2019-07-31", end: str = "2024-06-28",
              persist: bool = False) -> Dict[str, Any]:
    """Run the bounded production pilot: ~100 securities, monthly dates, 5 years.
    Returns a structured report with all metrics the mission asks for."""
    from xsection import ranking, evaluate as ev

    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end, freq="BM")]
    provider = ProductionPilotProvider()
    n_tickers = len(provider._tickers)

    print(f"=== Production Pilot: {n_tickers} securities, {len(dates)} dates ===")
    print(f"    dates: {dates[0]} .. {dates[-1]}")

    # 1. Pre-load prices
    print("\n[1/4] Loading prices...")
    price_stats = provider.preload_prices()

    # 2. Run one sample ranking (latest date) for detail
    print("\n[2/4] Sample ranking (latest date)...")
    sample_date = dates[-1]
    sample = ranking.run_ranking(sample_date, universe_id="production-pilot",
                                 persist=persist, provider=provider)
    print(f"  {sample_date}: {sample.get('n_ranked', 0)} ranked, "
          f"{sample.get('n_excluded', 0)} excluded, status={sample['status']}")

    # 3. Run schedule evaluation (all dates, primary horizon=20d)
    print(f"\n[3/4] Evaluating {len(dates)} monthly dates (horizon=20d)...")
    t0 = time.time()
    schedule = ev.evaluate_schedule(dates, universe_id="production-pilot",
                                    horizon=20, cost_bps=10.0, provider=provider)
    eval_time = time.time() - t0
    print(f"  done in {eval_time:.1f}s — {schedule.get('dates_evaluated', 0)} dates evaluated")

    # 4. Run single-date evaluations at all horizons for the sample date
    print("\n[4/4] Multi-horizon evaluation for sample date...")
    sample_ev = ev.evaluate_ranking(sample, provider) if sample.get("status") == "OK" else {}

    # Coverage stats from the sample ranking
    if sample.get("status") == "OK":
        coverages = [r["coverage"] for r in sample["ranked"]]
        feat_present = sum(1 for f in sample["ranked"][0]["features"]
                          if f["raw_value"] is not None) if sample["ranked"] else 0
        feat_total = len(sample["ranked"][0]["features"]) if sample["ranked"] else 0
        sectors_represented = len({r.get("sector") for r in sample["ranked"]})
        missing_rate = round(100 * (1 - sum(coverages) / len(coverages)), 2) if coverages else None
    else:
        coverages, feat_present, feat_total = [], 0, 0
        sectors_represented, missing_rate = 0, None

    # Reproducibility: re-run the sample date and compare fingerprint
    print("  reproducibility check...", end=" ", flush=True)
    sample2 = ranking.run_ranking(sample_date, universe_id="production-pilot",
                                   persist=False, provider=provider)
    reproducible = (sample.get("decision_fingerprint") == sample2.get("decision_fingerprint"))
    print("MATCH" if reproducible else "MISMATCH")

    report = {
        "status": schedule.get("status", "UNKNOWN"),
        "pilot_parameters": {
            "n_tickers": n_tickers, "n_dates": len(dates),
            "date_range": f"{dates[0]} to {dates[-1]}",
            "cadence": "monthly (last business day)",
            "benchmark": "SPY", "cost_bps": 10.0,
        },
        "input_audit": input_audit(),
        "survivorship_safe": False,
        "survivorship_note": "Curated current-ticker watchlist; delisted names ABSENT; "
                             "all metrics are inflated by survivorship bias.",
        "price_loading": price_stats,
        "sample_ranking": {
            "as_of": sample_date,
            "n_members": sample.get("n_members", 0),
            "n_ranked": sample.get("n_ranked", 0),
            "n_excluded": sample.get("n_excluded", 0),
            "excluded_reasons": [e.get("reason", "") for e in sample.get("excluded", [])],
            "feature_coverage_mean": round(sum(coverages) / len(coverages), 3) if coverages else None,
            "missing_data_rate_pct": missing_rate,
            "features_present": feat_present, "features_total": feat_total,
            "sectors_represented": sectors_represented,
            "top_5": [(r["ticker_as_of"], round(r["composite_raw"], 3))
                      for r in sample.get("ranked", [])[:5]],
            "bottom_5": [(r["ticker_as_of"], round(r["composite_raw"], 3))
                         for r in sample.get("ranked", [])[-5:]],
        },
        "schedule_evaluation": {
            "dates_evaluated": schedule.get("dates_evaluated", 0),
            "mean_rank_ic": schedule.get("mean_rank_ic"),
            "ic_information_ratio": schedule.get("ic_information_ratio"),
            "long_short_net_mean_pct": schedule.get("long_short_net_mean_pct"),
            "long_short_net_annualized_sharpe": schedule.get("long_short_net_annualized_sharpe"),
            "deflated_sharpe": schedule.get("deflated_sharpe"),
            "avg_turnover": schedule.get("avg_turnover"),
            "per_date_sample": schedule.get("per_date", [])[:5],
        },
        "multi_horizon": sample_ev.get("by_horizon", {}),
        "reproducibility": {
            "reproducible": reproducible,
            "fingerprint_1": sample.get("decision_fingerprint"),
            "fingerprint_2": sample2.get("decision_fingerprint"),
        },
        "eval_time_sec": round(eval_time, 1),
    }

    print(f"\n=== Pilot complete ===")
    print(f"  ranked/date:    {sample.get('n_ranked', 0)}")
    print(f"  coverage:       {report['sample_ranking']['feature_coverage_mean']}")
    print(f"  missing rate:   {missing_rate}%")
    print(f"  IC (mean):      {schedule.get('mean_rank_ic')}")
    print(f"  L/S net (mean): {schedule.get('long_short_net_mean_pct')}%")
    print(f"  dSR:            {schedule.get('deflated_sharpe')}")
    print(f"  reproducible:   {reproducible}")

    return report


if __name__ == "__main__":
    import json
    report = run_pilot()
    print(json.dumps(report, indent=2, default=str))
