# Real-Data Cross-Sectional Ranking Activation Report

**Date:** 2026-07-21 · **Pilot:** 94 securities, 60 monthly dates (Jul 2019 – Jun 2024), 10 GICS sectors.

## Input Audit

| Input | Classification | Source |
|---|---|---|
| Equity prices | **REAL_PIT** | yfinance auto_adjust (corp-action adjusted, truncated at as_of) |
| Fundamentals | **REAL_PIT** | SEC EDGAR companyfacts (filed-date governed, YTD→quarter differenced) |
| Benchmark (SPY) | **REAL_PIT** | yfinance |
| Universe membership | **REAL_REVISED** | Curated current-ticker watchlist — NOT survivorship-safe |
| Delisting returns | UNAVAILABLE | No delisted securities in watchlist |
| Analyst estimates | UNAVAILABLE | No free PIT estimate history |

## Pilot Results

| Metric | Value |
|---|---|
| Securities ranked per date | 94 (0 excluded) |
| Feature coverage (mean) | 77.8% |
| Missing data rate | 22.2% (fundamentals untagged in EDGAR for some names) |
| Sectors represented | 10 |
| Mean rank IC (@20d) | 0.0106 |
| IC information ratio | 0.053 |
| L/S net mean (@20d, 10bps cost) | +0.17%/month |
| L/S net annualized Sharpe | 0.10 |
| Deflated Sharpe ratio | 0.77 |
| Avg top-decile turnover | 38.4% |
| Reproducible | Yes (fingerprint `d21c00b00285e3a6` matches across re-runs) |
| Top 5 (Jun 2024) | NFLX 0.76, NVDA 0.75, CL 0.69, LLY 0.68, BAC 0.66 |
| Bottom 5 (Jun 2024) | CHTR 0.06, INTC 0.06, PPG 0.04, SBUX 0.03, NUE 0.03 |

**252-day horizon (Jun 2024):** top decile +36.9%, bottom decile +11.2%, L/S +25.6%, IC 0.25.

## Survivorship Status: NOT SAFE

This pilot uses today's tickers applied historically. Delisted companies are absent — the exact bias this engine exists to prevent. **All metrics above are inflated.** The Sharadar provider (`NASDAQ_DATA_LINK_API_KEY`) is integration-ready and STOPS with `UNIVERSE_INCOMPLETE` until provisioned; it adds delisted names, ticker history, and membership dates. No current-constituent substitute is fabricated.

## What Changed

- `ranking.run_ranking()` + `evaluate.evaluate_schedule()` accept optional `provider=` (backward-compatible; existing callers unchanged).
- `xsection/providers/production_pilot.py`: 94-ticker provider with price caching, input audit, and pilot runner.
- `universe.py`: registered `production-pilot` provider.
- Tests: 14 new offline checks (59 total in the production suite); all 16 suites green (494 checks).

## Constraint Compliance

No synthetic values outside tests. No history reconstruction from today's constituents (labelled REAL_REVISED). No current estimates substituted. No factors, agents, UI features, or weight tuning added. Survivorship safety not claimed.
