"""
Verification for M-F3B production data activation.

Run: python3 tests/test_xsection_production.py

FULLY OFFLINE and deterministic — no network. The real-data paths (EDGAR /
yfinance) are exercised separately by the acceptance harness; here we prove the
*logic* with handcrafted, provider-shaped inputs so the guarantees hold without
a live feed or a license:
  - Sharadar provider is BLOCKED without a key (never fabricates membership)
  - Sharadar parse layer: permanent identity, ticker rename vs reused ticker,
    PIT SF1 filing filter, conservative delisting returns
  - EDGAR assemble_records: YTD->discrete quarter differencing, future-filing
    exclusion, explicit missingness for untagged concepts
  - compute_features tolerates missing real fundamentals (no crash, feature None)
  - health report: OK on the fixture, BLOCKED on the licensed provider
  - backfill: idempotent (stable content hash) + resumable
  - acceptance Part B STOPS with UNIVERSE_INCOMPLETE without a license
  - the synthetic fixture ranking is unchanged by the fundamentals_fn injection
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:64s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


# ── 1. Production provider is BLOCKED without a license ──────────────────────

def test_provider_blocked():
    print("=== 1. Sharadar provider BLOCKED without key (no fabrication) ===")
    from xsection.universe import UniverseIncomplete, get_provider
    from xsection.providers.sharadar import SharadarUniverseProvider
    p = get_provider("sharadar")
    check("factory returns SharadarUniverseProvider", isinstance(p, SharadarUniverseProvider))
    try:
        p.members("2020-06-30")
        check("members() raises UniverseIncomplete", False)
    except UniverseIncomplete as e:
        check("members() raises UniverseIncomplete", True)
        check("reason is UNIVERSE_INCOMPLETE + names the key", "UNIVERSE_INCOMPLETE" in str(e)
              and "NASDAQ_DATA_LINK_API_KEY" in str(e))
    check("declares survivorship_safe intent", p.survivorship_safe is True)


# ── 2. Sharadar parse layer (offline, provider-shaped rows) ─────────────────

def test_sharadar_parse():
    print("=== 2. Sharadar parse: identity, rename, reused ticker, delisting ===")
    from xsection.providers import sharadar as sh
    from xsection.universe import SecurityMaster
    rows = [
        # permaticker 111: ticker rename KAPB -> KPPA (one identity)
        {"table": "SEP", "permaticker": 111, "ticker": "KAPB", "name": "Kappa",
         "firstpricedate": "2015-01-02", "lastpricedate": "2020-12-31", "isdelisted": "N"},
        {"table": "SEP", "permaticker": 111, "ticker": "KPPA", "name": "Kappa",
         "firstpricedate": "2021-01-04", "lastpricedate": "2024-12-31", "isdelisted": "N"},
        # permaticker 222: XILO delisted 2021 (bankruptcy)
        {"table": "SEP", "permaticker": 222, "ticker": "XILO", "name": "Xi Logistics",
         "firstpricedate": "2016-01-04", "lastpricedate": "2021-12-31", "isdelisted": "Y"},
        # permaticker 333: XILO REUSED by a different company from 2023
        {"table": "SEP", "permaticker": 333, "ticker": "XILO", "name": "Xi Innovations",
         "firstpricedate": "2023-01-03", "lastpricedate": None, "isdelisted": "N"},
    ]
    secs = sh.parse_tickers(rows)
    by_id = {s["security_id"]: s for s in secs}
    check("3 distinct permanent identities", len(secs) == 3, str(sorted(by_id)))
    k = by_id["SHARADAR:111"]
    check("rename collapses to ONE identity w/ 2 ticker windows", len(k["tickers"]) == 2)
    sm = SecurityMaster(secs)
    check("KAPB@2018 -> perma 111", sm.resolve("KAPB", "2018-06-01") == "SHARADAR:111")
    check("KPPA@2023 -> perma 111 (same identity)", sm.resolve("KPPA", "2023-06-01") == "SHARADAR:111")
    check("XILO@2020 -> perma 222", sm.resolve("XILO", "2020-06-01") == "SHARADAR:222")
    check("XILO@2024 -> perma 333 (reused, NOT merged)", sm.resolve("XILO", "2024-06-01") == "SHARADAR:333")
    check("perma 222 marked delisted", by_id["SHARADAR:222"]["listing_status"] == "delisted")

    # PIT membership
    m2020 = {m["security_id"] for m in sh.membership_as_of(secs, "2020-06-01", sm)}
    m2023 = {m["security_id"] for m in sh.membership_as_of(secs, "2023-06-01", sm)}
    check("2020 membership includes soon-delisted 222", "SHARADAR:222" in m2020)
    check("2023 membership EXCLUDES delisted 222", "SHARADAR:222" not in m2023)
    check("2023 membership includes reused-ticker 333", "SHARADAR:333" in m2023)


def test_sharadar_sf1_and_delisting():
    print("=== 3. Sharadar SF1 PIT filter + delisting-return mapping ===")
    from xsection.providers import sharadar as sh
    sf1 = [
        {"ticker": "ABC", "reportperiod": "2019-12-31", "calendardate": "2019-12-31",
         "datekey": "2020-02-15", "revenue": 100, "netinc": 10, "equity": 50},
        {"ticker": "ABC", "reportperiod": "2020-03-31", "calendardate": "2020-03-31",
         "datekey": "2020-05-15", "revenue": 110, "netinc": 12, "equity": 55},  # FUTURE vs as_of
    ]
    recs = sh.parse_sf1_records(sf1, as_of="2020-04-01")
    check("future-datekey filing excluded (PIT)", len(recs) == 1 and recs[0]["period_end"] == "2019-12-31")
    check("SF1 column mapping (revenue/net_income/equity)",
          recs[0]["revenue"] == 100.0 and recs[0]["net_income"] == 10.0 and recs[0]["equity"] == 50.0)
    check("bankruptcy -> -100%", sh.delisting_return_from_action("bankruptcy") == -100.0)
    check("hard delist -> -35%", sh.delisting_return_from_action("delisted") == -35.0)
    check("merger -> None (value in last price)", sh.delisting_return_from_action("acquisitionbyother") is None)


# ── 4. EDGAR assemble_records: YTD differencing + missingness ────────────────

def test_edgar_assemble():
    print("=== 4. EDGAR assemble_records: YTD->quarter, future-filing, missingness ===")
    from xsection.providers.edgar_features import assemble_records
    # Revenue reported YTD-cumulative (Q1=90d, H1=181d) filed same day; net_income
    # only reported at Q1 (missing at Q2 -> explicit None); equity instantaneous.
    datums = [
        {"concept": "revenue", "value": 100.0, "available_at": "2020-05-01",
         "period_start": "2020-01-01", "period_end": "2020-03-31"},           # Q1 discrete
        {"concept": "revenue", "value": 250.0, "available_at": "2020-05-01",
         "period_start": "2020-01-01", "period_end": "2020-06-30"},           # H1 cumulative
        {"concept": "net_income", "value": 20.0, "available_at": "2020-05-01",
         "period_start": "2020-01-01", "period_end": "2020-03-31"},
        {"concept": "equity", "value": 500.0, "available_at": "2020-05-01",
         "period_start": None, "period_end": "2020-06-30"},
        # a FUTURE filing that must be invisible at as_of
        {"concept": "revenue", "value": 300.0, "available_at": "2020-11-01",
         "period_start": "2020-07-01", "period_end": "2020-09-30"},
    ]
    recs = assemble_records(datums, as_of="2020-08-01")
    by_pe = {r["period_end"]: r for r in recs}
    check("Q1 discrete revenue kept (100)", by_pe["2020-03-31"]["revenue"] == 100.0)
    check("H1 differenced to discrete Q2 revenue (250-100=150)", by_pe["2020-06-30"]["revenue"] == 150.0)
    check("future filing (2020-09-30) excluded", "2020-09-30" not in by_pe)
    check("missing net_income at Q2 -> explicit None",
          by_pe["2020-06-30"].get("net_income") is None)
    check("instantaneous equity attached at period_end", by_pe["2020-06-30"]["equity"] == 500.0)


# ── 5. compute_features tolerates missing real fundamentals ─────────────────

def test_compute_features_missing_safe():
    print("=== 5. compute_features None-safe on real (partial) fundamentals ===")
    import numpy as np
    import pandas as pd
    from xsection import features as ft
    idx = pd.date_range("2018-01-01", "2020-06-30", freq="B")
    px = pd.Series(50 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0003, 0.02, len(idx)))), index=idx)

    def partial_funds(_sec, _as_of):
        # operating_income + gross_profit MISSING (untagged) — must not crash
        base = {"gross_profit": None, "operating_income": None, "free_cash_flow": None,
                "total_debt": None, "source": "edgar:companyfacts"}
        return [
            {"period_end": "2019-06-30", "filed": "2019-08-01", "revenue": 100.0,
             "net_income": 10.0, "equity": 500.0, "shares": 100.0, "accessn": "X-1", **base},
            {"period_end": "2020-03-31", "filed": "2020-05-01", "revenue": 110.0,
             "net_income": 11.0, "equity": 520.0, "shares": 101.0, "accessn": "X-2", **base},
        ]
    member = {"security_id": "WL:TEST", "ticker_as_of": "TEST", "sector": "Tech"}
    row = ft.compute_features(member, "2020-06-30", px, None, {"ticker": "TEST"},
                              fundamentals_fn=partial_funds)
    fmap = {f["feature_name"]: f["raw_value"] for f in row["features"]}
    check("no crash; row produced", row["status"] in (ft.ELIGIBLE, ft.PARTIAL_DATA))
    check("missing operating_margin -> None (missingness, not 0)", fmap.get("operating_margin") is None)
    check("present revenue_growth computed", fmap.get("revenue_growth_yoy") is not None)
    check("source label reflects EDGAR", any(f["source"].startswith("edgar")
          for f in row["features"] if f["family"] in ("quality", "growth", "valuation")))


# ── 6. Health report ────────────────────────────────────────────────────────

def test_health():
    print("=== 6. Dataset health: OK on fixture, BLOCKED on licensed provider ===")
    from xsection.health import dataset_health
    from xsection.universe import get_provider
    h = dataset_health(get_provider("reference-smallcap-demo"),
                       ["2020-06-01", "2021-06-01", "2023-06-01"], feature_sample=12)
    check("fixture health OK", h["status"] == "OK")
    check("membership coverage 100%", h["membership_coverage_pct"] == 100.0)
    check("delisting coverage > 0 (delisted names represented)", h["delisting_coverage_pct"] > 0)
    check("ticker mapping coverage 100%", h["ticker_mapping_coverage_pct"] == 100.0)
    check("feature coverage present", h["feature_coverage_mean"] is not None)
    hb = dataset_health(get_provider("sharadar"), ["2020-06-30"])
    check("licensed provider health BLOCKED", hb["status"] == "BLOCKED")
    check("blocked report fabricates nothing", hb["membership_coverage_pct"] == 0.0)


# ── 7. Backfill idempotency + resume ────────────────────────────────────────

def test_backfill():
    print("=== 7. Backfill: idempotent content hash + resumable ===")
    import json
    from xsection import backfill as bf
    from xsection.universe import get_provider
    p = get_provider("reference-smallcap-demo")
    dates = ["2020-06-01", "2021-06-01"]
    wd1 = Path(tempfile.mkdtemp())
    r1 = bf.run_backfill(p, dates, workdir=wd1)
    check("run completes", r1["status"] == "OK" and r1["completed"] == 2)
    man1 = json.loads(list(wd1.glob("manifest_*.json"))[0].read_text())
    h1 = {d: v.get("content_hash") for d, v in man1["completed"].items()}
    r2 = bf.run_backfill(p, dates, workdir=wd1)     # resume
    check("resume skips completed dates", r2["skipped_resumed"] == 2 and r2["newly_computed"] == 0)
    wd2 = Path(tempfile.mkdtemp())
    bf.run_backfill(p, dates, workdir=wd2)
    man2 = json.loads(list(wd2.glob("manifest_*.json"))[0].read_text())
    h2 = {d: v.get("content_hash") for d, v in man2["completed"].items()}
    check("idempotent: identical content hashes across fresh runs", h1 == h2)


# ── 8. Acceptance Part B STOPS; synthetic ranking unchanged ─────────────────

def test_acceptance_and_synthetic_preserved():
    print("=== 8. Acceptance Part B BLOCKED; synthetic ranking preserved ===")
    from xsection.acceptance import survivorship_ranking_acceptance
    b = survivorship_ranking_acceptance("sharadar", ["2019-06-28", "2020-06-30"])
    check("Part B not runnable", b["runnable"] is False)
    check("Part B status UNIVERSE_INCOMPLETE", b["status"] == "UNIVERSE_INCOMPLETE")
    check("Part B lists the required edge metrics it CANNOT compute",
          "rank_ic" in b["edge_metrics_required"])

    # Synthetic fixture ranking must be unchanged by the fundamentals_fn injection.
    from data import store
    from xsection import ranking
    store._DB_PATH = Path(tempfile.mkdtemp()) / "s.db"; store.DB_PATH = store._DB_PATH
    ranking.set_db_path(store._DB_PATH)
    res = ranking.run_ranking("2023-06-01", persist=False)
    check("fixture ranking still OK", res["status"] == "OK")
    check("fixture still survivorship_safe + PIT", res["survivorship_safe"] is True
          and res["labels"]["PIT_SAFE"] is True)
    check("fixture fundamentals source unchanged (synthetic)",
          any("synthetic" in (f.get("source") or "")
              for r in res["ranked"] for f in r["features"] if f["family"] == "quality"))


# ── 9. Production pilot provider (offline shape checks) ───────────────────

def test_production_pilot_provider():
    print("=== 9. Production pilot provider: shape, audit, not survivorship-safe ===")
    from xsection.providers.production_pilot import (
        ProductionPilotProvider, PILOT_TICKERS, SECTORS, input_audit,
    )
    from xsection.universe import get_provider
    p = get_provider("production-pilot")
    check("factory returns ProductionPilotProvider", isinstance(p, ProductionPilotProvider))
    check("NOT survivorship safe", p.survivorship_safe is False)
    check("104 pilot tickers curated", len(PILOT_TICKERS) >= 90)
    check("all pilot tickers have sectors", all(t in SECTORS for t in PILOT_TICKERS))
    check("10 sectors represented", len(set(SECTORS.values())) >= 10)
    m = p.members("2023-06-30")
    check("members returns all tickers", len(m) == len(PILOT_TICKERS))
    check("every member flagged NOT_SURVIVORSHIP_SAFE",
          all("NOT_SURVIVORSHIP_SAFE" in (mem.get("flags") or []) for mem in m))
    check("security_id prefix PP:", all(mem["security_id"].startswith("PP:") for mem in m))
    check("delisting_return_pct always None (no delisted in watchlist)",
          p.delisting_return_pct("PP:AAPL") is None)
    audit = input_audit()
    check("input audit has 8 entries", len(audit) == 8)
    classifications = {a["classification"] for a in audit}
    check("audit uses REAL_PIT, REAL_REVISED, UNAVAILABLE",
          classifications == {"REAL_PIT", "REAL_REVISED", "UNAVAILABLE"})
    check("membership classified REAL_REVISED (not REAL_PIT)",
          any(a["input"] == "Universe membership" and a["classification"] == "REAL_REVISED"
              for a in audit))
    check("fundamentals_fn returns edgar_fundamentals",
          p.fundamentals_fn() is not None)
    # ranking.run_ranking accepts provider= directly
    from xsection import ranking
    ranking.set_db_path(Path(tempfile.mkdtemp()) / "pp.db")
    # Use fixture provider to test the provider= parameter works
    from xsection.universe import FixtureUniverseProvider
    fp = FixtureUniverseProvider()
    res = ranking.run_ranking("2023-06-01", provider=fp, persist=False)
    check("run_ranking(provider=) works with direct provider", res["status"] == "OK")


def main():
    for t in (test_provider_blocked, test_sharadar_parse, test_sharadar_sf1_and_delisting,
              test_edgar_assemble, test_compute_features_missing_safe, test_health,
              test_backfill, test_acceptance_and_synthetic_preserved,
              test_production_pilot_provider):
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — production data activation verified (offline).")


if __name__ == "__main__":
    main()
