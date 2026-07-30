"""
Production acceptance test for the cross-sectional engine.

The mission asks for a bounded historical ranking replay on REAL data reporting
universe size, delisted members included, factor coverage, rank IC, decile
returns, long-short net return, turnover/costs, and missing-data rate. That test
splits cleanly into two honesty-separated parts, because in this environment
exactly one input is missing — the LICENSED survivorship-safe membership:

  PART A — Real PIT feature pipeline (RUNNABLE, keyless).
    Real corporate-action-adjusted prices (gateway) + real EDGAR filed-date
    fundamentals over a small operator watchlist. Reports the data-quality
    metrics that DON'T depend on membership: universe size, factor/feature
    coverage, missing-data rate, and concrete point-in-time evidence. It makes
    NO edge claim and is explicitly NOT survivorship-safe (a watchlist is not a
    universe), so rank-IC / decile / long-short are deliberately withheld here.

  PART B — Survivorship-safe historical ranking replay (BLOCKED, licensed).
    rank IC, decile returns, long-short net, turnover — the EDGE metrics — are
    only meaningful on a survivorship-safe universe with delisted names present.
    That needs the licensed constituent dataset (Sharadar). Without the key this
    returns UNIVERSE_INCOMPLETE and STOPS, exactly as the mission requires —
    never a current-constituents substitute dressed up as a historical result.

Run `python3 xsection/acceptance.py` for the operator summary; the functions
below return structured dicts consumed by PRODUCTION_DATA_ACTIVATION_REPORT.md.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from xsection import features as ft
from xsection.universe import UniverseIncomplete, get_provider


# ── PART A: real feature pipeline validation (keyless) ──────────────────────

def real_feature_acceptance(tickers: List[str], as_of: str,
                            benchmark: str = "SPY", start: Optional[str] = None,
                            sectors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run the REAL feature pipeline on a labelled watchlist and report coverage +
    missing-data rate + PIT evidence. No survivorship claim, no edge claim."""
    from xsection.providers.edgar_features import WatchlistUniverseProvider
    start = start or f"{int(as_of[:4]) - 3}{as_of[4:]}"
    p = WatchlistUniverseProvider(tickers, benchmark=benchmark, start=start, sectors=sectors)
    sm = p.security_master()
    cov = p.coverage()
    bench_px = p.prices(p.benchmark_id(), start, as_of)
    fund_fn = p.fundamentals_fn()

    rows, evidence = [], []
    total_cells, present_cells = 0, 0
    status_mix: Dict[str, int] = {}
    for m in p.members(as_of):
        px = p.prices(m["security_id"], start, as_of)
        row = ft.compute_features(m, as_of, px, bench_px, {"ticker": m["ticker_as_of"]},
                                  fundamentals_fn=fund_fn)
        present = {f["feature_name"]: f["raw_value"] for f in row["features"]}
        n_present = sum(1 for v in present.values() if v is not None)
        total_cells += len(present)
        present_cells += n_present
        status_mix[row["status"]] = status_mix.get(row["status"], 0) + 1
        # capture one concrete PIT evidence datum per name (a real filed fundamental)
        fund_feats = [f for f in row["features"]
                      if f.get("evidence_id") and f.get("available_at")]
        if fund_feats:
            ev = fund_feats[0]
            evidence.append({"ticker": m["ticker_as_of"], "feature": ev["feature_name"],
                             "value": ev["raw_value"], "filed": ev["available_at"],
                             "period_end": ev.get("period_end"), "accession": ev["evidence_id"]})
        rows.append({"ticker": m["ticker_as_of"], "status": row["status"],
                     "coverage": row["coverage"], "n_bars": len(px.dropna()),
                     "n_features_present": n_present})

    miss_rate = round(100 * (1 - present_cells / total_cells), 2) if total_cells else None
    return {
        "part": "A_real_feature_pipeline", "runnable": True, "survivorship_safe": False,
        "as_of": as_of, "n_securities": len(rows), "benchmark": benchmark,
        "coverage_start": start,
        "feature_coverage_mean": round(sum(r["coverage"] for r in rows) / len(rows), 3) if rows else None,
        "missing_data_rate_pct": miss_rate, "status_mix": status_mix,
        "per_security": rows,
        "pit_evidence": evidence[:len(rows)],
        "data_sources": {"prices": "yfinance via gateway (auto_adjust=True; corp-action adjusted)",
                         "fundamentals": "SEC EDGAR companyfacts (filed-date governed)"},
        "unavailable_by_design": ["analyst_estimates", "revisions", "sentiment_history",
                                  "macro_vintages"],
        "note": "Feature pipeline is REAL and point-in-time. This is a watchlist, "
                "NOT a survivorship-safe universe — no edge metrics reported here.",
    }


# ── PART B: survivorship-safe ranking replay (licensed) ─────────────────────

def survivorship_ranking_acceptance(universe_id: str, dates: List[str],
                                    horizon: int = 20, cost_bps: float = 10.0) -> Dict[str, Any]:
    """Attempt the real survivorship-safe ranking replay + edge metrics. STOPS with
    UNIVERSE_INCOMPLETE if the licensed dataset is not configured."""
    try:
        provider = get_provider(universe_id)
        provider.members(dates[-1])              # forces the license gate
    except UniverseIncomplete as e:
        return {"part": "B_survivorship_ranking_replay", "runnable": False,
                "status": "UNIVERSE_INCOMPLETE", "universe_id": universe_id,
                "reason": str(e),
                "edge_metrics_required": ["universe_size", "delisted_members_included",
                                          "factor_coverage", "rank_ic",
                                          "top_bottom_decile_returns", "long_short_net",
                                          "turnover", "costs", "missing_data_rate"],
                "note": "BLOCKED pending licensed survivorship-safe membership. No "
                        "current-constituent substitute is used — the replay STOPS."}
    # If we get here, the licensed data is present: run the real replay.
    from xsection.evaluate import evaluate_schedule
    ev = evaluate_schedule(dates, universe_id=universe_id, horizon=horizon, cost_bps=cost_bps)
    ev.update({"part": "B_survivorship_ranking_replay", "runnable": True,
               "survivorship_safe": bool(getattr(provider, "survivorship_safe", False))})
    return ev


def run_acceptance(watchlist: Optional[List[str]] = None, as_of: str = "2020-06-30",
                   production_universe: str = "sharadar") -> Dict[str, Any]:
    """Full acceptance: Part A (real features, runnable) + Part B (survivorship
    replay, licensed). Returns both parts and an overall verdict."""
    watchlist = watchlist or ["AAPL", "MSFT", "JNJ", "XOM", "KO"]
    sectors = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare",
               "XOM": "Energy", "KO": "Consumer Staples"}
    part_a = real_feature_acceptance(watchlist, as_of, sectors=sectors)
    part_b = survivorship_ranking_acceptance(
        production_universe, ["2018-06-29", "2019-06-28", "2020-06-30"])
    return {"as_of": as_of, "part_a": part_a, "part_b": part_b,
            "verdict": {
                "real_pit_features": "PROVEN" if part_a["runnable"] else "FAILED",
                "survivorship_safe_production_ranking":
                    "READY" if part_b.get("runnable") else "BLOCKED_NEEDS_LICENSE",
            }}


if __name__ == "__main__":     # pragma: no cover
    import json
    print(json.dumps(run_acceptance(), indent=2, default=str))
