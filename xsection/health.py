"""
Dataset health report for a UniverseProvider.

Before you trust a ranking, you must know what the underlying data actually
covers — silent gaps are how survivorship and look-ahead bias creep back in.
This module probes any provider across a schedule of dates and reports, without
guessing or hiding anything:

  - membership coverage   : which dates yield a point-in-time universe, and size
  - delisting coverage    : share of names carrying delisting metadata (the proof
                            that removed securities are represented, not dropped)
  - ticker-mapping coverage : share of members that resolve to a ticker as-of
  - feature coverage      : feature-completeness + data-quality status mix from a
                            real compute_features pass (sampled)
  - stale / conflicting   : stale-fundamentals flags and ambiguous ticker maps
  - excluded securities   : counts by exclusion reason

A blocked/licensed provider reports STATUS=BLOCKED cleanly (it raises
UniverseIncomplete) instead of pretending to have data. The report is a plain
dict — the operator (and PRODUCTION_DATA_ACTIVATION_REPORT.md) read it directly.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from xsection import features as ft
from xsection.universe import UniverseIncomplete


def dataset_health(provider, dates: List[str], feature_sample: int = 8) -> Dict[str, Any]:
    """Probe `provider` across `dates`. `feature_sample` caps how many securities
    get a full (possibly network-touching) feature computation per sampled date."""
    uid = getattr(provider, "universe_id", "unknown")
    surv = bool(getattr(provider, "survivorship_safe", False))
    try:
        sm = provider.security_master()
    except UniverseIncomplete as e:
        return _blocked(uid, surv, str(e))
    except Exception as e:                       # provider needs data it doesn't have
        return _blocked(uid, surv, f"provider unavailable: {e}")

    membership = []
    delist_have, delist_total = 0, 0
    ticker_have, ticker_total = 0, 0
    conflicts = 0
    status_mix: Counter = Counter()
    excl_reasons: Counter = Counter()
    stale = 0
    feat_cov_samples: List[float] = []
    dates_ok = 0

    cov = None
    try:
        cov = provider.coverage()
    except Exception:
        cov = {}

    for as_of in dates:
        try:
            members = provider.members(as_of)
        except UniverseIncomplete:
            membership.append({"as_of": as_of, "status": "OUT_OF_COVERAGE", "n": 0})
            continue
        except Exception as e:
            membership.append({"as_of": as_of, "status": "ERROR", "n": 0, "error": str(e)[:80]})
            continue
        dates_ok += 1
        membership.append({"as_of": as_of, "status": "OK", "n": len(members),
                           "n_delisted": sum(1 for m in members
                                             if m.get("listing_status") == "delisted")})
        for m in members:
            delist_total += 1
            if m.get("listing_status") == "delisted" or m.get("delisting_date"):
                delist_have += 1
            ticker_total += 1
            if m.get("ticker_as_of"):
                ticker_have += 1
            # ticker conflict: does the as-of ticker resolve back to THIS security?
            tk = m.get("ticker_as_of")
            if tk and sm.resolve(tk, as_of) not in (m["security_id"], None):
                conflicts += 1

        # sampled real feature computation (bounded to limit network cost)
        bench_id = provider.benchmark_id() if hasattr(provider, "benchmark_id") else None
        bench_px = None
        if bench_id and cov:
            try:
                bench_px = provider.prices(bench_id, cov.get("start", "2015-01-01"), as_of)
            except Exception:
                bench_px = None
        fund_fn = provider.fundamentals_fn() if hasattr(provider, "fundamentals_fn") else None
        for m in members[:feature_sample]:
            raw = sm.get(m["security_id"]) or {"ticker": m.get("ticker_as_of")}
            try:
                px = provider.prices(m["security_id"], (cov or {}).get("start", "2015-01-01"), as_of)
                row = ft.compute_features(m, as_of, px, bench_px, raw, fundamentals_fn=fund_fn)
            except Exception as e:
                excl_reasons[f"feature_error:{type(e).__name__}"] += 1
                continue
            status_mix[row["status"]] += 1
            feat_cov_samples.append(row["coverage"])
            for fl in row.get("flags", []):
                if fl.startswith("stale_fundamentals"):
                    stale += 1
            if row["status"] in (ft.INSUFFICIENT_HISTORY, ft.EXCLUDED, ft.PARTIAL_DATA):
                excl_reasons[row["status"]] += 1

    n_dates = len(dates)
    return {
        "universe_id": uid, "status": "OK" if dates_ok else "NO_COVERAGE",
        "survivorship_safe": surv, "coverage_declared": cov,
        "dates_probed": n_dates, "dates_with_membership": dates_ok,
        "membership_coverage_pct": round(100 * dates_ok / n_dates, 1) if n_dates else 0.0,
        "membership_by_date": membership,
        "delisting_coverage_pct": round(100 * delist_have / delist_total, 1) if delist_total else 0.0,
        "delisted_member_observations": delist_have,
        "ticker_mapping_coverage_pct": round(100 * ticker_have / ticker_total, 1) if ticker_total else 0.0,
        "ticker_conflicts": conflicts,
        "feature_coverage_mean": round(sum(feat_cov_samples) / len(feat_cov_samples), 3) if feat_cov_samples else None,
        "feature_samples": len(feat_cov_samples),
        "data_quality_status_mix": dict(status_mix),
        "stale_fundamental_flags": stale,
        "excluded_by_reason": dict(excl_reasons),
        "note": ("Survivorship-safe over declared coverage." if surv
                 else "NOT survivorship-safe — do not read as a production universe."),
    }


def _blocked(uid: str, surv: bool, reason: str) -> Dict[str, Any]:
    return {"universe_id": uid, "status": "BLOCKED", "survivorship_safe": surv,
            "reason": reason, "membership_coverage_pct": 0.0,
            "note": "Provider is licensed/unavailable; no data probed, nothing fabricated."}
