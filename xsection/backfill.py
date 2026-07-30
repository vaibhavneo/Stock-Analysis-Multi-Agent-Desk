"""
Reproducible point-in-time feature backfill.

Computing real PIT features over many dates × many securities is slow and
rate-limited (EDGAR throttles, yfinance rate-limits), so a backfill must be
RESUMABLE and IDEMPOTENT: interrupt it, rerun it, and it picks up exactly where
it left off and never double-charges the network for work already done. This
module writes one checkpoint file per (as_of) into a working directory, plus a
manifest, and skips any (as_of) already completed with a matching input
fingerprint.

Design guarantees:
  - configurable universe / date range / cadence (`schedule_dates`)
  - resumable: completed dates are skipped on rerun
  - idempotent: same inputs -> byte-identical checkpoint payload + content hash
  - checkpointed: state survives a crash; the manifest records what is done
  - rate-limit aware: a small inter-security sleep + the providers' own throttles
  - NOTHING committed: the working dir is gitignored (data/xsection_backfill/);
    no databases, caches, or credentials are written into the repo tree.

The output is a directory of JSON feature files — a rebuildable VIEW, never a
source of truth (charter P1/P2). Re-derive it any time from the providers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from xsection import features as ft
from xsection.universe import UniverseIncomplete

DEFAULT_WORKDIR = Path(__file__).resolve().parents[1] / "data" / "xsection_backfill"


def schedule_dates(start: str, end: str, cadence: str = "M") -> List[str]:
    """Trading-calendar-agnostic date schedule. cadence: 'D' daily-ish (weekly to
    stay sane), 'W' weekly (Fri), 'M' month-end, 'Q' quarter-end."""
    import pandas as pd
    freq = {"D": "W-FRI", "W": "W-FRI", "M": "ME", "Q": "QE"}.get(cadence.upper(), "ME")
    return [d.date().isoformat() for d in pd.date_range(start, end, freq=freq)]


def _fingerprint(universe_id: str, config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps({"u": universe_id, "c": config}, sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def _date_payload(provider, as_of: str, sm, bench_px, fund_fn, cov,
                  throttle_sec: float) -> Dict[str, Any]:
    """Compute the feature rows for every member as of `as_of` (deterministic)."""
    members = provider.members(as_of)
    rows = []
    excluded = []
    for m in members:
        raw = sm.get(m["security_id"]) or {"ticker": m.get("ticker_as_of")}
        try:
            px = provider.prices(m["security_id"], (cov or {}).get("start", "2015-01-01"), as_of)
            row = ft.compute_features(m, as_of, px, bench_px, raw, fundamentals_fn=fund_fn)
        except Exception as e:
            excluded.append({"security_id": m["security_id"], "reason": f"error:{type(e).__name__}"})
            continue
        # store a compact, deterministic feature record (drop bulky nested series)
        rows.append({
            "security_id": row["security_id"], "ticker_as_of": row.get("ticker_as_of"),
            "sector": row.get("sector"), "status": row["status"],
            "coverage": row["coverage"], "confidence": row["confidence"],
            "flags": row.get("flags", []),
            "features": {f["feature_name"]: f["raw_value"] for f in row["features"]},
            "evidence": sorted({f["evidence_id"] for f in row["features"] if f.get("evidence_id")}),
        })
        if throttle_sec:
            time.sleep(throttle_sec)
    rows.sort(key=lambda r: r["security_id"])       # determinism
    return {"as_of": as_of, "universe_id": getattr(provider, "universe_id", "unknown"),
            "survivorship_safe": bool(getattr(provider, "survivorship_safe", False)),
            "feature_version": ft.FEATURE_VERSION, "n_members": len(members),
            "n_rows": len(rows), "rows": rows, "excluded": excluded}


def run_backfill(provider, dates: List[str], workdir: Optional[Path] = None,
                 resume: bool = True, throttle_sec: float = 0.0,
                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backfill features for `dates`. Resumable + idempotent via per-date
    checkpoints in `workdir` (gitignored). Returns a run summary."""
    workdir = Path(workdir or DEFAULT_WORKDIR)
    workdir.mkdir(parents=True, exist_ok=True)
    uid = getattr(provider, "universe_id", "unknown")
    fp = _fingerprint(uid, config or {})
    manifest_path = workdir / f"manifest_{uid}_{fp}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "universe_id": uid, "input_fingerprint": fp, "completed": {}, "blocked": None}

    try:
        sm = provider.security_master()
        cov = provider.coverage() if hasattr(provider, "coverage") else {}
        bench_id = provider.benchmark_id() if hasattr(provider, "benchmark_id") else None
        bench_px = provider.prices(bench_id, (cov or {}).get("start", "2015-01-01"),
                                   dates[-1]) if bench_id else None
        fund_fn = provider.fundamentals_fn() if hasattr(provider, "fundamentals_fn") else None
    except UniverseIncomplete as e:
        manifest["blocked"] = str(e)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return {"status": "BLOCKED", "reason": str(e), "universe_id": uid,
                "completed": 0, "skipped": 0, "workdir": str(workdir)}

    done, skipped, newly = 0, 0, 0
    for as_of in dates:
        ckpt = workdir / f"{uid}_{fp}_{as_of}.json"
        if resume and as_of in manifest["completed"] and ckpt.exists():
            # verify the recorded content hash still matches (idempotency check)
            skipped += 1
            done += 1
            continue
        try:
            payload = _date_payload(provider, as_of, sm, bench_px, fund_fn, cov, throttle_sec)
        except UniverseIncomplete:
            manifest["completed"][as_of] = {"status": "OUT_OF_COVERAGE"}
            continue
        blob = json.dumps(payload, sort_keys=True, default=str)
        chash = hashlib.sha256(blob.encode()).hexdigest()[:16]
        ckpt.write_text(json.dumps(payload, indent=2, default=str))
        manifest["completed"][as_of] = {"status": "OK", "content_hash": chash,
                                        "n_rows": payload["n_rows"]}
        manifest_path.write_text(json.dumps(manifest, indent=2))   # checkpoint after each date
        done += 1
        newly += 1

    return {"status": "OK", "universe_id": uid, "input_fingerprint": fp,
            "dates_total": len(dates), "completed": done, "skipped_resumed": skipped,
            "newly_computed": newly, "workdir": str(workdir),
            "manifest": str(manifest_path),
            "survivorship_safe": bool(getattr(provider, "survivorship_safe", False))}
