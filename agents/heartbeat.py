"""
Daily heartbeat — the loop that turns an analysis tool into a learning system.

Nothing in this repo previously ran on a clock. Every capability for learning
existed (frozen predictions, matured outcomes, purged calibration) and none of
it accumulated, because accumulation requires something to run whether or not a
person opens the dashboard. This is that something.

One cycle, in this order — the order matters:

    1. GRADE    refresh_outcomes() first, so today's calibration fit sees
                everything that matured overnight rather than yesterday's view.
    2. FIT      re-fit calibration per horizon (purged, effective-N gated).
                Reported, not forced: no horizon is used unless it earns it.
    3. FORECAST a point-in-time snapshot per ticker -> pillars -> p_up at every
                ledger horizon.
    4. FREEZE   store the prediction, content-hashed and immutable.

NO LLM CALLS. Deliberately: the forecast path is deterministic arithmetic, so a
daily run over hundreds of tickers costs nothing but time and is reproducible.
Narration is a separate, opt-in concern (agents.stock_agents.run_decision_explainer).

WHAT GETS STORED, AND WHY IT IS THE RAW NUMBER
    horizon_probabilities stores the UNCALIBRATED p_up. Calibration must be
    fitted against the engine's own uncorrected output, or each day's fit would
    be learning from the previous day's correction and compound its own error.
    The calibrated number is returned for the reader in `calibrated_p_up`; the
    graded number is raw. This is the difference between measuring the engine
    and measuring the engine's corrections.

WHERE INDEPENDENT SAMPLES ACTUALLY COME FROM
    A daily cadence does NOT manufacture independence at long horizons: 252
    daily calls graded over the following year are still ~1 independent window.
    Only calendar time and cross-sectional breadth help, and breadth is limited
    by correlation (ten mega-caps on one day are not ten independent draws).
    So the short horizons (1d, 5d) are where evidence accrues fastest, and they
    are where calibration will legitimately activate first. Running this daily
    over a WIDE universe is the lever; running it more often than daily is not.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from data import prediction_ledger as pl
from data.prediction_ledger import HORIZONS


def _default_recommend(ticker: str):
    """Keyless deterministic recommendation + price history."""
    from web.app import _build_full_recommendation
    return _build_full_recommendation(ticker)


def already_frozen(ticker: str, day: str) -> bool:
    """Is there already a prediction for this ticker on this day?

    Day-stamping alone does NOT make the freeze idempotent: freeze_prediction
    hashes price_at_call too, and the price moves intraday, so two runs during
    market hours produce different hashes and two rows. Verified against the
    real ledger — a re-run added rows even with the day stamp pinned.

    That matters beyond tidiness. Two calls on the same ticker the same day are
    very nearly the same observation, and letting them both into the ledger
    inflates row counts while adding almost no independent evidence — exactly
    the error the effective-N gate exists to catch, reintroduced one level up.
    """
    try:
        conn = pl._conn()
        try:
            row = conn.execute(
                """SELECT 1 FROM prediction_snapshots
                    WHERE ticker=? AND substr(created_at,1,10)=? LIMIT 1""",
                (ticker.upper(), day[:10])).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False        # can't tell -> proceed; a duplicate beats a gap


def forecast_and_freeze(ticker: str,
                        recommend_fn: Optional[Callable[[str], Any]] = None,
                        calibrators: Optional[Dict[int, Dict[str, Any]]] = None,
                        as_of: Optional[str] = None,
                        force: bool = False) -> Dict[str, Any]:
    """One ticker: snapshot -> pillars -> forecast -> freeze.

    Returns a per-ticker record; never raises, because one bad ticker must not
    end a universe-wide run. Skips a ticker already predicted today unless
    `force` — see already_frozen().
    """
    from intelligence.prediction_engine import forecast_horizons

    day = (as_of or datetime.now().strftime("%Y-%m-%d"))[:10]
    if not force and already_frozen(ticker, day):
        return {"ticker": ticker, "status": "skipped",
                "reason": f"already_predicted_on_{day}"}

    try:
        rec, df = (recommend_fn or _default_recommend)(ticker)
    except Exception as e:
        return {"ticker": ticker, "status": "error",
                "reason": f"recommend:{str(e)[:80]}"}

    try:
        price = rec.get("current_price")
        if price is None and df is not None and len(df):
            price = float(df["Close"].iloc[-1])

        # RAW forecast — this is what gets frozen and later graded.
        raw = forecast_horizons(
            ticker, price, rec.get("pillars", {}), rec.get("algo_signals", {}) or {},
            atr_14=(rec.get("levels") or {}).get("atr_14"),
            horizons=tuple(HORIZONS),
            calibrators=None,
        )
        probs = {h["horizon_days"]: h["p_up"]
                 for h in (raw.get("horizons") or {}).values()}
        if not probs:
            return {"ticker": ticker, "status": "skipped",
                    "reason": "no_forecast (insufficient pillar data or price)"}

        rec["horizon_probabilities"] = probs
        # Stamp the prediction to the DAY, not the moment. freeze_prediction
        # content-hashes created_at, so a wall-clock timestamp would make every
        # re-run a new snapshot - a cron retry or a manual re-run would quietly
        # double-count the same call and bias calibration toward duplicated
        # observations. Day-level stamping makes the freeze genuinely idempotent
        # per (ticker, day), and matches what replay.py already does.
        rec["generated_at"] = f"{day}T00:00:00"

        snapshot_id = pl.freeze_prediction(rec)

        # Calibrated view for the reader only — never the graded number.
        calibrated = None
        if calibrators:
            try:
                from intelligence.calibration import calibrate
                calibrated = {h: round(calibrate(p, calibrators.get(h)), 3)
                              for h, p in probs.items()}
            except Exception:
                calibrated = None

        return {"ticker": ticker, "status": "done", "snapshot_id": snapshot_id,
                "action": rec.get("action"), "composite": rec.get("composite"),
                "price": price, "p_up": probs, "calibrated_p_up": calibrated}
    except Exception as e:
        return {"ticker": ticker, "status": "error", "reason": f"forecast:{str(e)[:80]}"}


def run_daily(tickers: Sequence[str],
              recommend_fn: Optional[Callable[[str], Any]] = None,
              grade: bool = True,
              refit: bool = True,
              as_of: Optional[str] = None,
              force: bool = False,
              progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """One full heartbeat cycle. Idempotent per (ticker, day) via an explicit
    already_frozen() guard — NOT via content hashing, which is not sufficient
    here because price_at_call moves intraday (see already_frozen). `force`
    overrides the guard. Verified against the real ledger by
    test_rerunning_the_same_day_does_not_duplicate.

    grade/refit exist so a test (or a backfill) can exercise the forecast half
    without touching the ledger's outcome tables.
    """
    t0 = time.time()
    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    out: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of,
        "tickers": len(tickers),
    }

    # 1. GRADE what matured overnight, BEFORE fitting.
    if grade:
        graded: Dict[str, Any] = {}
        for t in tickers:
            try:
                graded[t] = pl.refresh_outcomes(ticker=t)
            except Exception as e:
                graded[t] = {"error": str(e)[:80]}
        out["graded"] = {
            "tickers": len(graded),
            "matured": sum((g or {}).get("matured", 0) or 0
                           for g in graded.values() if isinstance(g, dict)),
        }

    # 2. FIT calibration on everything now known. Reported, never forced.
    calibrators = None
    if refit:
        try:
            from intelligence.calibration import load_calibrators
            calibrators = load_calibrators(HORIZONS)
            out["calibration"] = {
                int(h): {"applied": c.get("applied"), "n": c.get("n"),
                         "effective_n": c.get("effective_n"),
                         "reason": c.get("reason"),
                         "brier_improvement": c.get("brier_improvement")}
                for h, c in calibrators.items()
            }
            out["calibration_active_horizons"] = sorted(
                int(h) for h, c in calibrators.items() if c.get("applied"))
        except Exception as e:
            out["calibration"] = {"error": str(e)[:120]}

    # 3-4. FORECAST + FREEZE per ticker.
    results: List[Dict[str, Any]] = []
    for t in tickers:
        r = forecast_and_freeze(t, recommend_fn=recommend_fn,
                                calibrators=calibrators, as_of=as_of, force=force)
        results.append(r)
        if progress_cb:
            progress_cb(r)

    out["results"] = results
    out["frozen"] = sum(1 for r in results if r["status"] == "done")
    out["skipped"] = sum(1 for r in results if r["status"] == "skipped")
    out["errors"] = sum(1 for r in results if r["status"] == "error")
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def independence_report(horizons: Sequence[int] = HORIZONS) -> Dict[int, Dict[str, Any]]:
    """How much genuinely independent evidence exists per horizon, and how far
    it is from the gate.

    This is the number to watch while the flywheel spins up — row counts will
    climb every day and mean very little at long horizons, while effective_n
    is what actually decides whether calibration can ever activate.
    """
    from intelligence.calibration import (MIN_EFFECTIVE_OBS, _pairs_for_horizon,
                                          effective_sample_size)
    report: Dict[int, Dict[str, Any]] = {}
    for h in horizons:
        try:
            pairs = _pairs_for_horizon(int(h))
        except Exception:
            report[int(h)] = {"rows": 0, "effective_n": 0, "error": True}
            continue
        eff = effective_sample_size([d for _, _, d in pairs], int(h)) if pairs else 0
        report[int(h)] = {
            "rows": len(pairs),
            "effective_n": eff,
            "needed": MIN_EFFECTIVE_OBS,
            "gate_met": eff >= MIN_EFFECTIVE_OBS,
            "shortfall": max(0, MIN_EFFECTIVE_OBS - eff),
        }
    return report
