"""
Probability calibration — the arrow that closes the loop.

The ledger already froze what we predicted (`horizon_probabilities_json`) and
later scored what actually happened (`prediction_outcomes.direction_correct`).
Until now nothing read that back: the system measured its own accuracy and then
forecast the next day exactly as if it never had. This module is the feedback
edge — it learns a monotonic correction from realised outcomes and applies it to
`p_up`.

WHY AT THE PROBABILITY, NOT THE WEIGHTS
    Refitting pillar weights from outcome data is precisely the weight-shopping
    backtest/pillars.py forbids ("Fitting weights is allowed ONLY through
    walk_forward_cv with one ledger.record_trial per weight vector tried").
    A probability recalibration is a different, safer object:
      · MONOTONIC — isotonic regression cannot reorder picks. If A was ranked
        above B before, it still is. Only the stated confidence moves.
      · ONE NUMBER, ONE METRIC — Brier either improves or it doesn't.
      · NO NEW SIGNAL — it corrects systematic over/under-confidence in a
        forecast the engine already made; it never invents a view.

HONEST EVALUATION (the part that's easy to get wrong)
    Isotonic regression fit and scored on the same rows ALWAYS improves Brier —
    that's overfitting, not skill. So the improvement claim here is made on
    K-fold CROSS-VALIDATED Brier: fit on k-1 folds, score the held-out fold.
    A map is only marked applicable when it beats the raw probabilities
    out-of-sample. Same discipline as PBO/dSR elsewhere in this codebase.

    Isotonic is implemented here via PAVA (Pool Adjacent Violators) rather than
    sklearn, which is not a dependency of this project — it is ~20 lines, exact,
    and deterministic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Below this many matured outcomes at a horizon, a fitted map is noise. Stated
# prior, not tuned: ~30 is the conventional floor for a stable rate estimate.
MIN_CALIBRATION_OBS = 30
# Independent (non-overlapping) evaluation windows required before a map is
# trusted. Row count is not evidence when rows overlap in time: 143 monthly
# calls graded over a year are ~15 independent observations. Stated prior.
MIN_EFFECTIVE_OBS = 20
# Folds for the out-of-sample Brier check. 5 keeps ~24 training rows at the
# MIN_CALIBRATION_OBS floor, which is thin but honest; fewer folds would make
# the held-out estimate noisier still.
CV_FOLDS = 5


# ── Isotonic regression (PAVA) ─────────────────────────────────────────────

def _pava(ys: Sequence[float], ws: Sequence[float]) -> List[float]:
    """Pool Adjacent Violators: the least-squares non-decreasing fit to `ys`.

    Walks left to right maintaining blocks of a running weighted mean; whenever
    a new block would violate monotonicity against the previous one, the two are
    merged and the check repeats backwards. O(n).
    """
    vals: List[float] = []
    wts: List[float] = []
    for y, w in zip(ys, ws):
        vals.append(float(y))
        wts.append(float(w))
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            w_new = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w_new)
            wts.append(w_new)
    out: List[float] = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(round(w)))
    return out


def fit_isotonic(pairs: Sequence[Tuple[float, float]]) -> Optional[Dict[str, List[float]]]:
    """pairs: [(predicted_p, actual_outcome_0_or_1), ...] -> knot map.

    Returns {"x": [...], "y": [...]} — a step function, monotonically
    non-decreasing in x, that maps a stated probability to the empirically
    observed frequency. None when there is nothing to fit.
    """
    clean = [(float(p), float(o)) for p, o in pairs
             if p is not None and o is not None]
    if len(clean) < 2:
        return None
    clean.sort(key=lambda t: t[0])
    xs = [p for p, _ in clean]
    ys = [o for _, o in clean]
    fitted = _pava(ys, [1.0] * len(ys))
    if len(fitted) != len(xs):        # defensive: PAVA expansion must align
        return None
    # Collapse duplicate x values (keep the last fitted value for each x).
    knots_x: List[float] = []
    knots_y: List[float] = []
    for x, y in zip(xs, fitted):
        if knots_x and abs(x - knots_x[-1]) < 1e-12:
            knots_y[-1] = y
        else:
            knots_x.append(x)
            knots_y.append(y)
    return {"x": knots_x, "y": knots_y}


def apply_isotonic(p: float, cal_map: Optional[Dict[str, List[float]]]) -> float:
    """Map a raw probability through the fitted curve (linear interpolation
    between knots, clamped at the ends). Identity when there is no map."""
    if not cal_map or not cal_map.get("x"):
        return float(p)
    xs, ys = cal_map["x"], cal_map["y"]
    p = float(p)
    if p <= xs[0]:
        return float(ys[0])
    if p >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if p <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return float(y1)
            t = (p - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
    return float(ys[-1])


# ── Scoring ────────────────────────────────────────────────────────────────

def brier(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Mean squared error between stated probability and realised 0/1."""
    vals = [(float(p) - float(o)) ** 2 for p, o in pairs
            if p is not None and o is not None]
    return sum(vals) / len(vals) if vals else None


def cross_validated_brier(pairs: Sequence[Tuple[float, float]],
                          folds: int = CV_FOLDS) -> Optional[Dict[str, float]]:
    """Out-of-sample Brier, WITHOUT time-awareness. Correct only when rows are
    genuinely independent — kept for pure-function testing and for callers with
    no dates. Ledger-backed fitting uses purged_cv_brier() instead; see the
    warning in that function about why this one leaks on real data.
    """
    clean = [(float(p), float(o)) for p, o in pairs
             if p is not None and o is not None]
    if len(clean) < folds * 2:
        return None
    clean.sort(key=lambda t: t[0])

    raw_sq: List[float] = []
    cal_sq: List[float] = []
    for k in range(folds):
        test = [row for i, row in enumerate(clean) if i % folds == k]
        train = [row for i, row in enumerate(clean) if i % folds != k]
        if not test or len(train) < 2:
            continue
        cal_map = fit_isotonic(train)
        for p, o in test:
            raw_sq.append((p - o) ** 2)
            cal_sq.append((apply_isotonic(p, cal_map) - o) ** 2)
    if not raw_sq:
        return None
    return {"brier_raw": sum(raw_sq) / len(raw_sq),
            "brier_calibrated": sum(cal_sq) / len(cal_sq),
            "n": len(raw_sq)}


def effective_sample_size(dates: Sequence[str], horizon_days: int) -> int:
    """How many genuinely independent observations a set of calls represents.

    Two calls made a month apart but each graded over the following year share
    eleven twelfths of their price path — they are very nearly the same
    observation counted twice. Row count therefore massively overstates
    evidence at long horizons, which is precisely where an unpurged CV will
    manufacture an improvement that is not there.

    Two independent ceilings, and the answer is the smaller:

      SPAN     calendar span / horizon — how many non-overlapping evaluation
               windows the observation period can physically contain.
      MOMENTS  the number of distinct call DATES — you cannot have more
               independent observations than moments of observation.

    The second ceiling is not decorative. Span alone reported 666 independent
    windows at a 1-day horizon from a ledger holding only 20 distinct call
    dates, because it measured the calendar rather than the sampling. That
    inflates the gate exactly where it is closest to admitting a map, which is
    the failure this function exists to prevent.

    Still generous even so: it ignores cross-sectional correlation, so ten
    large-caps sampled on one day count as one moment, not one tenth of one.
    Generous is acceptable; wrong by an order of magnitude is not.
    """
    clean = sorted({d[:10] for d in dates if d})
    if len(clean) < 2:
        return len(clean)
    try:
        from datetime import date
        d0 = date.fromisoformat(clean[0])
        d1 = date.fromisoformat(clean[-1])
    except (TypeError, ValueError):
        return len(clean)
    if horizon_days <= 0:
        return len(clean)
    span_trading_days = (d1 - d0).days * (252.0 / 365.0)
    span_windows = max(1, int(span_trading_days // horizon_days))
    return min(span_windows, len(clean))


def purged_cv_brier(rows: Sequence[Tuple[float, float, str]], horizon_days: int,
                    folds: int = CV_FOLDS) -> Optional[Dict[str, Any]]:
    """Time-blocked, purged out-of-sample Brier. rows: [(p, outcome, as_of_date)].

    Two differences from cross_validated_brier, both load-bearing:

      BLOCKED   folds are contiguous CALENDAR blocks, not probability-rank
                slices. Shuffling by rank puts a test row's temporal neighbours
                in the training set, which is the leak.
      PURGED    training rows whose own outcome window overlaps the test block
                (within `horizon_days` on either side) are dropped, matching
                the purging backtest/validation.walk_forward_cv already does.

    Also reports effective_n so the caller can refuse a verdict computed from
    fewer independent observations than it appears to have.
    """
    clean = [(float(p), float(o), str(d)[:10]) for p, o, d in rows
             if p is not None and o is not None and d]
    if len(clean) < folds * 2:
        return None
    clean.sort(key=lambda t: t[2])

    dates = sorted({r[2] for r in clean})
    if len(dates) < folds:
        folds = max(2, len(dates))
    # Contiguous calendar blocks.
    per = max(1, len(dates) // folds)
    blocks = [dates[i:i + per] for i in range(0, len(dates), per)][:folds]

    from datetime import date, timedelta
    embargo = timedelta(days=int(horizon_days * 365.0 / 252.0))

    raw_sq: List[float] = []
    cal_sq: List[float] = []
    for block in blocks:
        if not block:
            continue
        try:
            lo = date.fromisoformat(block[0]) - embargo
            hi = date.fromisoformat(block[-1]) + embargo
        except (TypeError, ValueError):
            continue
        test = [r for r in clean if r[2] in set(block)]
        train = []
        for r in clean:
            try:
                d = date.fromisoformat(r[2])
            except (TypeError, ValueError):
                continue
            if d < lo or d > hi:          # outside the embargoed window
                train.append((r[0], r[1]))
        if not test or len(train) < 2:
            continue
        cal_map = fit_isotonic(train)
        for p, o, _ in test:
            raw_sq.append((p - o) ** 2)
            cal_sq.append((apply_isotonic(p, cal_map) - o) ** 2)

    if not raw_sq:
        return None
    return {"brier_raw": sum(raw_sq) / len(raw_sq),
            "brier_calibrated": sum(cal_sq) / len(cal_sq),
            "n": len(raw_sq),
            "effective_n": effective_sample_size([r[2] for r in clean], horizon_days)}


# ── Ledger-backed fitting ──────────────────────────────────────────────────

def _pairs_for_horizon(horizon: int, source: str = "all") -> List[Tuple[float, float]]:
    """(predicted p_up, realised direction_correct) for one horizon.

    Only rows where the snapshot actually stored a per-horizon probability are
    usable — older rows predate horizon_probabilities_json and are skipped
    rather than back-filled with a blended edge_score, which would calibrate
    against a number the engine never stated.
    """
    from data import prediction_ledger as pl

    conn = pl._conn()
    try:
        rows = [dict(r) for r in conn.execute(
            f"""SELECT o.direction_correct, o.as_of_date, s.created_at,
                       s.horizon_probabilities_json
                  FROM prediction_outcomes o
                  JOIN prediction_snapshots s ON s.snapshot_id = o.snapshot_id
                 WHERE o.horizon_days=? AND o.matured=1
                       AND o.direction_correct IS NOT NULL
                       AND s.horizon_probabilities_json IS NOT NULL
                       {pl._source_where(source)}""",
            (horizon,)).fetchall()]
    finally:
        conn.close()

    pairs: List[Tuple[float, float, str]] = []
    for r in rows:
        try:
            probs = json.loads(r["horizon_probabilities_json"] or "{}")
        except (TypeError, ValueError):
            continue
        p = probs.get(str(horizon), probs.get(horizon))
        if p is None:
            continue
        pairs.append((float(p), float(r["direction_correct"]),
                      str(r["as_of_date"] or r["created_at"] or "")[:10]))
    return pairs


def fit_calibration(horizon: int, source: str = "all",
                    min_obs: int = MIN_CALIBRATION_OBS) -> Dict[str, Any]:
    """Fit (and honestly grade) a calibration map for one horizon.

    Always returns a dict describing what happened — `applied` is the only field
    the forecast path needs, and it is False unless the map beat raw
    probabilities OUT OF SAMPLE. Never raises: a calibration failure must cost
    a forecast its correction, never its existence.
    """
    result: Dict[str, Any] = {"horizon_days": horizon, "applied": False,
                              "n": 0, "map": None, "reason": None}
    try:
        pairs = _pairs_for_horizon(horizon, source)
    except Exception as e:
        result["reason"] = f"ledger_unavailable: {e}"
        return result

    result["n"] = len(pairs)
    if len(pairs) < min_obs:
        result["reason"] = f"insufficient_outcomes (<{min_obs})"
        return result

    # Time-blocked and purged - an unpurged CV manufactures an improvement at
    # long horizons, where overlapping evaluation windows make consecutive
    # calls nearly the same observation counted many times.
    cv = purged_cv_brier(pairs, horizon)
    if not cv:
        result["reason"] = "cv_unavailable"
        return result
    result["brier_raw"] = round(cv["brier_raw"], 5)
    result["brier_calibrated"] = round(cv["brier_calibrated"], 5)
    result["effective_n"] = cv["effective_n"]

    # Gate on INDEPENDENT observations, not row count. 143 monthly calls graded
    # over a year are ~15 independent windows; fitting a map on that is noise
    # fitting no matter how many rows the query returned.
    if cv["effective_n"] < MIN_EFFECTIVE_OBS:
        result["reason"] = (f"insufficient_independent_windows "
                            f"({cv['effective_n']}<{MIN_EFFECTIVE_OBS}); "
                            f"{len(pairs)} rows overlap in time")
        return result

    if cv["brier_calibrated"] >= cv["brier_raw"]:
        # The correction does not survive out-of-sample. Report it and change
        # nothing - a dead result recorded, not tuned away.
        result["reason"] = "no_out_of_sample_improvement"
        return result

    # Final map uses ALL rows (the CV above only decided WHETHER to fit one).
    # Dates are dropped here: they were needed to judge independence, not to
    # fit the curve.
    cal_map = fit_isotonic([(p, o) for p, o, *_ in pairs])
    if not cal_map:
        result["reason"] = "fit_failed"
        return result
    result["map"] = cal_map
    result["applied"] = True
    result["brier_improvement"] = round(cv["brier_raw"] - cv["brier_calibrated"], 5)
    return result


def load_calibrators(horizons: Sequence[int], source: str = "all") -> Dict[int, Dict[str, Any]]:
    """fit_calibration() for each horizon, keyed by horizon days."""
    return {int(h): fit_calibration(int(h), source=source) for h in horizons}


def calibrate(p: float, calibrator: Optional[Dict[str, Any]]) -> float:
    """Apply a fit_calibration() result to one probability.

    Identity unless that horizon's map earned its place. Kept separate from
    apply_isotonic so the `applied` gate lives in exactly one place.
    """
    if not calibrator or not calibrator.get("applied"):
        return float(p)
    return apply_isotonic(p, calibrator.get("map"))
