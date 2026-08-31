#!/usr/bin/env python3
"""Tests for intelligence/calibration.py — the feedback edge that closes the loop.

The load-bearing property here is NOT "calibration improves Brier" (fitting and
scoring the same rows always does). It is that the module refuses to claim an
improvement it cannot demonstrate OUT OF SAMPLE, and that a correction can never
reorder picks. Those two are what make this safe to put in the decision path.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.calibration import (MIN_CALIBRATION_OBS, apply_isotonic,
                                      brier, calibrate, cross_validated_brier,
                                      fit_calibration, fit_isotonic)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _with_dates(pairs, step_days=1, start="2020-01-01"):
    """Attach call dates to (p, o) pairs. Daily spacing keeps windows
    non-overlapping at the short horizons these tests use."""
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    return [(p, o, (d0 + timedelta(days=i * step_days)).isoformat())
            for i, (p, o) in enumerate(pairs)]


# ── Isotonic core ──────────────────────────────────────────────────────────

def test_isotonic_output_is_monotonic():
    """The whole safety argument rests on this: a calibrated ranking is the
    same ranking. If the map could decrease, it could reorder picks."""
    rng = random.Random(7)
    pairs = [(rng.random(), float(rng.random() < 0.5)) for _ in range(300)]
    m = fit_isotonic(pairs)
    ys = m["y"]
    check("fitted curve is non-decreasing", all(ys[i] <= ys[i + 1] + 1e-12
                                                for i in range(len(ys) - 1)))
    probes = [i / 50 for i in range(51)]
    mapped = [apply_isotonic(p, m) for p in probes]
    check("applied curve is non-decreasing across the whole range",
          all(mapped[i] <= mapped[i + 1] + 1e-9 for i in range(len(mapped) - 1)))


def test_isotonic_learns_a_known_bias():
    """An engine that says 0.8 but is right 50% of the time should be corrected
    downward toward the truth."""
    pairs = []
    for _ in range(200):
        pairs.append((0.8, 1.0))
        pairs.append((0.8, 0.0))     # stated 0.8, realised 0.5
    m = fit_isotonic(pairs)
    out = apply_isotonic(0.8, m)
    check("overconfident 0.8 pulled toward realised 0.5", abs(out - 0.5) < 0.05,
          f"got {out}")


def test_apply_without_map_is_identity():
    check("no map -> unchanged", apply_isotonic(0.73, None) == 0.73)
    check("empty map -> unchanged", apply_isotonic(0.73, {"x": [], "y": []}) == 0.73)


# ── Honest evaluation ──────────────────────────────────────────────────────

def _well_calibrated(n=400, seed=99):
    """Honest probabilities: outcome really does occur with frequency p."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = rng.uniform(0.2, 0.8)
        out.append((p, float(rng.random() < p)))
    return out


def test_cv_brier_does_not_reward_already_calibrated_input():
    """THE overfitting guard. When probabilities are already honest there is
    nothing to learn, so a fitted map must not show an out-of-sample gain —
    even though fitting and scoring the same rows always looks like a win.

    (Deliberately NOT tested with random noise: there, flattening meaningless
    probabilities to the base rate is a genuine improvement, so an OOS gain is
    correct behaviour. See test_cv_brier_flattens_meaningless_probabilities.)"""
    pairs = _well_calibrated()
    cv = cross_validated_brier(pairs)
    in_sample_map = fit_isotonic(pairs)
    in_sample = brier([(apply_isotonic(p, in_sample_map), o) for p, o in pairs])
    check("in-sample Brier looks better (the trap)", in_sample < cv["brier_raw"],
          f"in={in_sample:.4f} raw={cv['brier_raw']:.4f}")
    check("cross-validated Brier does NOT improve on honest probabilities",
          cv["brier_calibrated"] >= cv["brier_raw"] - 0.005,
          f"raw={cv['brier_raw']:.4f} cal={cv['brier_calibrated']:.4f}")


def test_cv_brier_flattens_meaningless_probabilities():
    """Documents the converse, which is correct and worth locking in: if the
    stated probabilities carry no information, calibration SHOULD collapse them
    toward the base rate, and that is a real out-of-sample gain.
    Theory: uniform p against a coin flip scores 1/3; a constant 0.5 scores 1/4."""
    rng = random.Random(11)
    pairs = [(rng.random(), float(rng.random() < 0.5)) for _ in range(400)]
    cv = cross_validated_brier(pairs)
    check("raw noise Brier near the theoretical 1/3", abs(cv["brier_raw"] - 1 / 3) < 0.05,
          f"{cv['brier_raw']:.4f}")
    check("calibrated collapses toward the theoretical 1/4",
          abs(cv["brier_calibrated"] - 0.25) < 0.05, f"{cv['brier_calibrated']:.4f}")


def test_cv_brier_rewards_a_real_systematic_bias():
    """A genuinely miscalibrated engine SHOULD be correctable out of sample."""
    rng = random.Random(13)
    pairs = []
    for _ in range(400):
        stated = rng.choice([0.7, 0.8, 0.9])
        true_p = stated - 0.3            # systematically overconfident
        pairs.append((stated, float(rng.random() < true_p)))
    cv = cross_validated_brier(pairs)
    check("cross-validated Brier improves on a real bias",
          cv["brier_calibrated"] < cv["brier_raw"],
          f"raw={cv['brier_raw']:.4f} cal={cv['brier_calibrated']:.4f}")


def test_cv_is_deterministic():
    """This number gates a behaviour change - it must not wobble between runs."""
    rng = random.Random(17)
    pairs = [(rng.random(), float(rng.random() < 0.6)) for _ in range(200)]
    a = cross_validated_brier(pairs)
    b = cross_validated_brier(pairs)
    check("same input -> same verdict", a == b)


# ── The applied gate ───────────────────────────────────────────────────────

def test_calibrate_is_identity_unless_applied():
    m = fit_isotonic([(0.8, 1.0), (0.8, 0.0)] * 50)
    check("not-applied calibrator is identity",
          calibrate(0.8, {"applied": False, "map": m}) == 0.8)
    check("None calibrator is identity", calibrate(0.8, None) == 0.8)
    check("applied calibrator actually maps",
          calibrate(0.8, {"applied": True, "map": m}) != 0.8)


def test_fit_calibration_gates_on_sample_size(monkeypatch=None):
    """Below the floor, report why and change nothing."""
    import intelligence.calibration as cal
    orig = cal._pairs_for_horizon
    cal._pairs_for_horizon = lambda h, source="all": _with_dates([(0.8, 1.0)] * 5)
    try:
        r = cal.fit_calibration(20)
        check("thin sample not applied", r["applied"] is False)
        check("reason names the floor", "insufficient" in (r["reason"] or ""))
        check("n reported honestly", r["n"] == 5)
    finally:
        cal._pairs_for_horizon = orig


def test_fit_calibration_rejects_unimprovable_data():
    """Enough rows but no out-of-sample gain -> reported, not applied.

    Uses already-honest probabilities: there is genuinely nothing to learn, so
    the only way a map gets applied here is by overfitting."""
    import intelligence.calibration as cal
    noise = _with_dates(_well_calibrated(n=300, seed=23))
    orig = cal._pairs_for_horizon
    cal._pairs_for_horizon = lambda h, source="all": noise
    try:
        # h=1 against daily-spaced calls -> windows really are independent, so
        # the run reaches the improvement check instead of stopping at the
        # overlap gate. That is what this test is about.
        r = cal.fit_calibration(1)
        check("noise is not applied", r["applied"] is False)
        check("reason is the honest one",
              r["reason"] == "no_out_of_sample_improvement", str(r["reason"]))
        check("both Brier numbers still reported",
              r.get("brier_raw") is not None and r.get("brier_calibrated") is not None)
    finally:
        cal._pairs_for_horizon = orig


def test_fit_calibration_applies_on_real_bias():
    import intelligence.calibration as cal
    rng = random.Random(29)
    biased = []
    for _ in range(300):
        stated = rng.choice([0.7, 0.8, 0.9])
        biased.append((stated, float(rng.random() < stated - 0.3)))
    biased = _with_dates(biased)
    orig = cal._pairs_for_horizon
    cal._pairs_for_horizon = lambda h, source="all": biased
    try:
        r = cal.fit_calibration(1)   # daily calls, daily horizon -> independent
        check("real bias is applied", r["applied"] is True, str(r.get("reason")))
        check("improvement is reported", r.get("brier_improvement", 0) > 0)
        check("map returned", r.get("map") is not None)
    finally:
        cal._pairs_for_horizon = orig


def test_ledger_failure_degrades_to_no_calibration():
    """A broken ledger must cost the correction, never the forecast."""
    import intelligence.calibration as cal
    orig = cal._pairs_for_horizon

    def boom(h, source="all"):
        raise RuntimeError("db gone")

    cal._pairs_for_horizon = boom
    try:
        r = cal.fit_calibration(20)
        check("no exception escapes", r["applied"] is False)
        check("failure reason surfaced", "ledger_unavailable" in (r["reason"] or ""))
    finally:
        cal._pairs_for_horizon = orig


# ── Wiring into the forecast ───────────────────────────────────────────────

def _pillars():
    return {"technical": {"score": 70, "confidence": 1.0},
            "algo": {"score": 68, "confidence": 1.0},
            "fundamentals": {"score": 60, "confidence": 0.8}}


def test_forecast_without_calibrators_is_unchanged():
    from intelligence.prediction_engine import forecast_horizons
    a = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,))
    b = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,), calibrators=None)
    check("omitting calibrators changes nothing", a["horizons"] == b["horizons"])
    check("absence is flagged", "calibration_unavailable" in a["flags"])
    check("no calibrated marker when uncalibrated",
          "calibrated" not in list(a["horizons"].values())[0])


def test_forecast_applies_calibrator_and_keeps_audit_trail():
    from intelligence.prediction_engine import forecast_horizons
    base = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,))
    raw_p = list(base["horizons"].values())[0]["p_up"]

    # A map that pushes everything toward 0.5 (a plausible overconfidence fix).
    m = {"x": [0.0, 1.0], "y": [0.45, 0.55]}
    out = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,),
                            calibrators={20: {"applied": True, "map": m}})
    h = list(out["horizons"].values())[0]
    check("p_up actually moved", h["p_up"] != raw_p)
    check("uncalibrated value preserved for audit", h.get("p_up_uncalibrated") == raw_p)
    check("calibrated marker set", h.get("calibrated") is True)


def test_direction_follows_the_calibrated_probability():
    """Direction must be decided AFTER calibration, or the label would describe
    a number the engine no longer stands behind."""
    from intelligence.prediction_engine import forecast_horizons
    # Force everything down to 0.20 -> direction must read DOWN.
    m = {"x": [0.0, 1.0], "y": [0.20, 0.20]}
    out = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,),
                            calibrators={20: {"applied": True, "map": m}})
    h = list(out["horizons"].values())[0]
    check("calibrated p_up used", abs(h["p_up"] - 0.20) < 1e-6, str(h["p_up"]))
    check("direction reflects the calibrated number", h["direction"] == "DOWN")


def test_unapplied_calibrator_is_a_no_op_but_is_flagged():
    from intelligence.prediction_engine import forecast_horizons
    base = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,))
    out = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,),
                            calibrators={20: {"applied": False, "map": None,
                                              "reason": "insufficient_outcomes"}})
    check("unapplied calibrator leaves p_up alone",
          list(out["horizons"].values())[0]["p_up"]
          == list(base["horizons"].values())[0]["p_up"])
    check("distinguishes 'looked but did not apply'",
          "calibration_not_applied" in out["flags"])


def test_broken_calibrator_does_not_break_the_forecast():
    from intelligence.prediction_engine import forecast_horizons
    out = forecast_horizons("T", 100.0, _pillars(), {}, horizons=(20,),
                            calibrators={20: {"applied": True, "map": "garbage"}})
    check("forecast still produced", len(out["horizons"]) == 1)
    check("p_up still a real number",
          isinstance(list(out["horizons"].values())[0]["p_up"], float))


# ── Purged, time-blocked CV (the overlapping-window guard) ─────────────────

def _dated(n, horizon_days, seed=41, step_days=30):
    """n calls spaced `step_days` apart, each graded over `horizon_days`."""
    from datetime import date, timedelta
    rng = random.Random(seed)
    d0 = date(2024, 1, 2)
    return [(rng.random(), float(rng.random() < 0.5),
             (d0 + timedelta(days=i * step_days)).isoformat()) for i in range(n)]


def test_effective_sample_size_collapses_overlapping_windows():
    """The core insight: monthly calls graded over a year are not independent."""
    from intelligence.calibration import effective_sample_size
    dates = [d for _, _, d in _dated(18, 252)]
    long_h = effective_sample_size(dates, 252)
    short_h = effective_sample_size(dates, 1)
    check("18 monthly calls at a 1-year horizon are ~1 independent window",
          long_h <= 2, f"got {long_h}")
    check("effective n falls as horizon grows", long_h < short_h)


def test_effective_sample_size_cannot_exceed_the_number_of_calls():
    """Regression for a real defect: the original formula used calendar span
    alone and reported 666 independent windows at a 1-day horizon from a ledger
    holding 20 distinct call dates. You cannot have more independent
    observations than moments of observation.

    This mattered in production - h=5d was APPLYING a calibration map on a
    claimed 38 independent windows when only 11 distinct call dates existed."""
    from intelligence.calibration import effective_sample_size
    dates = [d for _, _, d in _dated(18, 252)]      # 18 distinct dates
    for horizon in (1, 2, 5, 20):
        eff = effective_sample_size(dates, horizon)
        check(f"h={horizon}d effective_n bounded by the 18 call dates",
              eff <= 18, f"got {eff}")
    check("a dense span still cannot beat its own sampling",
          effective_sample_size([d for _, _, d in _dated(5, 1, step_days=365)], 1) <= 5)


def test_purged_cv_reports_effective_n():
    from intelligence.calibration import purged_cv_brier
    out = purged_cv_brier(_dated(60, 20, step_days=7), horizon_days=20)
    check("purged CV returns a result", out is not None)
    check("effective_n reported alongside row count",
          out.get("effective_n") is not None and out["n"] > 0)
    check("effective_n is smaller than the row count",
          out["effective_n"] < out["n"], f"{out['effective_n']} vs {out['n']}")


def test_purged_cv_is_stricter_than_unpurged_on_overlapping_data():
    """The regression this whole section exists for. On heavily-overlapping
    long-horizon rows the unpurged CV finds an 'improvement'; the purged one
    should not be fooled as easily."""
    from intelligence.calibration import purged_cv_brier
    rows = _dated(140, 252, seed=5, step_days=4)      # dense, hugely overlapping
    unpurged = cross_validated_brier([(p, o) for p, o, _ in rows])
    purged = purged_cv_brier(rows, horizon_days=252)
    check("unpurged CV claims a gain on overlapping noise",
          unpurged["brier_calibrated"] < unpurged["brier_raw"],
          f"raw={unpurged['brier_raw']:.4f} cal={unpurged['brier_calibrated']:.4f}")
    check("purged CV flags the sample as effectively tiny",
          purged["effective_n"] <= 3, f"got {purged['effective_n']}")


def test_fit_calibration_refuses_when_windows_overlap(monkeypatch=None):
    """Enough ROWS but almost no independent windows -> refuse, and say why."""
    import intelligence.calibration as cal
    rows = _dated(150, 252, seed=9, step_days=5)
    orig = cal._pairs_for_horizon
    cal._pairs_for_horizon = lambda h, source="all": rows
    try:
        r = cal.fit_calibration(252)
        check("not applied despite 150 rows", r["applied"] is False)
        check("reason names the overlap",
              "independent_windows" in (r["reason"] or ""), str(r["reason"]))
        check("effective_n surfaced for the reader", r.get("effective_n") is not None)
    finally:
        cal._pairs_for_horizon = orig


def test_fit_calibration_still_applies_with_genuine_independent_evidence():
    """The gate must not be so strict that a real, well-sampled bias is refused."""
    import intelligence.calibration as cal
    from datetime import date, timedelta
    rng = random.Random(31)
    d0 = date(2020, 1, 1)
    rows = []
    for i in range(400):                       # daily calls, 1-day horizon
        stated = rng.choice([0.7, 0.8, 0.9])
        rows.append((stated, float(rng.random() < stated - 0.3),
                     (d0 + timedelta(days=i)).isoformat()))
    orig = cal._pairs_for_horizon
    cal._pairs_for_horizon = lambda h, source="all": rows
    try:
        r = cal.fit_calibration(1)
        check("real bias with independent windows IS applied",
              r["applied"] is True, str(r.get("reason")))
        check("effective_n comfortably above the floor",
              r.get("effective_n", 0) >= cal.MIN_EFFECTIVE_OBS)
    finally:
        cal._pairs_for_horizon = orig


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
