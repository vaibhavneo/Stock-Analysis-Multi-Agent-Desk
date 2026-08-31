#!/usr/bin/env python3
"""Tests for the replay -> calibration data path.

Replay is what makes the learning loop usable now instead of in a year: it
produces matured outcomes on demand. But a replayed snapshot is only useful for
calibration if it also froze the PROBABILITY it stated, and that probability
must be computable from information available at as_of. These tests guard both
halves — the data is produced, and it is produced without lookahead.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from agents.replay import _pit_horizon_probabilities
from data.prediction_ledger import HORIZONS

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _prices(n=300, start=50.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series([start + i * 0.1 for i in range(n)], index=idx)


def _rec():
    return {
        "ticker": "TEST",
        "pillars": {"technical": {"score": 70, "confidence": 1.0},
                    "algo": {"score": 66, "confidence": 1.0},
                    "fundamentals": {"score": 58, "confidence": 0.8}},
        "levels": {"atr_14": 2.5},
    }


def test_probabilities_are_produced_for_every_ledger_horizon():
    """The whole point: a replayed snapshot must carry a probability the
    calibration layer can later grade."""
    probs = _pit_horizon_probabilities("TEST", _rec(), {"algo_score": 66}, _prices())
    check("probabilities returned", probs is not None)
    check("one probability per LEDGER horizon", set(probs.keys()) == set(HORIZONS),
          f"got {sorted(probs.keys())} want {sorted(HORIZONS)}")
    check("all are real probabilities",
          all(0.0 < p < 1.0 for p in probs.values()), str(probs))


def test_horizons_match_what_the_outcome_engine_evaluates():
    """Storing a probability at a horizon the ledger never evaluates would
    journal a number that can never be scored."""
    probs = _pit_horizon_probabilities("TEST", _rec(), {"algo_score": 66}, _prices())
    check("no probability at an unevaluated horizon",
          all(h in HORIZONS for h in probs.keys()), str(sorted(probs.keys())))


def test_no_lookahead_regime_is_used():
    """compute_market_regime() reads TODAY's market. A replay of an old date
    must not see it, so the forecast is made with regime=None and says so."""
    import intelligence.prediction_engine as pe

    seen = {}
    orig = pe.forecast_horizons

    def spy(*a, **kw):
        seen.update(kw)
        return orig(*a, **kw)

    pe.forecast_horizons = spy
    try:
        _pit_horizon_probabilities("TEST", _rec(), {"algo_score": 66}, _prices())
    finally:
        pe.forecast_horizons = orig
    check("regime withheld from a replayed forecast", seen.get("regime") is None)
    check("calibrators withheld (would leak future outcomes)",
          seen.get("calibrators") is None)


def test_probability_depends_only_on_pit_inputs():
    """Same PIT inputs -> same probability, regardless of when the replay runs.
    Determinism here is what makes a replayed track record evidence."""
    a = _pit_horizon_probabilities("TEST", _rec(), {"algo_score": 66}, _prices())
    b = _pit_horizon_probabilities("TEST", _rec(), {"algo_score": 66}, _prices())
    check("replay probabilities are deterministic", a == b)


def test_stronger_pillars_give_a_higher_probability():
    """Sanity: the forecast must actually track the evidence, or calibration
    would be learning from noise."""
    weak = _rec()
    weak["pillars"] = {"technical": {"score": 35, "confidence": 1.0},
                       "algo": {"score": 30, "confidence": 1.0}}
    strong = _rec()
    strong["pillars"] = {"technical": {"score": 85, "confidence": 1.0},
                         "algo": {"score": 82, "confidence": 1.0}}
    pw = _pit_horizon_probabilities("T", weak, {"algo_score": 30}, _prices())
    ps = _pit_horizon_probabilities("T", strong, {"algo_score": 82}, _prices())
    h = sorted(HORIZONS)[0]
    check("stronger pillars -> higher p_up", ps[h] > pw[h], f"{ps[h]} vs {pw[h]}")


def test_failure_degrades_to_none_not_an_exception():
    """A replay row that cannot produce a probability must still freeze its
    action - degraded but honest beats aborting the row."""
    probs = _pit_horizon_probabilities("TEST", {"pillars": {}}, {}, _prices())
    check("no exception escapes", probs is None or isinstance(probs, dict))


def test_replay_one_attaches_probabilities_before_freezing():
    """The wiring itself: freeze_prediction must SEE horizon_probabilities,
    because that is the column calibration reads."""
    import agents.replay as rp

    captured = {}

    def fake_freeze(rec):
        captured["rec"] = rec
        return "snap-test"

    def fake_pit_inputs(ticker, as_of, full_prices, pit_fn=None):
        return (_prices(), {"current_price": 80.0}, {"score": 70},
                {"algo_score": 66}, {"available": True}, {}, {}, [], [])

    def fake_build(*a, **kw):
        return _rec()

    orig_freeze = rp.pl.freeze_prediction
    orig_pit = rp.pit_inputs
    import agents.recommendation as arec
    orig_build = arec.build_recommendation

    rp.pl.freeze_prediction = fake_freeze
    rp.pit_inputs = fake_pit_inputs
    arec.build_recommendation = fake_build
    try:
        out = rp.replay_one("TEST", "2024-06-03", _prices(), run_id="t")
        rec = captured.get("rec", {})
        check("replay_one completed", out.get("status") == "done", str(out))
        check("horizon_probabilities attached before freeze",
              rec.get("horizon_probabilities") is not None)
        check("probabilities cover the ledger horizons",
              set((rec.get("horizon_probabilities") or {}).keys()) == set(HORIZONS))
    finally:
        rp.pl.freeze_prediction = orig_freeze
        rp.pit_inputs = orig_pit
        arec.build_recommendation = orig_build


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
