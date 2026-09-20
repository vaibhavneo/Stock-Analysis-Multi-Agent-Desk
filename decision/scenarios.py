"""
Scenario engine (Phase 5) + what-would-change-my-mind (Phase 15).

A scenario is a CONDITIONAL, not a forecast: "if X holds, the path toward Y
becomes available". The distinction is load-bearing, because the app's previous
bull/bear prose was a fixed 2:1 ATR geometry dressed in sentences — every stock
got a target exactly twice its stop distance away, which means the "bull case"
carried no information about the company at all.

Here each scenario is built from:
  - the actual level ladder (decision/levels.py) for its price references, so a
    bull target is a resistance the market has traded at, not `price + 2·ATR`;
  - the named evidence that would have to hold (or break) for it;
  - the catalysts inside its horizon;
  - an explicit invalidation condition.

Probability rules, enforced in code rather than by convention:
  - A scenario carries a numeric probability ONLY if the forecast engine's
    probability for that horizon was actually corrected by a fitted calibration
    that beat raw probabilities out of sample.
  - Otherwise `probability` is None and `probability_basis` is
    "NOT_CALIBRATED", which the UI renders literally.
  - No LLM output ever becomes a probability. The narrative is quoted, never
    parsed for numbers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from decision.evidence import DecisionEvidence, directional

# A calibrated probability still needs a real sample behind it before it is
# shown as a number. Below this, the correction exists but is not evidence.
MIN_CALIBRATION_SAMPLE = 30


def _probability_for(forecast: Optional[Dict[str, Any]],
                     horizon_label: str,
                     cal_interp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The only place a scenario probability can come from.

    Returns `{probability, basis, source, detail}`. `probability` is None unless
    every condition is met, and the reason is always stated.
    """
    if not forecast or not forecast.get("horizons"):
        return {"probability": None, "basis": "NOT_CALIBRATED",
                "source": None,
                "detail": "No forecast was computed for this request."}

    h = (forecast["horizons"] or {}).get(horizon_label)
    if not h or h.get("p_up") is None:
        return {"probability": None, "basis": "NOT_CALIBRATED", "source": None,
                "detail": f"No forecast at the {horizon_label} horizon."}

    n = (cal_interp or {}).get("n_predictions") or 0
    if not h.get("calibrated"):
        return {
            "probability": None,
            "basis": "NOT_CALIBRATED",
            "source": "intelligence/prediction_engine.py (uncalibrated)",
            "detail": ("The engine produced a directional score for this horizon, but it has "
                       "not been corrected against measured outcomes, so it is not a "
                       "probability and is not shown as one. Uncorrected score: "
                       f"{h['p_up']:.2f}."),
            "uncalibrated_score": h["p_up"],
        }
    if n < MIN_CALIBRATION_SAMPLE:
        return {
            "probability": None,
            "basis": "INSUFFICIENT_CALIBRATION_SAMPLE",
            "source": "data/prediction_ledger.py",
            "detail": (f"A calibration was applied, but only {n} matured predictions back it "
                       f"(minimum {MIN_CALIBRATION_SAMPLE}). Not shown as a probability."),
            "uncalibrated_score": h.get("p_up_uncalibrated"),
        }
    return {
        "probability": h["p_up"],
        "basis": "CALIBRATED",
        "source": f"isotonic calibration over {n} matured predictions at this horizon",
        "detail": (f"Corrected from {h.get('p_up_uncalibrated')} using measured outcomes; "
                   f"the correction beat raw probabilities on purged cross-validation."),
    }


def _levels_for(level_map: Optional[Dict[str, Any]], kind: str,
                n: int = 2) -> List[Dict[str, Any]]:
    if not level_map or level_map.get("status") != "OK":
        return []
    key = "decision_relevant_supports" if kind == "SUPPORT" else "decision_relevant_resistances"
    return (level_map.get(key) or [])[:n]


def _catalysts_in(catalysts: Optional[Dict[str, Any]], max_days: int) -> List[Dict[str, Any]]:
    if not catalysts or catalysts.get("status") != "OK":
        return []
    return [{"date": e["date"], "event": e["event"], "days_away": e["days_away"],
             "relevance": e["relevance"]}
            for e in catalysts.get("events", []) if e["days_away"] <= max_days]


def build_scenarios(
    thesis: Dict[str, Any],
    edge: Dict[str, Any],
    items: List[DecisionEvidence],
    level_map: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    cal_interp: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
    horizon_days: int = 91,
    horizon_label: str = "3M",
    current_price: Optional[float] = None,
) -> Dict[str, Any]:
    """BULL / BASE / BEAR, plus CATALYST and FAILURE cases where the data
    supports them. Each is a conditional with named evidence and an explicit
    invalidation."""
    dir_items = directional(items)
    bulls = sorted([e for e in dir_items if e.direction == "BULLISH"],
                   key=lambda e: e.weight, reverse=True)
    bears = sorted([e for e in dir_items if e.direction == "BEARISH"],
                   key=lambda e: e.weight, reverse=True)

    prob = _probability_for(forecast, horizon_label, cal_interp)
    resistances = _levels_for(level_map, "RESISTANCE")
    supports = _levels_for(level_map, "SUPPORT")
    window_catalysts = _catalysts_in(catalysts, horizon_days)

    def _price_refs(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{"price": l["price"], "distance_pct": l["distance_pct"],
                 "basis": l["basis"], "sources": l["sources"],
                 "confidence": l["confidence"]} for l in levels]

    scenarios: List[Dict[str, Any]] = []

    # ── BULL ─────────────────────────────────────────────────────────────
    bull_target = resistances[0] if resistances else None
    scenarios.append({
        "name": "BULL",
        "thesis": ("The bullish evidence holds and the price works through the nearest "
                   "resistance the market has actually defended."
                   if bull_target else
                   "The bullish evidence holds, but no resistance level could be sourced, "
                   "so no upside reference is stated."),
        "evidence": [{"source": e.source, "observation": e.observation,
                      "weight": e.weight, "horizon": e.horizon} for e in bulls[:4]],
        "expected_direction": "UP",
        "horizon_days": horizon_days,
        "conditions": ([f"Price sustains above {resistances[0]['price']} "
                        f"({resistances[0]['distance_pct']:+.1f}%)"] if resistances else [])
                      + [f"{e.source} continues to read bullish" for e in bulls[:2]],
        "price_levels": _price_refs(resistances),
        "catalysts": window_catalysts,
        "risks": [e.observation for e in bears[:2]],
        "invalidation": (f"Price closes below {supports[0]['price']} "
                         f"({supports[0]['distance_pct']:+.1f}%)" if supports else
                         "No sourced support level — invalidation cannot be priced."),
        "probability": prob["probability"],
        "probability_basis": prob["basis"],
        "probability_source": prob["source"],
        "probability_detail": prob["detail"],
        "uncertainty": ("Upside references are levels the market has traded at, not targets "
                        "the price is expected to reach. No probability is attached unless "
                        "it has been calibrated against measured outcomes."),
    })

    # ── BASE ─────────────────────────────────────────────────────────────
    base_low = supports[0]["price"] if supports else None
    base_high = resistances[0]["price"] if resistances else None
    scenarios.append({
        "name": "BASE",
        "thesis": (f"Price oscillates between the nearest sourced support ({base_low}) and "
                   f"resistance ({base_high}) while the evidence stays mixed."
                   if base_low and base_high else
                   "Price continues without resolving the current evidence; no sourced range "
                   "is available to bound it."),
        "evidence": [{"source": e.source, "observation": e.observation,
                      "weight": e.weight, "horizon": e.horizon}
                     for e in sorted(dir_items, key=lambda e: e.weight, reverse=True)[:3]],
        "expected_direction": "SIDEWAYS",
        "horizon_days": horizon_days,
        "conditions": ["No named catalyst resolves"
                       if window_catalysts else "No scheduled catalyst in the window",
                       "Evidence balance does not shift materially"],
        "price_levels": _price_refs((supports[:1] + resistances[:1])),
        "catalysts": window_catalysts,
        "risks": ["Time decay of the thesis — a read that never resolves stops being a reason "
                  "to hold capital here."],
        "invalidation": "A decisive close outside the support/resistance band above.",
        "probability": prob["probability"] and None,   # base case is not a directional claim
        "probability_basis": "NOT_APPLICABLE",
        "probability_source": None,
        "probability_detail": ("The base case is the absence of resolution, not a directional "
                               "outcome; a directional probability does not apply to it."),
        "uncertainty": "Range bounds are observed levels, not guarantees of containment.",
    })

    # ── BEAR ─────────────────────────────────────────────────────────────
    scenarios.append({
        "name": "BEAR",
        "thesis": ("The bearish evidence dominates and the price loses the nearest sourced "
                   "support." if supports else
                   "The bearish evidence dominates; no support level could be sourced, so "
                   "no downside reference is stated."),
        "evidence": [{"source": e.source, "observation": e.observation,
                      "weight": e.weight, "horizon": e.horizon} for e in bears[:4]]
                    or [{"source": "none", "observation": "No bearish evidence carries weight "
                         "today — this scenario is the downside case, not a prediction.",
                         "weight": 0.0, "horizon": "ALL"}],
        "expected_direction": "DOWN",
        "horizon_days": horizon_days,
        "conditions": ([f"Price closes below {supports[0]['price']} "
                        f"({supports[0]['distance_pct']:+.1f}%)"] if supports else [])
                      + ([f"{e.source} turns bearish" for e in bulls[:2]] if not bears else
                         [f"{e.source} continues to read bearish" for e in bears[:2]]),
        "price_levels": _price_refs(supports),
        "catalysts": window_catalysts,
        "risks": [e.observation for e in bulls[:2]],
        "invalidation": (f"Price reclaims {resistances[0]['price']} "
                         f"({resistances[0]['distance_pct']:+.1f}%)" if resistances else
                         "No sourced resistance level — invalidation cannot be priced."),
        "probability": None,
        "probability_basis": prob["basis"] if prob["basis"] != "CALIBRATED" else "CALIBRATED",
        "probability_source": prob["source"],
        "probability_detail": (
            f"Downside probability would be {round(1 - prob['probability'], 3)} "
            f"if read as the complement of the calibrated up-probability; it is not stated "
            f"as a separate calibrated number because only the up-direction was calibrated."
            if prob["probability"] is not None else prob["detail"]),
        "uncertainty": ("Downside references are levels the market has traded at. Drawdowns "
                        "can and do exceed the deepest level on the ladder."),
    })

    # ── CATALYST case — only when a dated event exists in the window ─────
    if window_catalysts:
        nxt = window_catalysts[0]
        scenarios.append({
            "name": "CATALYST",
            "thesis": (f"{nxt['event']} on {nxt['date']} resolves in one direction and "
                       f"reprices the security. This is a scheduled increase in uncertainty, "
                       f"not a directional expectation."),
            "evidence": [{"source": "catalyst_calendar", "observation":
                          f"{nxt['event']} is {nxt['days_away']} days away.",
                          "weight": None, "horizon": "SHORT"}],
            "expected_direction": "EITHER",
            "horizon_days": nxt["days_away"],
            "conditions": [f"The event occurs on or near {nxt['date']}"],
            "price_levels": _price_refs(supports[:1] + resistances[:1]),
            "catalysts": window_catalysts,
            "risks": ["Entering immediately before a binary event means taking the event's "
                      "variance without any edge in predicting it."],
            "invalidation": "The event is postponed or already priced in.",
            "probability": None,
            "probability_basis": "NOT_CALIBRATED",
            "probability_source": None,
            "probability_detail": ("No catalyst-outcome probability is calibrated in this "
                                   "system, and none is stated."),
            "uncertainty": "Direction of a scheduled event is unknown by construction.",
        })

    # ── FAILURE case — only when a real failure mode is measurable ───────
    failure_reasons: List[str] = []
    if not edge.get("demonstrated"):
        failure_reasons.append("the statistical gate never clears, so the signal is never "
                               "shown to be worth acting on")
    worst_dd = (edge.get("backtest") or {}).get("max_drawdown")
    if worst_dd is not None and worst_dd > 0.3:
        failure_reasons.append(f"the strategy's historical max drawdown is {worst_dd:.0%}, "
                               f"which a position would have had to survive")
    if failure_reasons:
        scenarios.append({
            "name": "FAILURE",
            "thesis": ("The structural failure mode: " + "; and ".join(failure_reasons) + "."),
            "evidence": [{"source": "gate:statistical_edge",
                          "observation": edge.get("basis"), "weight": None, "horizon": "ALL"}],
            "expected_direction": "NOT_DIRECTIONAL",
            "horizon_days": horizon_days,
            "conditions": ["Continued failure of the walk-forward / PBO / dSR gates"],
            "price_levels": [],
            "catalysts": [],
            "risks": ["Capital committed on an unproven signal is exposed to the full "
                      "drawdown without a demonstrated compensating return."],
            "invalidation": "The gate clears on more data.",
            "probability": None,
            "probability_basis": "NOT_CALIBRATED",
            "probability_source": None,
            "probability_detail": "Not a market outcome; no probability applies.",
            "uncertainty": "This is a statement about the method, not about the security.",
        })

    return {
        "scenarios": scenarios,
        "horizon_days": horizon_days,
        "horizon_label": horizon_label,
        "probability_policy": (
            "A numeric probability appears only where an empirical calibration was applied "
            "AND at least "
            f"{MIN_CALIBRATION_SAMPLE} matured predictions back it. Otherwise the field reads "
            "'Probability not calibrated'. No probability is ever derived from narrative text."),
        "any_probability_stated": any(s["probability"] is not None for s in scenarios),
    }


def build_mind_changers(thesis: Dict[str, Any], edge: Dict[str, Any],
                        items: List[DecisionEvidence],
                        level_map: Optional[Dict[str, Any]] = None,
                        catalysts: Optional[Dict[str, Any]] = None,
                        conflict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Phase 15 — confirmation and invalidation triggers, each measurable.

    Every trigger must be checkable against data this system already has, so it
    can later be evaluated by the forward-validation framework. A trigger no one
    can test is a sentence, not a trigger.
    """
    supports = _levels_for(level_map, "SUPPORT", n=2)
    resistances = _levels_for(level_map, "RESISTANCE", n=2)
    dir_items = directional(items)
    bulls = sorted([e for e in dir_items if e.direction == "BULLISH"],
                   key=lambda e: e.weight, reverse=True)
    bears = sorted([e for e in dir_items if e.direction == "BEARISH"],
                   key=lambda e: e.weight, reverse=True)
    lead = bulls if thesis["direction"] == "BULLISH" else bears
    against = bears if thesis["direction"] == "BULLISH" else bulls

    confirmation: List[Dict[str, Any]] = []
    invalidation: List[Dict[str, Any]] = []

    if not edge.get("demonstrated"):
        confirmation.append({
            "trigger": "Statistical gate clears (walk-forward OOS positive + PBO < 0.5 + "
                       "net-of-cost positive + dSR ≥ 0.5 + sample ≥ 504 bars)",
            "measurable_as": "recommendation.confidence.statistical_edge.level == 'HIGH'",
            "currently": f"{edge['n_gates_passed']}/{edge['n_gates']} gates pass "
                         f"(level {edge['level']})",
            "category": "STATISTICAL",
        })
    if (edge.get("calibration") or {}).get("status") != "OK":
        confirmation.append({
            "trigger": "Live calibration reaches a readable sample",
            "measurable_as": "calibration_report(horizon).overall.n >= 30",
            "currently": f"{(edge.get('calibration') or {}).get('n_predictions', 0)} matured predictions",
            "category": "TRACK_RECORD",
        })
    if thesis["direction"] == "BULLISH" and resistances:
        confirmation.append({
            "trigger": f"Price sustains a close above {resistances[0]['price']} "
                       f"({resistances[0]['distance_pct']:+.1f}%)",
            "measurable_as": f"close > {resistances[0]['price']}",
            "currently": f"{resistances[0]['distance_pct']:+.1f}% away",
            "category": "PRICE",
            "level_basis": resistances[0]["basis"],
        })
    if thesis["direction"] == "BEARISH" and supports:
        confirmation.append({
            "trigger": f"Price closes below {supports[0]['price']} "
                       f"({supports[0]['distance_pct']:+.1f}%)",
            "measurable_as": f"close < {supports[0]['price']}",
            "currently": f"{supports[0]['distance_pct']:+.1f}% away",
            "category": "PRICE",
            "level_basis": supports[0]["basis"],
        })
    for e in lead[:2]:
        confirmation.append({
            "trigger": f"{e.source} strengthens further",
            "measurable_as": f"{e.metric} moves further from neutral",
            "currently": e.observation,
            "category": e.category,
        })

    if thesis["direction"] == "BULLISH" and supports:
        invalidation.append({
            "trigger": f"Price closes below {supports[0]['price']} "
                       f"({supports[0]['distance_pct']:+.1f}%), the nearest level the market "
                       f"has defended",
            "measurable_as": f"close < {supports[0]['price']}",
            "currently": f"{supports[0]['distance_pct']:+.1f}% away",
            "category": "PRICE",
            "level_basis": supports[0]["basis"],
        })
    if thesis["direction"] == "BEARISH" and resistances:
        invalidation.append({
            "trigger": f"Price reclaims {resistances[0]['price']} "
                       f"({resistances[0]['distance_pct']:+.1f}%)",
            "measurable_as": f"close > {resistances[0]['price']}",
            "currently": f"{resistances[0]['distance_pct']:+.1f}% away",
            "category": "PRICE",
            "level_basis": resistances[0]["basis"],
        })
    for e in lead[:2]:
        invalidation.append({
            "trigger": f"{e.source} flips direction",
            "measurable_as": f"{e.metric} crosses to the opposite side of neutral",
            "currently": e.observation,
            "category": e.category,
        })
    for e in against[:1]:
        invalidation.append({
            "trigger": f"{e.source} strengthens against the thesis",
            "measurable_as": f"{e.metric} moves further from neutral in the opposing direction",
            "currently": e.observation,
            "category": e.category,
        })
    nxt = (catalysts or {}).get("next_event")
    if nxt:
        invalidation.append({
            "trigger": f"{nxt['event']} on {nxt['date']} resolves against the thesis",
            "measurable_as": "post-event repricing beyond the nearest sourced level",
            "currently": f"{nxt['days_away']} days away",
            "category": "CATALYST",
        })
    for c in (conflict or {}).get("conflicts", []):
        if c.get("changes_decision") and c.get("kind") == "DIRECTIONAL":
            invalidation.append({
                "trigger": f"The '{c['name']}' conflict resolves against the thesis",
                "measurable_as": "the opposing source gains weight while the leading source loses it",
                "currently": c["why"],
                "category": "CONFLICT",
            })

    return {
        "confirmation": confirmation,
        "invalidation": invalidation,
        "note": ("Every trigger above is stated as something checkable against data this "
                 "system already computes, so it can be evaluated after the fact rather "
                 "than merely read."),
    }
