"""
Position context, risk budget, and add analysis (Phases 8, 10, 11).

Three things this layer refuses to do, each because the alternative is worse
than silence:

  1. **It never infers ownership.** If no position is supplied, the answer is
     `POSITION_CONTEXT_NOT_PROVIDED`, not an assumed zero and not an assumed
     holding. An existing position and a new position are different decision
     problems, and guessing which one the reader has makes both answers wrong.

  2. **It never sizes in dollars without a stated budget.** Without a portfolio
     value or a configured paper risk policy, the answer is
     `POSITION_SIZING_NOT_COMPUTABLE`. A percentage of an unknown number is not
     a number.

  3. **It never treats a lower price as a reason to add.** This is the single
     most important rule in the module. A price that has fallen is CHEAPER; it
     is only BETTER if the reason for owning it has strengthened, or at minimum
     has not weakened. The two are separated explicitly in `analyze_add()` and
     the distinction is what the output leads with.

The previous behaviour is worth stating plainly, because this module replaces
it: at −15% unrealized on a positive signal, the old brief said the thesis
"supports adding — this pulls your average cost down while you're underwater".
That sentence recommends averaging down on the basis of the drawdown itself.
Nothing in it checks whether the thesis improved, whether risk rose, or whether
the invalidation level moved.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Concentration bands as a share of a stated portfolio value. Stated priors,
# used only to LABEL concentration — never to compute a dollar recommendation.
CONCENTRATION_BANDS = ((25.0, "CONCENTRATED"), (10.0, "MEANINGFUL"), (0.0, "SMALL"))

# Unrealized-P&L thresholds that trigger a named observation. These describe
# the position, they do not decide anything on their own.
LARGE_GAIN_PCT = 25.0
LARGE_LOSS_PCT = -20.0


def build_position_context(current_price: Optional[float],
                           position: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize what the caller told us about their holding.

    `status`:
      PROVIDED                      — usable cost basis
      POSITION_CONTEXT_NOT_PROVIDED — nothing supplied; ownership unknown
      POSITION_CONTEXT_INSUFFICIENT — something supplied but unusable
    """
    if not position:
        return {"status": "POSITION_CONTEXT_NOT_PROVIDED",
                "owns": None,
                "message": ("No position information was supplied. Ownership is unknown, so "
                            "both the owned and not-owned branches are shown and neither is "
                            "assumed."),
                "missing": ["avg_cost", "shares", "portfolio_value"]}

    avg_cost = position.get("avg_cost")
    try:
        avg_cost = float(avg_cost) if avg_cost is not None else None
    except (TypeError, ValueError):
        avg_cost = None

    if not avg_cost or avg_cost <= 0 or not current_price or current_price <= 0:
        missing = []
        if not avg_cost or avg_cost <= 0:
            missing.append("avg_cost")
        if not current_price:
            missing.append("current_price")
        return {"status": "POSITION_CONTEXT_INSUFFICIENT",
                "owns": None,
                "message": ("Position information was supplied but is not usable: "
                            + ", ".join(missing) + " missing or non-positive."),
                "missing": missing}

    shares = position.get("shares")
    try:
        shares = float(shares) if shares is not None else None
    except (TypeError, ValueError):
        shares = None

    portfolio_value = position.get("portfolio_value")
    try:
        portfolio_value = float(portfolio_value) if portfolio_value is not None else None
    except (TypeError, ValueError):
        portfolio_value = None

    pl_pct = (current_price - avg_cost) / avg_cost * 100
    market_value = current_price * shares if shares else None
    cost_value = avg_cost * shares if shares else None
    pl_dollars = (market_value - cost_value) if (market_value and cost_value) else None

    weight_pct = None
    concentration = None
    if market_value and portfolio_value and portfolio_value > 0:
        weight_pct = round(market_value / portfolio_value * 100, 2)
        for bar, label in CONCENTRATION_BANDS:
            if weight_pct >= bar:
                concentration = label
                break

    observations: List[str] = []
    if pl_pct >= LARGE_GAIN_PCT:
        observations.append(
            f"Large unrealized gain ({pl_pct:+.1f}%). Trimming and holding are both "
            "defensible; which is right depends on whether the thesis still supports the "
            "remaining upside, not on the size of the gain.")
    if pl_pct <= LARGE_LOSS_PCT:
        observations.append(
            f"Large unrealized loss ({pl_pct:+.1f}%). The loss is already incurred and is "
            "not itself evidence about the future — the only live question is whether the "
            "original reason for owning it still holds.")
    if concentration == "CONCENTRATED":
        observations.append(
            f"This position is {weight_pct:.1f}% of the stated portfolio — concentrated. "
            "Adding increases single-name risk regardless of how good the setup looks.")

    return {
        "status": "PROVIDED",
        "owns": True,
        "avg_cost": round(avg_cost, 4),
        "current_price": round(current_price, 4),
        "shares": shares,
        "market_value": round(market_value, 2) if market_value else None,
        "cost_value": round(cost_value, 2) if cost_value else None,
        "unrealized_pl_pct": round(pl_pct, 2),
        "unrealized_pl_dollars": round(pl_dollars, 2) if pl_dollars is not None else None,
        "portfolio_value": portfolio_value,
        "weight_pct": weight_pct,
        "concentration": concentration,
        "underwater": current_price < avg_cost,
        # Asymmetric by construction: recovering a −50% loss needs +100%, not +50%.
        "recovery_required_pct": (round((avg_cost / current_price - 1) * 100, 2)
                                  if current_price < avg_cost else 0.0),
        "observations": observations,
        "missing": [k for k, v in (("shares", shares), ("portfolio_value", portfolio_value))
                    if v is None],
    }


def build_risk_budget(current_price: Optional[float],
                      position_context: Dict[str, Any],
                      level_map: Optional[Dict[str, Any]],
                      rec: Optional[Dict[str, Any]] = None,
                      algo_signals: Optional[Dict[str, Any]] = None,
                      historical_context: Optional[Dict[str, Any]] = None,
                      max_portfolio_risk_pct: Optional[float] = None) -> Dict[str, Any]:
    """Deterministic risk quantities (Phase 11).

    Everything computable from what was supplied is computed; everything that
    needs a portfolio value or a risk policy and did not get one is reported as
    NOT_COMPUTABLE with the missing input named. No dollar amount is ever
    recommended without a stated budget.
    """
    rec = rec or {}
    algo = algo_signals or {}
    out: Dict[str, Any] = {"inputs_missing": [], "notes": []}

    invalidation = None
    invalidation_basis = None
    if level_map and level_map.get("status") == "OK" and level_map.get("nearest_support"):
        invalidation = level_map["nearest_support"]["price"]
        invalidation_basis = level_map["nearest_support"]["basis"]
    elif (rec.get("levels") or {}).get("stop_loss"):
        invalidation = rec["levels"]["stop_loss"]
        invalidation_basis = "CURRENT_PRICE_DERIVED"
        out["notes"].append("No sourced support level; falling back to the ATR-derived stop, "
                            "which moves with the price and is therefore weaker as an "
                            "invalidation marker.")

    risk_to_stop_pct = None
    if invalidation and current_price:
        risk_to_stop_pct = round((invalidation / current_price - 1) * 100, 2)

    out["invalidation_level"] = invalidation
    out["invalidation_basis"] = invalidation_basis
    out["risk_to_invalidation_pct"] = risk_to_stop_pct

    out["volatility"] = {
        "atr_14": (rec.get("levels") or {}).get("atr_14"),
        "atr_pct_of_price": (round((rec["levels"]["atr_14"] / current_price) * 100, 2)
                             if current_price and (rec.get("levels") or {}).get("atr_14") else None),
        "historical_volatility_20d": algo.get("historical_volatility_20d"),
        "historical_volatility_60d": algo.get("historical_volatility_60d"),
        "vol_regime": algo.get("vol_regime"),
        "vol_expanding": algo.get("vol_expanding"),
    }

    worst_dd = None
    if historical_context:
        cands = [(lbl, h.get("max_drawdown_pct")) for lbl, h in
                 (historical_context.get("horizons") or {}).items()
                 if h.get("data_available") and h.get("max_drawdown_pct") is not None]
        if cands:
            lbl, dd = min(cands, key=lambda t: t[1])
            worst_dd = {"window": lbl, "max_drawdown_pct": dd}
    out["drawdown"] = worst_dd or {"status": "NOT_COMPUTABLE",
                                   "missing": "historical_context"}

    # Liquidity — from what the data layer already reports.
    avg_vol = (rec.get("liquidity") or {}).get("avg_volume_20d") or algo.get("volume_avg_20d")
    out["liquidity"] = ({"avg_volume_20d": avg_vol} if avg_vol else
                        {"status": "NOT_COMPUTABLE", "missing": "average volume"})

    # ── Position-level risk. Needs the holding. ──────────────────────────
    if position_context.get("status") != "PROVIDED":
        out["position_risk"] = {"status": "POSITION_SIZING_NOT_COMPUTABLE",
                                "reason": position_context.get("message")}
        out["inputs_missing"].extend(["avg_cost", "shares", "portfolio_value"])
        out["status"] = "PARTIAL"
        return out

    shares = position_context.get("shares")
    pv = position_context.get("portfolio_value")
    mv = position_context.get("market_value")

    pos_risk: Dict[str, Any] = {"status": "OK"}
    if shares and invalidation and current_price:
        loss_per_share = current_price - invalidation
        pos_risk["dollars_at_risk_to_invalidation"] = round(loss_per_share * shares, 2)
    else:
        pos_risk["dollars_at_risk_to_invalidation"] = None
        if not shares:
            out["inputs_missing"].append("shares")

    if pv and pv > 0 and pos_risk.get("dollars_at_risk_to_invalidation") is not None:
        pos_risk["portfolio_risk_if_invalidated_pct"] = round(
            pos_risk["dollars_at_risk_to_invalidation"] / pv * 100, 2)
    else:
        pos_risk["portfolio_risk_if_invalidated_pct"] = None
        if not pv:
            out["inputs_missing"].append("portfolio_value")

    pos_risk["weight_pct"] = position_context.get("weight_pct")
    pos_risk["concentration"] = position_context.get("concentration")

    if max_portfolio_risk_pct is not None and pos_risk.get("portfolio_risk_if_invalidated_pct") is not None:
        pos_risk["risk_budget_pct"] = max_portfolio_risk_pct
        pos_risk["within_risk_budget"] = (
            pos_risk["portfolio_risk_if_invalidated_pct"] <= max_portfolio_risk_pct)
        pos_risk["budget_note"] = (
            f"At the invalidation level this position would cost "
            f"{pos_risk['portfolio_risk_if_invalidated_pct']:.2f}% of the stated portfolio, "
            f"against a {max_portfolio_risk_pct:.2f}% stated budget.")
    else:
        pos_risk["risk_budget_pct"] = None
        pos_risk["within_risk_budget"] = None
        pos_risk["budget_note"] = ("No risk budget was supplied, so whether this exposure is "
                                   "acceptable cannot be answered — only described.")

    out["position_risk"] = pos_risk
    out["status"] = "OK" if not out["inputs_missing"] else "PARTIAL"
    if mv and pv:
        out["notes"].append(f"Position is {position_context['weight_pct']:.1f}% of the "
                            f"stated ${pv:,.0f} portfolio.")
    return out


def analyze_add(
    thesis: Dict[str, Any],
    edge: Dict[str, Any],
    entry: Dict[str, Any],
    position_context: Dict[str, Any],
    risk_budget: Dict[str, Any],
    conflict: Optional[Dict[str, Any]] = None,
    prior_thesis: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Phase 8 — should more be added, and if so on what kind of logic?

    Classifies the CANDIDATE add into one of the named kinds (averaging down,
    averaging up, pyramiding after confirmation, adding after a pullback,
    adding after catalyst confirmation) and then answers the ten questions the
    specification lists, each with a value or an explicit "cannot be answered".

    The verdict is never "add because it is cheaper". `cheaper_not_better` is
    computed first and, when true, is the headline.
    """
    if position_context.get("status") != "PROVIDED":
        return {
            "status": "POSITION_CONTEXT_NOT_PROVIDED",
            "verdict": "NOT_APPLICABLE",
            "message": ("Adding to a position can only be analysed against an existing "
                        "position. None was supplied."),
            "checks": [], "conditions": [],
        }

    pl = position_context["unrealized_pl_pct"]
    underwater = position_context["underwater"]

    # What KIND of add is being contemplated — determined by the position's own
    # state and the entry read, not chosen by the caller.
    if underwater and entry.get("status") in ("ATTRACTIVE", "ACCEPTABLE"):
        add_kind = "AVERAGING_DOWN"
    elif underwater:
        add_kind = "AVERAGING_DOWN"
    elif entry.get("status") == "ATTRACTIVE" and (entry.get("geometry") or {}).get("range_position", 1) <= 0.4:
        add_kind = "ADDING_AFTER_PULLBACK"
    elif thesis.get("direction") == "BULLISH" and pl > 0:
        add_kind = "AVERAGING_UP"
    else:
        add_kind = "PYRAMIDING_AFTER_CONFIRMATION"

    # ── Thesis strengthened / weakened / merely cheaper ──────────────────
    thesis_delta = "UNKNOWN"
    thesis_delta_detail = ("No prior decision record for this ticker is available, so whether "
                           "the thesis has strengthened or weakened since the position was "
                           "opened cannot be established. It is recorded from this decision "
                           "forward.")
    if prior_thesis:
        prev = prior_thesis.get("net_weight")
        now = thesis.get("net_weight")
        if prev is not None and now is not None:
            delta = now - prev
            if delta > 0.15:
                thesis_delta = "STRENGTHENED"
            elif delta < -0.15:
                thesis_delta = "WEAKENED"
            else:
                thesis_delta = "UNCHANGED"
            thesis_delta_detail = (f"Net directional weight moved {delta:+.2f} "
                                   f"({prev:+.2f} → {now:+.2f}) since the last recorded "
                                   f"decision.")

    cheaper_not_better = bool(
        underwater and thesis_delta in ("WEAKENED", "UNKNOWN"))

    checks: List[Dict[str, Any]] = [
        {"question": "Has the original thesis strengthened?",
         "answer": thesis_delta, "detail": thesis_delta_detail},
        {"question": "Has the thesis weakened?",
         "answer": "YES" if thesis_delta == "WEAKENED" else
                   ("NO" if thesis_delta in ("STRENGTHENED", "UNCHANGED") else "UNKNOWN"),
         "detail": thesis_delta_detail},
        {"question": "Has price merely fallen?",
         "answer": "YES" if underwater else "NO",
         "detail": (f"Position is {pl:+.1f}% against cost. A lower price improves the "
                    f"entry arithmetic; it says nothing on its own about the business."
                    if underwater else f"Position is {pl:+.1f}% against cost.")},
        {"question": "Has the statistical edge improved?",
         "answer": "YES" if edge.get("demonstrated") else "NO",
         "detail": (edge.get("sizing_consequence") or "") + " " +
                   f"{edge.get('n_gates_passed')}/{edge.get('n_gates')} gates pass."},
        {"question": "Has risk increased?",
         "answer": ("YES" if (risk_budget.get("volatility") or {}).get("vol_expanding")
                    else "NO" if (risk_budget.get("volatility") or {}).get("vol_expanding") is False
                    else "UNKNOWN"),
         "detail": (f"Volatility regime {(risk_budget.get('volatility') or {}).get('vol_regime')}, "
                    f"expanding={(risk_budget.get('volatility') or {}).get('vol_expanding')}.")},
        {"question": "Has volatility changed?",
         "answer": (risk_budget.get("volatility") or {}).get("vol_regime") or "UNKNOWN",
         "detail": (f"20d HV {(risk_budget.get('volatility') or {}).get('historical_volatility_20d')}, "
                    f"60d HV {(risk_budget.get('volatility') or {}).get('historical_volatility_60d')}.")},
        {"question": "Has the invalidation level changed?",
         "answer": str(risk_budget.get("invalidation_level")),
         "detail": (f"Invalidation sits at {risk_budget.get('invalidation_level')} "
                    f"({risk_budget.get('risk_to_invalidation_pct')}% away), basis "
                    f"{risk_budget.get('invalidation_basis')}. Whether it MOVED requires a "
                    f"prior decision record.")
                   if risk_budget.get("invalidation_level") else
                   "No invalidation level could be sourced."},
        {"question": "Has expected reward:risk improved?",
         "answer": str((entry.get("geometry") or {}).get("reward_risk_to_levels")),
         "detail": (entry.get("geometry") or {}).get("reward_risk_basis", "")},
        {"question": "Is the move caused by new information or simply price movement?",
         "answer": ("PRICE_MOVEMENT_ONLY" if cheaper_not_better else
                    "NEW_INFORMATION" if thesis_delta == "STRENGTHENED" else "UNKNOWN"),
         "detail": ("Nothing in the evidence has strengthened; only the price is lower."
                    if cheaper_not_better else thesis_delta_detail)},
        {"question": "Would adding push concentration past the risk budget?",
         "answer": ("NO" if (risk_budget.get("position_risk") or {}).get("within_risk_budget") is True
                    else "YES" if (risk_budget.get("position_risk") or {}).get("within_risk_budget") is False
                    else "NOT_COMPUTABLE"),
         "detail": (risk_budget.get("position_risk") or {}).get(
             "budget_note", "No risk budget supplied.")},
    ]

    blockers: List[str] = []
    if cheaper_not_better:
        blockers.append(
            "CHEAPER IS NOT BETTER: the price has fallen but nothing in the evidence has "
            "improved. A lower price improves the entry arithmetic and simultaneously may "
            "be the market pricing in a deterioration this analysis has not yet measured. "
            "Averaging down on price alone is not supported.")
    if not edge.get("demonstrated"):
        blockers.append(
            "No demonstrated statistical edge — position sizing is gated at 0%, which "
            "applies to an addition exactly as it applies to a new position.")
    if thesis.get("direction") != "BULLISH":
        blockers.append(f"The directional thesis is {thesis.get('direction')}, not bullish.")
    if entry.get("status") in ("OVEREXTENDED", "HIGH_RISK", "INVALIDATED"):
        blockers.append(f"Entry location reads {entry['status']}.")
    if (risk_budget.get("position_risk") or {}).get("within_risk_budget") is False:
        blockers.append("The position already exceeds the stated risk budget.")
    if position_context.get("concentration") == "CONCENTRATED":
        blockers.append(f"Position is already {position_context.get('weight_pct')}% of the "
                        f"stated portfolio.")
    for c in (conflict or {}).get("conflicts", []):
        if c.get("changes_decision"):
            blockers.append(f"Unresolved decision-changing conflict: {c['name']}.")
    nxt = (catalysts or {}).get("next_event")
    if nxt and nxt.get("days_away", 999) <= 14:
        blockers.append(f"{nxt['event']} is {nxt['days_away']} days away — adding immediately "
                        f"before a scheduled repricing increases exposure to an event this "
                        f"system cannot predict.")

    if blockers:
        verdict = "DO_NOT_ADD"
    elif thesis_delta == "STRENGTHENED" and entry.get("status") in ("ATTRACTIVE", "ACCEPTABLE"):
        verdict = "ADD_CONDITIONALLY"
    else:
        verdict = "NO_ADD_CASE_ESTABLISHED"

    conditions: List[Dict[str, Any]] = [{
        "label": "All of these must hold before adding",
        "all_of": [
            "The statistical gate has cleared (sizing is no longer 0%)",
            "The thesis has strengthened on evidence, not merely become cheaper",
            "Entry location reads ATTRACTIVE or ACCEPTABLE against sourced levels",
            "The addition keeps portfolio risk inside a stated budget",
            "No decision-changing conflict is unresolved",
        ],
        "why": ("Each of these is a separate failure mode. Adding when any one of them is "
                "false is how a position grows precisely as the reasons for holding it "
                "shrink."),
    }]

    return {
        "status": "OK",
        "add_kind": add_kind,
        "verdict": verdict,
        "cheaper_not_better": cheaper_not_better,
        "headline": (
            "CHEAPER, NOT BETTER — the price is lower, the case is not stronger."
            if cheaper_not_better else
            {"ADD_CONDITIONALLY": "An addition is defensible, but only under the conditions below.",
             "DO_NOT_ADD": "Adding is not supported right now.",
             "NO_ADD_CASE_ESTABLISHED": "No affirmative case for adding has been established."}[verdict]),
        "thesis_delta": thesis_delta,
        "blockers": blockers,
        "checks": checks,
        "conditions": conditions,
        "rule": ("PRICE FALLING ALONE IS NEVER A REASON TO ADD. A lower price may improve "
                 "valuation while simultaneously reflecting a deterioration in the thesis."),
    }
