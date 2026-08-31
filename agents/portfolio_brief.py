"""
Portfolio Decision Brief v2 — position-aware, weight-aware action plan per holding
plus a portfolio-level stance that is NOT an average of stock scores.

Architecture:
    seven-agent analysis (per holding)
        -> objective stock verdict          (decision_brief / decision_synthesis)
        + user position (avg cost, shares)   -> position-aware decision
        + portfolio weight & concentration   -> ADD / HOLD / TRIM / EXIT + price zones
        -> portfolio stance                  (structural, not a score average)

This is a PURE CONSUMER: it reuses `agents.decision_brief.build_decision_brief`
for every holding and adds only deterministic weight/level arithmetic. It creates
no agents, strategies, providers, databases, or scores, and never lets LLM prose
change an action, level, weight, or sizing decision.

Position actions:
    ADD   — objective thesis positive, statistical + allocation edge proven, NO risk
            veto, position below its target weight, and price inside a supported add
            zone. Blocked whenever the position already meets/exceeds its max weight.
    TRIM  — overweight (above max permitted weight) even if the stock is strong, OR
            price at/above the trim (resistance/valuation) zone, OR a neutral-tier
            position sitting on a large unrealized gain (profit-taking).
    EXIT  — thesis break or risk veto (a WEAK stock exits even if it's below cost;
            the exit level is an ATR/risk stop, never the user's cost basis).
    HOLD  — none of the above; keep the current position.
    WAIT  — a would-be ADD whose price levels cannot be computed (LEVEL_UNAVAILABLE).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.decision_brief import build_decision_brief

# Concentration ceiling: no single position may exceed this share of the book.
# A stated prior (like every threshold here), not a tuned value.
DEFAULT_MAX_WEIGHT_PCT = 25.0

# Cross-position correlation: two holdings above this pairwise correlation are
# treated as partly the same bet. Stated prior, matching backtest.risk's own
# default — NOT fitted, and deliberately not tuned against outcomes (that would
# be weight-shopping; see backtest/pillars.py's module docstring).
CORR_THRESHOLD = 0.7
# Below this many overlapping return observations a correlation estimate is
# noise, so no haircut is applied and the holding is flagged instead.
MIN_CORR_OBS = 30

# Sector ceiling: no single sector may exceed this share of the book. Looser
# than the per-position cap because a sector is a legitimate thesis, tighter
# than "anything goes" because a 60%-tech book is one macro bet. Stated prior.
DEFAULT_MAX_SECTOR_PCT = 40.0


def build_portfolio_brief(
    holdings: List[Dict[str, Any]],
    max_weight_pct: float = DEFAULT_MAX_WEIGHT_PCT,
    returns: Optional[Dict[str, Any]] = None,
    max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT,
) -> Dict[str, Any]:
    """holdings: list of dicts, each with at least:
        {"ticker", "shares", "avg_cost", "recommendation": <rec dict>}
      and optionally "backtest_all", "calibration", "xsec_ranking",
      "prediction_summary", "orchestrator_result".

    returns: optional {ticker: pd.Series of daily returns}. When supplied, a
      cross-position correlation haircut is applied to TARGET weights, so a
      basket of names that all move together can't quietly become one oversized
      bet that passes the per-position concentration ceiling. Omitted (the
      default) reproduces the previous behaviour exactly.

      Passed IN rather than fetched here on purpose: this module is a pure
      consumer that creates no providers (see the module docstring), and the
      caller already holds price history.

    Returns {"holdings": [<per-holding brief>...], "portfolio": {<summary>}}.
    """
    now = datetime.now().isoformat(timespec="seconds")

    # ── Pass 1: market values + total (needed before any weight can be known) ──
    priced = []
    total_value = 0.0
    total_cost = 0.0
    for h in holdings:
        rec = h.get("recommendation") or {}
        shares = _num(h.get("shares"))
        avg_cost = _num(h.get("avg_cost"))
        current_price = _num(rec.get("current_price"))
        mv = (current_price * shares) if (current_price and shares) else None
        cost_value = (avg_cost * shares) if (avg_cost and shares) else None
        if mv is not None:
            total_value += mv
        elif cost_value is not None:
            total_value += cost_value  # fall back to cost until priced
        if cost_value is not None:
            total_cost += cost_value
        priced.append({"h": h, "rec": rec, "shares": shares, "avg_cost": avg_cost,
                       "current_price": current_price, "mv": mv, "cost_value": cost_value})

    # ── Pass 2: per-holding position-aware, weight-aware brief ────────────────
    haircuts = _correlation_haircuts([str(p["h"].get("ticker", "")).upper().strip()
                                      for p in priced], returns)
    out_holdings: List[Dict[str, Any]] = []
    for p in priced:
        cur_weight = (100.0 * p["mv"] / total_value) if (p["mv"] and total_value) else 0.0
        ticker = str(p["h"].get("ticker", "")).upper().strip()
        out_holdings.append(_holding_brief(p, cur_weight, max_weight_pct,
                                           haircut=haircuts.get(ticker)))

    portfolio = _portfolio_summary(out_holdings, total_value, total_cost, max_weight_pct)
    portfolio["exposure"] = _exposure(out_holdings, total_value, max_sector_pct)
    portfolio["generated_at"] = now
    portfolio["max_weight_pct"] = max_weight_pct
    portfolio["disclaimer"] = ("Portfolio brief synthesized from computed outputs — "
                                "suggested actions, not automatic trades. Not financial advice.")
    return {"holdings": out_holdings, "portfolio": portfolio}


# ── Portfolio-level exposure ───────────────────────────────────────────────

def _exposure(holdings: List[Dict[str, Any]], total_value: float,
              max_sector_pct: float) -> Dict[str, Any]:
    """Gross/net/cash exposure and sector concentration for the whole book.

    This is the layer the per-position ceiling structurally cannot provide:
    every holding can sit comfortably under its own cap while the book is
    still a single sector bet. Long-only by design (see backtest/pillars.py),
    so gross == net today; both are reported anyway so the shape of this
    output does not have to change if shorts are ever added.

    Invested weight is measured from CURRENT weights, not targets — this
    describes the book as it actually stands, not as it is meant to become.
    """
    invested = sum(h.get("current_weight_pct") or 0.0 for h in holdings)
    by_sector: Dict[str, float] = {}
    unknown_sector = 0.0
    for h in holdings:
        w = h.get("current_weight_pct") or 0.0
        if w <= 0:
            continue
        sector = ((h.get("detail_brief") or {}).get("sector")
                  or h.get("sector") or "Unknown")
        sector = str(sector).strip() or "Unknown"
        by_sector[sector] = by_sector.get(sector, 0.0) + w
        if sector == "Unknown":
            unknown_sector += w

    sectors = [{"sector": s, "weight_pct": round(w, 2),
                "over_max": w > max_sector_pct}
               for s, w in sorted(by_sector.items(), key=lambda kv: kv[1], reverse=True)]
    breaches = [s for s in sectors if s["over_max"] and s["sector"] != "Unknown"]

    return {
        "gross_exposure_pct": round(invested, 2),
        "net_exposure_pct": round(invested, 2),     # long-only: equal by construction
        "cash_pct": round(max(0.0, 100.0 - invested), 2),
        "max_sector_pct": max_sector_pct,
        "by_sector": sectors,
        "sector_breaches": breaches,
        # Sector attribution is only as good as the sector labels; say so rather
        # than letting an unknown-heavy book read as diversified.
        "unclassified_weight_pct": round(unknown_sector, 2),
        "total_value": round(total_value, 2),
    }


# ── Cross-position correlation ─────────────────────────────────────────────

def _correlation_haircuts(tickers: List[str],
                          returns: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """{ticker: multiplier in (0, 1]} — how much of its standalone target weight
    a holding keeps once overlap with the REST of the book is accounted for.

    Delegates the arithmetic to backtest.risk.correlation_aware_position_size
    rather than re-deriving it: feeding that function a uniform 1.0 per name
    makes its output the pure haircut factor, so the portfolio layer and the
    strategy layer provably de-duplicate correlation the same way. Two copies
    of this formula drifting apart is exactly the bug worth designing out.

    Returns {} (no adjustment anywhere) whenever correlation can't be judged:
    no series supplied, fewer than two holdings, or too little overlapping
    history. Silence here means "unknown", never "uncorrelated".
    """
    if not returns or len(tickers) < 2:
        return {}
    try:
        import pandas as pd

        from backtest.risk import correlation_aware_position_size

        series = {t: returns[t] for t in tickers
                  if t in returns and returns[t] is not None}
        if len(series) < 2:
            return {}
        overlap = pd.DataFrame(series).dropna()
        if len(overlap) < MIN_CORR_OBS:
            return {}
        adjusted = correlation_aware_position_size(
            {t: 1.0 for t in series}, series, corr_threshold=CORR_THRESHOLD)
        # Only meaningful haircuts are reported; 1.0 means "no overlap found".
        return {t: float(v) for t, v in adjusted.items() if float(v) < 0.999}
    except Exception:
        # Correlation is an enhancement to sizing, never a precondition for
        # producing a brief - a failure here must degrade to today's behaviour.
        return {}


# ── Per-holding brief ──────────────────────────────────────────────────────

def _holding_brief(p: Dict[str, Any], cur_weight: float,
                   max_weight: float,
                   haircut: Optional[float] = None) -> Dict[str, Any]:
    h, rec = p["h"], p["rec"]
    ticker = str(h.get("ticker", "")).upper().strip()
    shares, avg_cost = p["shares"], p["avg_cost"]
    current_price, mv = p["current_price"], p["mv"]

    position = None
    if avg_cost and avg_cost > 0:
        position = {"avg_cost": avg_cost, "shares": shares}

    # Objective + position-aware stock verdict — the deterministic backbone.
    brief = build_decision_brief(
        ticker, rec,
        backtest_all=h.get("backtest_all"),
        calibration=h.get("calibration"),
        xsec_ranking=h.get("xsec_ranking"),
        prediction_summary=h.get("prediction_summary"),
        orchestrator_result=h.get("orchestrator_result"),
        owns_position=True if position else None,
        position=position,
    )

    if brief.get("final_action") == "INSUFFICIENT_EVIDENCE":
        return {
            "ticker": ticker, "shares": shares, "avg_cost": avg_cost,
            "current_price": current_price, "market_value": _round(mv),
            "current_weight_pct": round(cur_weight, 2),
            "position_action": "WAIT",
            "levels": {"status": "LEVEL_UNAVAILABLE"},
            "reasons": ["Required analysis inputs are missing for this holding."],
            "detail_brief": brief,
        }

    target_weight = _target_weight(rec, brief, cur_weight, max_weight)

    # Correlation haircut: shrink only the ADD headroom, never below what is
    # already held. Rationale — portfolio-level correlation has never been
    # validated through the honest stack (no walk-forward, no dSR), so by this
    # codebase's own rule an unvalidated signal may tilt but must not drive.
    # Letting it push target BELOW cur_weight would manufacture TRIM/EXIT
    # actions - real sells - off an unbacktested estimate. It stays visible in
    # the holding and the portfolio summary either way.
    correlation_note = None
    if haircut is not None and target_weight > cur_weight:
        adjusted = max(cur_weight, target_weight * haircut)
        if adjusted < target_weight:
            correlation_note = {
                "haircut": round(haircut, 3),
                "pre_correlation_target_pct": round(target_weight, 2),
                "reason": (f"Overlaps other holdings above {CORR_THRESHOLD:.2f} "
                           f"correlation; add headroom cut to avoid one bet held "
                           f"under several tickers."),
            }
            target_weight = adjusted

    levels = _compute_position_levels(rec, cur_weight, max_weight)
    action = _position_action(rec, brief, cur_weight, target_weight, max_weight, levels)
    urgency = _urgency(action, rec, cur_weight, max_weight, levels)

    pl_pct = ((current_price - avg_cost) / avg_cost * 100) if (avg_cost and current_price) else None
    pl_dollar = ((current_price - avg_cost) * shares) if (avg_cost and current_price and shares) else None

    return {
        "ticker": ticker,
        "sector": rec.get("sector") or "Unknown",   # feeds portfolio exposure
        "shares": shares,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "market_value": _round(mv),
        "unrealized_pl_pct": _round(pl_pct),
        "unrealized_pl_dollar": _round(pl_dollar),

        "current_weight_pct": round(cur_weight, 2),
        "target_weight_pct": round(target_weight, 2),
        "max_weight_pct": max_weight,
        "overweight": cur_weight > max_weight,
        "correlation": correlation_note,

        "position_action": action,
        "objective_action": brief.get("objective_action"),
        "urgency": urgency,
        "levels": levels,

        "reasons": brief.get("decisive_insights", [])[:3],
        "upgrade_conditions": (brief.get("triggers") or {}).get("upgrade", []),
        "downgrade_conditions": (brief.get("triggers") or {}).get("downgrade", []),
        "confidence": brief.get("evidence_status", {}),
        "review_note": (brief.get("action_plan") or {}).get("review_note"),

        # Full seven-agent-derived brief kept for the expandable detail section.
        "detail_brief": brief,
    }


# ── Target weight (deterministic; edge-gated) ───────────────────────────────

def _target_weight(rec: Dict[str, Any], brief: Dict[str, Any],
                   cur_weight: float, max_weight: float) -> float:
    """What weight this position deserves.

    Negative tier -> 0 (exit). Proven positive (edge HIGH, not gated) -> the
    Kelly-sized position, capped at the concentration ceiling. Positive but
    unproven -> hold what's there but don't grow it (target = current), so ADD
    only ever fires when the edge is actually established."""
    objective = brief.get("objective_action")
    if objective in ("REDUCE", "AVOID"):
        return 0.0

    conf = rec.get("confidence") or {}
    stat = (conf.get("statistical_edge") or {}).get("level")
    gated = rec.get("position_size_gated", True)
    size_pct = _num(rec.get("position_size_pct")) or 0.0

    if stat == "HIGH" and not gated and size_pct > 0:
        return min(size_pct, max_weight)
    # Positive but unproven: hold, don't add.
    return min(cur_weight, max_weight)


# ── Price zones (deterministic; never LLM-invented) ─────────────────────────

def _compute_position_levels(rec: Dict[str, Any], cur_weight: float,
                             max_weight: float) -> Dict[str, Any]:
    """Add / trim / exit zones from existing ATR-anchored levels + gates.

    Add zone requires no risk veto and sufficient statistical + allocation
    confidence. Trim zone is the resistance/valuation ceiling (target price).
    Exit is the ATR/risk stop — a risk break, never the user's cost basis.
    Returns status LEVEL_UNAVAILABLE when the needed prices aren't present."""
    levels = rec.get("levels") or {}
    entry_low = _num(levels.get("entry_zone_low"))
    entry_high = _num(levels.get("entry_zone_high"))
    target = _num(levels.get("target_price"))
    stop = _num(levels.get("stop_loss"))
    atr = _num(levels.get("atr_14"))
    cp = _num(rec.get("current_price"))

    # Need a current price plus the ATR-anchored stop (risk exit) and target
    # (trim ceiling) for any supported zone to exist.
    if not cp or stop is None or target is None:
        return {"status": "LEVEL_UNAVAILABLE",
                "note": "Supported price levels could not be computed for this holding."}

    conf = rec.get("confidence") or {}
    stat = (conf.get("statistical_edge") or {}).get("level")
    alloc = (conf.get("allocation") or {}).get("level")
    risk_veto = bool(rec.get("risk_veto"))
    gated = rec.get("position_size_gated", True)

    add_supported = (
        not risk_veto
        and stat in ("MEDIUM", "HIGH")
        and not gated
        and alloc in ("MEDIUM", "HIGH")
        and cur_weight < max_weight
        and entry_low is not None and entry_high is not None
    )

    add_zone = [round(entry_low, 2), round(entry_high, 2)] if add_supported else None
    # Trim ceiling = valuation/resistance target and above.
    trim_zone_low = round(target, 2)
    trim_zone_high = round(target + atr, 2) if atr else None
    exit_level = round(stop, 2)  # risk break

    return {
        "status": "OK",
        "add_zone": add_zone,                 # None => not a supported add right now
        "add_supported": add_supported,
        "trim_zone_low": trim_zone_low,
        "trim_zone_high": trim_zone_high,
        "exit_level": exit_level,
        "current_price": cp,
        "in_trim_zone": cp >= trim_zone_low,
        "below_exit": cp <= exit_level,
    }


# ── Position action (deterministic) ─────────────────────────────────────────

def _position_action(rec: Dict[str, Any], brief: Dict[str, Any],
                     cur_weight: float, target_weight: float,
                     max_weight: float, levels: Dict[str, Any]) -> str:
    objective = brief.get("objective_action")          # BUY/ACCUMULATE/HOLD/WATCH/REDUCE/AVOID
    final = brief.get("final_action")                  # position-aware (may be REDUCE for profit-take)
    risk_veto = bool(rec.get("risk_veto"))

    # EXIT: thesis break or risk veto with a negative thesis. A weak stock exits
    # even if it's below cost — decision is never anchored to recovering cost basis.
    if objective in ("REDUCE", "AVOID"):
        return "EXIT"

    # Risk veto never allows adding; if the position is non-trivial, trim it down.
    if risk_veto:
        return "TRIM" if cur_weight > 0 else "HOLD"

    # Overweight beats a positive thesis: a strong stock is still TRIM when it
    # exceeds its maximum permitted weight (concentration control).
    if cur_weight > max_weight:
        return "TRIM"

    # Position-aware profit-taking on a neutral objective signal (HOLD -> REDUCE
    # inside the stock brief) surfaces as TRIM at the portfolio level.
    if objective == "HOLD" and final == "REDUCE":
        return "TRIM"

    # In the trim (resistance/valuation) zone -> TRIM.
    if levels.get("status") == "OK" and levels.get("in_trim_zone"):
        return "TRIM"

    # ADD only when the edge is proven, a supported add zone exists, and there's
    # room below the target weight. Blocked when already at/over max weight (above).
    if objective in ("BUY", "ACCUMULATE"):
        if levels.get("status") != "OK":
            return "WAIT"          # would add, but no supported levels
        if levels.get("add_supported") and cur_weight < target_weight:
            return "ADD"
        return "HOLD"

    return "HOLD"


# ── Urgency (for ranking the top-3 portfolio actions) ───────────────────────

def _urgency(action: str, rec: Dict[str, Any], cur_weight: float,
             max_weight: float, levels: Dict[str, Any]) -> int:
    """0–100. Bigger positions and harder triggers rank higher. Weight is added
    so the same action on a larger holding is more urgent to address."""
    base = {"EXIT": 60, "TRIM": 45, "WAIT": 25, "ADD": 30, "HOLD": 0}.get(action, 0)
    if action == "EXIT" and rec.get("risk_veto"):
        base = 90
    if action == "TRIM" and cur_weight > max_weight:
        base = 70
    return min(100, int(base + cur_weight))


# ── Portfolio summary (structural, NOT a score average) ─────────────────────

def _portfolio_summary(holdings: List[Dict[str, Any]], total_value: float,
                       total_cost: float, max_weight: float) -> Dict[str, Any]:
    ranked_by_weight = sorted(holdings, key=lambda h: h.get("current_weight_pct", 0), reverse=True)
    overweight = [h for h in holdings if h.get("overweight")]
    n_exit = sum(1 for h in holdings if h["position_action"] == "EXIT")
    n_trim = sum(1 for h in holdings if h["position_action"] == "TRIM")
    n_add = sum(1 for h in holdings if h["position_action"] == "ADD")

    # Weight sitting in EXIT/TRIM vs ADD/HOLD — a structural read of the book.
    weight_needs_action = sum(h.get("current_weight_pct", 0) for h in holdings
                              if h["position_action"] in ("EXIT", "TRIM"))

    # Concentration risks: positions that are the largest and/or over the cap.
    concentration_risks = [
        {"ticker": h["ticker"], "weight_pct": h.get("current_weight_pct", 0),
         "over_max": h.get("overweight", False)}
        for h in ranked_by_weight[:3] if h.get("current_weight_pct", 0) > 0
    ]

    # Correlation risks: named holdings whose add headroom was cut because they
    # move with the rest of the book. Reported even when the haircut changed no
    # action, because the concentration ceiling alone cannot see this - five
    # names at 20% each clear every per-position check while being one bet.
    correlation_risks = [
        {"ticker": h["ticker"],
         "weight_pct": h.get("current_weight_pct", 0),
         "haircut": (h.get("correlation") or {}).get("haircut"),
         "pre_correlation_target_pct": (h.get("correlation") or {}).get("pre_correlation_target_pct")}
        for h in ranked_by_weight if h.get("correlation")
    ]
    # Correlated weight = share of the book already sitting in names that
    # overlap each other. This is the number the per-position ceiling misses.
    correlated_weight_pct = round(
        sum(r["weight_pct"] for r in correlation_risks), 2)

    # Best supported add opportunity: an ADD with the highest objective composite.
    adds = [h for h in holdings if h["position_action"] == "ADD"]
    best_add = max(adds, key=lambda h: _composite(h), default=None)

    # Highest-priority trim/exit by urgency.
    trims_exits = [h for h in holdings if h["position_action"] in ("TRIM", "EXIT")]
    top_trim_exit = max(trims_exits, key=lambda h: h.get("urgency", 0), default=None)

    # Top 3 actions by urgency (excluding plain HOLD).
    actionable = sorted([h for h in holdings if h["position_action"] != "HOLD"],
                        key=lambda h: h.get("urgency", 0), reverse=True)
    top_3 = [{"ticker": h["ticker"], "action": h["position_action"],
              "urgency": h.get("urgency", 0),
              "reason": (h.get("reasons") or ["—"])[0]} for h in actionable[:3]]

    stance = _portfolio_stance(n_exit, n_trim, n_add, overweight, weight_needs_action)

    # Portfolio-level confidence: the weakest data confidence across holdings is
    # the honest ceiling; calibration is shared (insufficient live history).
    data_levels = [(h.get("confidence") or {}).get("data_confidence") for h in holdings]
    data_conf = _weakest(data_levels)
    cal_conf = "insufficient history"
    for h in holdings:
        cm = (h.get("confidence") or {}).get("calibration_maturity")
        if cm and "insufficient" not in cm:
            cal_conf = cm
            break

    total_pl = total_value - total_cost
    return {
        "stance": stance,
        "n_holdings": len(holdings),
        "total_value": _round(total_value),
        "total_cost": _round(total_cost),
        "total_pl": _round(total_pl),
        "total_return_pct": _round((total_pl / total_cost * 100) if total_cost else None),
        "concentration_risks": concentration_risks,
        "correlation_risks": correlation_risks,
        "correlated_weight_pct": correlated_weight_pct,
        "overweight_holdings": [
            {"ticker": h["ticker"], "weight_pct": h.get("current_weight_pct", 0),
             "max_weight_pct": max_weight} for h in overweight],
        "best_add_opportunity": ({"ticker": best_add["ticker"],
                                   "reason": (best_add.get("reasons") or ["—"])[0]}
                                  if best_add else None),
        "top_priority_trim_exit": ({"ticker": top_trim_exit["ticker"],
                                     "action": top_trim_exit["position_action"],
                                     "reason": (top_trim_exit.get("reasons") or ["—"])[0]}
                                    if top_trim_exit else None),
        "top_3_actions": top_3,
        "data_confidence": data_conf,
        "calibration_confidence": cal_conf,
        "counts": {"add": n_add, "trim": n_trim, "exit": n_exit,
                   "overweight": len(overweight)},
    }


def _portfolio_stance(n_exit, n_trim, n_add, overweight, weight_needs_action) -> Dict[str, str]:
    """A structural label — derived from position actions and concentration, NOT
    from averaging the underlying stock scores."""
    if n_exit >= 1 and weight_needs_action >= 20:
        return {"label": "ELEVATED RISK",
                "note": "One or more holdings warrant an exit and a large share of the "
                        "book needs action — de-risk before adding anywhere."}
    if overweight:
        return {"label": "CONCENTRATED",
                "note": "One or more positions exceed the maximum weight — trim to "
                        "restore balance before deploying new capital."}
    if n_add > (n_trim + n_exit):
        return {"label": "CONSTRUCTIVE",
                "note": "More holdings support adding than trimming — selective, "
                        "edge-gated adds are reasonable within weight limits."}
    if (n_trim + n_exit) == 0 and n_add == 0:
        return {"label": "STABLE",
                "note": "No holding is flashing an action — hold and monitor review dates."}
    return {"label": "BALANCED",
            "note": "A mix of trims and holds — rebalance opportunistically, no forced moves."}


# ── Small helpers ──────────────────────────────────────────────────────────

def _composite(h: Dict[str, Any]) -> float:
    db = h.get("detail_brief") or {}
    return _num(db.get("composite_score")) or 0.0


def _weakest(levels: List[Optional[str]]) -> str:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    present = [l for l in levels if l in order]
    if not present:
        return "—"
    return min(present, key=lambda l: order[l])


def _num(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _round(v, nd: int = 2):
    return round(v, nd) if isinstance(v, (int, float)) else None
