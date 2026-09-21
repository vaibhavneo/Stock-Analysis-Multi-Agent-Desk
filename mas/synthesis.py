"""Sub-agent results -> one answer, and back into the evidence stream.

TWO DIRECTIONS, NOT ONE
-----------------------
The easy half is forward: the orchestrator holds a thesis, hands the direction
to the options specialist, and gets structures back. That is a hand-off, and
it was already working.

The half that makes this a system rather than a menu is the return path. A
specialist's finding has to be able to CHANGE the orchestrator's read, not
just decorate it. So findings are emitted as `DecisionEvidence` — the same
normalized shape the seven pillars already produce — which means they flow
through conflict analysis, horizon partitioning and confidence exactly like
every other observation, instead of sitting in a panel nothing reads. That
was the recurring failure this codebase keeps finding, and a new layer is
precisely where it would recur.

The sharpest example is `benchmark_relation`. When bitcoin explains 82% of
ETH's variance, an ETH-specific thesis is mostly a bitcoin bet wearing a
different ticker. That is not a footnote — it caps how much independent
conviction the rest of the analysis is entitled to, and it is emitted as
NOT_DIRECTIONAL evidence for the same reason the risk pillar is: "this moves
with bitcoin" is not a claim that it goes up.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .asset_class import spec_for

# Above this share of variance explained by the benchmark, a symbol-specific
# thesis is materially a benchmark bet. Chosen, not fitted: at r-squared 0.6
# the benchmark accounts for more of the movement than everything else
# combined, which is the point where the claim changes character.
BENCHMARK_DOMINANCE_R2 = 0.60


def _article(word: str) -> str:
    """'an ACCUMULATE view', not 'a ACCUMULATE view'. Verdict words arrive
    from a closed vocabulary that happens to include vowel-initial members,
    so the article cannot be hard-coded."""
    return "an" if (word or "")[:1].upper() in "AEIOU" else "a"


def _evidence_from_benchmark(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    r2 = data.get("r_squared")
    corr = data.get("correlation")
    if r2 is None or corr is None:
        return None
    return {
        "source": "cross_asset",
        "category": "RELATIVE",
        "metric": "benchmark_r_squared",
        "observation": data.get("verdict") or "",
        # A relationship is not a direction, exactly as a risk score is not.
        "direction": "NOT_DIRECTIONAL",
        "magnitude": min(1.0, abs(float(corr))),
        "horizon": "ALL",
        "reliability": 0.8,
        "validation_status": "TRACKED_FORWARD",
        "data_quality": "OK" if data.get("overlapping_returns", 0) >= 120 else "PARTIAL",
        "decision_relevance": ("DECISIVE" if r2 >= BENCHMARK_DOMINANCE_R2
                               else "CONTEXT"),
        "raw_value": r2,
        "provenance": {"benchmark": data.get("benchmark"),
                       "overlapping_returns": data.get("overlapping_returns"),
                       "period": data.get("period")},
        "flags": (["thesis_is_largely_a_benchmark_bet"]
                  if r2 >= BENCHMARK_DOMINANCE_R2 else []),
    }


def _evidence_from_options(data: Dict[str, Any], basis: Optional[str]) -> Optional[Dict[str, Any]]:
    """Only a REAL chain can say anything about what the market charges.

    A model engine's realized volatility is not evidence about market
    expectations — emitting it as such would be the single most misleading
    thing this layer could do, because it would look exactly like the real
    signal.
    """
    if basis != "TRADED_PRICE":
        return None
    surface = data.get("surface") or {}
    rank = surface.get("iv_rank") or surface.get("rank")
    if rank is None:
        return None
    try:
        rank = float(rank)
    except (TypeError, ValueError):
        return None
    rank_pct = rank * 100.0 if rank <= 1.0 else rank
    return {
        "source": "options_pilot",
        "category": "RISK",
        "metric": "implied_volatility_rank",
        "observation": (
            f"The options market is pricing volatility at the "
            f"{rank_pct:.0f}th percentile of its own recent range."
            + (" It expects a larger move than a quiet tape would imply."
               if rank_pct >= 70 else
               " It expects an unusually quiet stretch." if rank_pct <= 30 else "")),
        "direction": "NOT_DIRECTIONAL",
        "magnitude": abs(rank_pct - 50.0) / 50.0,
        "horizon": "SHORT",
        "reliability": 0.75,
        "validation_status": "TRACKED_FORWARD",
        "data_quality": "OK",
        "decision_relevance": "SUPPORTING",
        "raw_value": rank_pct,
        "provenance": {"service": "optionspilot", "basis": "TRADED_PRICE"},
        "flags": [],
    }


def to_evidence(execution: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sub-agent findings as DecisionEvidence dicts.

    Returned as dicts rather than DecisionEvidence objects so this module has
    no import dependency on `decision/` — the orchestration layer must not
    require the decision layer to be loadable, or a crypto symbol (which has
    no seven-pillar recommendation at all) could not be answered.
    """
    out: List[Dict[str, Any]] = []
    for cap, o in (execution.get("by_capability") or {}).items():
        res = o.get("result")
        if not res or res.get("status") != "OK":
            continue
        data = res.get("data") or {}
        if cap == "benchmark_relation":
            e = _evidence_from_benchmark(data)
        elif cap == "implied_volatility":
            e = _evidence_from_options(data, res.get("price_basis"))
        else:
            e = None
        if e:
            out.append(e)
    return out


def _options_section(o: Dict[str, Any]) -> Dict[str, Any]:
    res = o.get("result") or {}
    data = res.get("data") or {}
    basis = res.get("price_basis")
    cands = data.get("candidates") or []

    rows = []
    for c in cands[:4]:
        s = c.get("structure") or {}
        rows.append({
            "name": s.get("name"),
            "legs": s.get("legs"),
            "rationale": s.get("rationale"),
            "expiry_days": s.get("expiry_days"),
            "cost": c.get("net_cost"),
            "cash_direction": c.get("direction_of_cash"),
            "max_profit": c.get("max_profit"),
            "max_profit_unbounded": c.get("max_profit_unbounded"),
            "max_loss": c.get("max_loss"),
            "breakevens": c.get("breakevens"),
            "probability_of_profit_risk_neutral":
                c.get("probability_of_profit_risk_neutral"),
        })
    return {
        "answered_by": o.get("answered_by"),
        "basis": basis,
        "is_model_priced": basis == "MODEL",
        "fell_back": o.get("fell_back"),
        "view": data.get("view"),
        "spot": data.get("spot"),
        "volatility_annualized_pct": data.get("volatility_annualized_pct"),
        "volatility_basis": data.get("volatility_basis"),
        "annualization_days": data.get("annualization_days"),
        "structures": rows,
        "honesty": data.get("honesty") or [],
    }


def synthesize(execution: Dict[str, Any],
               core: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One answer from all the specialists that answered.

    `core` is the orchestrator's own read (a recommendation / decision
    intelligence object) when there is one. It is optional on purpose: a
    crypto symbol has no seven-pillar recommendation, and the specialists
    still have a useful answer about it.
    """
    caps = execution.get("by_capability") or {}
    symbol = execution.get("symbol")
    asset_class = execution.get("asset_class")
    spec = spec_for(asset_class or "UNKNOWN")

    sections: Dict[str, Any] = {}
    tensions: List[Dict[str, str]] = []
    refinements: List[str] = []

    bench = caps.get("benchmark_relation") or {}
    if (bench.get("result") or {}).get("status") == "OK":
        d = bench["result"]["data"]
        sections["benchmark"] = d
        r2 = d.get("r_squared")
        if r2 is not None and r2 >= BENCHMARK_DOMINANCE_R2:
            refinements.append(
                f"{d['benchmark_label'].capitalize()} explains "
                f"{r2:.0%} of {symbol}'s movement. A view specific to "
                f"{symbol} is mostly a view on {d['benchmark_label']}, and "
                f"deserves less independent conviction than it looks like.")
            if core and (core.get("action") or "").upper() not in ("", "HOLD"):
                tensions.append({
                    "between": "the symbol-specific read and the benchmark relation",
                    "tension": (
                        f"The desk has {_article(core.get('action'))} "
                        f"{core.get('action')} view on {symbol} "
                        f"specifically, but {r2:.0%} of its movement comes from "
                        f"{d['benchmark_label']}. Sizing it as an independent "
                        f"idea would double up on an exposure already held."),
                })

    opts = caps.get("option_structures") or {}
    if (opts.get("result") or {}).get("status") == "OK":
        sections["options"] = _options_section(opts)
        if opts.get("fell_back"):
            refinements.append(
                "OptionsPilot could not answer, so these structures are priced "
                "from a model on realized volatility rather than from live "
                "quotes. Treat the prices as reference values, not as costs.")
    elif opts:
        sections["options"] = {"answered_by": None,
                               "reason": opts.get("unanswered_reason")}

    iv = caps.get("implied_volatility") or {}
    if (iv.get("result") or {}).get("status") == "OK":
        sections["implied_volatility"] = iv["result"]["data"]

    # Standing caveats that belong to the asset class itself, not to a result.
    class_caveats: List[str] = []
    if spec.note:
        class_caveats.append(spec.note)
    if spec.options == "NO_ACCESSIBLE_VENUE":
        class_caveats.append(
            f"No venue this system reads lists options on a {spec.label}. Any "
            "structure shown is an analytical shape, not something orderable "
            "from here.")
    if spec.inapplicable_pillars:
        class_caveats.append(
            "Not scored for " + ", ".join(sorted(spec.inapplicable_pillars))
            + f": a {spec.label} has no such input, so those pillars are "
            "excluded from the composite rather than scored neutral.")

    answered = execution.get("answered") or []
    unanswered = execution.get("unanswered") or []
    if answered:
        headline = (f"{symbol} is {spec.label}. "
                    f"{len(answered)} specialist"
                    f"{'s' if len(answered) != 1 else ''} answered"
                    + (f"; {len(unanswered)} could not." if unanswered else "."))
    else:
        headline = (f"{symbol} is {spec.label}, and no specialist could answer. "
                    "See the trace for why.")

    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "asset_label": spec.label,
        "headline": headline,
        "sections": sections,
        "refinements": refinements,
        "tensions": tensions,
        "class_caveats": class_caveats,
        "evidence": to_evidence(execution),
        "answered": answered,
        "unanswered": unanswered,
        "trace": execution.get("trace"),
        "elapsed_ms": execution.get("elapsed_ms"),
    }
