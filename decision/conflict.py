"""
Contradiction / agreement / evidence-quality engine (Phase 12).

The problem this exists for is visible in the app's own output: a ticker can
simultaneously show a technical score of 60 ("neutral"), an algo read of
bearish, a negative thesis, a "MEDIUM" statistical edge, a dSR of 0.00, and
"calibration: insufficient" — six statements that cannot all be acted on the
same way, presented side by side with nothing reconciling them.

Averaging them is the wrong fix. An average of "strongly bullish" and "strongly
bearish" is "neutral", which is a claim neither input supports and which hides
the single most decision-relevant fact: that the evidence is in conflict and
conviction should therefore be lower.

So every disagreement is NAMED and answered against six questions:

    WHAT disagrees?   the two sources, by name
    WHY?               the mechanical rule that fired
    WHICH HORIZON?     both sources' horizon bands
    WHICH DATA SOURCE? provenance module for each
    HOW RELIABLE?      both sources' reliability
    DOES IT CHANGE THE DECISION?  whether it is severe enough to cap conviction

intelligence/evidence_synthesis.py::detect_contradictions already implements
four named pairwise rules over the raw pillars, and those are reused verbatim
rather than reimplemented — this module adds the evidence-level conflicts that
only become visible once everything is on one normalized scale (directional
disagreement between any two DECISIVE items, and the "strong thesis / no
demonstrated edge" split that no pillar-pair rule can see).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from decision.evidence import DecisionEvidence, directional

# Two directional items conflict when both carry real weight. A high-weight
# bull against a near-zero-weight bear is not a conflict — it is one opinion
# and one shrug, and reporting it as a conflict would make genuine conflicts
# unreadable by burying them.
MIN_CONFLICT_WEIGHT = 0.25

# Combined weight at which a directional conflict is severe enough to actually
# cap conviction rather than merely be reported.
DECISION_CHANGING_WEIGHT = 0.9


def _fmt(e: DecisionEvidence) -> Dict[str, Any]:
    return {
        "source": e.source,
        "direction": e.direction,
        "horizon": e.horizon,
        "reliability": e.reliability,
        "weight": e.weight,
        "validation_status": e.validation_status,
        "data_source": (e.provenance or {}).get("module"),
        "observation": e.observation,
    }


def _pairwise_conflicts(items: List[DecisionEvidence]) -> List[Dict[str, Any]]:
    """Directional disagreements between DECISIVE evidence items.

    Restricted to DECISIVE items on purpose: every pair of a dozen items would
    otherwise produce dozens of "conflicts", most of them between a context
    reading and a supporting one, and the list would stop being information.
    """
    decisive = [e for e in directional(items)
                if e.decision_relevance == "DECISIVE" and e.weight >= MIN_CONFLICT_WEIGHT]
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(decisive):
        for b in decisive[i + 1:]:
            if a.direction == b.direction:
                continue
            combined = round(a.weight + b.weight, 3)
            same_horizon = a.horizon == b.horizon
            out.append({
                "name": f"{a.source}_vs_{b.source}",
                "kind": "DIRECTIONAL",
                "what_disagrees": [_fmt(a), _fmt(b)],
                "why": (f"{a.source} reads {a.direction.lower()} while {b.source} reads "
                        f"{b.direction.lower()}; both carry enough weight "
                        f"(≥{MIN_CONFLICT_WEIGHT}) that neither can be dismissed."),
                "horizon": (f"both {a.horizon}" if same_horizon
                            else f"{a.source} is {a.horizon}, {b.source} is {b.horizon}"),
                "same_horizon": same_horizon,
                "combined_weight": combined,
                "severity": "HIGH" if combined >= DECISION_CHANGING_WEIGHT else "MEDIUM",
                "changes_decision": combined >= DECISION_CHANGING_WEIGHT,
                "resolution": (
                    "Same horizon — these are competing claims about the same future, "
                    "so conviction must be reduced rather than one side chosen."
                    if same_horizon else
                    "Different horizons — not strictly a contradiction; which one governs "
                    "depends on the intended holding period."),
            })
    return out


def _thesis_vs_edge_conflict(items: List[DecisionEvidence]) -> Optional[Dict[str, Any]]:
    """The single most consequential split in this whole system: a directional
    thesis with real weight, and no demonstrated statistical edge behind it.

    This is not a bug in either input. A thesis can be well-supported by current
    evidence while the historical record still fails to show that acting on such
    theses produces abnormal returns. Both are true; the error is letting the
    first imply the second.
    """
    gate = next((e for e in items if e.source == "gate:statistical_edge"), None)
    if gate is None:
        return None
    if gate.raw_value == "HIGH":
        return None

    dir_items = [e for e in directional(items) if e.decision_relevance == "DECISIVE"]
    if not dir_items:
        return None
    net = sum(e.weight if e.direction == "BULLISH" else -e.weight for e in dir_items)
    if abs(net) < 0.4:
        return None

    lean = "bullish" if net > 0 else "bearish"
    return {
        "name": "thesis_without_demonstrated_edge",
        "kind": "THESIS_VS_EDGE",
        "what_disagrees": [
            {"source": "thesis", "direction": lean.upper(), "horizon": "ALL",
             "reliability": None, "weight": round(abs(net), 3),
             "validation_status": "UNVALIDATED",
             "data_source": "decision/thesis.py (aggregate of decisive evidence)",
             "observation": f"Decisive evidence nets {lean} (weight {abs(net):.2f})."},
            _fmt(gate),
        ],
        "why": ("The current evidence leans directionally, but the statistical gate "
                "(walk-forward + embargo + PBO + net-of-cost + minimum sample) has NOT "
                "established that acting on this engine's signal produces abnormal "
                "returns. A supported thesis and a demonstrated edge are different "
                "claims; only the second justifies sizing."),
        "horizon": "ALL — the gate is horizon-independent",
        "same_horizon": False,
        "combined_weight": round(abs(net) + gate.weight, 3),
        "severity": "HIGH",
        "changes_decision": True,
        "resolution": ("Thesis may inform WATCH/monitoring and conditional planning; it "
                       "may not unlock position sizing. Sizing stays gated at 0%."),
    }


def _quality_issues(items: List[DecisionEvidence]) -> List[Dict[str, Any]]:
    """Evidence-quality problems that are not disagreements: missing inputs,
    stale data, unvalidated sources carrying decisive weight."""
    out: List[Dict[str, Any]] = []
    for e in items:
        if e.data_quality in ("MISSING", "STALE"):
            out.append({
                "source": e.source, "issue": e.data_quality,
                "detail": f"{e.source} data quality is {e.data_quality}"
                          + (f" ({', '.join(e.flags)})" if e.flags else ""),
                "decision_relevance": e.decision_relevance,
            })
        elif e.decision_relevance == "DECISIVE" and e.validation_status == "UNVALIDATED" \
                and e.direction in ("BULLISH", "BEARISH"):
            out.append({
                "source": e.source, "issue": "UNVALIDATED_BUT_DECISIVE",
                "detail": (f"{e.source} is treated as decisive but has never been "
                           f"validated out of sample."),
                "decision_relevance": e.decision_relevance,
            })
    return out


def analyze_conflicts(items: List[DecisionEvidence],
                      pillar_contradictions: Optional[List[Dict[str, Any]]] = None,
                      horizon_conflicts: Optional[List[Dict[str, Any]]] = None
                      ) -> Dict[str, Any]:
    """Full agreement / conflict / quality read.

    `pillar_contradictions` is intelligence/evidence_synthesis.py's existing
    four-rule output, passed through unchanged and merged rather than
    recomputed — two modules deriving the same contradiction independently is
    how they eventually disagree.
    """
    dir_items = directional(items)
    bullish = sorted([e for e in dir_items if e.direction == "BULLISH"],
                     key=lambda e: e.weight, reverse=True)
    bearish = sorted([e for e in dir_items if e.direction == "BEARISH"],
                     key=lambda e: e.weight, reverse=True)

    bull_weight = round(sum(e.weight for e in bullish), 3)
    bear_weight = round(sum(e.weight for e in bearish), 3)
    total = round(bull_weight + bear_weight, 3)

    conflicts: List[Dict[str, Any]] = []
    conflicts.extend(_pairwise_conflicts(items))
    tve = _thesis_vs_edge_conflict(items)
    if tve:
        conflicts.append(tve)

    # Existing pillar-pair rules, normalized into this module's shape so a
    # consumer reads one list, not two.
    for c in (pillar_contradictions or []):
        conflicts.append({
            "name": c.get("name"),
            "kind": "PILLAR_PAIR",
            "what_disagrees": [{"source": f"pillar:{k}", "direction": None,
                                "horizon": None, "reliability": None,
                                "weight": None, "validation_status": None,
                                "data_source": "intelligence/evidence_synthesis.py",
                                "observation": f"{k} = {v}"}
                               for k, v in (c.get("signals") or {}).items()],
            "why": c.get("description"),
            "horizon": "mixed",
            "same_horizon": False,
            "combined_weight": None,
            "severity": c.get("severity", "MEDIUM"),
            "changes_decision": c.get("severity") == "HIGH",
            "resolution": ("Named and carried into conviction rather than averaged "
                           "into the composite."),
        })

    for hc in (horizon_conflicts or []):
        conflicts.append({
            "name": f"horizon_conflict_{'_'.join(hc['horizons'])}",
            "kind": "HORIZON",
            "what_disagrees": [{"source": f"horizon:{h}", "direction": None,
                                "horizon": h, "reliability": None,
                                "weight": None, "validation_status": None,
                                "data_source": "decision/horizons.py",
                                "observation": f"lean {hc['leans'][h]:+.2f}"}
                               for h in hc["horizons"]],
            "why": hc["description"],
            "horizon": " vs ".join(hc["horizons"]),
            "same_horizon": False,
            "combined_weight": hc.get("gap"),
            "severity": "MEDIUM",
            "changes_decision": False,
            "resolution": "Resolved by the reader's holding period, not by the engine.",
        })

    quality = _quality_issues(items)

    # Agreement ratio on DIRECTIONAL weight only. 1.0 = unanimous, 0.5 = evenly
    # split. Reported rather than converted into a score, because "how divided
    # is the evidence" is a different fact from "which way does it point".
    if total > 0:
        agreement_ratio = round(max(bull_weight, bear_weight) / total, 3)
    else:
        agreement_ratio = None

    decision_changing = [c for c in conflicts if c.get("changes_decision")]

    if agreement_ratio is None:
        consensus = "NO_DIRECTIONAL_EVIDENCE"
    elif agreement_ratio >= 0.85:
        consensus = "STRONG_AGREEMENT"
    elif agreement_ratio >= 0.65:
        consensus = "LEANING"
    else:
        consensus = "CONFLICTED"

    return {
        "consensus": consensus,
        "agreement_ratio": agreement_ratio,
        "agrees": {"direction": "BULLISH" if bull_weight >= bear_weight else "BEARISH",
                   "weight": max(bull_weight, bear_weight),
                   "sources": [e.source for e in (bullish if bull_weight >= bear_weight else bearish)]},
        "disagrees": {"direction": "BEARISH" if bull_weight >= bear_weight else "BULLISH",
                      "weight": min(bull_weight, bear_weight),
                      "sources": [e.source for e in (bearish if bull_weight >= bear_weight else bullish)]},
        "bullish_weight": bull_weight,
        "bearish_weight": bear_weight,
        "conflicts": conflicts,
        "n_conflicts": len(conflicts),
        "n_decision_changing": len(decision_changing),
        "evidence_quality_issues": quality,
        "summary": _summary(consensus, agreement_ratio, conflicts, decision_changing),
    }


def _summary(consensus: str, ratio: Optional[float],
             conflicts: List[Dict[str, Any]],
             decision_changing: List[Dict[str, Any]]) -> str:
    if consensus == "NO_DIRECTIONAL_EVIDENCE":
        return "No directional evidence carries meaningful weight."
    base = {
        "STRONG_AGREEMENT": f"Evidence strongly agrees ({ratio:.0%} of directional weight on one side)",
        "LEANING": f"Evidence leans one way ({ratio:.0%} of directional weight)",
        "CONFLICTED": f"Evidence is genuinely split ({ratio:.0%} of directional weight on the larger side)",
    }[consensus]
    if decision_changing:
        names = ", ".join(c["name"] for c in decision_changing[:2])
        return f"{base}. {len(decision_changing)} conflict(s) cap conviction: {names}."
    if conflicts:
        return f"{base}. {len(conflicts)} conflict(s) noted, none decision-changing."
    return f"{base}, with no named conflicts."
