"""
Four things this app used to compress into one (Phase 3).

  THESIS            What the current evidence says about the business and the
                    price. Directional. Can be strong and still be wrong.
  STATISTICAL EDGE  Whether acting on theses like this one has been shown to
                    produce abnormal returns. NOT directional. Can be absent
                    while the thesis is strong.
  SCENARIO          What the price may do under stated conditions. Conditional,
                    not a forecast. (decision/scenarios.py)
  POSITION DECISION What to do given ownership, cost basis and risk budget.
                    Depends on all three above plus facts about the holder.
                    (decision/position.py, decision/state.py)

Collapsing them produces exactly the errors the UI currently invites:
a strong thesis reading as a proven edge, a bearish thesis reading as a short,
and a positive target price reading as a buy. This module builds the first two
as separate objects and states their relationship explicitly, so no consumer
downstream has to infer it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from decision.evidence import DecisionEvidence, directional

# Net directional weight required before the thesis is called directional at
# all. Below this the honest answer is "no thesis", not "a weak bullish thesis".
THESIS_MIN_WEIGHT = 0.30

STRENGTH_BANDS = ((1.2, "STRONG"), (0.6, "MODERATE"), (THESIS_MIN_WEIGHT, "WEAK"))


def _strength(net_abs: float) -> str:
    for bar, label in STRENGTH_BANDS:
        if net_abs >= bar:
            return label
    return "NONE"


def build_thesis(items: List[DecisionEvidence],
                 horizon_read: Optional[Dict[str, Any]] = None,
                 llm_prose: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The directional read, and nothing else.

    Deliberately says nothing about sizing, edge, or what to do. `llm_prose`
    (the prediction agent's summary/bull/bear text, when a run produced one) is
    attached verbatim under `narrative` and is never allowed to alter
    `direction`, `strength` or `net_weight` — it is quoted, not consulted.
    """
    dir_items = directional(items)
    bullish = [e for e in dir_items if e.direction == "BULLISH"]
    bearish = [e for e in dir_items if e.direction == "BEARISH"]
    bull_w = sum(e.weight for e in bullish)
    bear_w = sum(e.weight for e in bearish)
    net = bull_w - bear_w

    if abs(net) < THESIS_MIN_WEIGHT:
        direction = "NEUTRAL"
        strength = "NONE"
    else:
        direction = "BULLISH" if net > 0 else "BEARISH"
        strength = _strength(abs(net))

    supporting = sorted(bullish if direction == "BULLISH" else bearish,
                        key=lambda e: e.weight, reverse=True)
    opposing = sorted(bearish if direction == "BULLISH" else bullish,
                      key=lambda e: e.weight, reverse=True)

    if direction == "NEUTRAL":
        statement = ("No directional thesis: bullish and bearish evidence are close enough "
                     f"in weight ({bull_w:.2f} vs {bear_w:.2f}) that neither side carries "
                     "the read.")
    else:
        lead = supporting[0].observation if supporting else ""
        statement = (f"{strength.title()} {direction.lower()} thesis "
                     f"(net directional weight {net:+.2f}). Led by: {lead}")

    horizon_note = None
    if horizon_read and horizon_read.get("dominant_horizon"):
        dom = horizon_read["dominant_horizon"]
        horizon_note = (f"This thesis is primarily a {dom.lower()}-term read — "
                        f"{horizon_read['dominant_reason']}")

    return {
        "direction": direction,
        "strength": strength,
        "net_weight": round(net, 3),
        "bullish_weight": round(bull_w, 3),
        "bearish_weight": round(bear_w, 3),
        "statement": statement,
        "supporting_evidence": [e.source for e in supporting[:5]],
        "opposing_evidence": [e.source for e in opposing[:5]],
        "dominant_horizon": (horizon_read or {}).get("dominant_horizon"),
        "horizon_note": horizon_note,
        "narrative": ({"summary": (llm_prose or {}).get("summary"),
                       "bull_case": (llm_prose or {}).get("bull_case"),
                       "bear_case": (llm_prose or {}).get("bear_case")}
                      if llm_prose else None),
        "narrative_authority": ("Narrative is quoted from the prediction agent and has no "
                                "effect on direction, strength or weight."),
        "caveat": ("A thesis is a reading of current evidence. It is NOT a demonstrated "
                   "edge and NOT a recommendation to take a position — see `statistical_edge` "
                   "and `decision_state`."),
    }


def build_statistical_edge(rec: Dict[str, Any],
                           cal_interp: Optional[Dict[str, Any]] = None,
                           backtest_interp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The edge object — whether this engine's signal has been *shown* to work.

    Reads the gate's own verdict rather than re-deriving it, so this can never
    disagree with the thing that actually controls sizing.
    """
    edge = ((rec.get("confidence") or {}).get("statistical_edge") or {})
    bt = rec.get("backtest") or {}
    checks = edge.get("checks") or {}
    level = edge.get("level", "NONE")
    demonstrated = level == "HIGH"

    gate_rows = []
    for name in ("min_sample", "net_cost_positive", "dsr", "walk_forward", "pbo"):
        c = checks.get(name) or {}
        gate_rows.append({
            "gate": name,
            "passed": bool(c.get("pass")),
            "detail": {k: v for k, v in c.items() if k != "pass"},
        })

    cal_status = (cal_interp or {}).get("status", "INSUFFICIENT_HISTORY")
    cal_n = (cal_interp or {}).get("n_predictions", 0)
    cal_wr = (cal_interp or {}).get("win_rate")
    cal_ece = (cal_interp or {}).get("calibration_error_ece")

    honesty: List[str] = []
    if not demonstrated:
        honesty.append("NO_DEMONSTRATED_EDGE: the statistical gate has not been cleared. "
                       "Nothing here should be read as a proven edge.")
    dsr = bt.get("dsr")
    if dsr is not None and dsr < 0.5:
        honesty.append(f"Deflated Sharpe is {dsr:.2f}, below the 0.5 bar — after correcting "
                       f"for {bt.get('n_trials')} pre-registered strategy variants, the "
                       f"historical result is not distinguishable from luck.")
    n_obs = (checks.get("min_sample") or {}).get("n_obs")
    required = (checks.get("min_sample") or {}).get("required")
    if n_obs is not None and required is not None and n_obs < required:
        honesty.append(f"Sample is insufficient: {n_obs} bars against a {required}-bar minimum.")
    if cal_status != "OK":
        honesty.append(f"Calibration insufficient: only {cal_n} matured predictions — this "
                       f"engine's live accuracy has not been measured.")
    elif cal_ece is not None and cal_ece >= 0.2:
        honesty.append(f"Stated confidence is unreliable: calibration error {cal_ece:.2f} "
                       f"over {cal_n} matured predictions.")
    if not (rec.get("honesty_flags") or {}).get("survivorship_safe", False):
        honesty.append("Backtest is not survivorship-safe — delisted companies are excluded.")
    if (rec.get("honesty_flags") or {}).get("backtest_covers_core_only"):
        honesty.append("The backtest covers only the technical+algo core; the fundamentals, "
                       "social and research pillars have no tested history.")

    return {
        "level": level,
        "demonstrated": demonstrated,
        "verdict": ("DEMONSTRATED_EDGE" if demonstrated else
                    "INSUFFICIENT_SAMPLE" if level == "NONE" else "NO_DEMONSTRATED_EDGE"),
        "basis": edge.get("basis"),
        "gate_required": edge.get("gate_required"),
        "gates": gate_rows,
        "n_gates_passed": sum(1 for g in gate_rows if g["passed"]),
        "n_gates": len(gate_rows),
        "backtest": {"sharpe": bt.get("sharpe"), "dsr": dsr,
                     "max_drawdown": bt.get("max_drawdown"),
                     "n_trades": bt.get("n_trades"), "n_trials": bt.get("n_trials"),
                     "cost_model": bt.get("cost_model"),
                     "total_cost_pct": bt.get("total_cost_pct"),
                     "interpretation": (backtest_interp or {}).get("interpretation")},
        "calibration": {"status": cal_status, "n_predictions": cal_n,
                        "win_rate": cal_wr, "ece": cal_ece},
        "honesty": honesty,
        "sizing_consequence": ("Position sizing is unlocked." if demonstrated else
                               "Position sizing stays gated at 0%: no demonstrated edge."),
        "caveat": ("Statistical edge is not directional. 'No demonstrated edge' does not mean "
                   "the stock falls; it means this engine has not shown that acting on its "
                   "signal beats the alternative."),
    }


def relate_thesis_and_edge(thesis: Dict[str, Any], edge: Dict[str, Any]) -> Dict[str, Any]:
    """The explicit statement of how the two relate — the sentence whose absence
    lets a reader assume a strong thesis is a proven edge."""
    t_dir, t_str = thesis["direction"], thesis["strength"]
    demonstrated = edge["demonstrated"]

    if t_dir == "NEUTRAL" and not demonstrated:
        relation, text = "NO_THESIS_NO_EDGE", (
            "There is neither a directional thesis nor a demonstrated edge. This is the "
            "'nothing to do here' case, and it is a legitimate result.")
    elif t_dir == "NEUTRAL":
        relation, text = "EDGE_WITHOUT_THESIS", (
            "The engine's signal has a demonstrated edge, but current evidence gives no "
            "directional read on this security. The edge is about the method, not about "
            "this stock today.")
    elif demonstrated:
        relation, text = "THESIS_WITH_EDGE", (
            f"A {t_str.lower()} {t_dir.lower()} thesis AND a demonstrated statistical edge. "
            "This is the only configuration in which sizing is supported.")
    else:
        relation, text = "THESIS_WITHOUT_EDGE", (
            f"A {t_str.lower()} {t_dir.lower()} thesis with NO demonstrated statistical edge. "
            "These are different claims: the evidence leans one way, but this engine has not "
            "shown that acting on such a lean produces abnormal returns. The thesis can "
            "justify watching and planning; it cannot justify sizing.")

    return {
        "relation": relation,
        "statement": text,
        "thesis_direction": t_dir,
        "thesis_strength": t_str,
        "edge_demonstrated": demonstrated,
        "common_misreadings_blocked": [
            "strong thesis = proven edge",
            "bearish thesis = automatic short",
            "positive target = automatic buy",
        ],
    }
