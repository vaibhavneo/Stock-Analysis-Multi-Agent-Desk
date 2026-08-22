"""
Evidence ledger + contradiction engine (items 3 + 4) - combines pillars,
market regime, historical analogs, and (when available) cost-basis context
into one coherent, reliability-weighted read, and explicitly names
contradictions between signals rather than blending them away.

Item 3's explicit constraint - "do NOT simply average scores, weight by
reliability" - is why `weighted_score` and `simple_average_score` are BOTH
exposed in the output: making the two visibly different (and provably so,
see tests/test_evidence_synthesis.py) is what proves this isn't secretly
just an average with extra steps. Reliability weights are each source's own
already-computed confidence (backtest/pillars.py's per-pillar confidence,
intelligence/regime.py's regime confidence, intelligence/analog_engine.py's
match confidence) - nothing here invents a new trust score per source.

detect_contradictions() names exactly the four situations item 4 asks for.
Each is a simple, explainable threshold rule on data already computed
elsewhere in this session's work - no LLM judgment call, no black box - so
"why was this flagged" always has a one-line, mechanical answer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

BULLISH_THRESHOLD = 65.0
BEARISH_THRESHOLD = 35.0
HIGH_SEVERITY_GAP = 40.0


def _signal_label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 60:
        return "bullish"
    if score <= 40:
        return "bearish"
    return "neutral"


def _severity(gap: float) -> str:
    return "HIGH" if gap >= HIGH_SEVERITY_GAP else "MEDIUM"


def detect_contradictions(pillars: Dict[str, Any], regime: Optional[Dict[str, Any]] = None,
                           historical_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Four named, deterministic contradiction checks - item 4's exact list.
    Returns an empty list when nothing conflicts; never fabricates a
    contradiction where the underlying scores don't actually support one."""
    pillars = pillars or {}

    def score_of(name: str) -> Optional[float]:
        p = pillars.get(name)
        return p.get("score") if p else None

    fundamentals = score_of("fundamentals")
    technical = score_of("technical")
    algo = score_of("algo")
    social = score_of("social")

    contradictions: List[Dict[str, Any]] = []

    if fundamentals is not None and technical is not None \
            and fundamentals >= BULLISH_THRESHOLD and technical <= BEARISH_THRESHOLD:
        contradictions.append({
            "name": "strong_fundamentals_bearish_technicals",
            "description": ("Fundamentals look strong but the technical setup is bearish - the market "
                             "isn't yet pricing in the fundamental strength, or the technicals are "
                             "flagging a risk fundamentals don't capture."),
            "severity": _severity(fundamentals - technical),
            "signals": {"fundamentals": fundamentals, "technical": technical},
        })

    if social is not None and fundamentals is not None \
            and social >= BULLISH_THRESHOLD and fundamentals <= BEARISH_THRESHOLD:
        contradictions.append({
            "name": "bullish_sentiment_deteriorating_fundamentals",
            "description": ("Sentiment is bullish while fundamentals are weak - social enthusiasm may "
                             "be running ahead of the underlying business."),
            "severity": _severity(social - fundamentals),
            "signals": {"social": social, "fundamentals": fundamentals},
        })

    if algo is not None and fundamentals is not None \
            and algo >= BULLISH_THRESHOLD and fundamentals <= BEARISH_THRESHOLD:
        contradictions.append({
            "name": "strong_momentum_excessive_valuation",
            "description": ("Momentum is strong while fundamentals (the closest available valuation/"
                             "quality proxy) are weak - the move may be running on momentum alone, "
                             "without fundamental support."),
            "severity": _severity(algo - fundamentals),
            "signals": {"algo": algo, "fundamentals": fundamentals},
        })

    if technical is not None and algo is not None and regime and regime.get("risk_stance") == "RISK_OFF":
        stock_composite = (technical + algo) / 2.0
        if stock_composite >= BULLISH_THRESHOLD - 5:   # 60 - a slightly softer bar than the pairwise checks
            contradictions.append({
                "name": "bullish_stock_bearish_regime",
                "description": ("The stock's own technical/momentum read is bullish, but the broader "
                                 "market regime is risk-off - a bullish individual setup is more likely "
                                 "to be dragged down or fail in a risk-off market."),
                "severity": "HIGH" if stock_composite >= 75 else "MEDIUM",
                "signals": {"stock_composite": round(stock_composite, 1),
                            "regime_risk_stance": regime["risk_stance"]},
            })

    return contradictions


def build_evidence_ledger(
    ticker: str,
    pillars: Dict[str, Any],
    regime: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    analog_result: Optional[Dict[str, Any]] = None,
    risk_profile: Optional[Dict[str, Any]] = None,
    position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One coherent, reliability-weighted evidence read: every pillar, plus
    market regime and historical analogs when available, each carrying its
    own already-computed reliability/confidence as its weight - never a
    flat average. Contradictions are detected and listed explicitly, and
    reduce a conviction_multiplier rather than being silently absorbed into
    a single blended number.
    """
    pillars = pillars or {}
    items: List[Dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for name, p in pillars.items():
        if not p or p.get("score") is None:
            continue
        reliability = p.get("confidence", 0.0) or 0.0
        items.append({
            "source": f"pillar:{name}", "score": p["score"], "reliability": reliability,
            "signal": _signal_label(p["score"]), "flags": p.get("flags", []),
        })
        weighted_sum += p["score"] * reliability
        weight_total += reliability

    if regime and regime.get("risk_stance") is not None:
        regime_score = {"RISK_ON": 70.0, "NEUTRAL": 50.0, "RISK_OFF": 30.0}.get(regime["risk_stance"], 50.0)
        reliability = regime.get("confidence", 0.5) or 0.0
        items.append({
            "source": "market_regime", "score": regime_score, "reliability": reliability,
            "signal": _signal_label(regime_score), "flags": regime.get("flags", []),
        })
        weighted_sum += regime_score * reliability
        weight_total += reliability

    if analog_result and analog_result.get("status") == "ok":
        outcome = analog_result.get("outcome_by_horizon") or {}
        rep = outcome.get(63) or (next(iter(outcome.values())) if outcome else None)
        if rep and rep.get("pct_positive") is not None:
            analog_score = 50.0 + (rep["pct_positive"] - 0.5) * 100.0
            reliability = analog_result.get("confidence", 0.0) or 0.0
            items.append({
                "source": "historical_analogs", "score": round(analog_score, 1), "reliability": reliability,
                "signal": _signal_label(analog_score), "flags": ["price_percentile_valuation_proxy"],
            })
            weighted_sum += analog_score * reliability
            weight_total += reliability

    weighted_score = round(weighted_sum / weight_total, 1) if weight_total > 0 else None
    simple_average_score = round(sum(i["score"] for i in items) / len(items), 1) if items else None

    contradictions = detect_contradictions(pillars, regime, historical_context)
    conviction_penalty = sum(0.15 if c["severity"] == "HIGH" else 0.08 for c in contradictions)
    conviction_multiplier = round(max(0.4, 1.0 - conviction_penalty), 2)

    flags: List[str] = []
    if not items:
        flags.append("no_evidence_available")
    elif weight_total == 0:
        flags.append("no_reliable_evidence")

    result: Dict[str, Any] = {
        "ticker": ticker,
        "evidence": items,
        "weighted_score": weighted_score,
        "simple_average_score": simple_average_score,
        "contradictions": contradictions,
        "conviction_multiplier": conviction_multiplier,
        "flags": flags,
    }

    if risk_profile and risk_profile.get("cost_basis"):
        cb = risk_profile["cost_basis"]
        result["cost_basis"] = {
            "gain_loss_pct": cb.get("gain_loss_pct"),
            "underwater": cb.get("underwater"),
            "recovery_required_pct": cb.get("recovery_required_pct"),
        }

    return result
