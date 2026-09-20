"""
DecisionEvidence — the normalized evidence contract every Decision Intelligence
consumer reads (Phase 2).

This is a NORMALIZER, not a new analyst. Every item it emits is a restatement of
a number some existing module already computed: a pillar score, the statistical
gate's own verdict, the market regime, the forecast engine's p_up, the ledger's
measured win rate. Nothing here scores anything, and nothing here is allowed to
invent a value that was not already present in its input.

Why a new structure rather than reusing intelligence/evidence_synthesis.py's
item shape directly: that shape (source / score / reliability / signal / flags)
was built for one job — a reliability-weighted blend — and it carries no horizon,
no validation status, no as-of date and no provenance. Every one of those is
load-bearing for the decisions built above this layer:

  - `horizon` is what stops a 3-day RSI reading from cancelling a 5-year
    fundamental trend inside a single averaged number (Phase 13).
  - `validation_status` is what stops a tracked-forward pillar from being read as
    a demonstrated edge (Phase 17).
  - `as_of` + `data_quality` are what let a stale input be labelled instead of
    silently trusted.
  - `decision_relevance` is what makes "computed but never consulted" detectable:
    if an item is DECISIVE it must be able to change the output, and the mutation
    tests assert exactly that.

`from_ledger_item()` adapts the existing evidence_synthesis items rather than
replacing them, so the two layers can never disagree about a score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ── Controlled vocabularies ────────────────────────────────────────────────
# Small and closed on purpose. A vocabulary that grows per-feature stops being
# a vocabulary; these are the distinctions the decision layer actually branches
# on, and nothing else.

DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL", "NOT_DIRECTIONAL")

HORIZONS = ("SHORT", "MEDIUM", "LONG", "ALL")
HORIZON_DESCRIPTION = {
    "SHORT": "days to ~1 month",
    "MEDIUM": "~1 to 6 months",
    "LONG": "6 months and beyond",
    "ALL": "horizon-independent",
}

CATEGORIES = (
    "FUNDAMENTAL", "TECHNICAL", "MOMENTUM", "SENTIMENT", "RISK",
    "STATISTICAL", "MACRO", "RELATIVE", "POSITION", "CATALYST", "TRACK_RECORD",
)

# What kind of backing the item has. This is the single most important field in
# the whole structure: it is what keeps "a pillar leaned bullish" from ever
# being reported in the same breath as "this has a demonstrated edge".
VALIDATION_STATUSES = (
    "VALIDATED",            # survived walk-forward + embargo + PBO + net-cost
    "BACKTESTED_ONLY",      # has a historical backtest that did NOT clear the gate
    "TRACKED_FORWARD",      # no testable history; measured only going forward
    "MEASURED_LIVE",        # a live track record exists (the ledger)
    "INSUFFICIENT_SAMPLE",  # would be measurable, but n is too small to read
    "UNVALIDATED",          # computed, never validated at all
)

DATA_QUALITY = ("OK", "PARTIAL", "STALE", "MISSING")

RELEVANCE = ("DECISIVE", "SUPPORTING", "CONTEXT", "NOT_RELEVANT")


@dataclass
class DecisionEvidence:
    """One normalized analytical observation.

    `magnitude` is 0..1 distance from neutral, NOT a score: it answers "how far
    from 'no opinion' is this?" so items measured on different scales (a 0-100
    pillar, a 0-1 probability, a percentage return) can be compared without
    pretending they share units.
    """
    source: str
    category: str
    metric: str
    observation: str
    direction: str = "NEUTRAL"
    magnitude: float = 0.0
    horizon: str = "ALL"
    reliability: float = 0.0
    validation_status: str = "UNVALIDATED"
    as_of: Optional[str] = None
    data_quality: str = "OK"
    provenance: Dict[str, Any] = field(default_factory=dict)
    decision_relevance: str = "SUPPORTING"
    raw_value: Optional[Any] = None
    flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction {self.direction!r} not in {DIRECTIONS}")
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon {self.horizon!r} not in {HORIZONS}")
        if self.category not in CATEGORIES:
            raise ValueError(f"category {self.category!r} not in {CATEGORIES}")
        if self.validation_status not in VALIDATION_STATUSES:
            raise ValueError(f"validation_status {self.validation_status!r} invalid")
        if self.data_quality not in DATA_QUALITY:
            raise ValueError(f"data_quality {self.data_quality!r} invalid")
        if self.decision_relevance not in RELEVANCE:
            raise ValueError(f"decision_relevance {self.decision_relevance!r} invalid")
        self.magnitude = max(0.0, min(1.0, float(self.magnitude or 0.0)))
        self.reliability = max(0.0, min(1.0, float(self.reliability or 0.0)))

    @property
    def weight(self) -> float:
        """Reliability × magnitude — how much this item should actually move a
        read. An extreme number nobody should trust weighs nearly nothing, and
        so does a highly reliable number that says 'no opinion'."""
        return round(self.reliability * self.magnitude, 4)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["weight"] = self.weight
        d["horizon_description"] = HORIZON_DESCRIPTION[self.horizon]
        return d


# ── Helpers ────────────────────────────────────────────────────────────────

def direction_from_score(score: Optional[float], bullish_at: float = 60.0,
                         bearish_at: float = 40.0) -> str:
    """0-100 pillar convention, matching backtest/pillars.py's own bands and
    intelligence/evidence_synthesis.py's _signal_label so the three never
    disagree about what 'bullish' means."""
    if score is None:
        return "NOT_DIRECTIONAL"
    if score >= bullish_at:
        return "BULLISH"
    if score <= bearish_at:
        return "BEARISH"
    return "NEUTRAL"


def magnitude_from_score(score: Optional[float]) -> float:
    """Distance from 50 on a 0-100 scale, normalized to 0..1."""
    if score is None:
        return 0.0
    return min(1.0, abs(float(score) - 50.0) / 50.0)


def magnitude_from_probability(p: Optional[float]) -> float:
    """Distance from 0.5 on a probability, normalized so p=0.5 -> 0, p=1 -> 1."""
    if p is None:
        return 0.0
    return min(1.0, abs(float(p) - 0.5) * 2.0)


# Pillar → (category, horizon, validation_status). These assignments are the
# whole point of Phase 13: a technical read and a fundamental read are NOT
# statements about the same future, and blending them is what produced a single
# composite that could not be interrogated.
#
# `technical` and `algo` are BACKTESTED_ONLY rather than VALIDATED at the item
# level: they are the two pillars the core strategy backtest actually covers,
# but whether that backtest CLEARED the gate is a per-ticker fact, so the
# statistical item below carries the verdict and these carry only "testable".
PILLAR_PROFILE: Dict[str, Dict[str, str]] = {
    "technical":    {"category": "TECHNICAL",   "horizon": "SHORT",
                     "validation_status": "BACKTESTED_ONLY"},
    "algo":         {"category": "MOMENTUM",    "horizon": "MEDIUM",
                     "validation_status": "BACKTESTED_ONLY"},
    "fundamentals": {"category": "FUNDAMENTAL", "horizon": "LONG",
                     "validation_status": "TRACKED_FORWARD"},
    "social":       {"category": "SENTIMENT",   "horizon": "SHORT",
                     "validation_status": "TRACKED_FORWARD"},
    "research":     {"category": "FUNDAMENTAL", "horizon": "MEDIUM",
                     "validation_status": "TRACKED_FORWARD"},
    "risk":         {"category": "RISK",        "horizon": "ALL",
                     "validation_status": "TRACKED_FORWARD"},
}

PILLAR_FORMULA_HINT = {
    "technical": "backtest/pillars.py technical_score (RSI/MACD/MA/BB/volume/ADX vote)",
    "algo": "backtest/pillars.py algo_score (z-score, momentum composite, linreg)",
    "fundamentals": "backtest/pillars.py fundamentals_score (EDGAR PIT when available)",
    "social": "backtest/pillars.py social_score (Reddit + StockTwits)",
    "research": "backtest/pillars.py research_score (analyst consensus)",
    "risk": "backtest/pillars.py _risk_score (HV20, beta, drawdown)",
}


def _quality_from_flags(flags: List[str]) -> str:
    """Flags already name their own data problems; this reads them rather than
    re-deriving quality from the values."""
    flags = flags or []
    joined = " ".join(flags).lower()
    if "unavailable" in joined or "missing" in joined or "no_data" in joined:
        return "MISSING"
    if "stale" in joined:
        return "STALE"
    if "fallback" in joined or "partial" in joined or "proxy" in joined or "yfinance" in joined:
        return "PARTIAL"
    return "OK"


def evidence_from_pillar(name: str, pillar: Dict[str, Any],
                         as_of: Optional[str] = None,
                         claim_id: Optional[str] = None) -> Optional[DecisionEvidence]:
    """One pillar → one DecisionEvidence. Returns None (never a neutral
    placeholder) when the pillar has no score at all, so a missing pillar is
    distinguishable from a pillar that genuinely reads 50."""
    if not pillar or pillar.get("score") is None:
        return None
    profile = PILLAR_PROFILE.get(name)
    if profile is None:
        return None
    score = float(pillar["score"])
    flags = list(pillar.get("flags") or [])
    direction = direction_from_score(score)

    # The risk pillar is not a directional call on the stock: a high risk score
    # means "less risky", which is not the same claim as "goes up". Reporting it
    # as BULLISH is how a low-volatility name ends up looking like a buy.
    if name == "risk":
        direction = "NOT_DIRECTIONAL"
        observation = f"Risk pillar {score:.0f}/100 ({'lower' if score >= 60 else 'elevated'} measured risk)"
    else:
        observation = f"{name.capitalize()} pillar {score:.0f}/100 ({direction.lower()})"

    return DecisionEvidence(
        source=f"pillar:{name}",
        category=profile["category"],
        metric=f"{name}_pillar_score",
        observation=observation,
        direction=direction,
        magnitude=magnitude_from_score(score),
        horizon=profile["horizon"],
        reliability=float(pillar.get("confidence") or 0.0),
        validation_status=profile["validation_status"],
        as_of=as_of,
        data_quality=_quality_from_flags(flags),
        provenance={"module": "backtest/pillars.py",
                    "formula": PILLAR_FORMULA_HINT.get(name),
                    "claim_id": claim_id or pillar.get("claim_id")},
        decision_relevance="DECISIVE" if name in ("technical", "algo", "fundamentals")
                            else "SUPPORTING",
        raw_value=score,
        flags=flags,
    )


def evidence_from_statistical_edge(rec: Dict[str, Any]) -> DecisionEvidence:
    """The statistical gate's own verdict, as evidence in its own right.

    This item is deliberately NOT_DIRECTIONAL: "there is a demonstrated edge" is
    not a claim that the price goes up. Conflating the two is precisely the
    error Phase 17 exists to prevent, and giving this item a direction would
    reintroduce it through the back door.
    """
    edge = ((rec.get("confidence") or {}).get("statistical_edge") or {})
    bt = rec.get("backtest") or {}
    level = edge.get("level", "NONE")
    dsr = bt.get("dsr")
    sharpe = bt.get("sharpe")
    n_obs = ((edge.get("checks") or {}).get("min_sample") or {}).get("n_obs")

    if level == "HIGH":
        status = "VALIDATED"
        obs = (f"Statistical edge HIGH: walk-forward OOS positive, PBO < 0.5, "
               f"net-of-cost positive, dSR {dsr if dsr is not None else '—'} ≥ 0.5.")
    elif level == "NONE":
        status = "INSUFFICIENT_SAMPLE"
        obs = f"Statistical edge cannot be assessed — sample too small ({n_obs} bars)."
    else:
        status = "BACKTESTED_ONLY"
        obs = (f"Statistical edge {level}: the core strategy has a backtest "
               f"(Sharpe {sharpe}, dSR {dsr}) that does NOT clear the "
               f"demonstrated-edge bar. This is not a proven edge.")

    return DecisionEvidence(
        source="gate:statistical_edge",
        category="STATISTICAL",
        metric="statistical_edge_level",
        observation=obs,
        direction="NOT_DIRECTIONAL",
        magnitude=float(edge.get("score") or 0.0),
        horizon="ALL",
        reliability=1.0,          # the gate's own verdict about itself is exact
        validation_status=status,
        as_of=rec.get("data_asof"),
        data_quality="OK" if n_obs else "PARTIAL",
        provenance={"module": "agents/recommendation.py::_assess_statistical_edge",
                    "formula": edge.get("gate_required"),
                    "checks": {k: v.get("pass") for k, v in (edge.get("checks") or {}).items()}},
        decision_relevance="DECISIVE",
        raw_value=level,
        flags=[] if level == "HIGH" else ["edge_not_demonstrated"],
    )


def evidence_from_backtest(rec: Dict[str, Any]) -> Optional[DecisionEvidence]:
    """The raw historical performance of the core strategy on THIS ticker.

    Separate from the gate item above because they answer different questions:
    this one is "what happened historically", the gate is "does that generalize".
    A strongly negative Sharpe is real, decision-relevant information even when
    the gate has already said 'unproven'.
    """
    bt = rec.get("backtest") or {}
    sharpe = bt.get("sharpe")
    if sharpe is None:
        return None
    max_dd = bt.get("max_drawdown")
    direction = "BEARISH" if sharpe < 0 else ("BULLISH" if sharpe > 0.5 else "NEUTRAL")
    return DecisionEvidence(
        source="backtest:seven_pillar_core",
        category="STATISTICAL",
        metric="core_strategy_sharpe",
        observation=(f"Core strategy on this ticker: Sharpe {sharpe:.2f}, "
                     f"max drawdown {max_dd:.0%}, {bt.get('n_trades')} trades, "
                     f"net of {bt.get('total_cost_pct')}% costs."
                     if max_dd is not None else
                     f"Core strategy Sharpe {sharpe:.2f} on this ticker."),
        direction=direction,
        magnitude=min(1.0, abs(float(sharpe)) / 1.5),
        horizon="MEDIUM",
        reliability=0.6,     # a single-ticker in-sample backtest; not the gate
        validation_status="BACKTESTED_ONLY",
        as_of=rec.get("data_asof"),
        data_quality="OK",
        provenance={"module": "backtest/engine.py::run_vectorized_backtest",
                    "cost_model": bt.get("cost_model"),
                    "n_trials": bt.get("n_trials")},
        decision_relevance="SUPPORTING",
        raw_value=sharpe,
        flags=["in_sample_single_ticker"],
    )


def evidence_from_regime(regime: Optional[Dict[str, Any]]) -> Optional[DecisionEvidence]:
    if not regime or regime.get("risk_stance") is None:
        return None
    stance = regime["risk_stance"]
    direction = {"RISK_ON": "BULLISH", "RISK_OFF": "BEARISH"}.get(stance, "NEUTRAL")
    magnitude = 0.0 if stance == "NEUTRAL" else 0.5
    vix = regime.get("vix_level")
    return DecisionEvidence(
        source="market_regime",
        category="MACRO",
        metric="risk_stance",
        observation=(f"Market regime {stance} — SPY trend {regime.get('trend')}, "
                     f"VIX {vix if vix is not None else 'unavailable'}, "
                     f"volatility regime {regime.get('volatility_regime')}."),
        direction=direction,
        magnitude=magnitude,
        horizon="SHORT",
        reliability=float(regime.get("confidence") or 0.0),
        validation_status="TRACKED_FORWARD",
        as_of=regime.get("as_of"),
        data_quality=_quality_from_flags(regime.get("flags") or []),
        provenance={"module": "intelligence/regime.py",
                    "formula": "SPY technical score + VIX bands (25/15) + QQQ-SPY 1M gap"},
        decision_relevance="CONTEXT",
        raw_value=stance,
        flags=list(regime.get("flags") or []),
    )


def evidence_from_relative_strength(historical_context: Optional[Dict[str, Any]]
                                    ) -> List[DecisionEvidence]:
    """Excess return vs the benchmark, one item per horizon that has data.

    Emitted per horizon rather than averaged because a name beating the
    benchmark over 1Y while losing to it over 1M is exactly the horizon
    conflict Phase 13 requires the synthesis to state rather than net out.
    """
    if not historical_context:
        return []
    rel = historical_context.get("relative_performance") or {}
    horizon_map = {"1M": "SHORT", "3M": "MEDIUM", "6M": "MEDIUM",
                   "1Y": "LONG", "3Y": "LONG", "5Y": "LONG"}
    out: List[DecisionEvidence] = []
    for label, h in rel.items():
        if not h or not h.get("data_available"):
            continue
        excess = h.get("excess_return_pct")
        if excess is None:
            continue
        direction = "BULLISH" if excess > 2 else ("BEARISH" if excess < -2 else "NEUTRAL")
        out.append(DecisionEvidence(
            source=f"relative_strength:{label}",
            category="RELATIVE",
            metric=f"excess_return_{label}",
            observation=(f"{label}: {h['stock_return_pct']:+.1f}% vs "
                         f"{h['benchmark_symbol']} {h['benchmark_return_pct']:+.1f}% "
                         f"({excess:+.1f}% excess)."),
            direction=direction,
            magnitude=min(1.0, abs(excess) / 25.0),
            horizon=horizon_map.get(label, "MEDIUM"),
            reliability=0.7,
            validation_status="UNVALIDATED",
            as_of=None,
            data_quality="OK",
            provenance={"module": "intelligence/historical_context.py",
                        "formula": "stock return − benchmark return over the window"},
            decision_relevance="SUPPORTING",
            raw_value=excess,
            flags=[],
        ))
    return out


def evidence_from_cross_section(xsec_interp: Optional[Dict[str, Any]]
                                ) -> Optional[DecisionEvidence]:
    if not xsec_interp or xsec_interp.get("status") != "OK":
        return None
    pct = xsec_interp.get("percentile")
    if pct is None:
        return None
    direction = "BULLISH" if pct >= 0.7 else ("BEARISH" if pct <= 0.3 else "NEUTRAL")
    return DecisionEvidence(
        source="cross_sectional_rank",
        category="RELATIVE",
        metric="composite_percentile",
        observation=xsec_interp.get("interpretation", ""),
        direction=direction,
        magnitude=min(1.0, abs(pct - 0.5) * 2.0),
        horizon="MEDIUM",
        reliability=0.75 if xsec_interp.get("survivorship_safe") else 0.5,
        validation_status="TRACKED_FORWARD",
        as_of=xsec_interp.get("as_of"),
        data_quality="OK",
        provenance={"module": "xsection/ranking.py",
                    "universe_id": xsec_interp.get("universe_id"),
                    "survivorship_safe": xsec_interp.get("survivorship_safe")},
        decision_relevance="SUPPORTING",
        raw_value=pct,
        flags=[] if xsec_interp.get("survivorship_safe") else ["not_survivorship_safe"],
    )


def evidence_from_calibration(cal_interp: Optional[Dict[str, Any]]
                              ) -> Optional[DecisionEvidence]:
    """This engine's own measured live track record. NOT_DIRECTIONAL for the
    same reason the gate item is: 'the engine has been right 51% of the time'
    says nothing about which way this particular stock goes."""
    if not cal_interp:
        return None
    if cal_interp.get("status") != "OK":
        return DecisionEvidence(
            source="calibration",
            category="TRACK_RECORD",
            metric="live_win_rate",
            observation=("No mature live track record yet — this engine's forward "
                         "accuracy has not been measured."),
            direction="NOT_DIRECTIONAL",
            magnitude=0.0,
            horizon="ALL",
            reliability=1.0,
            validation_status="INSUFFICIENT_SAMPLE",
            as_of=None,
            data_quality="MISSING",
            provenance={"module": "data/prediction_ledger.py::calibration_report"},
            decision_relevance="DECISIVE",
            raw_value=None,
            flags=["calibration_insufficient"],
        )
    n = cal_interp.get("n_predictions") or 0
    wr = cal_interp.get("win_rate")
    ece = cal_interp.get("calibration_error_ece")
    return DecisionEvidence(
        source="calibration",
        category="TRACK_RECORD",
        metric="live_win_rate",
        observation=cal_interp.get("interpretation", ""),
        direction="NOT_DIRECTIONAL",
        magnitude=magnitude_from_probability(wr),
        horizon="ALL",
        reliability=1.0,
        validation_status="MEASURED_LIVE",
        as_of=None,
        data_quality="OK" if n >= 30 else "PARTIAL",
        provenance={"module": "data/prediction_ledger.py::calibration_report",
                    "n": n, "ece": ece},
        decision_relevance="DECISIVE",
        raw_value=wr,
        flags=(["calibration_unreliable"] if (ece is not None and ece >= 0.2) else []),
    )


def evidence_from_forecast(forecast: Optional[Dict[str, Any]]) -> List[DecisionEvidence]:
    """The forecast engine's p_up per horizon.

    `validation_status` is MEASURED_LIVE only for horizons the engine actually
    applied a fitted calibration to; everything else is UNVALIDATED, which is
    what stops a decayed heuristic probability from reading as a measured one.
    """
    if not forecast or not forecast.get("horizons"):
        return []
    label_to_horizon = {"1W": "SHORT", "1M": "SHORT", "3M": "MEDIUM",
                        "6M": "MEDIUM", "1Y": "LONG"}
    out: List[DecisionEvidence] = []
    for label, h in forecast["horizons"].items():
        p = h.get("p_up")
        if p is None:
            continue
        direction = {"UP": "BULLISH", "DOWN": "BEARISH"}.get(h.get("direction"), "NEUTRAL")
        out.append(DecisionEvidence(
            source=f"forecast:{label}",
            category="STATISTICAL",
            metric=f"p_up_{label}",
            observation=(f"{label} directional probability {p:.0%} "
                         f"({h.get('direction')})."),
            direction=direction,
            magnitude=magnitude_from_probability(p),
            horizon=label_to_horizon.get(label, "MEDIUM"),
            reliability=float(h.get("confidence") or 0.0),
            validation_status="MEASURED_LIVE" if h.get("calibrated") else "UNVALIDATED",
            as_of=None,
            data_quality="OK",
            provenance={"module": "intelligence/prediction_engine.py",
                        "formula": "pillar composite → confidence-scaled → regime → "
                                   "analog blend → horizon decay → optional isotonic calibration",
                        "p_up_uncalibrated": h.get("p_up_uncalibrated")},
            decision_relevance="SUPPORTING",
            raw_value=p,
            flags=list(forecast.get("flags") or []),
        ))
    return out


def evidence_from_position(position_context: Optional[Dict[str, Any]]
                           ) -> Optional[DecisionEvidence]:
    """The user's own unrealized P&L — a fact about the portfolio, never a
    directional claim about the security. Marked NOT_DIRECTIONAL so it can
    never be counted as agreeing or disagreeing with the thesis."""
    if not position_context:
        return None
    pl = position_context.get("unrealized_pl_pct")
    if pl is None:
        return None
    return DecisionEvidence(
        source="position:cost_basis",
        category="POSITION",
        metric="unrealized_pl_pct",
        observation=(f"Existing position is {pl:+.1f}% against an average cost of "
                     f"${position_context.get('avg_cost')}."),
        direction="NOT_DIRECTIONAL",
        magnitude=min(1.0, abs(pl) / 50.0),
        horizon="ALL",
        reliability=1.0,
        validation_status="UNVALIDATED",
        as_of=None,
        data_quality="OK",
        provenance={"module": "decision/position.py", "formula": "(price − avg_cost)/avg_cost"},
        decision_relevance="CONTEXT",
        raw_value=pl,
        flags=[],
    )


def build_decision_evidence(
    rec: Dict[str, Any],
    regime: Optional[Dict[str, Any]] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    forecast: Optional[Dict[str, Any]] = None,
    xsec_interp: Optional[Dict[str, Any]] = None,
    cal_interp: Optional[Dict[str, Any]] = None,
    position_context: Optional[Dict[str, Any]] = None,
    catalysts: Optional[Dict[str, Any]] = None,
) -> List[DecisionEvidence]:
    """The full normalized evidence set for one security.

    Order is stable (pillars, gate, backtest, regime, relative, cross-section,
    calibration, forecast, position, catalysts) so two runs on identical inputs
    produce identical output — the repeatability contract the decision
    fingerprint depends on.
    """
    as_of = rec.get("data_asof")
    items: List[DecisionEvidence] = []

    for name in ("technical", "algo", "fundamentals", "research", "social", "risk"):
        e = evidence_from_pillar(name, (rec.get("pillars") or {}).get(name), as_of=as_of)
        if e:
            items.append(e)

    items.append(evidence_from_statistical_edge(rec))
    bt = evidence_from_backtest(rec)
    if bt:
        items.append(bt)

    reg = evidence_from_regime(regime)
    if reg:
        items.append(reg)

    items.extend(evidence_from_relative_strength(historical_context))

    xs = evidence_from_cross_section(xsec_interp)
    if xs:
        items.append(xs)

    cal = evidence_from_calibration(cal_interp)
    if cal:
        items.append(cal)

    items.extend(evidence_from_forecast(forecast))

    pos = evidence_from_position(position_context)
    if pos:
        items.append(pos)

    if catalysts:
        from decision.catalysts import evidence_from_catalysts
        items.extend(evidence_from_catalysts(catalysts))

    return items


def directional(items: List[DecisionEvidence]) -> List[DecisionEvidence]:
    """Only the items that make a directional claim. Everything downstream that
    asks 'what agrees / what disagrees' must go through this, because counting
    a NOT_DIRECTIONAL item as agreement is how 'the data is reliable' turns
    into 'the stock goes up'."""
    return [e for e in items if e.direction in ("BULLISH", "BEARISH")]


def to_dicts(items: List[DecisionEvidence]) -> List[Dict[str, Any]]:
    return [e.to_dict() for e in items]
