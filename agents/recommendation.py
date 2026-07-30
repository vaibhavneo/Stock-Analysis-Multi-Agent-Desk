"""
Recommendation engine — the 7-pillar composite turned into a structured,
evidence-backed, tracked investing recommendation.

What "recommendation" means here, precisely:
  - The ACTION VERB and every NUMBER (composite, pillar scores, levels, size,
    horizon) are computed by formulas in backtest/pillars.py and here — never
    by an LLM (P4). The LLM's contribution is the optional `thesis` prose,
    rendered AROUND the computed numbers.
  - Every pillar score and the composite are recorded as EvidenceLedger CLAIMS
    with their exact formula and input datums, so /api/evidence/<claim_id> can
    trace any number a user sees. Snapshot inputs (yfinance/reddit/stocktwits)
    carry available_at = retrieval time and as_of_honored=False — honest: they
    are reconstructable in provenance but not in time, unlike EDGAR claims.
  - Conviction is DETERMINISTIC (pillar agreement + the core strategy's honest
    dSR + the risk veto), not an LLM mood.
  - Every recommendation is persisted to the recommendations table with
    grounding_strategy="seven_pillar_composite", so the existing outcome
    tracker (check_outcome / get_historical_hit_rate) measures this engine's
    real forward hit rate — the only evidence that ultimately matters.

NOT financial advice; a decision-support artifact with its uncertainty and
data-quality flags attached (honesty_flags), never hidden.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from backtest import experiments
from backtest.costs import DEFAULT_COST_MODEL
from backtest.engine import (compute_performance_metrics, deflated_sharpe_ratio,
                             run_vectorized_backtest)
from backtest.pillars import compute_pillar_scores, seven_pillar_core_strategy
from backtest.risk import compute_atr, safe_kelly_fraction
from backtest.strategies import STRATEGIES_NEEDING_FULL_DF, STRATEGY_REGISTRY
from backtest.validation import (probability_of_backtest_overfitting,
                                 walk_forward_cv)

# ── Pre-registered statistical-edge thresholds (NOT tuned on eval data) ──────
DSR_CREDIBLE_BAR = 0.5      # López de Prado's "more likely skill than luck"
MIN_OBS = 504              # ~2 trading years — the minimum sample to claim edge
WF_FOLDS = 5               # walk-forward folds
WF_EMBARGO_FRAC = 0.02     # embargo gap after each test block (leakage guard)
WF_PURGE_OBS = 50          # purge bars at the train/test boundary
WF_CONSISTENCY_MIN = 0.5   # >= half the OOS folds must have positive Sharpe
PBO_MAX = 0.5              # Probability of Backtest Overfitting must be below this


def build_recommendation(
    ticker: str,
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    signal_summary: Dict[str, Any],
    algo_signals: Dict[str, Any],
    fundamentals: Dict[str, Any],
    reddit: Optional[Dict[str, Any]] = None,
    stocktwits: Optional[Dict[str, Any]] = None,
    pit: Optional[Dict[str, Any]] = None,
    llm_prose: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the full structured recommendation. Keyless — works without any
    LLM; `llm_prose` (e.g. the prediction agent's summary/bull/bear text) only
    enriches the `thesis` field when present."""
    from data import ledger

    ticker = ticker.upper().strip()
    run_id = run_id or f"rec-{ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    current_price = float(indicators.get("current_price") or df["Close"].iloc[-1])

    # ── 1. Pillars + composite (deterministic; fundamentals are SEC-only) ──
    snap = compute_pillar_scores(ticker, indicators, signal_summary, algo_signals,
                                 fundamentals, reddit=reddit, stocktwits=stocktwits,
                                 pit=pit, strict_fundamentals=True)

    # ── 2. Honest backtest of the CORE, deflated against the IMMUTABLE
    # pre-registered variant count. NOT the per-ticker execution count — that is
    # the whole point: n_trials is fixed, so the recommendation cannot change
    # because a user ran more backtests (experiments.deflation_n()). The
    # execution is still LOGGED for the forward audit, idempotently per variant.
    core_sig = seven_pillar_core_strategy(df)
    volume = df["Volume"] if "Volume" in df.columns else None
    bt = run_vectorized_backtest(df["Close"], core_sig,
                                 cost_model=DEFAULT_COST_MODEL, volume=volume)
    m = compute_performance_metrics(bt.strategy_returns)
    n_trials = experiments.deflation_n()          # FIXED — interaction-independent
    dsr = deflated_sharpe_ratio(m["sharpe_ratio"], n_trials=n_trials,
                                skewness=m["skewness"], kurtosis=m["kurtosis"],
                                n_obs=m["n_observations"])
    core_now = int(core_sig.iloc[-1]) if len(core_sig) else 0

    # ── 2b. Statistical-edge assessment: walk-forward + embargo + PBO +
    # net-cost + minimum sample. This — not the raw dSR — decides whether the
    # edge is PROVEN, and it gates Kelly sizing below.
    edge = _assess_statistical_edge(df, core_sig, bt, m, dsr, volume)

    # Kelly is GATED: non-zero size requires a proven edge (HIGH statistical
    # confidence). A promising-but-unproven backtest sizes to ZERO, by rule.
    raw_kelly = safe_kelly_fraction(bt.strategy_returns)
    kelly = raw_kelly if edge["level"] == "HIGH" else 0.0

    # ── 3. Levels — investing-horizon ATR geometry ─────────────────────────
    # Wider than the trading path (2x ATR stop vs 1.5x): an investor holding
    # months must survive ordinary weekly noise a swing trader would exit on.
    atr = compute_atr(df, window=14)
    stop_distance = max(2.0 * atr, current_price * 0.02)
    levels = {
        "entry_zone_low": round(current_price - 0.5 * atr, 2),
        "entry_zone_high": round(current_price + 0.25 * atr, 2),
        "stop_loss": round(current_price - stop_distance, 2),
        "target_price": round(current_price + 2.0 * stop_distance, 2),   # 2:1 R:R
        "atr_14": round(atr, 2),
        "formula": "entry [p-0.5*ATR, p+0.25*ATR]; stop p-max(2*ATR,2%); target 2:1 R:R",
    }

    # ── 4. Horizon — deterministic map from the volatility regime ──────────
    vol_regime = algo_signals.get("vol_regime", "MEDIUM")
    time_horizon_days = {"HIGH": 45, "MEDIUM": 91, "LOW": 126}.get(vol_regime, 91)

    # ── 5. FOUR SEPARATE, CALIBRATED CONFIDENCE DIMENSIONS ─────────────────
    # The old single "conviction" blended a good narrative with a proven edge —
    # exactly what let a nice story inflate a bet. These are now distinct, and
    # only the DETERMINISTIC ones gate anything. LLM prose informs `thesis`
    # only, and `thesis` gates nothing.
    action = snap["action"]
    core_pillars_agreeing = sum(
        1 for k in ("technical", "algo", "fundamentals")
        if snap["pillars"][k]["score"] >= 60)
    bullish_action = action in ("BUY", "ACCUMULATE")
    core_agrees = (core_now == 1) == bullish_action if action != "HOLD" else True

    confidence = {
        "thesis": _thesis_confidence(snap, core_agrees, llm_prose),
        "data": _data_confidence(snap, pit, reddit, stocktwits),
        "statistical_edge": edge,
        "allocation": _allocation_confidence(edge, kelly, snap["risk_veto"]),
    }
    # The headline `conviction` now TRACKS PROVEN EDGE, not narrative — it is
    # the statistical-edge level, never higher. A great story with no edge reads
    # LOW/NONE, as it should.
    conviction = edge["level"]

    # ── 6. Evidence claims — every number a reader sees is traceable ───────
    claims = _record_claims(ledger, ticker, snap, run_id, confidence)

    # Log the execution to the ledger for the forward audit — idempotent per
    # (ticker, variant), so repeated runs / Compare-All never inflate anything.
    experiments.record_execution(
        ticker, "seven_pillar_core",
        outcome="alive" if edge["level"] == "HIGH" else "dead",
        sharpe=m["sharpe_ratio"], dsr=round(dsr, 4), cost_model=bt.cost_model)

    # ── 7. Historical hit rate of THIS engine (forward track record) ───────
    try:
        from data.store import get_historical_hit_rate
        hit = get_historical_hit_rate(strategy_source="seven_pillar_composite",
                                      ticker=ticker)
    except Exception:
        hit = {"no_data": True}

    # Additive metadata for the Prediction Ledger (NOT gate inputs — purely for
    # freezing/attribution downstream). Sourced from data already in hand.
    try:
        data_asof = str(df.index[-1].date())
    except Exception:
        data_asof = None
    expected_return_pct = round(
        (levels["target_price"] / current_price - 1.0) * 100, 2) if current_price else None

    rec: Dict[str, Any] = {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "current_price": round(current_price, 2),
        "sector": (fundamentals or {}).get("sector"),
        "regime": vol_regime,
        "data_asof": data_asof,
        "benchmark": "SPY",
        "expected_return_pct": expected_return_pct,
        "action": action,
        "conviction": conviction,              # == statistical-edge level (proven, not narrative)
        "confidence": confidence,              # the four separate dimensions
        "composite": snap["composite"],
        "core_score": snap["core_score"],
        "risk_multiplier": snap["risk_multiplier"],
        "risk_veto": snap["risk_veto"],
        "pillars": {k: {"score": v["score"], "confidence": v["confidence"],
                        "backtestable": v["backtestable"], "flags": v["flags"],
                        "claim_id": claims.get(f"pillar_{k}")}
                    for k, v in snap["pillars"].items()},
        "levels": levels,
        "position_size_pct": round(kelly * 100, 1),
        "position_size_gated": edge["level"] != "HIGH",   # true = Kelly forced to 0 (no proven edge)
        "raw_kelly_pct": round(raw_kelly * 100, 1),        # what Kelly WOULD say, ungated (for transparency)
        "time_horizon_days": time_horizon_days,
        "backtest": {
            "strategy": "seven_pillar_core",
            "sharpe": round(m["sharpe_ratio"], 3),
            "dsr": round(dsr, 3),
            "n_trials": n_trials,
            "n_trials_basis": "pre-registered variants (fixed; interaction-independent)",
            "max_drawdown": round(m["max_drawdown"], 3),
            "n_trades": bt.n_trades,
            "cost_model": bt.cost_model,
            "total_cost_pct": round(bt.total_cost * 100, 2),
            "core_signal_now": core_now,
        },
        "hit_rate": hit,
        "experiment_manifest_hash": experiments.manifest_hash(),
        "honesty_flags": {
            "survivorship_safe": False,
            "pit_fundamentals": bool((pit or {}).get("available")),
            "fundamentals_source": snap["pillars"]["fundamentals"]["inputs"].get("source", "unavailable"),
            "cost_model": bt.cost_model,
            "social_research_tracked_forward_only": True,
            "backtest_covers_core_only": True,
            "interaction_independent": True,
        },
        "claims": claims,
        "thesis": _thesis(llm_prose),
        "disclaimer": ("Computed decision support from historical data — not financial "
                       "advice. The backtest covers only the technical+algo+risk core; "
                       "social/research pillars have no testable history."),
    }
    rec["decision_fingerprint"] = decision_fingerprint(rec)
    return rec


def decision_fingerprint(rec: Dict[str, Any]) -> str:
    """A stable hash of exactly the DECISION-determining fields of a
    recommendation — the reproducibility contract.

    Excludes audit pointers and timestamps (claim_ids embed the retrieval
    microsecond; generated_at/run_id/hit_rate are metadata). Two recommendations
    with the same fingerprint are the SAME investment decision. The guarantee
    this module now makes: identical input data => identical fingerprint,
    regardless of how many backtests were run, in what order, or what is in the
    trial ledger.
    """
    import hashlib
    import json as _json
    d = {
        "action": rec["action"], "conviction": rec["conviction"],
        "composite": rec["composite"], "core_score": rec["core_score"],
        "risk_multiplier": rec["risk_multiplier"], "risk_veto": rec["risk_veto"],
        "pillars": {k: {"score": v["score"], "flags": v["flags"]}
                    for k, v in rec["pillars"].items()},
        "levels": rec["levels"],
        "position_size_pct": rec["position_size_pct"],
        "position_size_gated": rec["position_size_gated"],
        "raw_kelly_pct": rec["raw_kelly_pct"],
        "time_horizon_days": rec["time_horizon_days"],
        "backtest": {k: rec["backtest"][k] for k in
                     ("sharpe", "dsr", "n_trials", "max_drawdown", "n_trades",
                      "cost_model", "total_cost_pct", "core_signal_now")},
        "confidence": {dim: {"level": c["level"], "score": c["score"],
                             "checks": {g: c["checks"][g]["pass"] for g in c["checks"]}
                             if "checks" in c else None}
                       for dim, c in rec["confidence"].items()},
        "experiment_manifest_hash": rec["experiment_manifest_hash"],
    }
    return hashlib.sha1(_json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _level(score: float, hi: float = 0.75, mid: float = 0.5) -> str:
    return "HIGH" if score >= hi else "MEDIUM" if score >= mid else "LOW"


def _assess_statistical_edge(df, core_sig, bt, m, dsr, volume) -> Dict[str, Any]:
    """The strict, DETERMINISTIC gate: is there a PROVEN out-of-sample edge?

    HIGH requires ALL of — walk-forward (positive mean OOS Sharpe, consistent),
    embargo (applied), PBO below 0.5 (the selection generalizes), net-of-cost
    positive return, minimum sample size, and dSR >= 0.5. Anything less is not
    "a weaker buy" — it is "edge not established", and it forbids non-zero Kelly.
    This is what separates a good story from a proven strategy.
    """
    checks: Dict[str, Any] = {}
    n_obs = int(m.get("n_observations") or 0)
    checks["min_sample"] = {"pass": n_obs >= MIN_OBS, "n_obs": n_obs, "required": MIN_OBS}
    checks["net_cost_positive"] = {"pass": m.get("annualized_return", 0) > 0,
                                   "annualized_return_pct": round(m.get("annualized_return", 0) * 100, 2),
                                   "cost_model": bt.cost_model}
    checks["dsr"] = {"pass": dsr >= DSR_CREDIBLE_BAR, "value": round(dsr, 3),
                     "bar": DSR_CREDIBLE_BAR, "n_trials": experiments.deflation_n()}

    # Walk-forward with embargo + purge (leakage-controlled OOS test).
    wf_pass = False
    wf_detail: Dict[str, Any] = {"applied": False}
    try:
        def _sig_fn(prices_slice):
            return seven_pillar_core_strategy(df.loc[prices_slice.index]).reindex(
                prices_slice.index).fillna(0.0)
        wf = walk_forward_cv(df["Close"], _sig_fn, n_folds=WF_FOLDS,
                             embargo_frac=WF_EMBARGO_FRAC, purge_obs=WF_PURGE_OBS,
                             cost_model=DEFAULT_COST_MODEL, volume=volume)
        wf_pass = (wf.n_folds >= 3 and wf.mean_test_sharpe > 0
                   and wf.consistency >= WF_CONSISTENCY_MIN)
        wf_detail = {"applied": True, "n_folds": wf.n_folds,
                     "mean_oos_sharpe": round(wf.mean_test_sharpe, 3),
                     "consistency": round(wf.consistency, 2),
                     "embargo_frac": WF_EMBARGO_FRAC, "purge_obs": WF_PURGE_OBS}
    except Exception as e:
        wf_detail = {"applied": False, "error": str(e)[:80]}
    checks["walk_forward"] = {"pass": wf_pass, **wf_detail}
    checks["embargo"] = {"pass": bool(wf_detail.get("applied")), "frac": WF_EMBARGO_FRAC}

    # PBO across the pre-registered library on THIS ticker: if selecting the
    # in-sample best generalizes poorly, confidence in any single pick is capped.
    pbo_pass = False
    pbo_detail: Dict[str, Any] = {"computed": False}
    try:
        returns = {}
        for v in experiments.all_variants():
            name = v["strategy"]
            fn = STRATEGY_REGISTRY[name]
            sig = (fn(df) if name in STRATEGIES_NEEDING_FULL_DF else fn(df["Close"])).fillna(0)
            r = run_vectorized_backtest(df["Close"], sig,
                                        cost_model=DEFAULT_COST_MODEL, volume=volume)
            returns[name] = r.strategy_returns
        rmat = pd.DataFrame(returns).dropna(how="any")
        if rmat.shape[1] >= 2 and len(rmat) >= 40:
            pbo = probability_of_backtest_overfitting(rmat, n_splits=8)
            pbo_pass = pbo.pbo < PBO_MAX
            pbo_detail = {"computed": True, "pbo": round(pbo.pbo, 3),
                          "is_overfit": pbo.is_overfit, "bar": PBO_MAX}
    except Exception as e:
        pbo_detail = {"computed": False, "error": str(e)[:80]}
    checks["pbo"] = {"pass": pbo_pass, **pbo_detail}

    gates = ["min_sample", "net_cost_positive", "dsr", "walk_forward", "pbo"]
    n_pass = sum(1 for g in gates if checks[g]["pass"])
    all_pass = all(checks[g]["pass"] for g in gates)

    if not checks["min_sample"]["pass"]:
        level = "NONE"      # not enough data to make ANY statistical claim
        basis = f"insufficient sample ({n_obs} < {MIN_OBS} bars) — edge cannot be assessed"
    elif all_pass:
        level = "HIGH"
        basis = "walk-forward OOS positive + PBO<0.5 + net-cost positive + dSR>=0.5 + sample OK"
    elif checks["net_cost_positive"]["pass"] and checks["walk_forward"]["pass"]:
        level = "MEDIUM"
        basis = "profitable + walk-forward positive, but PBO or dSR not yet convincing"
    else:
        level = "LOW"
        basis = "sample sufficient but out-of-sample edge not established"

    return {"level": level, "score": round(n_pass / len(gates), 2), "basis": basis,
            "checks": checks, "gate_required": "walk-forward + embargo + PBO + net-cost + min-sample"}


def _data_confidence(snap, pit, reddit, stocktwits) -> Dict[str, Any]:
    """How trustworthy is the DATA feeding the pillars? PIT (SEC) fundamentals,
    coverage, and social sample size — separate from whether the edge is real."""
    reasons = []
    score = 0.4
    fsrc = snap["pillars"]["fundamentals"]["inputs"].get("source")
    if fsrc == "sec-edgar":
        score += 0.35; reasons.append("fundamentals via SEC/EDGAR (point-in-time)")
    else:
        reasons.append("no SEC fundamentals (non-filer or unavailable)")
    if snap["pillars"]["technical"]["score"] is not None and snap["pillars"]["algo"]["confidence"] > 0:
        score += 0.15; reasons.append("price/technical data present")
    soc = snap["pillars"]["social"]
    if "no_social_data" not in soc["flags"] and soc["confidence"] >= 0.5:
        score += 0.1; reasons.append("social sample adequate")
    elif "no_social_data" in soc["flags"]:
        reasons.append("no social data")
    score = min(1.0, score)
    return {"level": _level(score), "score": round(score, 2), "basis": "; ".join(reasons)}


def _thesis_confidence(snap, core_agrees, llm_prose) -> Dict[str, Any]:
    """The NARRATIVE dimension: how coherent is the qualitative case? Informed by
    pillar agreement and (optionally) LLM prose. Display-only — it GATES NOTHING
    (P4 + the goal's 'LLM prose must not override deterministic gates')."""
    agree = sum(1 for k in ("technical", "algo", "fundamentals", "research", "social")
                if snap["pillars"][k]["score"] >= 55)
    score = 0.2 + 0.13 * agree                     # 0..5 pillars leaning in
    if core_agrees:
        score += 0.1
    if llm_prose and (llm_prose.get("summary") or llm_prose.get("bull_case")):
        score += 0.05
    score = min(1.0, score)
    return {"level": _level(score), "score": round(score, 2),
            "basis": f"{agree}/5 pillars lean the same way"
                     + ("; LLM thesis present" if llm_prose else "; no LLM thesis (computed only)"),
            "note": "narrative only — does not affect sizing or the statistical gate"}


def _allocation_confidence(edge, kelly, risk_veto) -> Dict[str, Any]:
    """Confidence in the POSITION SIZE. Follows the statistical gate: non-zero
    Kelly is permitted only with a proven edge, so allocation confidence can
    only be HIGH when statistical-edge is HIGH."""
    if edge["level"] != "HIGH":
        return {"level": "NONE", "score": 0.0,
                "basis": "size gated to 0 — statistical edge not established"}
    if risk_veto:
        return {"level": "LOW", "score": 0.3,
                "basis": "edge proven but risk veto active — size heavily constrained"}
    score = min(1.0, 0.6 + kelly * 4.0)      # more Kelly -> more allocation confidence
    return {"level": _level(score), "score": round(score, 2),
            "basis": f"proven edge; half-Kelly {round(kelly*100,1)}% (capped 10%)"}


def _thesis(llm_prose: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The LLM's seat at the table: narrative only, clearly labelled."""
    if not llm_prose:
        return None
    return {
        "summary": llm_prose.get("summary"),
        "bull_case": llm_prose.get("bull_case"),
        "bear_case": llm_prose.get("bear_case"),
        "key_catalysts": llm_prose.get("key_catalysts"),
        "source": "llm_prose (narrative only — every number above is computed)",
    }


def _record_claims(ledger, ticker: str, snap: Dict[str, Any], run_id: str,
                   confidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One EvidenceLedger claim per pillar + one for the composite.

    Input datums are built from each pillar's `inputs` snapshot. available_at is
    the retrieval moment and as_of_honored=False — these are honest-provenance
    (you can see exactly what fed the number) but not time-reconstructable,
    unlike EDGAR claims. A ledger failure degrades to claim_id=None (the number
    still shows; only its stored audit row is lost) — same pattern as
    fundamentals_pit._safe_claim.
    """
    from financial_data.schemas import make_datum, make_source

    now = datetime.now()
    claims: Dict[str, Any] = {}
    pillar_datums = {}

    for name, p in snap["pillars"].items():
        try:
            d = make_datum(
                kind="derived_metric",
                value=p["score"], available_at=now,
                source=make_source(provider="stock_agent_pillars",
                                   document=f"{ticker}:{name}:{now.date()}",
                                   ref=p["formula"][:120]),
                symbol=ticker, concept=f"pillar_{name}", unit="score_0_100",
                confidence=p["confidence"] if p["confidence"] > 0 else 0.1,
                extra={"inputs": p["inputs"], "flags": p["flags"]})
            pillar_datums[name] = d
            claims[f"pillar_{name}"] = ledger.record_claim(
                f"pillar_{name}", p["score"], p["formula"], {name: d},
                symbol=ticker, unit="score_0_100", as_of=None,
                as_of_honored=False, run_id=run_id)
        except Exception:
            claims[f"pillar_{name}"] = None

    try:
        formula = (f"50 + (clip({'+'.join(f'{w}*{k}' for k, w in snap['weights'].items())}"
                   f" + modifiers, 0, 100) - 50) * risk_mult")
        claims["composite"] = ledger.record_claim(
            "composite_score", snap["composite"], formula,
            {k: v for k, v in pillar_datums.items()},
            symbol=ticker, unit="score_0_100", as_of=None,
            as_of_honored=False, run_id=run_id)
    except Exception:
        claims["composite"] = None

    # Record the statistical-edge assessment as its own claim: the number is the
    # edge score, the formula is the gate, and the evidence is the walk-forward /
    # PBO / cost checks — so a reader can trace WHY a size was (or was not) taken.
    if confidence:
        try:
            from financial_data.schemas import make_datum as _mk, make_source as _src
            edge = confidence["statistical_edge"]
            ed = _mk(kind="derived_metric", value=edge["score"], available_at=now,
                     source=_src(provider="stock_agent_stat_edge",
                                 document=f"{ticker}:statistical_edge:{now.date()}",
                                 ref=edge["gate_required"][:120]),
                     symbol=ticker, concept="statistical_edge", unit="pass_fraction",
                     confidence=0.9, extra={"level": edge["level"], "checks": edge["checks"]})
            claims["statistical_edge"] = ledger.record_claim(
                "statistical_edge", edge["score"], edge["gate_required"], {"edge": ed},
                symbol=ticker, unit="pass_fraction", as_of=None,
                as_of_honored=False, run_id=run_id)
        except Exception:
            claims["statistical_edge"] = None
    return claims


def log_composite_recommendation(rec: Dict[str, Any]) -> Optional[int]:
    """Persist through the EXISTING store so the outcome tracker just works.

    grounding_strategy="seven_pillar_composite" lets get_historical_hit_rate
    filter this engine's calls without any schema change. Returns the row id
    for /api/track/check, or None if the store hiccups (a logging failure must
    never break the user's analysis — but it is surfaced in the result)."""
    from data.store import log_recommendation
    # Freeze an immutable prediction snapshot for the Prediction Ledger. Additive
    # and failure-isolated — a freeze hiccup must never break the analysis, and
    # it never touches the recommendation itself (measurement only).
    try:
        from data.prediction_ledger import freeze_prediction
        rec["snapshot_id"] = freeze_prediction(rec)
    except Exception:
        rec["snapshot_id"] = None
    try:
        payload = {
            "ticker": rec["ticker"],
            "current_price": rec["current_price"],
            "prediction": {
                "action": rec["action"],
                "conviction": rec["conviction"],
                "entry_price": rec["levels"]["entry_zone_high"],
                "target_price": rec["levels"]["target_price"],
                "stop_loss": rec["levels"]["stop_loss"],
                "time_horizon_days": rec["time_horizon_days"],
                "composite": rec["composite"],
                "pillars": {k: v["score"] for k, v in rec["pillars"].items()},
                "confidence": {k: v["level"] for k, v in rec.get("confidence", {}).items()},
                "position_size_gated": rec.get("position_size_gated"),
                "experiment_manifest_hash": rec.get("experiment_manifest_hash"),
                "honesty_flags": rec["honesty_flags"],
                "claims": rec["claims"],
            },
            "grounding": {
                "grounding_strategy": "seven_pillar_composite",
                "position_size": rec["position_size_pct"] / 100.0,
            },
        }
        return log_recommendation(payload)
    except Exception:
        return None
