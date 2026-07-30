"""
ExperimentRegistry — the IMMUTABLE, pre-registered catalog of strategy variants.

The multiple-comparisons problem the Deflated Sharpe guards against is a
property of the RESEARCH DESIGN — "how many strategy/parameter combinations did
we build and could have selected?" — NOT of how many times a user clicked a
button. The old code conflated the two: the dSR denominator came from
`ledger.effective_n_trials(ticker)`, a per-ticker count that GREW every time a
backtest ran. So a fresh ticker saw n_trials=1 and scored dSR≈1.0, while the
same ticker after a few Compare-All clicks saw n_trials=16 and scored the SAME
data dSR≈0.0. The recommendation changed because the user interacted — the exact
bug this module kills.

Here every variant is PRE-REGISTERED before any evaluation: strategy id +
canonical parameter set, declared in code, content-hashed into a frozen
manifest. The dSR deflation count is |variants| — FIXED, and independent of
ticker, click count, and API-call history. Parameters are asserted to equal the
strategy functions' shipped defaults (a validator enforces it), so "changed a
parameter" is impossible to do silently and tuning-on-eval-data would change the
manifest hash.

Executions are still LOGGED (for the forward audit / hit-rate tracking) via
record_execution(), but keyed by (ticker, variant) so re-running is idempotent —
they never change the deflation count and never inflate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

# ── The pre-registered universe (the ONLY place the multiple-testing count is
# defined). Each entry: strategy id + its canonical parameter set. Params MUST
# equal the strategy function's shipped defaults — validate_manifest() enforces
# it, so this doubles as a guard against tuning parameters on evaluation data.
PREREGISTERED_VARIANTS: Tuple[Dict[str, Any], ...] = (
    {"strategy": "sma_crossover",       "params": {"fast": 42, "slow": 252}},
    {"strategy": "momentum",            "params": {"bullish_threshold": 3.0, "bearish_threshold": -3.0}},
    {"strategy": "mean_reversion",      "params": {"window": 20, "threshold": 0.8}},
    {"strategy": "stat_arb",            "params": {"window": 60, "p_threshold": 0.05}},
    {"strategy": "trend_following",     "params": {"sma_window": 252, "breakout_window": 20}},
    {"strategy": "candlestick_filtered","params": {"rsi_oversold": 40.0, "rsi_overbought": 60.0}},
    {"strategy": "rsi_macd",            "params": {"rsi_oversold": 30.0, "rsi_overbought": 70.0}},
    {"strategy": "seven_pillar_core",   "params": {"enter": 65.0, "exit": 55.0}},
)


def _canon(params: Dict[str, Any]) -> str:
    return json.dumps(params or {}, sort_keys=True)


def variant_id(strategy: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Stable content id for a (strategy, params) pair — the audit key."""
    if params is None:
        params = canonical_params(strategy)
    return hashlib.sha1(f"{strategy}|{_canon(params)}".encode()).hexdigest()[:16]


def manifest_hash() -> str:
    """Fingerprint of the entire frozen registry. Changes iff a variant is
    added/removed or a parameter is edited — an immutability tripwire recorded
    with every recommendation so a reader can verify which universe deflated it."""
    blob = json.dumps(
        [{"strategy": v["strategy"], "params": v["params"]} for v in PREREGISTERED_VARIANTS],
        sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def n_preregistered_variants() -> int:
    return len(PREREGISTERED_VARIANTS)


def deflation_n() -> int:
    """THE dSR denominator, everywhere. Fixed = |pre-registered variants|.
    Interaction-, ticker-, and history-independent by construction."""
    return n_preregistered_variants()


def canonical_params(strategy: str) -> Dict[str, Any]:
    for v in PREREGISTERED_VARIANTS:
        if v["strategy"] == strategy:
            return dict(v["params"])
    raise KeyError(f"{strategy!r} is not pre-registered")


def is_preregistered(strategy: str, params: Optional[Dict[str, Any]] = None) -> bool:
    for v in PREREGISTERED_VARIANTS:
        if v["strategy"] == strategy and (params is None or v["params"] == params):
            return True
    return False


def all_variants() -> Tuple[Dict[str, Any], ...]:
    return PREREGISTERED_VARIANTS


def validate_manifest() -> List[str]:
    """Return a list of problems (empty = valid). Enforces the two invariants
    that make this registry trustworthy:
      1. every pre-registered strategy exists in STRATEGY_REGISTRY;
      2. every pre-registered param set EQUALS the function's shipped defaults
         (proves params were not tuned on evaluation data — the manifest is the
         library's own declared design, not a fitted artifact).
    """
    from backtest.strategies import STRATEGY_REGISTRY
    problems: List[str] = []
    seen = set()
    for v in PREREGISTERED_VARIANTS:
        s, params = v["strategy"], v["params"]
        if s in seen:
            problems.append(f"duplicate pre-registered variant: {s}")
        seen.add(s)
        fn = STRATEGY_REGISTRY.get(s)
        if fn is None:
            problems.append(f"pre-registered {s!r} missing from STRATEGY_REGISTRY")
            continue
        defaults = {name: p.default for name, p in inspect.signature(fn).parameters.items()
                    if p.default is not inspect.Parameter.empty}
        for k, want in params.items():
            if k not in defaults:
                problems.append(f"{s}: param {k!r} not in function signature")
            elif defaults[k] != want:
                problems.append(f"{s}: param {k}={want} != shipped default {defaults[k]} (tuning?)")
    # Every registered strategy should be pre-registered, or the deflation count
    # under-states the real search space.
    for s in STRATEGY_REGISTRY:
        if not any(v["strategy"] == s for v in PREREGISTERED_VARIANTS):
            problems.append(f"registered {s!r} is not pre-registered (deflation under-counts)")
    return problems


# ── Pre-registered cross-sectional RANKING configuration (M-F3) ─────────────
# The composite ranking weights are declared HERE, versioned and content-hashed,
# so they cannot be tuned on the evaluation period (a validator + the ranking's
# fingerprint pin them). Risk enters as a NEGATIVE-weighted / veto term, never as
# accidental positive alpha. Weights sum to 1.0 across the alpha factors; risk is
# a separate penalty. Changing any weight changes ranking_config_hash().
RANKING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "xsec-v1": {
        "version": "xsec-v1",
        "weights": {"quality": 0.22, "growth": 0.18, "valuation": 0.22,
                    "momentum": 0.28, "revisions": 0.10},
        "risk_penalty": 0.25,          # risk score (0=safe..1=risky) subtracts up to this
        "risk_veto_percentile": 0.95,  # top-5% riskiest are vetoed out of high conviction
        "min_coverage": 0.40,          # a security needs >=40% feature coverage to rank
        "winsor_pct": 0.05,            # 5% two-sided winsorization
        "notes": "Fixed priors, NOT fitted. Risk is a penalty+veto, never positive alpha.",
    },
}


def ranking_config(version: str = "xsec-v1") -> Dict[str, Any]:
    if version not in RANKING_CONFIGS:
        raise KeyError(f"ranking config {version!r} is not pre-registered")
    return json.loads(json.dumps(RANKING_CONFIGS[version]))   # deep copy


def ranking_config_hash(version: str = "xsec-v1") -> str:
    return hashlib.sha1(
        json.dumps(RANKING_CONFIGS[version], sort_keys=True).encode()).hexdigest()[:16]


def validate_ranking_config(version: str = "xsec-v1") -> List[str]:
    """Weights sum to 1.0 (alpha factors) and risk is non-positive-alpha."""
    problems: List[str] = []
    cfg = RANKING_CONFIGS.get(version)
    if not cfg:
        return [f"ranking config {version!r} missing"]
    total = sum(cfg["weights"].values())
    if abs(total - 1.0) > 1e-9:
        problems.append(f"alpha weights sum to {total}, not 1.0")
    if cfg.get("risk_penalty", 0) < 0:
        problems.append("risk_penalty must be >= 0 (risk is a penalty, not alpha)")
    if "risk" in cfg["weights"]:
        problems.append("risk must NOT be a positively-weighted alpha factor")
    return problems


def record_execution(ticker: str, strategy: str, outcome: str = "inconclusive",
                     sharpe: Optional[float] = None, params: Optional[Dict[str, Any]] = None,
                     dsr: Optional[float] = None, cost_model: Optional[str] = None) -> Optional[str]:
    """Log an execution to the ledger for the forward audit — idempotent per
    (ticker, variant). Deliberately keyed by variant_id, NOT by period or
    caller, so Compare-All, repeated API calls, and the CLI all collapse to one
    row per variant-on-ticker. Never affects deflation_n(). Returns the trial id
    (== ledger row), or None if the ledger is unavailable (logging must never
    break an analysis)."""
    from data import ledger
    vid = variant_id(strategy, params)
    try:
        return ledger.record_trial(
            hypothesis=f"{strategy} beats buy-and-hold on {ticker.upper()}",
            params={"variant_id": vid, "strategy": strategy},   # period-free -> idempotent
            outcome=outcome, family=ledger.strategy_family(ticker),
            strategy=strategy, universe=ticker.upper(), sharpe=sharpe,
            flags={"variant_id": vid, "dsr": dsr, "cost_model": cost_model,
                   "survivorship_safe": False, "pit_fundamentals": False,
                   "manifest_hash": manifest_hash()})
    except Exception:
        return None
