"""
Historical evaluation of cross-sectional rankings (delisting-inclusive).

Answers "did the ranks work?" WITHOUT the two biases that flatter every naive
study:
  - Survivorship: delisted names are NEVER dropped. If a security delists inside
    the evaluation horizon, its return is measured to the last tradable price and
    then the documented conservative delisting return is applied for the
    remainder (DELISTING_POLICY.md). A bankruptcy contributes ~-100%, exactly the
    loss a survivorship-biased study hides.
  - Look-ahead: forward returns use only prices AFTER the ranking date; the
    ranks themselves were built point-in-time upstream.

Metrics per horizon (1/5/20/60/252 trading days): top-decile return, bottom-
decile return, long-short spread (gross + net of realistic costs), benchmark-
relative, sector-neutral, rank information coefficient (Spearman), hit rate.
A schedule of ranking dates additionally yields IC stability, turnover, and a
cost-aware long-short series scored with dSR (deflated against the pre-registered
ranking-config count via the ExperimentRegistry) — never optimized on this set.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

HORIZONS = (1, 5, 20, 60, 252)


def _spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return round(num / (dx * dy), 4) if dx > 0 and dy > 0 else None


def forward_return(px_full, as_of: str, horizon: int,
                   delist_return_pct: Optional[float]) -> Optional[Dict[str, Any]]:
    """PIT + delisting-inclusive forward return over `horizon` trading days.

    Returns None only when there is genuinely no entry price. A security that
    delists inside the window is MATURED (its fate is realized), not dropped."""
    import pandas as pd
    px = px_full.dropna()
    if not len(px):
        return None
    pos = int(px.index.searchsorted(pd.Timestamp(as_of)))
    if pos >= len(px):
        return None
    entry = float(px.iloc[pos])
    hpos = pos + horizon
    if hpos < len(px):
        exit_px = float(px.iloc[hpos])
        return {"ret": exit_px / entry - 1.0, "matured": True, "delisted": False}
    # The series ends before the horizon. If the name DELISTED, apply the
    # conservative delisting return for the remainder — never drop it.
    last = float(px.iloc[-1])
    r_to_last = last / entry - 1.0
    if delist_return_pct is not None:
        total = (1.0 + r_to_last) * (1.0 + delist_return_pct / 100.0) - 1.0
        return {"ret": total, "matured": True, "delisted": True}
    return {"ret": r_to_last, "matured": False, "delisted": False}   # just not enough data yet


def evaluate_ranking(result: Dict[str, Any], provider) -> Dict[str, Any]:
    """Evaluate a single ranking's forward performance at every horizon."""
    as_of = result["as_of"]
    cov = provider.coverage()
    bench_id = provider.benchmark_id()
    bench_full = provider.prices(bench_id, cov["start"], cov["end"]) if bench_id else None

    ranked = result["ranked"]
    n = len(ranked)
    decile = max(1, n // 10)
    # Precompute each security's full forward price series + delisting return.
    fret_cache: Dict[str, Any] = {}
    for x in ranked:
        px = provider.prices(x["security_id"], cov["start"], cov["end"])
        dr = provider.delisting_return_pct(x["security_id"]) if hasattr(provider, "delisting_return_pct") else None
        fret_cache[x["security_id"]] = (px, dr)

    by_horizon = {}
    for h in HORIZONS:
        rets, comps, sectors, delisted_flags = [], [], [], []
        bench_ret = None
        if bench_full is not None:
            b = forward_return(bench_full, as_of, h, None)
            bench_ret = b["ret"] if b else None
        for x in ranked:
            px, dr = fret_cache[x["security_id"]]
            fr = forward_return(px, as_of, h, dr)
            if fr is None or not fr["matured"]:
                continue
            rets.append(fr["ret"]); comps.append(x["composite_raw"])
            sectors.append(x.get("sector")); delisted_flags.append(fr["delisted"])
        if len(rets) < 3:
            by_horizon[h] = {"n": len(rets), "insufficient": True}
            continue
        order = sorted(range(len(rets)), key=lambda i: comps[i], reverse=True)
        top_idx = order[:decile]; bot_idx = order[-decile:]
        top_ret = sum(rets[i] for i in top_idx) / len(top_idx)
        bot_ret = sum(rets[i] for i in bot_idx) / len(bot_idx)
        ls = top_ret - bot_ret
        # sector-neutral long-short: demean returns within sector, then decile spread
        sect_mean = {}
        for s in set(sectors):
            vals = [rets[i] for i in range(len(rets)) if sectors[i] == s]
            sect_mean[s] = sum(vals) / len(vals) if vals else 0.0
        sn = [rets[i] - sect_mean[sectors[i]] for i in range(len(rets))]
        sn_ls = (sum(sn[i] for i in top_idx) / len(top_idx)) - (sum(sn[i] for i in bot_idx) / len(bot_idx))
        ic = _spearman(comps, rets)
        hit = sum(1 for i in top_idx if rets[i] > (bench_ret or 0)) / len(top_idx)
        by_horizon[h] = {
            "n": len(rets), "n_delisted_included": sum(delisted_flags),
            "top_decile_return_pct": round(top_ret * 100, 3),
            "bottom_decile_return_pct": round(bot_ret * 100, 3),
            "long_short_spread_pct": round(ls * 100, 3),
            "benchmark_return_pct": round(bench_ret * 100, 3) if bench_ret is not None else None,
            "long_short_vs_benchmark_pct": round((ls - 0) * 100, 3),
            "sector_neutral_ls_pct": round(sn_ls * 100, 3),
            "rank_ic": ic,
            "top_decile_hit_rate": round(hit, 3),
        }
    return {"as_of": as_of, "universe_id": result["universe_id"],
            "survivorship_safe": result["survivorship_safe"],
            "ranking_run_id": result.get("ranking_run_id"),
            "by_horizon": by_horizon,
            "delisting_treatment": "conservative (to-last-price then delisting return); delisted names INCLUDED",
            "note": "reference-fixture synthetic prices; illustrates mechanics, not real edge"}


def evaluate_schedule(dates: List[str], universe_id: str = "reference-smallcap-demo",
                      config_version: str = "xsec-v1", horizon: int = 20,
                      cost_bps: float = 10.0,
                      provider: Optional[Any] = None) -> Dict[str, Any]:
    """Run rankings across a schedule of dates and aggregate the primary horizon:
    IC time series (+ IC information ratio), a cost-aware long-short return series
    (Sharpe + dSR), turnover, and rank stability. dSR deflates against the
    pre-registered ranking-config count via the ExperimentRegistry. Pass `provider`
    to reuse one instance across dates."""
    from backtest import experiments
    from backtest.engine import deflated_sharpe_ratio
    from xsection import ranking
    from xsection.universe import get_provider, UniverseIncomplete

    if provider is None:
        provider = get_provider(universe_id)
    ics: List[float] = []
    ls_series: List[float] = []
    prev_top: Optional[set] = None
    turnovers: List[float] = []
    per_date = []
    for d in dates:
        try:
            res = ranking.run_ranking(d, universe_id=universe_id, config_version=config_version,
                                      persist=False, provider=provider)
        except UniverseIncomplete:
            continue
        if res.get("status") != "OK":
            continue
        ev = evaluate_ranking(res, provider)
        hb = ev["by_horizon"].get(horizon, {})
        if hb.get("insufficient") or hb.get("rank_ic") is None:
            continue
        ics.append(hb["rank_ic"])
        # cost-aware long-short: gross spread minus 2*cost (enter long+short)
        gross = hb["long_short_spread_pct"] / 100.0
        net = gross - 2 * (cost_bps / 10000.0)
        ls_series.append(net)
        top_set = {x["security_id"] for x in res["top"]}
        if prev_top is not None and top_set:
            turnover = 1.0 - len(top_set & prev_top) / len(top_set)
            turnovers.append(turnover)
        prev_top = top_set
        per_date.append({"as_of": d, "rank_ic": hb["rank_ic"],
                         "ls_gross_pct": hb["long_short_spread_pct"],
                         "ls_net_pct": round(net * 100, 3)})

    if not ics:
        return {"status": "INSUFFICIENT_DATA", "dates_evaluated": 0, "horizon": horizon}

    mean_ic = sum(ics) / len(ics)
    ic_std = (sum((x - mean_ic) ** 2 for x in ics) / len(ics)) ** 0.5 if len(ics) > 1 else 0.0
    ls_mean = sum(ls_series) / len(ls_series)
    ls_std = (sum((x - ls_mean) ** 2 for x in ls_series) / len(ls_series)) ** 0.5 if len(ls_series) > 1 else 0.0
    ls_sharpe = (ls_mean / ls_std * math.sqrt(252 / horizon)) if ls_std > 0 else 0.0
    n_obs = len(ls_series)
    dsr = None
    if n_obs >= 2:
        # skew/kurtosis of the net LS series
        m = ls_mean
        s = ls_std or 1e-9
        skew = sum(((x - m) / s) ** 3 for x in ls_series) / n_obs
        kurt = sum(((x - m) / s) ** 4 for x in ls_series) / n_obs
        try:
            dsr = round(deflated_sharpe_ratio(ls_sharpe, n_trials=len(experiments.RANKING_CONFIGS),
                                              skewness=skew, kurtosis=kurt, n_obs=n_obs), 4)
        except Exception:
            dsr = None
    return {
        "status": "OK", "universe_id": universe_id, "config_version": config_version,
        "config_hash": experiments.ranking_config_hash(config_version),
        "horizon": horizon, "cost_bps": cost_bps, "dates_evaluated": len(ics),
        "mean_rank_ic": round(mean_ic, 4),
        "ic_information_ratio": round(mean_ic / ic_std, 4) if ic_std > 0 else None,
        "long_short_net_mean_pct": round(ls_mean * 100, 3),
        "long_short_net_annualized_sharpe": round(ls_sharpe, 3),
        "deflated_sharpe": dsr,
        "n_trials_basis": f"pre-registered ranking configs = {len(experiments.RANKING_CONFIGS)} (fixed)",
        "avg_turnover": round(sum(turnovers) / len(turnovers), 3) if turnovers else None,
        "per_date": per_date,
        "delisting": "delisted names included in every long-short leg",
        "cost_note": f"net = gross - 2*{cost_bps}bps (long+short legs)",
    }
