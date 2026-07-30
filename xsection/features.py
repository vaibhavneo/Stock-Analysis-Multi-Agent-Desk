"""
Point-in-time cross-sectional feature store.

Every feature is computed from data available ON OR BEFORE the ranking date —
prices truncated at `as_of`, and fundamentals whose FILED date is <= `as_of`
(a filing filed after the ranking date can never enter an earlier rank). Each
feature record carries the full provenance the mission requires: security_id,
ticker_as_of, ranking_date, feature_name, raw_value, available_at, period_end,
source, evidence_id, data_quality status, missingness reason, and feature
version. Normalized values are filled in later (normalize.py) — a raw feature
record is source-truthful on its own.

Honest coverage on the REFERENCE fixture:
  - Momentum & risk families: computed from the (synthetic) price series —
    genuinely point-in-time (truncation is the guarantee).
  - Quality / growth / valuation: computed from SYNTHETIC point-in-time
    fundamentals generated deterministically per security, each with a
    period_end and a FILED date lagging it (~45d), so as-of filtering is real
    and the "future filing cannot enter an earlier rank" property is testable.
  - Estimates / revisions: UNAVAILABLE. There is no point-in-time estimate
    history in free data, and current-vintage estimates are look-ahead — so they
    are marked UNAVAILABLE and excluded, never substituted (a required test).

For a REAL universe with real CIKs, quality/growth/valuation would be sourced
from EDGAR via analyze_fundamentals_pit(as_of=...) through the gateway; the
reference fixture uses synthetic fundamentals because its CIKs are not real.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

FEATURE_VERSION = "xsec-features-v1"

# Data-quality statuses (mission §4).
ELIGIBLE = "ELIGIBLE"
PARTIAL_DATA = "PARTIAL_DATA"
STALE_DATA = "STALE_DATA"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
ILLIQUID = "ILLIQUID"
DELISTED_BUT_VALID_HISTORICALLY = "DELISTED_BUT_VALID_HISTORICALLY"
EXCLUDED = "EXCLUDED"

MIN_BARS = 130            # ~6 months of trading days to compute momentum/risk
TRADING_DAYS_Y = 252


def _feat(name: str, value: Optional[float], family: str, formula: str,
          available_at: Optional[str], source: str, period_end: Optional[str] = None,
          missing_reason: Optional[str] = None, evidence_id: Optional[str] = None) -> Dict[str, Any]:
    return {"feature_name": name, "family": family, "raw_value": value,
            "normalized_value": None, "available_at": available_at, "period_end": period_end,
            "source": source, "evidence_id": evidence_id,
            "data_quality": ELIGIBLE if value is not None else PARTIAL_DATA,
            "missingness_reason": missing_reason, "feature_version": FEATURE_VERSION,
            "formula": formula}


# ── Synthetic point-in-time fundamentals (reference fixture only) ────────────

def synthetic_fundamentals(security: Dict[str, Any], as_of: str) -> List[Dict[str, Any]]:
    """Deterministic quarterly fundamentals for a reference security, each with a
    period_end and a FILED date (~45 days after period_end). Only quarters FILED
    on or before `as_of` are returned — the point-in-time guarantee. Labelled
    synthetic; never presented as real SEC data."""
    import numpy as np
    import pandas as pd
    pm = security.get("price_model", {})
    rng = np.random.default_rng(int(pm.get("seed", 1)) + 777)
    first = security.get("first_tradable") or "2019-01-02"
    quarters = pd.date_range(first, as_of, freq="Q")
    base_rev = 200.0 + (int(pm.get("seed", 1)) % 7) * 40.0
    out = []
    for i, q_end in enumerate(quarters):
        filed = (q_end + pd.Timedelta(days=45)).date().isoformat()
        if filed > as_of:            # FUTURE filing — not knowable at as_of
            continue
        growth = 1.0 + pm.get("drift", 0.0) * 60 + rng.normal(0, 0.02)
        rev = base_rev * (growth ** i)
        margin = 0.12 + (int(pm.get("seed", 1)) % 5) * 0.03 + rng.normal(0, 0.01)
        out.append({
            "period_end": q_end.date().isoformat(), "filed": filed,
            "revenue": round(rev, 2),
            "gross_profit": round(rev * (margin + 0.25), 2),
            "operating_income": round(rev * margin, 2),
            "net_income": round(rev * margin * 0.8, 2),
            "free_cash_flow": round(rev * margin * 0.7, 2),
            "total_debt": round(rev * (0.4 + (int(pm.get("seed", 1)) % 4) * 0.15), 2),
            "equity": round(rev * 1.2, 2),
            "shares": round(100.0 * (1.0 + 0.01 * i), 2),   # mild dilution
            "accessn": f"SYNTH-{security['security_id']}-{q_end.date().isoformat()}",
        })
    return out


# ── Feature generation per security ─────────────────────────────────────────

def compute_features(member: Dict[str, Any], as_of: str, prices, bench_prices,
                     security_raw: Dict[str, Any], fundamentals_fn=None) -> Dict[str, Any]:
    """Compute all point-in-time features for one security as of `as_of`.

    prices/bench_prices: adjusted-close series TRUNCATED at as_of (caller slices).
    fundamentals_fn(security_raw, as_of) -> list of quarterly records (same shape
    as `synthetic_fundamentals`). Defaults to the synthetic generator so the
    reference fixture is unchanged; a production provider injects an EDGAR-backed
    function so quality/growth/valuation come from real filed-date-governed data.
    Returns {security_id, ticker_as_of, status, coverage, confidence, features[],
             flags[], exclusion_reason}."""
    import numpy as np
    import pandas as pd

    sid = member["security_id"]
    px = prices.dropna()
    px = px[px.index <= pd.Timestamp(as_of)]
    feats: List[Dict[str, Any]] = []
    flags: List[str] = []

    delisted_valid = member.get("listing_status") == "delisted" and \
        member.get("delisting_date") and as_of <= member["delisting_date"]

    if len(px) < MIN_BARS:
        status = INSUFFICIENT_HISTORY
        return {"security_id": sid, "ticker_as_of": member.get("ticker_as_of"),
                "sector": member.get("sector"), "industry": member.get("industry"),
                "status": status, "coverage": 0.0, "confidence": 0.0, "features": [],
                "flags": ["insufficient_price_history"],
                "exclusion_reason": f"{INSUFFICIENT_HISTORY}: {len(px)} < {MIN_BARS} bars"}

    last_date = str(px.index[-1].date())
    ret = px.pct_change().dropna()
    src = f"gateway:prices({member.get('provenance','fixture')})"

    def mom(n):
        return (float(px.iloc[-1] / px.iloc[-n] - 1.0) if len(px) > n else None)

    # ── Momentum family (price-derived, genuinely PIT) ──────────────────────
    feats += [
        _feat("mom_1m", mom(21), "momentum", "P_t/P_{t-21}-1", last_date, src),
        _feat("mom_3m", mom(63), "momentum", "P_t/P_{t-63}-1", last_date, src),
        _feat("mom_6m", mom(126), "momentum", "P_t/P_{t-126}-1", last_date, src),
        _feat("mom_12m", mom(252), "momentum", "P_t/P_{t-252}-1", last_date, src),
    ]
    # Skip-recent-month 12-1 momentum (classic; avoids short-term reversal)
    m12_1 = (float(px.iloc[-21] / px.iloc[-252] - 1.0) if len(px) > 252 else None)
    feats.append(_feat("mom_12_1", m12_1, "momentum", "P_{t-21}/P_{t-252}-1 (skip 1m)", last_date, src))
    vol20 = float(ret.tail(20).std() * math.sqrt(TRADING_DAYS_Y)) if len(ret) >= 20 else None
    voladj = (mom(126) / vol20 if (mom(126) is not None and vol20) else None)
    feats.append(_feat("mom_voladj_6m", voladj, "momentum", "mom_6m / ann_vol_20d", last_date, src))
    # Relative strength vs benchmark
    rs = None
    if bench_prices is not None and len(bench_prices):
        b = bench_prices.dropna(); b = b[b.index <= pd.Timestamp(as_of)]
        if len(b) > 126 and len(px) > 126:
            rs = float((px.iloc[-1] / px.iloc[-126]) / (b.iloc[-1] / b.iloc[-126]) - 1.0)
    feats.append(_feat("rs_vs_bench_6m", rs, "momentum", "sec_6m_ret - bench_6m_ret (ratio)", last_date, src))
    # Trend persistence: fraction of last 60 days above the 50d SMA
    sma50 = px.rolling(50).mean()
    persist = float((px.tail(60) > sma50.tail(60)).mean()) if len(px) >= 60 else None
    feats.append(_feat("trend_persistence", persist, "momentum", "frac(last 60d > SMA50)", last_date, src))

    # ── Risk family (price-derived) ─────────────────────────────────────────
    feats.append(_feat("realized_vol", vol20, "risk", "std(daily ret,20d)*sqrt(252)", last_date, src))
    downside = ret[ret < 0]
    dvol = float(downside.tail(60).std() * math.sqrt(TRADING_DAYS_Y)) if len(downside) >= 5 else None
    feats.append(_feat("downside_vol", dvol, "risk", "std(neg daily ret,60d)*sqrt(252)", last_date, src))
    cummax = px.cummax()
    mdd = float((px / cummax - 1.0).min()) if len(px) else None
    feats.append(_feat("max_drawdown", mdd, "risk", "min(P/cummax - 1)", last_date, src))
    # drawdown state (how far below the running peak now)
    dds = float(px.iloc[-1] / cummax.iloc[-1] - 1.0) if len(px) else None
    feats.append(_feat("drawdown_state", dds, "risk", "P_t/peak - 1", last_date, src))
    # beta vs benchmark
    beta = None
    if bench_prices is not None and len(bench_prices):
        b = bench_prices.dropna(); b = b[b.index <= pd.Timestamp(as_of)]
        br = b.pct_change().dropna()
        joined = pd.concat([ret, br], axis=1, join="inner").dropna()
        if len(joined) > 60 and joined.iloc[:, 1].var() > 0:
            beta = float(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var())
    feats.append(_feat("beta", beta, "risk", "cov(sec,bench)/var(bench)", last_date, src))
    # liquidity proxy: inverse of realized vol * (no dollar volume on synthetic) — flagged
    feats.append(_feat("gap_risk", float(ret.abs().tail(60).max()) if len(ret) >= 60 else None,
                       "risk", "max(|daily ret|,60d)", last_date, src))

    # ── Fundamental families (PIT; filed<=as_of only) ───────────────────────
    # Default = synthetic (reference fixture); a production provider injects a
    # real EDGAR-backed fundamentals_fn. The source label is read off the
    # records so provenance is truthful either way.
    funds = (fundamentals_fn or synthetic_fundamentals)(security_raw, as_of)
    fsrc = (funds[-1].get("source") if funds and funds[-1].get("source")
            else "synthetic_pit_fundamentals(reference_fixture)")
    if len(funds) >= 2:
        latest, prev = funds[-1], funds[-4] if len(funds) >= 4 else funds[0]
        pe_end, filed = latest["period_end"], latest["filed"]
        eid = latest["accessn"]

        # Real EDGAR records can leave individual line items untagged (None); the
        # synthetic fixture never does. Guard every division so a missing datum
        # becomes explicit missingness (feature None -> PARTIAL_DATA) rather than
        # a crash or a fabricated number.
        def _div(a, b):
            return (a / b) if (a is not None and b) else None

        def _ttm(key):
            vals = [f.get(key) for f in funds[-4:]] if len(funds) >= 4 else [latest.get(key)]
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            s = sum(vals)
            return s * (4 / len(vals)) if len(funds) >= 4 else s * 4   # annualize partial coverage

        rg = _div(latest.get("revenue"), prev.get("revenue"))
        rev_g = (rg - 1.0) if rg is not None else None
        feats.append(_feat("revenue_growth_yoy", rev_g, "growth", "rev_t/rev_{t-4q}-1", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("gross_margin", _div(latest.get("gross_profit"), latest.get("revenue")),
                           "quality", "gross_profit/revenue", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("operating_margin", _div(latest.get("operating_income"), latest.get("revenue")),
                           "quality", "operating_income/revenue", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("fcf_margin", _div(latest.get("free_cash_flow"), latest.get("revenue")),
                           "quality", "fcf/revenue", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("leverage", _div(latest.get("total_debt"), latest.get("equity")),
                           "risk", "total_debt/equity", filed, fsrc, pe_end, evidence_id=eid))
        dl = _div(latest.get("shares"), prev.get("shares"))
        feats.append(_feat("dilution", (dl - 1.0) if dl is not None else None,
                           "quality", "shares_t/shares_{t-4q}-1", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("cash_conversion", _div(latest.get("free_cash_flow"), latest.get("operating_income")),
                           "quality", "fcf/operating_income", filed, fsrc, pe_end, evidence_id=eid))
        # Valuation: market cap = price * shares (None if shares untagged)
        mktcap = (float(px.iloc[-1]) * latest["shares"]) if latest.get("shares") else None
        ttm_rev, ttm_fcf, ttm_ni = _ttm("revenue"), _ttm("free_cash_flow"), _ttm("net_income")
        feats.append(_feat("price_to_sales", _div(mktcap, ttm_rev), "valuation", "mktcap/ttm_revenue", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("fcf_yield", _div(ttm_fcf, mktcap), "valuation", "ttm_fcf/mktcap", filed, fsrc, pe_end, evidence_id=eid))
        feats.append(_feat("earnings_yield", _div(ttm_ni, mktcap), "valuation", "ttm_ni/mktcap", filed, fsrc, pe_end, evidence_id=eid))
    else:
        flags.append("no_pit_fundamentals")

    # ── Estimates/revisions: UNAVAILABLE (no PIT estimate history; never faked)
    feats.append(_feat("earnings_revision", None, "revisions", "n/a", None,
                       "unavailable", missing_reason="no_pit_estimate_history"))
    flags.append("estimates_unavailable_no_pit_history")

    # ── Data-quality gate ───────────────────────────────────────────────────
    n_total = len(feats)
    n_present = sum(1 for f in feats if f["raw_value"] is not None)
    coverage = round(n_present / n_total, 3) if n_total else 0.0
    # staleness: is the most recent fundamental filing badly out of date at as_of?
    if funds:
        stale_days = (pd.Timestamp(as_of) - pd.Timestamp(funds[-1]["filed"])).days
        if stale_days > 200:
            flags.append(f"stale_fundamentals_{stale_days}d")
    confidence = round(coverage * (0.85 if "no_pit_fundamentals" not in flags else 0.5), 3)

    if delisted_valid:
        status = DELISTED_BUT_VALID_HISTORICALLY
    elif coverage < 0.4:
        status = PARTIAL_DATA
    else:
        status = ELIGIBLE

    return {"security_id": sid, "ticker_as_of": member.get("ticker_as_of"),
            "name": member.get("name"), "sector": member.get("sector"),
            "industry": member.get("industry"), "listing_status": member.get("listing_status"),
            "status": status, "coverage": coverage, "confidence": confidence,
            "features": feats, "flags": flags, "exclusion_reason": None,
            "last_data_date": last_date}
