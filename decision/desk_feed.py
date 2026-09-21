"""
The desk half of the OptionsPilot loop — the direction that needs no credential.

OptionsPilot reads this service for direction and scores it at 45% of its
composite. Measured against its live universe on 2026-09-21: **2 of 31 names
had any view at all**, and none was fresh. Its runs therefore record "0 ideas
from 31 decisions", which reads like a quiet market and is actually this
service saying nothing.

The cause is not an outage. `/api/predictions` serves the prediction ledger,
and the only things that write to it are `log_composite_recommendation` (which
fires when somebody analyses a ticker by hand) and the heartbeat — and the
heartbeat is a LaunchAgent that runs on a laptop, not on Railway. So the
deployed desk accumulates a view only for whatever a human happened to look at.

This module lets the desk be asked to form a view on a named set of symbols and
freeze it, which is the input OptionsPilot is starving for. It adds no new
scoring: `forecast_and_freeze` is the heartbeat's own per-ticker path, reused
unchanged, so a view served to OptionsPilot is the same object the calibration
ledger will later grade. A feed that produced numbers nothing grades would be
worse than no feed.

`coverage()` deliberately mirrors OptionsPilot's own vocabulary — absent /
stale / neutral / directional, ten days — so the two services agree on what
"covered" means instead of each computing its own answer and disagreeing.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Callable, Dict, List, Optional

# OptionsPilot's own staleness bar. Matching it is the point: a desk that calls
# a view fresh while the consumer calls it stale is worse than either rule.
MAX_AGE_DAYS = 10

# Actions that carry a direction. Copied from OptionsPilot's `_BULLISH` /
# `_BEARISH` sets so a name counted as directional here is counted the same way
# there.
BULLISH = {"buy", "strong_buy", "accumulate", "overweight", "long"}
BEARISH = {"sell", "strong_sell", "reduce", "underweight", "short"}


def stance_for(action: Optional[str]) -> str:
    a = str(action or "").strip().lower()
    if a in BULLISH:
        return "bullish"
    if a in BEARISH:
        return "bearish"
    return "neutral"


def _latest_by_ticker(limit: int = 500) -> Dict[str, Dict[str, Any]]:
    """The most recent frozen view per ticker."""
    from data import prediction_ledger as pl
    latest: Dict[str, Dict[str, Any]] = {}
    for row in pl.list_snapshots(limit=limit):
        ticker = (row.get("ticker") or "").upper()
        if not ticker:
            continue
        current = latest.get(ticker)
        if current is None or (row.get("created_at") or "") > (current.get("created_at") or ""):
            latest[ticker] = row
    return latest


def _age_days(created_at: Optional[str], as_of: Optional[dt.datetime] = None) -> Optional[int]:
    if not created_at:
        return None
    try:
        when = dt.datetime.fromisoformat(str(created_at))
    except ValueError:
        return None
    return ((as_of or dt.datetime.now()) - when).days


def coverage(symbols: List[str], as_of: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """How much of a universe this desk can actually speak to.

    Reachable and useful are different facts, and this answers the second. A
    name is only `directional` if it has a view, the view is fresher than the
    shared staleness bar, AND the action carries a direction — a HOLD scores
    zero on OptionsPilot's directional component exactly as an absent view does.
    """
    symbols = [s.upper().strip() for s in (symbols or []) if s and s.strip()]
    latest = _latest_by_ticker()

    absent, stale, neutral, directional = [], [], [], []
    detail: Dict[str, Any] = {}
    for symbol in symbols:
        row = latest.get(symbol)
        if row is None:
            absent.append(symbol)
            detail[symbol] = {"state": "absent"}
            continue
        age = _age_days(row.get("created_at"), as_of)
        stance = stance_for(row.get("action"))
        if age is None or age > MAX_AGE_DAYS:
            stale.append(symbol)
            state = "stale"
        elif stance == "neutral":
            neutral.append(symbol)
            state = "neutral"
        else:
            directional.append(symbol)
            state = "directional"
        detail[symbol] = {"state": state, "action": row.get("action"),
                          "stance": stance, "age_days": age,
                          "created_at": row.get("created_at")}

    n = len(symbols)
    usable = len(directional)
    return {
        "universe": n,
        "absent": absent,
        "stale": stale,
        "neutral": neutral,
        "directional": directional,
        "detail": detail,
        "usable_fraction": round(usable / n, 3) if n else None,
        "max_age_days": MAX_AGE_DAYS,
        "means": _means(n, usable, len(absent), len(stale), len(neutral)),
    }


def _means(n: int, usable: int, absent: int, stale: int, neutral: int) -> str:
    if not n:
        return "No symbols were asked about."
    if usable == 0:
        return (f"NO name carries a fresh directional view. Anything consuming this desk for "
                f"direction is getting nothing usable: {absent} absent, {stale} older than "
                f"{MAX_AGE_DAYS} days, {neutral} with no direction.")
    if usable < n // 2:
        return (f"{usable} of {n} names carry a fresh directional view. The rest are "
                f"{absent} absent, {stale} stale, {neutral} neutral.")
    return (f"{usable} of {n} names carry a fresh directional view "
            f"({absent} absent, {stale} stale, {neutral} neutral).")


def refresh(symbols: List[str], limit: int = 10, force: bool = False,
            recommend_fn: Optional[Callable] = None,
            as_of: Optional[str] = None) -> Dict[str, Any]:
    """Form and freeze a view for each symbol, newest-needed first.

    `limit` bounds the work per call because each symbol builds a full
    recommendation. The caller is expected to loop; the operation is idempotent
    per (ticker, day), so repeating it is free rather than duplicating
    observations.

    Symbols that already have a fresh view are skipped before any work is done,
    so a second pass costs nothing and the ordering puts absent names first.
    """
    from agents.heartbeat import forecast_and_freeze

    symbols = [s.upper().strip() for s in (symbols or []) if s and s.strip()]
    before = coverage(symbols)

    # Absent first, then stale, then neutral — spend a bounded budget where it
    # buys the most coverage.
    queue = (list(before["absent"]) + list(before["stale"]) + list(before["neutral"])) \
        if not force else list(symbols)
    queue = queue[:max(0, int(limit))]

    results: List[Dict[str, Any]] = []
    for symbol in queue:
        try:
            record = forecast_and_freeze(symbol, recommend_fn=recommend_fn,
                                         as_of=as_of, force=force)
        except Exception as e:                      # forecast_and_freeze does not raise,
            record = {"ticker": symbol, "status": "error",   # but a caller must not depend
                      "reason": f"{type(e).__name__}"}       # on that promise.
        results.append({"ticker": symbol,
                        "status": record.get("status"),
                        "reason": record.get("reason"),
                        "action": (record.get("recommendation") or {}).get("action")
                        if isinstance(record.get("recommendation"), dict) else record.get("action")})

    after = coverage(symbols)
    gained = sorted(set(after["directional"]) - set(before["directional"]))

    return {
        "requested": len(symbols),
        "attempted": len(queue),
        "remaining": max(0, len(before["absent"]) + len(before["stale"])
                         + len(before["neutral"]) - len(queue)),
        "results": results,
        "coverage_before": {k: before[k] for k in
                            ("universe", "usable_fraction", "means")},
        "coverage_after": {k: after[k] for k in
                           ("universe", "usable_fraction", "means")},
        "newly_directional": gained,
        "note": ("Views are frozen through the same path the calibration ledger grades, so "
                 "anything served to a consumer is measured later rather than merely shown."),
    }
