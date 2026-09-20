"""
Forward validation (Phase 23) — grading the decision, not the score.

The prediction ledger already grades whether a recommendation's DIRECTION was
right. That is a narrower question than the one this layer creates: a HOLD that
survived a 12% drawdown and recovered, and a HOLD that drifted sideways, are the
same direction and completely different decisions.

So this evaluates the things the decision actually claimed:

  - subsequent return and excess return over the benchmark
  - maximum favourable and maximum adverse excursion (the ledger already
    computes and stores both, and until now nothing read them)
  - whether the stated INVALIDATION level was hit, and how many days it took
  - whether the stated CONFIRMATION level was hit, and how many days it took
  - whether the state was appropriate in hindsight, by a stated rule

The last one needs care. "Was HOLD correct?" has no ground truth, so instead of
inventing one, `state_was_correct` is defined by an explicit, narrow rule per
state and the rule is reported alongside the verdict. A reader who disagrees
with the rule can see exactly what was applied.

And the hard line, carried over from the existing research safeguards: none of
this is called an edge. `summarize_forward_validation()` reports
NO_DEMONSTRATED_EDGE until the sample and significance requirements the rest of
this codebase already enforces are met.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from decision import journal

# Horizons at which a journaled decision is graded. Matches the prediction
# ledger's own so the two can be joined without resampling.
HORIZONS = (5, 20, 60, 126, 252)

# Minimum matured, INDEPENDENT decisions before any aggregate is reported as
# more than descriptive. Overlapping windows on the same name share a price
# path, so the count that matters is distinct (ticker, decision date) pairs.
MIN_SAMPLE_FOR_INFERENCE = 30

DEFAULT_BENCHMARK = "SPY"

# How each state is graded. Stated here, in one place, so the rule is auditable
# rather than buried in branches.
STATE_RULES = {
    "WATCH": ("correct when NOT entering avoided a loss, i.e. the subsequent return "
              "was negative or the invalidation level was hit"),
    "NO_TRADE": ("correct when the subsequent absolute excess return was small "
                 "(|excess| < 5%), i.e. there was genuinely nothing to capture"),
    "AVOID_NEW_POSITION": ("correct when the subsequent return was negative or the "
                           "invalidation level was hit"),
    "WAIT_FOR_ENTRY": ("correct when the stated entry condition was reachable — the "
                       "price traded into the entry zone at some point in the window"),
    "HOLD": ("correct when the position was not invalidated, i.e. the invalidation "
             "level was never hit"),
    "ADD_CONDITIONALLY": ("correct when the subsequent excess return was positive AND "
                          "the invalidation level was never hit"),
    "REDUCE": ("correct when the subsequent return was negative or the maximum adverse "
               "excursion exceeded the maximum favourable excursion"),
    "EXIT_CONDITIONALLY": ("correct when the invalidation level was hit, i.e. the named "
                           "exit trigger actually fired"),
    "CONFLICTED": ("not gradeable — declining to call a split is not a directional "
                   "claim, so no outcome confirms or refutes it"),
    "INSUFFICIENT_DATA": ("not gradeable — refusing to decide on missing inputs is not a "
                          "directional claim"),
}
NOT_GRADEABLE = ("CONFLICTED", "INSUFFICIENT_DATA")


def _level_from_triggers(triggers: Optional[List[Dict[str, Any]]]) -> Optional[float]:
    """Pull the first numeric price out of a trigger list's `measurable_as`
    field, which is written as e.g. "daily close < 96.56"."""
    import re
    for t in (triggers or []):
        m = re.search(r"([<>])\s*(\d+(?:\.\d+)?)", str(t.get("measurable_as") or ""))
        if m:
            try:
                return float(m.group(2))
            except ValueError:
                continue
    return None


def evaluate_decision(frozen: Dict[str, Any], prices, benchmark=None,
                      as_of: Optional[str] = None) -> Dict[str, Any]:
    """Grade one journaled decision at every horizon with enough price history.

    `prices` is a date-indexed Close series for the ticker starting at or before
    the decision date; `benchmark` the same for the benchmark.
    """
    import pandas as pd

    created = str(frozen.get("created_at"))[:10]
    try:
        start = pd.Timestamp(created)
    except Exception:
        return {"error": "unparseable created_at", "horizons": {}}

    fwd = prices[prices.index >= start]
    if len(fwd) < 2:
        return {"error": "no forward price data", "horizons": {}}

    p0 = float(fwd.iloc[0])
    bench_fwd = benchmark[benchmark.index >= start] if benchmark is not None else None
    b0 = float(bench_fwd.iloc[0]) if bench_fwd is not None and len(bench_fwd) else None

    invalidation_level = _level_from_triggers(frozen.get("invalidation"))
    confirmation_level = _level_from_triggers(frozen.get("confirmation"))
    entry_plan = frozen.get("entry_plan") or {}
    entry_low = entry_plan.get("entry_low")

    state = frozen.get("headline_state")
    out: Dict[str, Any] = {"decision_id": None, "ticker": frozen.get("ticker"),
                           "created_at": frozen.get("created_at"),
                           "headline_state": state, "horizons": {}}

    for h in HORIZONS:
        if len(fwd) <= h:
            out["horizons"][h] = {"matured": False, "reason": "window not yet complete"}
            continue

        window = fwd.iloc[:h + 1]
        p1 = float(window.iloc[-1])
        raw = (p1 / p0 - 1) * 100
        mfe = (float(window.max()) / p0 - 1) * 100
        mae = (float(window.min()) / p0 - 1) * 100

        excess = None
        bench_ret = None
        if bench_fwd is not None and len(bench_fwd) > h and b0:
            bench_ret = (float(bench_fwd.iloc[h]) / b0 - 1) * 100
            excess = raw - bench_ret

        hit_inval, days_inval = False, None
        if invalidation_level is not None:
            below = window[window <= invalidation_level]
            if len(below):
                hit_inval = True
                days_inval = int(window.index.get_loc(below.index[0]))

        hit_conf, days_conf = False, None
        if confirmation_level is not None:
            above = window[window >= confirmation_level]
            if len(above):
                hit_conf = True
                days_conf = int(window.index.get_loc(above.index[0]))

        reached_entry, days_entry = None, None
        if entry_low is not None:
            at_or_below = window[window <= entry_low]
            reached_entry = bool(len(at_or_below))
            if reached_entry:
                days_entry = int(window.index.get_loc(at_or_below.index[0]))

        correct = _grade_state(state, raw, excess, mfe, mae, hit_inval, reached_entry)

        out["horizons"][h] = {
            "matured": True,
            "as_of_date": str(window.index[-1].date()),
            "price_at_horizon": round(p1, 4),
            "raw_return_pct": round(raw, 3),
            "benchmark_return_pct": round(bench_ret, 3) if bench_ret is not None else None,
            "excess_return_pct": round(excess, 3) if excess is not None else None,
            "mfe_pct": round(mfe, 3),
            "mae_pct": round(mae, 3),
            "hit_invalidation": hit_inval,
            "days_to_invalidation": days_inval,
            "hit_confirmation": hit_conf,
            "days_to_confirmation": days_conf,
            "reached_entry_zone": reached_entry,
            "days_to_entry_zone": days_entry,
            "state_was_correct": correct,
            "grading_rule": STATE_RULES.get(state, "no rule defined for this state"),
        }

    return out


def _grade_state(state: Optional[str], raw: float, excess: Optional[float],
                 mfe: float, mae: float, hit_inval: bool,
                 reached_entry: Optional[bool]) -> Optional[bool]:
    """Apply the stated rule. None when the state is not a directional claim —
    which is a legitimate answer, not a gap."""
    if state in NOT_GRADEABLE or state is None:
        return None
    if state in ("WATCH", "AVOID_NEW_POSITION"):
        return bool(raw < 0 or hit_inval)
    if state == "NO_TRADE":
        return bool(abs(excess if excess is not None else raw) < 5.0)
    if state == "WAIT_FOR_ENTRY":
        return None if reached_entry is None else bool(reached_entry)
    if state == "HOLD":
        return bool(not hit_inval)
    if state == "ADD_CONDITIONALLY":
        return bool((excess if excess is not None else raw) > 0 and not hit_inval)
    if state == "REDUCE":
        return bool(raw < 0 or abs(mae) > abs(mfe))
    if state == "EXIT_CONDITIONALLY":
        return bool(hit_inval)
    return None


def refresh_outcomes(ticker: Optional[str] = None,
                     fetch_fn: Optional[Callable] = None,
                     limit: int = 500) -> Dict[str, Any]:
    """Evaluate every journaled decision against the price path since, and
    upsert the results into `decision_outcomes`.

    Only `decision_outcomes` is written — the journal rows themselves stay
    immutable, which is what lets a decision be graded repeatedly as more of its
    window completes without ever rewriting what it said.
    """
    import pandas as pd

    if fetch_fn is None:
        from tools.market_data import fetch_price_history

        def fetch_fn(t):                      # noqa: E306
            return fetch_price_history(t, period="5y")

    rows = journal.list_decisions(ticker, limit=limit)
    if not rows:
        return {"evaluated": 0, "matured": 0, "tickers": 0, "errors": []}

    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(r)

    bench = None
    try:
        b = fetch_fn(DEFAULT_BENCHMARK)
        bench = b["Close"].astype(float) if b is not None and not b.empty else None
    except Exception:
        bench = None

    evaluated = matured = 0
    errors: List[str] = []
    conn = journal._conn()
    try:
        for t, decisions in by_ticker.items():
            try:
                df = fetch_fn(t)
                prices = df["Close"].astype(float)
            except Exception as e:
                errors.append(f"{t}: {type(e).__name__}")
                continue
            for row in decisions:
                frozen = json.loads(row["frozen_json"])
                res = evaluate_decision(frozen, prices, bench)
                if res.get("error"):
                    continue
                evaluated += 1
                for h, o in res["horizons"].items():
                    if not o.get("matured"):
                        continue
                    matured += 1
                    conn.execute(
                        """INSERT INTO decision_outcomes
                           (decision_id, horizon_days, evaluated_at, as_of_date, matured,
                            price_at_horizon, raw_return_pct, benchmark_return_pct,
                            excess_return_pct, mae_pct, mfe_pct, hit_invalidation,
                            days_to_invalidation, hit_confirmation, days_to_confirmation,
                            state_was_correct)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                           ON CONFLICT(decision_id, horizon_days) DO UPDATE SET
                             evaluated_at=excluded.evaluated_at,
                             as_of_date=excluded.as_of_date, matured=excluded.matured,
                             price_at_horizon=excluded.price_at_horizon,
                             raw_return_pct=excluded.raw_return_pct,
                             benchmark_return_pct=excluded.benchmark_return_pct,
                             excess_return_pct=excluded.excess_return_pct,
                             mae_pct=excluded.mae_pct, mfe_pct=excluded.mfe_pct,
                             hit_invalidation=excluded.hit_invalidation,
                             days_to_invalidation=excluded.days_to_invalidation,
                             hit_confirmation=excluded.hit_confirmation,
                             days_to_confirmation=excluded.days_to_confirmation,
                             state_was_correct=excluded.state_was_correct""",
                        (row["decision_id"], h, datetime.now().isoformat(timespec="seconds"),
                         o["as_of_date"], 1, o["price_at_horizon"], o["raw_return_pct"],
                         o["benchmark_return_pct"], o["excess_return_pct"], o["mae_pct"],
                         o["mfe_pct"], 1 if o["hit_invalidation"] else 0,
                         o["days_to_invalidation"], 1 if o["hit_confirmation"] else 0,
                         o["days_to_confirmation"],
                         None if o["state_was_correct"] is None else int(o["state_was_correct"])))
        conn.commit()
    finally:
        conn.close()

    return {"evaluated": evaluated, "matured": matured,
            "tickers": len(by_ticker), "errors": errors}


def summarize_forward_validation(horizon: int = 20) -> Dict[str, Any]:
    """Aggregate what the journal has learned so far.

    Reports NO_DEMONSTRATED_EDGE until the sample requirement is met, and says
    so in the same language the rest of this codebase uses, so nobody can read
    an early, thin result as validation.
    """
    conn = journal._conn()
    try:
        rows = [dict(r) for r in conn.execute(
            """SELECT o.*, d.headline_state, d.ticker, d.created_at, d.edge_demonstrated,
                      d.decision_confidence, d.thesis_direction
               FROM decision_outcomes o
               JOIN decision_journal d ON d.decision_id = o.decision_id
               WHERE o.horizon_days=? AND o.matured=1""", (horizon,)).fetchall()]
    finally:
        conn.close()

    n = len(rows)
    independent = len({(r["ticker"], str(r["created_at"])[:10]) for r in rows})

    by_state: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        s = r["headline_state"]
        b = by_state.setdefault(s, {"n": 0, "graded": 0, "correct": 0,
                                    "returns": [], "excess": [], "mae": [], "mfe": []})
        b["n"] += 1
        if r["state_was_correct"] is not None:
            b["graded"] += 1
            b["correct"] += int(r["state_was_correct"])
        if r["raw_return_pct"] is not None:
            b["returns"].append(r["raw_return_pct"])
        if r["excess_return_pct"] is not None:
            b["excess"].append(r["excess_return_pct"])
        if r["mae_pct"] is not None:
            b["mae"].append(r["mae_pct"])
        if r["mfe_pct"] is not None:
            b["mfe"].append(r["mfe_pct"])

    def _avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {}
    for s, b in sorted(by_state.items()):
        summary[s] = {
            "n": b["n"],
            "n_graded": b["graded"],
            "correct_rate": round(b["correct"] / b["graded"], 3) if b["graded"] else None,
            "avg_return_pct": _avg(b["returns"]),
            "avg_excess_return_pct": _avg(b["excess"]),
            "avg_mae_pct": _avg(b["mae"]),
            "avg_mfe_pct": _avg(b["mfe"]),
            "grading_rule": STATE_RULES.get(s),
        }

    sufficient = independent >= MIN_SAMPLE_FOR_INFERENCE
    return {
        "horizon_days": horizon,
        "n_outcomes": n,
        "n_independent_decisions": independent,
        "by_state": summary,
        "verdict": "SUFFICIENT_SAMPLE" if sufficient else "NO_DEMONSTRATED_EDGE",
        "statement": (
            f"{independent} independent matured decisions at the {horizon}-day horizon. "
            + (f"Above the {MIN_SAMPLE_FOR_INFERENCE} minimum, so these rates are "
               f"descriptive of a real sample — they are still NOT a demonstrated edge "
               f"until they survive the same walk-forward and multiple-testing correction "
               f"the strategy gate applies."
               if sufficient else
               f"Below the {MIN_SAMPLE_FOR_INFERENCE} minimum. NO_DEMONSTRATED_EDGE: these "
               f"numbers describe too few decisions to support any inference.")),
        "safeguard": ("The existing research safeguards remain authoritative. Nothing here "
                      "is called alpha, an edge, or a validated rule."),
    }
