"""
What this engine has actually done — the evidence behind the "fine print".

The panel this replaces listed eight chips and then restated four of them as
five warning lines. Everything in it was true and none of it helped, because
the facts that would have helped were computed and discarded:

  - The **strategy library race** (`web/app.py::_build_backtest_all`) tests
    eight strategies against buy-and-hold on the same price history. It was
    gated behind `deep=true` alongside a 70-second cross-sectional ranking; it
    costs 0.58s. It answers the one question the panel should have led with:
    does the strategy this engine runs beat simply holding the stock? On INTC,
    seven_pillar_core returns a Sharpe of 0.018 against buy-and-hold's 0.59,
    and two of the eight tested strategies beat holding. That is the most
    decision-relevant fact available and it was never displayed.

  - The **per-horizon live record** (`calibration_report_all_horizons`) holds
    1,416 matured 1-day calls, 298 at 20 days, 256 at a year. It was reachable
    only from `/api/intelligence`.

  - **Trade count is the real sample.** The min-sample gate checks BARS —
    1,254 against a 504 minimum — and reports a comfortable pass. But a
    strategy that held six times over five years has a sample of six, and the
    panel was reassuring on the wrong number.

One thing this module refuses to do: imply that any of it forecasts the next
session. It is a record of what happened, and `what_this_cannot_tell_you`
says so in the panel rather than leaving a reader to discover it by being
disappointed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Round trips, not bars. Stated priors, documented here rather than tuned:
# below 30 the trade sample cannot support an inference about the average
# trade; 100 is where a win rate starts to mean something.
TRADES_TOO_FEW = 30
TRADES_THIN = 100

# A measured win rate needs a real sample before it is worth reading.
LIVE_SAMPLE_READABLE = 100

CORE_STRATEGY = "seven_pillar_core"


def _verdict_for_trades(n: Optional[int]) -> Dict[str, Any]:
    if n is None:
        return {"level": "UNKNOWN",
                "statement": "The number of completed trades is not available."}
    if n < TRADES_TOO_FEW:
        return {"level": "TOO_FEW",
                "statement": (f"The strategy completed {n} trades in the whole test. That is the "
                              f"real sample size — not the bar count — and {n} trades cannot "
                              f"support a conclusion about how it performs on average.")}
    if n < TRADES_THIN:
        return {"level": "THIN",
                "statement": (f"{n} completed trades. Enough to describe what happened, not "
                              f"enough to be confident it repeats.")}
    return {"level": "ADEQUATE",
            "statement": (f"{n} completed trades — a large enough sample that the average "
                          f"result is worth reading.")}


def build_strategy_comparison(rec: Dict[str, Any],
                              backtest_all: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Does the strategy this engine runs beat simply holding the stock?"""
    bt = rec.get("backtest") or {}
    core_sharpe = bt.get("sharpe")

    if not backtest_all or not backtest_all.get("rows"):
        return {"status": "UNAVAILABLE",
                "statement": ("The strategy library was not run for this request, so there is "
                              "nothing to compare this strategy against."),
                "rows": []}

    rows = sorted(backtest_all["rows"], key=lambda r: r.get("sharpe", 0), reverse=True)
    hold = backtest_all.get("buy_hold") or {}
    hold_sharpe = hold.get("sharpe")
    hold_annual = hold.get("annualized_return_pct")
    hold_max_dd = hold.get("max_dd_pct")

    core = next((r for r in rows if r.get("strategy") == CORE_STRATEGY), None)
    rank = next((i + 1 for i, r in enumerate(rows) if r.get("strategy") == CORE_STRATEGY), None)
    n_beating_hold = sum(1 for r in rows if r.get("beats_hold"))
    core_beats_hold = bool(core and core.get("beats_hold"))

    if core is None or hold_sharpe is None:
        statement = "The comparison against buy-and-hold could not be computed."
    elif core_beats_hold:
        statement = (f"On this stock's history, the strategy this engine runs did better than "
                     f"simply holding: {core['annualized_return_pct']:+.1f}% a year against "
                     f"{hold_annual:+.1f}% for holding"
                     if hold_annual is not None else
                     f"On this stock's history, the strategy this engine runs beat simply "
                     f"holding it.")
        statement += (f". It ranked {rank} of {len(rows)} strategies tested, and "
                      f"{n_beating_hold} of {len(rows)} beat holding.")
    else:
        lead = (f"On this stock's history, the strategy this engine runs did WORSE than simply "
                f"holding the stock")
        if core.get("annualized_return_pct") is not None and hold_annual is not None:
            lead += (f": {core['annualized_return_pct']:+.1f}% a year against "
                     f"{hold_annual:+.1f}% for holding")
        statement = (f"{lead}. It ranked {rank} of {len(rows)} strategies tested; "
                     f"{n_beating_hold} of {len(rows)} beat holding. This is the single most "
                     f"important number in this panel.")

    return {
        "status": "OK",
        "core_strategy": CORE_STRATEGY,
        "core_sharpe": core_sharpe,
        "core_annual_return_pct": (core or {}).get("annualized_return_pct"),
        "core_beats_hold": core_beats_hold,
        "core_rank": rank,
        "n_strategies": len(rows),
        "n_beating_hold": n_beating_hold,
        "buy_hold_sharpe": hold_sharpe,
        "buy_hold_annual_return_pct": hold_annual,
        "buy_hold_max_drawdown_pct": hold_max_dd,
        "statement": statement,
        "rows": [{"strategy": r.get("strategy"),
                  "sharpe": r.get("sharpe"),
                  "annual_return_pct": r.get("annualized_return_pct"),
                  "max_drawdown_pct": r.get("max_dd_pct"),
                  "n_trades": r.get("n_trades"),
                  "beats_hold": r.get("beats_hold"),
                  "is_core": r.get("strategy") == CORE_STRATEGY} for r in rows],
        "caveat": ("Every one of these was tested on this one stock's past. A strategy that "
                   "won here won one comparison, not an argument."),
    }


def build_live_record(calibration_by_horizon: Optional[Dict[Any, Any]]) -> Dict[str, Any]:
    """This engine's own measured forward record, per horizon.

    The only evidence here that is about the future rather than the past — and
    even then it measures the ENGINE, not this stock.
    """
    if not calibration_by_horizon:
        return {"status": "UNAVAILABLE",
                "statement": ("No matured predictions have been recorded here, so this engine's "
                              "live accuracy has never been measured. That is a gap in the "
                              "evidence, not a neutral result."),
                "rows": []}

    rows: List[Dict[str, Any]] = []
    for horizon, report in sorted(calibration_by_horizon.items(), key=lambda kv: int(kv[0])):
        overall = (report or {}).get("overall") or report or {}
        n = overall.get("n")
        if not n:
            continue
        rows.append({
            "horizon_days": int(horizon),
            "n": n,
            "win_rate": overall.get("win_rate"),
            "avg_excess_return_pct": overall.get("avg_excess_return_pct"),
            "brier": overall.get("brier"),
            "readable": n >= LIVE_SAMPLE_READABLE,
        })

    if not rows:
        return {"status": "UNAVAILABLE",
                "statement": ("No prediction has matured yet, so this engine's live accuracy "
                              "has never been measured."),
                "rows": []}

    with_excess = [r for r in rows if r["avg_excess_return_pct"] is not None]
    n_negative = sum(1 for r in with_excess if r["avg_excess_return_pct"] < 0)
    best = max(with_excess, key=lambda r: r["avg_excess_return_pct"], default=None)

    if with_excess and n_negative == len(with_excess):
        summary = (f"At every horizon measured, the average call has trailed the benchmark. "
                   f"This engine has a track record and it is not yet a good one.")
    elif with_excess and n_negative:
        summary = (f"The average call has trailed the benchmark at {n_negative} of "
                   f"{len(with_excess)} horizons measured"
                   + (f"; the exception is {best['horizon_days']} days at "
                      f"{best['avg_excess_return_pct']:+.1f}%." if best else "."))
    else:
        summary = "The average call has beaten the benchmark at every horizon measured."

    return {
        "status": "OK",
        "rows": rows,
        "total_matured": sum(r["n"] for r in rows),
        "statement": summary,
        "caveat": ("Win rate counts direction; average excess return counts money. They can "
                   "disagree, and when they do the second one is the one that matters."),
    }


def build_track_record(rec: Dict[str, Any],
                       edge: Dict[str, Any],
                       backtest_all: Optional[Dict[str, Any]] = None,
                       calibration_by_horizon: Optional[Dict[Any, Any]] = None
                       ) -> Dict[str, Any]:
    """The whole panel: comparison, sample honesty, live record, and the limit."""
    bt = rec.get("backtest") or {}
    comparison = build_strategy_comparison(rec, backtest_all)
    live = build_live_record(calibration_by_horizon)
    trades = _verdict_for_trades(bt.get("n_trades"))

    checks = {g["gate"]: g["passed"] for g in (edge.get("gates") or [])}
    n_obs = ((edge.get("gates") or [{}])[0].get("detail") or {}).get("n_obs")

    # The headline: one sentence, chosen by what is actually wrong rather than
    # by listing everything that could be.
    if comparison.get("status") == "OK" and not comparison.get("core_beats_hold"):
        headline = ("This strategy did not beat simply holding the stock over the period "
                    "tested.")
    elif trades["level"] == "TOO_FEW":
        headline = ("There are too few completed trades to judge this strategy, whatever the "
                    "other numbers say.")
    elif not edge.get("demonstrated"):
        headline = ("This strategy has not cleared its statistical tests, so nothing here is "
                    "a proven edge.")
    else:
        headline = "This strategy cleared its statistical tests on the history available."

    return {
        "headline": headline,
        "comparison": comparison,
        "live_record": live,
        "sample": {
            "bars_tested": n_obs,
            "bars_required": ((edge.get("gates") or [{}])[0].get("detail") or {}).get("required"),
            "completed_trades": bt.get("n_trades"),
            "verdict": trades["level"],
            "statement": trades["statement"],
            "why_bars_mislead": ("The gate counts bars of price history, which is why it passes "
                                 "on a long history with very few trades. Trades are the sample "
                                 "that matters."),
        },
        "gates": [{"gate": g["gate"], "passed": g["passed"]} for g in (edge.get("gates") or [])],
        "n_gates_passed": sum(1 for v in checks.values() if v),
        "n_gates": len(checks),
        "what_this_cannot_tell_you": (
            "None of this predicts the next session, the next week, or this stock specifically. "
            "It is a record of how a mechanical strategy behaved on past prices and how this "
            "engine's own past calls resolved. Short-horizon price moves are dominated by news "
            "and flow that nothing here observes."),
        "what_this_is_good_for": (
            "Deciding how much weight to put on everything else in this report. A strategy that "
            "loses to buy-and-hold on its own history is a reason to size smaller, not a reason "
            "to read the rest more carefully."),
    }
