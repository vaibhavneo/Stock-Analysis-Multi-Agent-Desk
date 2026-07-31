from __future__ import annotations

import statistics
from typing import List

from .models import PriceBar
from .prediction import next_day_estimate

MIN_INDICATOR_BARS = 15
MIN_TRIALS_FOR_SCORE = 10


def backtest_next_day(symbol: str, bars: List[PriceBar], lookback_days: int = 63) -> dict:
    """Walk-forward backtest of the next-day price model.

    For each historical trading day t, predicts day t+1's close using ONLY
    bars[:t+1] (no lookahead), then compares against the actual close. This
    scores the same `next_day_estimate` formula used by the live predict
    endpoint, so the score reflects real predictive skill, not a different
    model dressed up as a "backtest".
    """
    bars = sorted(bars, key=lambda b: b.day)
    n = len(bars)
    usable_days = n - MIN_INDICATOR_BARS - 1
    window_days = min(lookback_days, usable_days)

    if window_days < MIN_TRIALS_FOR_SCORE:
        return {
            "symbol": symbol,
            "error": (
                f"Not enough price history for a reliable backtest — only {n} bars available, "
                f"need at least {MIN_INDICATOR_BARS + MIN_TRIALS_FOR_SCORE + 1}. "
                "Switch the data feed to Live to pull real daily history."
            ),
            "bars_available": n,
        }

    start_idx = n - window_days - 1
    trials = []
    for t in range(start_idx, n - 1):
        window = bars[: t + 1]
        if len(window) < MIN_INDICATOR_BARS:
            continue
        actual_bar = bars[t + 1]
        prior_close = bars[t].close
        pred = next_day_estimate(window)

        actual_close = actual_bar.close
        if actual_close > prior_close:
            actual_direction = "UP"
        elif actual_close < prior_close:
            actual_direction = "DOWN"
        else:
            actual_direction = "FLAT"

        pct_error = (
            abs(pred["predicted_close"] - actual_close) / actual_close * 100
            if actual_close
            else 0.0
        )

        trials.append(
            {
                "date": actual_bar.day.isoformat(),
                "predicted_close": pred["predicted_close"],
                "actual_close": round(actual_close, 4),
                "pct_error": round(pct_error, 3),
                "predicted_direction": pred["predicted_direction"],
                "actual_direction": actual_direction,
                "direction_hit": pred["predicted_direction"] == actual_direction,
            }
        )

    if len(trials) < MIN_TRIALS_FOR_SCORE:
        return {
            "symbol": symbol,
            "error": "Not enough sequential trading days to score a backtest.",
            "bars_available": n,
        }

    errors = [t["pct_error"] for t in trials]
    hits = [t for t in trials if t["direction_hit"]]
    mape = statistics.fmean(errors)
    median_error = statistics.median(errors)
    direction_accuracy = len(hits) / len(trials) * 100

    # Composite 0-100 score: direction-calling skill (60%) + price precision (40%).
    # ~12.5% average error zeroes out the precision component.
    price_accuracy_score = max(0.0, 100 - mape * 8)
    accuracy_score = round(0.6 * direction_accuracy + 0.4 * price_accuracy_score, 1)

    if accuracy_score >= 75:
        grade = "STRONG"
    elif accuracy_score >= 60:
        grade = "MODERATE"
    elif accuracy_score >= 45:
        grade = "WEAK"
    else:
        grade = "UNRELIABLE"

    worst = max(trials, key=lambda t: t["pct_error"])
    best = min(trials, key=lambda t: t["pct_error"])

    return {
        "symbol": symbol,
        "trials": len(trials),
        "requested_days": lookback_days,
        "days_covered": f"{trials[0]['date']} to {trials[-1]['date']}",
        "mape_pct": round(mape, 3),
        "median_error_pct": round(median_error, 3),
        "direction_accuracy_pct": round(direction_accuracy, 1),
        "accuracy_score": accuracy_score,
        "grade": grade,
        "best_call": best,
        "worst_call": worst,
        "history": trials,
    }
