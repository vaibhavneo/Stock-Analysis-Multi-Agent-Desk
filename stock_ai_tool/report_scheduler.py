from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, List

from .live_data import load_market_inputs
from .multiagent import MultiAgentOrchestrator
from .prediction import prediction_for_all
from .portfolio import load_portfolio, portfolio_report


PRICES_CSV = str(Path(__file__).resolve().parents[1] / "sample_data" / "thematic_prices.csv")
PROFILES_CSV = str(Path(__file__).resolve().parents[1] / "sample_data" / "thematic_profiles.csv")
FUNDAMENTALS_CSV = str(Path(__file__).resolve().parents[1] / "sample_data" / "thematic_fundamentals.csv")


def _pct_change(bars, lookback: int) -> float:
    closes = [b.close for b in bars]
    if len(closes) < lookback + 1:
        return 0.0
    base = closes[-(lookback + 1)]
    current = closes[-1]
    return (current - base) / base * 100 if base else 0.0


def generate_report(
    period: str = "daily",
    bars_by_symbol: dict | None = None,
    profiles: dict | None = None,
    fundamentals: dict | None = None,
    portfolio_path: str = "portfolio.json",
    mode: str = "sample",
) -> dict:
    if period not in ("daily", "weekly", "monthly"):
        raise ValueError(f"period must be daily/weekly/monthly, got {period!r}")

    if bars_by_symbol is None or profiles is None or fundamentals is None:
        bars_by_symbol, profiles, fundamentals, _ = load_market_inputs(
            PRICES_CSV, PROFILES_CSV, FUNDAMENTALS_CSV, mode=mode
        )

    orch_result = MultiAgentOrchestrator().run(
        bars_by_symbol=bars_by_symbol,
        profiles=profiles,
        fundamentals=fundamentals,
        top_n=10,
    )

    predictions = prediction_for_all(bars_by_symbol, fundamentals)

    # flatten top_by_theme into a single list of recommendations
    all_recs = []
    for theme_recs in orch_result.get("top_by_theme", {}).values():
        all_recs.extend(theme_recs)
    # deduplicate by symbol, keep highest composite_score
    seen: dict = {}
    for r in all_recs:
        sym = r.get("symbol", "")
        if sym not in seen or r.get("composite_score", 0) > seen[sym].get("composite_score", 0):
            seen[sym] = r
    all_recs = list(seen.values())

    # signals sorted
    buy_signals = sorted(
        [p for p in predictions.values() if p.get("signal") == "BUY_WINDOW"],
        key=lambda x: -x.get("composite_score", 0),
    )[:5]
    take_profit = sorted(
        [p for p in predictions.values() if p.get("signal") == "TAKE_PROFIT_ZONE"],
        key=lambda x: -x.get("composite_score", 0),
    )[:5]

    reduce_signals = [r for r in all_recs if r.get("action") == "REDUCE"]

    # portfolio section
    portfolio_section = None
    portfolio = load_portfolio(portfolio_path)
    if portfolio.positions:
        portfolio_section = portfolio_report(
            portfolio,
            bars_by_symbol,
            profiles,
            fundamentals,
            orch_result,
        )

    report = {
        "period": period,
        "generated": date.today().isoformat(),
        "buy_windows": [
            {"symbol": p["symbol"], "last_price": p["last_price"],
             "signal_note": p["signal_note"], "rsi": p["indicators"]["rsi"]}
            for p in buy_signals
        ],
        "take_profit_zones": [
            {"symbol": p["symbol"], "last_price": p["last_price"],
             "signal_note": p["signal_note"], "rsi": p["indicators"]["rsi"]}
            for p in take_profit
        ],
        "top_buys": sorted(
            [
                {"symbol": r["symbol"], "action": r["action"], "composite_score": r["composite_score"]}
                for r in all_recs
                if r.get("action") in ("STRONG_BUY", "BUY")
            ],
            key=lambda x: -x["composite_score"],
        )[:5],
        "reduce_signals": [
            {"symbol": r["symbol"], "composite_score": r["composite_score"]}
            for r in reduce_signals
        ][:5],
        "portfolio": portfolio_section,
    }

    if period in ("weekly", "monthly"):
        # sector rotation
        sector_perf: Dict[str, List[float]] = {}
        for sym, bars in bars_by_symbol.items():
            profile = profiles.get(sym)
            if not profile:
                continue
            chg = _pct_change(bars, 5)
            sector_perf.setdefault(profile.sector, []).append(chg)

        from statistics import fmean
        sector_table = sorted(
            [
                {"sector": s, "week_pct": round(fmean(v), 2)}
                for s, v in sector_perf.items()
            ],
            key=lambda x: -x["week_pct"],
        )

        # best/worst this week
        weekly_changes = {
            sym: _pct_change(bars, 5)
            for sym, bars in bars_by_symbol.items()
        }
        sorted_changes = sorted(weekly_changes.items(), key=lambda x: -x[1])
        best_5 = [{"symbol": s, "week_pct": round(c, 2)} for s, c in sorted_changes[:5]]
        worst_5 = [{"symbol": s, "week_pct": round(c, 2)} for s, c in sorted_changes[-5:]]

        # option bias
        option_recs = orch_result.get("actions", {}).get("option_recommendations", [])
        call_count = sum(1 for r in option_recs if r.get("action") == "CALL")
        put_count = sum(1 for r in option_recs if r.get("action") == "PUT")

        report["sector_rotation"] = sector_table
        report["best_performers_week"] = best_5
        report["worst_performers_week"] = worst_5
        report["option_bias"] = {"call": call_count, "put": put_count}

    if period == "monthly":
        # fundamental quality ranking
        quality_rows = []
        for sym, fund in sorted(fundamentals.items()):
            quality_score = (
                max(0.0, fund.revenue_growth) * 0.3
                + max(0.0, fund.eps_growth) * 0.2
                + max(0.0, fund.fcf_margin) * 0.3
                + max(0.0, fund.gross_margin) * 0.2
            )
            quality_rows.append({"symbol": sym, "quality_score": round(quality_score, 3)})
        quality_rows.sort(key=lambda x: -x["quality_score"])
        report["fundamental_quality_ranking"] = quality_rows

        # suggested allocation (top 10 by composite score)
        top10 = sorted(all_recs, key=lambda x: -x.get("composite_score", 0))[:10]
        report["suggested_allocation"] = [
            {
                "symbol": r["symbol"],
                "action": r["action"],
                "composite_score": r["composite_score"],
                "target_weight_pct": round(10.0, 1),  # equal-weight suggestion
            }
            for r in top10
        ]

        # 3-month price target progress
        target_progress = []
        for sym, pred in predictions.items():
            if "targets" in pred:
                target_progress.append({
                    "symbol": sym,
                    "last_price": pred["last_price"],
                    "3_month_target": pred["targets"]["3_month"],
                    "upside_pct": round(
                        (pred["targets"]["3_month"] - pred["last_price"]) / pred["last_price"] * 100
                        if pred["last_price"] else 0, 1
                    ),
                })
        target_progress.sort(key=lambda x: -x["upside_pct"])
        report["price_target_progress"] = target_progress[:15]

    return report


def save_report(report: dict, output_path: str) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
