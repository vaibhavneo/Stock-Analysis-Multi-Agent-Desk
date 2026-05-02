from __future__ import annotations

import argparse
import json
from typing import Dict, List

from .analysis import analyze_symbol
from .broker import PaperBroker
from .data import (
    fill_missing_price_bars,
    load_fundamentals,
    load_price_bars,
    load_symbol_profiles,
)
from .multiagent import MultiAgentOrchestrator
from .models import AnalysisReport, OrderRequest
from .symbols import parse_symbol_filter


def _select_symbols(
    data: Dict[str, list], symbols_filter: str | None
) -> Dict[str, list]:
    if not symbols_filter:
        return data
    requested = parse_symbol_filter(symbols_filter)
    return {symbol: bars for symbol, bars in data.items() if symbol in requested}


def _scan_reports(symbol_data: Dict[str, list]) -> List[AnalysisReport]:
    reports: List[AnalysisReport] = []
    for symbol, bars in symbol_data.items():
        if len(bars) < 5:
            continue
        reports.append(analyze_symbol(bars))
    reports.sort(key=lambda r: r.score, reverse=True)
    return reports


def _position_size(capital: float, confidence: float, price: float) -> int:
    allocation = capital * min(0.2, max(0.03, confidence * 0.15))
    return max(0, int(allocation // price))


def cmd_scan(args: argparse.Namespace) -> int:
    data = load_price_bars(args.csv_path)
    data = _select_symbols(data, args.symbols)
    reports = _scan_reports(data)
    payload = [
        {
            "symbol": r.symbol,
            "signal": r.signal,
            "score": r.score,
            "confidence": r.confidence,
            "last_price": r.last_price,
            "indicators": {
                "sma_short": r.indicators.sma_short,
                "sma_long": r.indicators.sma_long,
                "momentum": r.indicators.momentum,
                "rsi": r.indicators.rsi,
                "volatility": r.indicators.volatility,
            },
            "rationale": r.rationale,
        }
        for r in reports
    ]
    print(json.dumps(payload, indent=2))
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    data = load_price_bars(args.csv_path)
    data = _select_symbols(data, args.symbols)
    reports = _scan_reports(data)
    broker = PaperBroker(starting_cash=args.capital)

    for report in reports:
        price = report.last_price
        if report.signal == "BUY":
            qty = _position_size(broker.cash, report.confidence, price)
            if qty > 0:
                broker.execute(
                    OrderRequest(
                        symbol=report.symbol,
                        side="BUY",
                        quantity=qty,
                        price=price,
                    )
                )
        elif report.signal == "SELL":
            position = broker.positions.get(report.symbol)
            if position and position.quantity > 0:
                qty = max(1, int(position.quantity * min(1.0, report.confidence)))
                broker.execute(
                    OrderRequest(
                        symbol=report.symbol,
                        side="SELL",
                        quantity=qty,
                        price=price,
                    )
                )

    summary = {
        "starting_cash": args.capital,
        "final_state": broker.snapshot(),
        "reports": [
            {
                "symbol": r.symbol,
                "signal": r.signal,
                "confidence": r.confidence,
                "score": r.score,
                "price": r.last_price,
            }
            for r in reports
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_multiagent(args: argparse.Namespace) -> int:
    bars = load_price_bars(args.prices_csv)
    profiles = load_symbol_profiles(args.profiles_csv)
    fundamentals = load_fundamentals(args.fundamentals_csv)
    bars = fill_missing_price_bars(bars, profiles, fundamentals)

    if args.symbols:
        allowed = parse_symbol_filter(args.symbols)
        bars = {s: v for s, v in bars.items() if s in allowed}
        profiles = {s: v for s, v in profiles.items() if s in allowed}
        fundamentals = {s: v for s, v in fundamentals.items() if s in allowed}

    orchestrator = MultiAgentOrchestrator()
    result = orchestrator.run(
        bars_by_symbol=bars,
        profiles=profiles,
        fundamentals=fundamentals,
        top_n=args.top,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Stock Analysis and Trading Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Analyze symbols and output ranked signals")
    scan.add_argument("csv_path", help="Path to OHLCV CSV")
    scan.add_argument("--symbols", help="Comma-separated symbol filter")
    scan.set_defaults(func=cmd_scan)

    trade = subparsers.add_parser(
        "trade", help="Run one paper-trading cycle using generated signals"
    )
    trade.add_argument("csv_path", help="Path to OHLCV CSV")
    trade.add_argument("--symbols", help="Comma-separated symbol filter")
    trade.add_argument("--capital", type=float, default=10000.0, help="Starting cash")
    trade.set_defaults(func=cmd_trade)

    multi = subparsers.add_parser(
        "multiagent",
        help="Run 5-agent pipeline for AI and quantum stocks",
    )
    multi.add_argument("prices_csv", help="Path to OHLCV CSV")
    multi.add_argument("profiles_csv", help="Path to symbol profile CSV")
    multi.add_argument("fundamentals_csv", help="Path to fundamentals CSV")
    multi.add_argument("--symbols", help="Comma-separated symbol filter")
    multi.add_argument("--top", type=int, default=40, help="Number of top picks")
    multi.set_defaults(func=cmd_multiagent)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
