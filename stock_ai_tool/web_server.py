from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .deep_analysis import build_deep_analysis
from .data import load_fundamentals, load_symbol_profiles
from .live_data import fetch_live_quote, load_market_inputs as load_provider_inputs
from .multiagent import MultiAgentOrchestrator
from .prediction import price_target as calc_price_target, prediction_for_all
from .portfolio import load_portfolio, save_portfolio, add_position, remove_position, portfolio_report
from .report_scheduler import generate_report
from .symbols import parse_symbol_filter


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
PRICES_CSV = ROOT / "sample_data" / "thematic_prices.csv"
PROFILES_CSV = ROOT / "sample_data" / "thematic_profiles.csv"
FUNDAMENTALS_CSV = ROOT / "sample_data" / "thematic_fundamentals.csv"
PORTFOLIO_JSON = ROOT / "portfolio.json"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def run_multiagent(symbols: str | None, top: int, mode: str = "sample") -> dict:
    allowed = parse_symbol_filter(symbols) if symbols else None
    bars, profiles, fundamentals, source_meta = load_provider_inputs(
        str(PRICES_CSV),
        str(PROFILES_CSV),
        str(FUNDAMENTALS_CSV),
        mode=mode,
        symbols=allowed,
    )

    result = MultiAgentOrchestrator().run(
        bars_by_symbol=bars,
        profiles=profiles,
        fundamentals=fundamentals,
        top_n=top,
    )
    result["meta"] = {
        "symbols": sorted(set(profiles.keys()) & set(bars.keys()) & set(fundamentals.keys())),
        "top": top,
        "data_mode": source_meta["mode"],
        "provider": source_meta["provider"],
        "sources": source_meta["sources"],
        "warnings": source_meta["warnings"],
    }
    return result


def load_market_inputs(mode: str = "sample", symbols: set[str] | None = None) -> tuple[dict, dict, dict, dict]:
    return load_provider_inputs(
        str(PRICES_CSV),
        str(PROFILES_CSV),
        str(FUNDAMENTALS_CSV),
        mode=mode,
        symbols=symbols,
    )


def run_deep_analysis(symbol: str, mode: str = "sample") -> dict:
    resolved = next(iter(parse_symbol_filter(symbol)), symbol.strip().upper())
    bars, profiles, fundamentals, source_meta = load_market_inputs(mode=mode)
    payload = build_deep_analysis(
        symbol=resolved,
        bars_by_symbol=bars,
        profiles=profiles,
        fundamentals=fundamentals,
    )
    payload["data_source"] = {
        "mode": source_meta["mode"],
        "provider": source_meta["provider"],
        "symbol_source": source_meta["sources"].get(resolved, "sample"),
        "warnings": source_meta["warnings"],
    }
    return payload


def load_benchmarks() -> dict:
    fundamentals = load_fundamentals(str(FUNDAMENTALS_CSV))
    profiles = load_symbol_profiles(str(PROFILES_CSV))
    rows = []
    for symbol, fundamental in sorted(fundamentals.items()):
        profile = profiles.get(symbol)
        if not profile:
            continue
        rows.append(
            {
                "symbol": symbol,
                "sector": profile.sector,
                "theme": profile.theme,
                "pe_ratio": fundamental.pe_ratio,
                "industry_pe": fundamental.industry_pe,
                "sales_millions": fundamental.sales_millions,
                "revenue_growth": fundamental.revenue_growth,
                "industry_revenue_growth": fundamental.industry_revenue_growth,
                "fcf_margin": fundamental.fcf_margin,
                "industry_fcf_margin": fundamental.industry_fcf_margin,
            }
        )
    return {"benchmarks": rows}


def load_universe() -> dict:
    profiles = load_symbol_profiles(str(PROFILES_CSV))
    return {
        "symbols": [
            {
                "symbol": profile.symbol,
                "company": profile.company,
                "sector": profile.sector,
                "theme": profile.theme,
            }
            for profile in sorted(profiles.values(), key=lambda p: p.symbol)
        ]
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "StockMultiAgentUI/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            self._handle_analyze(parsed.query)
            return
        if parsed.path == "/api/deep-analysis":
            self._handle_deep_analysis(parsed.query)
            return
        if parsed.path == "/api/live-quote":
            self._handle_live_quote(parsed.query)
            return
        if parsed.path == "/api/benchmarks":
            self._send_json(load_benchmarks())
            return
        if parsed.path == "/api/universe":
            self._send_json(load_universe())
            return
        if parsed.path == "/api/predict":
            self._handle_predict(parsed.query)
            return
        if parsed.path == "/api/portfolio":
            self._handle_portfolio_get(parsed.query)
            return
        if parsed.path == "/api/report":
            self._handle_report(parsed.query)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/portfolio":
            self._handle_portfolio_post()
            return
        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_analyze(self, query: str) -> None:
        params = parse_qs(query)
        symbols = params.get("symbols", [""])[0].strip() or None
        top_raw = params.get("top", ["5"])[0]
        mode = params.get("mode", ["sample"])[0].strip().lower()
        try:
            top = max(1, min(100, int(top_raw)))
            payload = run_multiagent(symbols=symbols, top=top, mode=mode)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(payload)

    def _handle_deep_analysis(self, query: str) -> None:
        params = parse_qs(query)
        symbol = params.get("symbol", [""])[0].strip()
        mode = params.get("mode", ["sample"])[0].strip().lower()
        if not symbol:
            self._send_json({"error": "symbol is required"}, status=400)
            return
        try:
            payload = run_deep_analysis(symbol, mode=mode)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(payload)

    def _handle_live_quote(self, query: str) -> None:
        params = parse_qs(query)
        symbol = params.get("symbol", [""])[0].strip().upper()
        if not symbol:
            self._send_json({"error": "symbol is required"}, status=400)
            return
        try:
            resolved = next(iter(parse_symbol_filter(symbol)), symbol)
            payload = fetch_live_quote(resolved)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(payload)

    def _handle_predict(self, query: str) -> None:
        params = parse_qs(query)
        symbol = params.get("symbol", [""])[0].strip().upper()
        mode = params.get("mode", ["sample"])[0].strip().lower()
        if not symbol:
            self._send_json({"error": "symbol is required"}, status=400)
            return
        try:
            resolved = next(iter(parse_symbol_filter(symbol)), symbol)
            bars, _, fundamentals, _ = load_market_inputs(mode=mode, symbols={resolved})
            sym_bars = bars.get(resolved)
            if not sym_bars:
                self._send_json({"error": f"No data for {resolved}"}, status=404)
                return
            from .analysis import analyze_symbol
            report = analyze_symbol(sym_bars)
            result = calc_price_target(resolved, sym_bars, fundamentals.get(resolved), report)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def _handle_portfolio_get(self, query: str) -> None:
        params = parse_qs(query)
        mode = params.get("mode", ["sample"])[0].strip().lower()
        try:
            bars, profiles, fundamentals, _ = load_market_inputs(mode=mode)
            portfolio = load_portfolio(str(PORTFOLIO_JSON))
            orch_result = MultiAgentOrchestrator().run(
                bars_by_symbol=bars,
                profiles=profiles,
                fundamentals=fundamentals,
                top_n=10,
            )
            result = portfolio_report(portfolio, bars, profiles, fundamentals, orch_result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def _handle_portfolio_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
            return
        action = body.get("action", "").lower()
        symbol = body.get("symbol", "").upper()
        shares = float(body.get("shares", 0))
        cost = float(body.get("cost", 0))
        if not symbol or shares <= 0:
            self._send_json({"error": "symbol and shares are required"}, status=400)
            return
        try:
            portfolio = load_portfolio(str(PORTFOLIO_JSON))
            if action == "add":
                portfolio = add_position(portfolio, symbol, shares, cost)
            elif action == "remove":
                portfolio = remove_position(portfolio, symbol, shares)
            else:
                self._send_json({"error": "action must be add or remove"}, status=400)
                return
            save_portfolio(portfolio, str(PORTFOLIO_JSON))
            self._send_json({"status": "ok", "positions": len(portfolio.positions)})
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _handle_report(self, query: str) -> None:
        params = parse_qs(query)
        period = params.get("period", ["daily"])[0].strip().lower()
        mode = params.get("mode", ["sample"])[0].strip().lower()
        if period not in ("daily", "weekly", "monthly"):
            self._send_json({"error": "period must be daily, weekly, or monthly"}, status=400)
            return
        try:
            result = generate_report(period=period, portfolio_path=str(PORTFOLIO_JSON), mode=mode)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(result)

    def _serve_static(self, path: str) -> None:
        if path == "/":
            target = WEB_ROOT / "index.html"
        else:
            target = (WEB_ROOT / path.lstrip("/")).resolve()

        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists():
            self.send_error(404)
            return

        body = target.read_bytes()
        content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the multi-agent stock UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving multi-agent stock UI at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
