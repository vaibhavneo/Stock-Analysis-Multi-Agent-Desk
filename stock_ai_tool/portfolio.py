from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .models import FundamentalSnapshot, PriceBar, SymbolProfile


@dataclass
class PortfolioPosition:
    symbol: str
    shares: float
    avg_cost: float
    date_added: str


@dataclass
class Portfolio:
    positions: List[PortfolioPosition] = field(default_factory=list)
    cash: float = 0.0


def load_portfolio(path: str = "portfolio.json") -> Portfolio:
    p = Path(path)
    if not p.exists():
        return Portfolio()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        positions = [PortfolioPosition(**pos) for pos in data.get("positions", [])]
        return Portfolio(positions=positions, cash=float(data.get("cash", 0.0)))
    except Exception:
        return Portfolio()


def save_portfolio(portfolio: Portfolio, path: str = "portfolio.json") -> None:
    data = {
        "positions": [asdict(p) for p in portfolio.positions],
        "cash": portfolio.cash,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_position(portfolio: Portfolio, symbol: str, shares: float, cost_per_share: float) -> Portfolio:
    symbol = symbol.upper()
    for pos in portfolio.positions:
        if pos.symbol == symbol:
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + cost_per_share * shares) / total_shares
            pos.shares = total_shares
            return portfolio
    from datetime import date
    portfolio.positions.append(
        PortfolioPosition(
            symbol=symbol,
            shares=shares,
            avg_cost=cost_per_share,
            date_added=date.today().isoformat(),
        )
    )
    return portfolio


def remove_position(portfolio: Portfolio, symbol: str, shares: float) -> Portfolio:
    symbol = symbol.upper()
    for pos in portfolio.positions:
        if pos.symbol == symbol:
            pos.shares -= shares
            if pos.shares <= 0:
                portfolio.positions = [p for p in portfolio.positions if p.symbol != symbol]
            return portfolio
    return portfolio


def portfolio_report(
    portfolio: Portfolio,
    bars_by_symbol: Dict[str, List[PriceBar]],
    profiles: Dict[str, SymbolProfile],
    fundamentals: Dict[str, FundamentalSnapshot],
    orchestrator_result: Optional[dict] = None,
) -> dict:
    action_map: Dict[str, str] = {}
    if orchestrator_result:
        for theme_recs in orchestrator_result.get("top_by_theme", {}).values():
            for rec in theme_recs:
                sym = rec.get("symbol", "")
                if sym and sym not in action_map:
                    action_map[sym] = rec.get("action", "HOLD")

    positions_out = []
    total_cost = 0.0
    total_value = 0.0

    for pos in portfolio.positions:
        bars = bars_by_symbol.get(pos.symbol, [])
        current_price = bars[-1].close if bars else pos.avg_cost
        market_value = pos.shares * current_price
        cost_basis = pos.shares * pos.avg_cost
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0

        positions_out.append({
            "symbol": pos.symbol,
            "shares": pos.shares,
            "avg_cost": round(pos.avg_cost, 2),
            "current_price": round(current_price, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "action": action_map.get(pos.symbol, "HOLD"),
        })
        total_cost += cost_basis
        total_value += market_value

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

    # --- sector allocation ---
    sector_value: Dict[str, float] = {}
    theme_value: Dict[str, float] = {}
    for pos_out in positions_out:
        sym = pos_out["symbol"]
        val = pos_out["market_value"]
        profile = profiles.get(sym)
        if profile:
            sector_value[profile.sector] = sector_value.get(profile.sector, 0.0) + val
            theme_value[profile.theme] = theme_value.get(profile.theme, 0.0) + val

    def _pct(v: float) -> float:
        return round(v / total_value * 100, 1) if total_value else 0.0

    sector_allocation = {s: _pct(v) for s, v in sector_value.items()}
    theme_allocation = {t: _pct(v) for t, v in theme_value.items()}

    # --- top / bottom performers ---
    sorted_by_pnl = sorted(positions_out, key=lambda x: x["pnl_pct"])
    top_performer = sorted_by_pnl[-1]["symbol"] if sorted_by_pnl else None
    bottom_performer = sorted_by_pnl[0]["symbol"] if sorted_by_pnl else None

    # --- risk flags ---
    risk_flags: List[str] = []
    for pos_out in positions_out:
        pct = _pct(pos_out["market_value"])
        if pct > 25:
            risk_flags.append(f"Concentration: {pos_out['symbol']} is {pct}% of portfolio")
        if pos_out["action"] == "REDUCE":
            risk_flags.append(f"Holding REDUCE signal: {pos_out['symbol']}")
        bars = bars_by_symbol.get(pos_out["symbol"], [])
        if len(bars) >= 20:
            from .analysis import _volatility
            closes = [b.close for b in bars]
            vol = _volatility(closes, 20)
            if vol > 0.04:
                risk_flags.append(f"High volatility: {pos_out['symbol']} ({vol:.1%} daily σ)")

    # --- rebalancing suggestions ---
    suggestions: List[str] = []
    for sector, pct in sector_allocation.items():
        if pct > 40:
            suggestions.append(f"Consider trimming {sector} sector (currently {pct}% of portfolio)")
    for pos_out in positions_out:
        if pos_out["action"] == "REDUCE" and pos_out["pnl_pct"] > 15:
            suggestions.append(
                f"Take some profit in {pos_out['symbol']} (+{pos_out['pnl_pct']}%, REDUCE signal)"
            )
    for pos_out in positions_out:
        if pos_out["action"] in ("BUY", "STRONG_BUY") and _pct(pos_out["market_value"]) < 3:
            suggestions.append(
                f"Consider adding to {pos_out['symbol']} (BUY signal, underweight at {_pct(pos_out['market_value'])}%)"
            )

    return {
        "positions": positions_out,
        "cash": round(portfolio.cash, 2),
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "sector_allocation": sector_allocation,
        "theme_allocation": theme_allocation,
        "top_performer": top_performer,
        "bottom_performer": bottom_performer,
        "risk_flags": risk_flags,
        "rebalancing_suggestions": suggestions,
    }
