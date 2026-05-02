from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from .analysis import analyze_symbol
from .models import FundamentalSnapshot, PriceBar, SymbolProfile
from .multiagent import MultiAgentOrchestrator


def _fmt_money_m(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}T"
    if value >= 1_000:
        return f"${value / 1_000:.1f}B"
    return f"${value:.0f}M"


def _score_label(score: float) -> str:
    if score >= 0.72:
        return "strong"
    if score >= 0.60:
        return "constructive"
    if score >= 0.45:
        return "mixed"
    return "weak"


def _agent_item(actions: dict, key: str, symbol: str, fallback: str) -> dict:
    return next(
        (item for item in actions.get(key, []) if item["symbol"] == symbol),
        {"symbol": symbol, "score": 0.0, "summary": fallback},
    )


def _project_sales(fundamental: FundamentalSnapshot) -> List[dict]:
    sales = fundamental.sales_millions
    growth = fundamental.revenue_growth
    projections = []
    for year in range(1, 4):
        sales = sales * (1 + growth)
        projections.append(
            {
                "year": year,
                "sales_millions": round(sales, 1),
                "sales_display": _fmt_money_m(sales),
                "growth_assumption": round(growth, 3),
            }
        )
        growth = max(fundamental.industry_revenue_growth, growth * 0.82)
    return projections


def _cash_flow_status(fundamental: FundamentalSnapshot) -> str:
    if fundamental.fcf_margin >= fundamental.industry_fcf_margin:
        return "above sector benchmark"
    if fundamental.fcf_margin >= 0:
        return "positive but below benchmark"
    return "negative cash flow profile"


def _peer_rows(
    symbol: str,
    profile: SymbolProfile,
    fundamentals: Dict[str, FundamentalSnapshot],
    recommendations: List[dict],
) -> List[dict]:
    by_symbol = {item["symbol"]: item for item in recommendations}
    peers = []
    for peer_symbol, peer in fundamentals.items():
        if peer_symbol == symbol:
            continue
        peer_rec = by_symbol.get(peer_symbol)
        peers.append(
            {
                "symbol": peer_symbol,
                "same_sector": peer_symbol in by_symbol,
                "pe_ratio": peer.pe_ratio,
                "sales_display": _fmt_money_m(peer.sales_millions),
                "revenue_growth": peer.revenue_growth,
                "fcf_margin": peer.fcf_margin,
                "composite_score": peer_rec["composite_score"] if peer_rec else 0.0,
                "action": peer_rec["action"] if peer_rec else "N/A",
            }
        )

    ranked = sorted(peers, key=lambda row: row["composite_score"], reverse=True)
    same_sector = [
        row for row in ranked if row["symbol"] != symbol and row["same_sector"]
    ]
    if same_sector:
        return same_sector[:5]
    return ranked[:5]


def build_deep_analysis(
    symbol: str,
    bars_by_symbol: Dict[str, List[PriceBar]],
    profiles: Dict[str, SymbolProfile],
    fundamentals: Dict[str, FundamentalSnapshot],
) -> dict:
    symbol = symbol.upper()
    if symbol not in profiles or symbol not in fundamentals or symbol not in bars_by_symbol:
        raise ValueError(f"No complete deep-analysis data for {symbol}")

    profile = profiles[symbol]
    fundamental = fundamentals[symbol]
    report = analyze_symbol(bars_by_symbol[symbol])
    result = MultiAgentOrchestrator().run(
        bars_by_symbol=bars_by_symbol,
        profiles=profiles,
        fundamentals=fundamentals,
        top_n=max(1, len(profiles)),
    )
    actions = result["actions"]

    stock = next(
        (item for item in actions["stock_recommendation"] if item["symbol"] == symbol),
        None,
    )
    option = next(
        (item for item in actions["option_recommendation"] if item["symbol"] == symbol),
        None,
    )
    sector_agent = _agent_item(
        actions,
        "sector_analysis",
        symbol,
        "No sector agent assessment is available for this symbol.",
    )
    fundamental_agent = _agent_item(
        actions,
        "fundamental_analysis",
        symbol,
        "No fundamental agent assessment is available for this symbol.",
    )
    technical_agent = _agent_item(
        actions,
        "technical_analysis",
        symbol,
        "No technical agent assessment is available for this symbol.",
    )
    benchmark_pe = fundamental.industry_pe if fundamental.industry_pe > 0 else 30.0
    pe_premium = None if fundamental.pe_ratio <= 0 else fundamental.pe_ratio / benchmark_pe - 1.0
    growth_spread = fundamental.revenue_growth - fundamental.industry_revenue_growth
    fcf_spread = fundamental.fcf_margin - fundamental.industry_fcf_margin
    recommendations = actions["stock_recommendation"]
    sector_symbols = {
        peer_symbol
        for peer_symbol, peer_profile in profiles.items()
        if peer_profile.sector == profile.sector
    }
    sector_recommendations = [
        item for item in recommendations if item["symbol"] in sector_symbols
    ]
    sector_leaders = sorted(
        sector_recommendations,
        key=lambda item: item["composite_score"],
        reverse=True,
    )[:5]

    risk_flags = []
    if fundamental.pe_ratio <= 0:
        risk_flags.append("No positive P/E; valuation depends on future profitability.")
    elif pe_premium and pe_premium > 0.5:
        risk_flags.append("P/E trades more than 50% above industry benchmark.")
    if fundamental.fcf_margin < 0:
        risk_flags.append("Negative free-cash-flow margin.")
    if fundamental.debt_to_equity > 0.8:
        risk_flags.append("Elevated leverage versus growth-stock profile.")
    if report.signal == "SELL":
        risk_flags.append("Technical agent is currently bearish.")
    if report.indicators.volatility > 0.04:
        risk_flags.append("High near-term price volatility.")

    valuation_view = "fair"
    if fundamental.pe_ratio <= 0:
        valuation_view = "speculative"
    elif pe_premium is not None and pe_premium > 0.35:
        valuation_view = "expensive"
    elif pe_premium is not None and pe_premium < -0.20:
        valuation_view = "discounted"

    positives = []
    if sector_agent["score"] >= 0.60:
        positives.append(f"{profile.sector} sector score is constructive at {sector_agent['score']:.3f}.")
    if growth_spread > 0:
        positives.append(f"Revenue growth is {growth_spread:.1%} above the industry benchmark.")
    if fcf_spread > 0:
        positives.append(f"Free-cash-flow margin is {fcf_spread:.1%} above the industry benchmark.")
    if valuation_view in {"discounted", "fair"} and fundamental.pe_ratio > 0:
        positives.append(f"Valuation screens {valuation_view} versus sector P/E.")
    if report.signal == "BUY":
        positives.append("Technical agent currently has a BUY signal.")
    if not positives:
        positives.append("No dominant positive factor; this is a watchlist-quality setup until live data improves.")

    next_checks = [
        "Confirm latest quote, volume, and gap risk before trading.",
        "Compare forward estimates and guidance against sector leaders.",
        "Check option-chain bid/ask, open interest, and implied volatility before selecting a contract.",
    ]
    if profile.theme == "QUANTUM":
        next_checks.insert(1, "Validate quantum roadmap milestones, bookings, dilution risk, and cash runway.")

    action = stock["action"] if stock else "HOLD"
    option_action = option["action"] if option else "NO_TRADE"
    confidence = stock["confidence"] if stock else 0.0
    option_summary = (
        f"{option_action} bias at ${option['strike']:.2f} with {option['expiry_days']} days to expiry."
        if option
        else "No option recommendation is available for this symbol."
    )
    estimated_eps = (
        report.last_price / fundamental.pe_ratio
        if fundamental.pe_ratio and fundamental.pe_ratio > 0
        else None
    )
    forward_pe = (
        fundamental.pe_ratio / (1 + fundamental.eps_growth)
        if fundamental.pe_ratio > 0 and fundamental.eps_growth > -0.95
        else None
    )
    free_cash_flow_millions = (
        fundamental.sales_millions * fundamental.fcf_margin
        if fundamental.sales_millions
        else None
    )
    thesis = (
        f"{symbol} is a {profile.theme} stock with a {action.replace('_', ' ')} equity stance; "
        f"the strongest current lens is "
        f"{max([('sector', sector_agent['score']), ('fundamental', fundamental_agent['score']), ('technical', technical_agent['score'])], key=lambda item: item[1])[0]}."
    )
    agent_breakdown = {
        "sector": {
            "label": "Sector Agent",
            "score": sector_agent["score"],
            "verdict": _score_label(sector_agent["score"]),
            "summary": sector_agent["summary"],
            "bullets": [
                f"Theme classification: {profile.theme}; sector: {profile.sector}.",
                f"Sector score is {sector_agent['score']:.3f}, which maps to a {_score_label(sector_agent['score'])} sector read.",
                f"Top same-sector benchmark: {sector_leaders[0]['symbol']} at {sector_leaders[0]['composite_score']:.3f}." if sector_leaders else "No same-sector benchmark leader available.",
            ],
        },
        "fundamental": {
            "label": "Fundamental Agent",
            "score": fundamental_agent["score"],
            "verdict": _score_label(fundamental_agent["score"]),
            "summary": fundamental_agent["summary"],
            "bullets": [
                f"P/E is {fundamental.pe_ratio:.1f}x versus industry {benchmark_pe:.1f}x.",
                f"Revenue growth is {fundamental.revenue_growth:.1%} versus industry {fundamental.industry_revenue_growth:.1%}.",
                f"Cash-flow status is { _cash_flow_status(fundamental)}.",
            ],
        },
        "technical": {
            "label": "Technical Agent",
            "score": technical_agent["score"],
            "verdict": report.signal,
            "summary": technical_agent["summary"],
            "bullets": [
                f"Last price is ${report.last_price:.2f}; model signal is {report.signal}.",
                f"Short SMA ${report.indicators.sma_short:.2f} versus long SMA ${report.indicators.sma_long:.2f}.",
                f"Momentum {report.indicators.momentum:.3f}; volatility {report.indicators.volatility:.4f}; RSI {report.indicators.rsi:.1f}.",
            ],
        },
        "recommendation": {
            "label": "Recommendation Agent",
            "score": stock["composite_score"] if stock else 0.0,
            "verdict": action,
            "summary": stock["rationale"] if stock else "No stock recommendation is available.",
            "bullets": [
                f"Composite score is {stock['composite_score']:.3f} with {confidence:.1%} confidence." if stock else "Composite score is unavailable.",
                "Weights: 20% sector, 35% fundamentals, 45% technicals.",
                f"Equity action: {action.replace('_', ' ')}.",
            ],
        },
        "option": {
            "label": "Options Agent",
            "score": option["confidence"] if option else 0.0,
            "verdict": option_action,
            "summary": option["rationale"] if option else "No option recommendation is available.",
            "bullets": [
                option_summary,
                "Use live bid/ask, open interest, and implied volatility before placing any option order.",
                "Defined-risk spreads can reduce premium risk versus naked long options.",
            ],
        },
    }

    return {
        "symbol": symbol,
        "profile": asdict(profile),
        "decision_summary": {
            "action": action,
            "option_action": option_action,
            "confidence": confidence,
            "thesis": thesis,
            "positives": positives[:4],
            "risks": risk_flags or ["No major sample-data risk flags; still validate live data and liquidity."],
            "next_checks": next_checks[:4],
        },
        "agent_breakdown": agent_breakdown,
        "recommendation": stock,
        "option_recommendation": option,
        "valuation": {
            "pe_ratio": fundamental.pe_ratio,
            "industry_pe": benchmark_pe,
            "pe_premium_to_industry": None if pe_premium is None else round(pe_premium, 3),
            "valuation_view": valuation_view,
        },
        "fundamentals": {
            "market_cap_millions": None,
            "estimated_eps": None if estimated_eps is None else round(estimated_eps, 3),
            "eps_growth": fundamental.eps_growth,
            "forward_pe": None if forward_pe is None else round(forward_pe, 2),
            "debt_to_equity": fundamental.debt_to_equity,
            "free_cash_flow_millions": (
                None if free_cash_flow_millions is None else round(free_cash_flow_millions, 1)
            ),
        },
        "sales_quality": {
            "sales_millions": fundamental.sales_millions,
            "sales_display": _fmt_money_m(fundamental.sales_millions),
            "revenue_growth": fundamental.revenue_growth,
            "industry_revenue_growth": fundamental.industry_revenue_growth,
            "growth_spread": round(growth_spread, 3),
            "gross_margin": fundamental.gross_margin,
            "operating_margin": fundamental.operating_margin,
            "fcf_margin": fundamental.fcf_margin,
            "industry_fcf_margin": fundamental.industry_fcf_margin,
            "fcf_spread": round(fcf_spread, 3),
            "projected_sales": _project_sales(fundamental),
        },
        "cash_flow": {
            "gross_margin": fundamental.gross_margin,
            "operating_margin": fundamental.operating_margin,
            "free_cash_flow_margin": fundamental.fcf_margin,
            "industry_free_cash_flow_margin": fundamental.industry_fcf_margin,
            "status": _cash_flow_status(fundamental),
        },
        "technical": {
            "last_price": report.last_price,
            "signal": report.signal,
            "score": report.score,
            "confidence": report.confidence,
            "indicators": asdict(report.indicators),
            "rationale": report.rationale,
        },
        "benchmark_assessment": {
            "growth_vs_industry": "above" if growth_spread > 0 else "below",
            "cash_flow_vs_industry": "above" if fcf_spread > 0 else "below",
            "overall_quality": _score_label(stock["composite_score"] if stock else 0.0),
            "risk_flags": risk_flags,
        },
        "sector_leaders": sector_leaders,
        "peer_comparison": _peer_rows(symbol, profile, fundamentals, sector_recommendations),
    }
