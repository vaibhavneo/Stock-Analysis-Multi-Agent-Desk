from __future__ import annotations

from dataclasses import asdict
import math
from statistics import fmean
from typing import Dict, Iterable, List

from .analysis import analyze_symbol
from .models import (
    AgentAssessment,
    FundamentalSnapshot,
    OptionRecommendation,
    PriceBar,
    StockRecommendation,
    SymbolProfile,
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean_close(bars: List[PriceBar], window: int) -> float:
    closes = [b.close for b in bars[-window:]] if len(bars) >= window else [b.close for b in bars]
    return fmean(closes) if closes else 0.0


class SectorAnalysisAgent:
    name = "sector-analysis-agent"

    def run(
        self, bars_by_symbol: Dict[str, List[PriceBar]], profiles: Dict[str, SymbolProfile]
    ) -> Dict[str, AgentAssessment]:
        momentum_by_symbol: Dict[str, float] = {}
        for symbol, bars in bars_by_symbol.items():
            if len(bars) < 2 or symbol not in profiles:
                continue
            current = _mean_close(bars, 5)
            base = _mean_close(bars, 10)
            if base <= 0:
                continue
            momentum_by_symbol[symbol] = (current - base) / base

        sector_momentum: Dict[str, List[float]] = {}
        for symbol, mom in momentum_by_symbol.items():
            sector = profiles[symbol].sector
            sector_momentum.setdefault(sector, []).append(mom)

        sector_strength = {
            sector: fmean(values) if values else 0.0
            for sector, values in sector_momentum.items()
        }

        output: Dict[str, AgentAssessment] = {}
        for symbol, profile in profiles.items():
            strength = sector_strength.get(profile.sector, 0.0)
            score = _clamp(0.5 + strength * 4, 0.0, 1.0)
            summary = (
                f"{profile.sector} sector trend score={strength:.3f}. "
                f"Relative sector strength supports {profile.theme} theme conviction."
            )
            output[symbol] = AgentAssessment(
                symbol=symbol,
                score=round(score, 3),
                summary=summary,
            )
        return output


class FundamentalAnalysisAgent:
    name = "fundamental-analysis-agent"

    def run(
        self, fundamentals: Dict[str, FundamentalSnapshot]
    ) -> Dict[str, AgentAssessment]:
        output: Dict[str, AgentAssessment] = {}
        for symbol, f in fundamentals.items():
            growth_score = _clamp((f.revenue_growth + f.eps_growth) / 2, -0.5, 1.0)
            leverage_penalty = _clamp(f.debt_to_equity / 2.0, 0.0, 1.0)
            benchmark_pe = f.industry_pe if f.industry_pe > 0 else 30.0
            valuation_penalty = 0.30 if f.pe_ratio <= 0 else _clamp(
                (f.pe_ratio / benchmark_pe - 1.0) / 1.5, 0.0, 1.0
            )
            fcf_bonus = _clamp(f.fcf_margin, -0.2, 0.6)
            sales_scale = _clamp(math.log10(max(1.0, f.sales_millions)) / 6, 0.0, 1.0)
            benchmark_growth_delta = f.revenue_growth - f.industry_revenue_growth

            raw = (
                0.40 * (0.5 + growth_score / 2)
                + 0.22 * (0.5 + fcf_bonus / 1.2)
                + 0.13 * sales_scale
                + 0.12 * (0.5 + _clamp(benchmark_growth_delta, -0.5, 0.5))
                - 0.10 * leverage_penalty
                - 0.17 * valuation_penalty
            )
            score = _clamp(raw, 0.0, 1.0)
            summary = (
                f"P/E={f.pe_ratio:.1f} vs industry {benchmark_pe:.1f}; "
                f"sales=${f.sales_millions:.0f}M; rev growth={f.revenue_growth:.1%} "
                f"vs industry {f.industry_revenue_growth:.1%}."
            )
            output[symbol] = AgentAssessment(
                symbol=symbol, score=round(score, 3), summary=summary
            )
        return output


class TechnicalAnalysisAgent:
    name = "technical-analysis-agent"

    def run(self, bars_by_symbol: Dict[str, List[PriceBar]]) -> Dict[str, AgentAssessment]:
        output: Dict[str, AgentAssessment] = {}
        for symbol, bars in bars_by_symbol.items():
            report = analyze_symbol(bars)
            score = _clamp((report.score + 1) / 2, 0.0, 1.0)
            summary = (
                f"Signal={report.signal}, momentum={report.indicators.momentum:.3f}, "
                f"RSI={report.indicators.rsi:.1f}, vol={report.indicators.volatility:.4f}"
            )
            output[symbol] = AgentAssessment(
                symbol=symbol, score=round(score, 3), summary=summary
            )
        return output


class StockRecommendationAgent:
    name = "stock-recommendation-agent"

    def run(
        self,
        profiles: Dict[str, SymbolProfile],
        sector_scores: Dict[str, AgentAssessment],
        fundamental_scores: Dict[str, AgentAssessment],
        technical_scores: Dict[str, AgentAssessment],
        top_n: int,
    ) -> List[StockRecommendation]:
        recs: List[StockRecommendation] = []
        for symbol, profile in profiles.items():
            if symbol not in sector_scores or symbol not in fundamental_scores:
                continue
            if symbol not in technical_scores:
                continue
            sector = sector_scores[symbol].score
            fundamental = fundamental_scores[symbol].score
            technical = technical_scores[symbol].score
            composite = 0.20 * sector + 0.35 * fundamental + 0.45 * technical
            confidence = _clamp(abs(composite - 0.5) * 2, 0.05, 0.99)

            if composite >= 0.72:
                action = "STRONG_BUY"
            elif composite >= 0.60:
                action = "BUY"
            elif composite >= 0.45:
                action = "HOLD"
            else:
                action = "REDUCE"

            rationale = (
                f"Sector={sector:.2f}, Fundamental={fundamental:.2f}, "
                f"Technical={technical:.2f} -> Composite={composite:.2f}"
            )
            recs.append(
                StockRecommendation(
                    symbol=symbol,
                    company=profile.company,
                    theme=profile.theme,
                    sector=profile.sector,
                    action=action,
                    composite_score=round(composite, 3),
                    confidence=round(confidence, 3),
                    rationale=rationale,
                )
            )
        recs.sort(key=lambda r: r.composite_score, reverse=True)
        return recs[:top_n]


class OptionRecommendationAgent:
    name = "option-recommendation-agent"

    def run(
        self,
        picks: Iterable[StockRecommendation],
        bars_by_symbol: Dict[str, List[PriceBar]],
    ) -> List[OptionRecommendation]:
        output: List[OptionRecommendation] = []
        for pick in picks:
            bars = bars_by_symbol.get(pick.symbol, [])
            if not bars:
                continue
            report = analyze_symbol(bars)
            px = report.last_price
            vol = report.indicators.volatility
            confidence = _clamp((pick.confidence + min(0.4, vol * 10)) / 1.2, 0.05, 0.99)

            if pick.action in {"STRONG_BUY", "BUY"}:
                action = "CALL"
                strike = round(px * 1.03, 2)
                rationale = "Bullish composite and trend profile favor upside exposure."
            elif pick.action == "REDUCE":
                action = "PUT"
                strike = round(px * 0.97, 2)
                rationale = "Weak composite profile favors downside hedge/speculation."
            else:
                action = "NO_TRADE"
                strike = round(px, 2)
                rationale = "Hold bias; option edge is weak versus premium decay."

            expiry_days = 30 if vol < 0.04 else 21
            output.append(
                OptionRecommendation(
                    symbol=pick.symbol,
                    action=action,  # type: ignore[arg-type]
                    expiry_days=expiry_days,
                    strike=strike,
                    confidence=round(confidence, 3),
                    rationale=rationale,
                )
            )
        return output


class MultiAgentOrchestrator:
    def __init__(self) -> None:
        self.sector_agent = SectorAnalysisAgent()
        self.fundamental_agent = FundamentalAnalysisAgent()
        self.technical_agent = TechnicalAnalysisAgent()
        self.stock_agent = StockRecommendationAgent()
        self.option_agent = OptionRecommendationAgent()

    def run(
        self,
        bars_by_symbol: Dict[str, List[PriceBar]],
        profiles: Dict[str, SymbolProfile],
        fundamentals: Dict[str, FundamentalSnapshot],
        top_n: int = 5,
    ) -> dict:
        symbols = set(profiles.keys()) & set(bars_by_symbol.keys()) & set(fundamentals.keys())
        filtered_bars = {s: bars_by_symbol[s] for s in symbols}
        filtered_profiles = {s: profiles[s] for s in symbols}
        filtered_fundamentals = {s: fundamentals[s] for s in symbols}

        sector_scores = self.sector_agent.run(filtered_bars, filtered_profiles)
        fundamental_scores = self.fundamental_agent.run(filtered_fundamentals)
        technical_scores = self.technical_agent.run(filtered_bars)

        picks = self.stock_agent.run(
            filtered_profiles,
            sector_scores,
            fundamental_scores,
            technical_scores,
            top_n=top_n,
        )
        options = self.option_agent.run(picks, filtered_bars)

        by_theme: Dict[str, List[dict]] = {}
        for pick in picks:
            by_theme.setdefault(pick.theme, []).append(asdict(pick))

        return {
            "actions": {
                "sector_analysis": [asdict(sector_scores[s]) for s in sorted(sector_scores.keys())],
                "fundamental_analysis": [
                    asdict(fundamental_scores[s]) for s in sorted(fundamental_scores.keys())
                ],
                "technical_analysis": [
                    asdict(technical_scores[s]) for s in sorted(technical_scores.keys())
                ],
                "stock_recommendation": [asdict(p) for p in picks],
                "option_recommendation": [asdict(o) for o in options],
            },
            "top_by_theme": by_theme,
        }
