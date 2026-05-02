from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


Signal = Literal["BUY", "HOLD", "SELL"]
Side = Literal["BUY", "SELL"]
OptionAction = Literal["CALL", "PUT", "NO_TRADE"]


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IndicatorSnapshot:
    sma_short: float
    sma_long: float
    momentum: float
    rsi: float
    volatility: float


@dataclass(frozen=True)
class AnalysisReport:
    symbol: str
    signal: Signal
    confidence: float
    score: float
    last_price: float
    indicators: IndicatorSnapshot
    rationale: str


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: int
    price: float


@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: float


@dataclass(frozen=True)
class TradeExecution:
    symbol: str
    side: Side
    quantity: int
    fill_price: float
    cash_after: float
    note: str


@dataclass(frozen=True)
class SymbolProfile:
    symbol: str
    company: str
    sector: str
    theme: str


@dataclass(frozen=True)
class FundamentalSnapshot:
    symbol: str
    pe_ratio: float
    sales_millions: float
    revenue_growth: float
    eps_growth: float
    debt_to_equity: float
    gross_margin: float
    operating_margin: float
    fcf_margin: float
    industry_pe: float
    industry_revenue_growth: float
    industry_fcf_margin: float


@dataclass(frozen=True)
class AgentAssessment:
    symbol: str
    score: float
    summary: str


@dataclass(frozen=True)
class StockRecommendation:
    symbol: str
    company: str
    theme: str
    sector: str
    action: str
    composite_score: float
    confidence: float
    rationale: str


@dataclass(frozen=True)
class OptionRecommendation:
    symbol: str
    action: OptionAction
    expiry_days: int
    strike: float
    confidence: float
    rationale: str
