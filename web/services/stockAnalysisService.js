(function attachStockAnalysisService(global) {
  const fmt = {
    pct(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return `${(Number(value) * 100).toFixed(1)}%`;
    },
    moneyMillions(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      const n = Number(value);
      if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}T`;
      if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}B`;
      return `$${n.toFixed(0)}M`;
    },
    multiple(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return `${Number(value).toFixed(1)}x`;
    },
    price(value) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return `$${Number(value).toFixed(2)}`;
    },
    number(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "N/A";
      return Number(value).toFixed(digits);
    },
  };

  function clamp(value, low = 0, high = 1) {
    return Math.max(low, Math.min(high, Number(value) || 0));
  }

  function titleCase(value) {
    return String(value || "N/A")
      .replaceAll("_", " ")
      .toLowerCase()
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function ratingFromScore(score) {
    const value = Number(score);
    if (!Number.isFinite(value)) return "Neutral";
    if (value >= 0.62) return "Bullish";
    if (value <= 0.42) return "Bearish";
    return "Neutral";
  }

  function valuationRating(view) {
    const normalized = String(view || "").toLowerCase();
    if (normalized.includes("discount")) return "Bullish";
    if (normalized.includes("expensive") || normalized.includes("speculative")) return "Bearish";
    return "Neutral";
  }

  function trendRating(signal) {
    if (signal === "BUY") return "Bullish";
    if (signal === "SELL") return "Bearish";
    return "Neutral";
  }

  function watchlistAction({ overallRating, action, riskCount, optionAction }) {
    const normalizedAction = String(action || "").toUpperCase();
    if (riskCount >= 3 && overallRating !== "Bullish") return "High Risk Speculative";
    if (normalizedAction.includes("STRONG_BUY") || (overallRating === "Bullish" && optionAction === "CALL")) return "Accumulate";
    if (normalizedAction.includes("REDUCE") || overallRating === "Bearish") return "Avoid";
    if (normalizedAction.includes("BUY")) return "Accumulate";
    if (normalizedAction.includes("HOLD") && riskCount <= 1) return "Watch";
    return "Watch";
  }

  function resolvePayload(ticker, context = {}) {
    const symbol = String(ticker || "").toUpperCase();
    return context.payload || context.deepPayloads?.get?.(symbol) || {
      symbol,
      profile: { symbol, company: symbol, sector: "Unknown", theme: "Unknown" },
      valuation: {},
      fundamentals: {},
      sales_quality: {},
      cash_flow: {},
      technical: { indicators: {} },
      benchmark_assessment: {},
      decision_summary: {
        action: "HOLD",
        option_action: "NO_TRADE",
        positives: [],
        risks: ["No complete backend analysis is available for this ticker yet."],
        next_checks: ["Run analysis or connect a live provider."],
      },
      agent_breakdown: {},
      peer_comparison: [],
      sector_leaders: [],
    };
  }

  function peerRows(payload) {
    return (payload.peer_comparison || []).map((row) => ({
      symbol: row.symbol,
      peRatio: fmt.multiple(row.pe_ratio),
      sales: row.sales_display || "N/A",
      growth: fmt.pct(row.revenue_growth),
      fcfMargin: fmt.pct(row.fcf_margin),
      score: fmt.number(row.composite_score, 3),
      action: titleCase(row.action || "N/A"),
    }));
  }

  function sectorLeaderRows(payload) {
    return (payload.sector_leaders || []).map((row) => ({
      symbol: row.symbol,
      action: titleCase(row.action),
      score: fmt.number(row.composite_score, 3),
      rationale: row.rationale || "No rationale available.",
    }));
  }

  function getSectorAnalysis(ticker, context = {}) {
    const payload = resolvePayload(ticker, context);
    const profile = payload.profile || {};
    const sectorAgent = payload.agent_breakdown?.sector || {};
    const benchmark = payload.benchmark_assessment || {};
    const sales = payload.sales_quality || {};
    const risks = payload.decision_summary?.risks || benchmark.risk_flags || [];
    const relativeStrength = clamp(sectorAgent.score ?? payload.recommendation?.composite_score ?? 0.5);
    const theme = profile.theme || "Unknown";
    const sector = profile.sector || "Unknown";

    return {
      ticker: payload.symbol || ticker,
      title: "Sector Analysis",
      rating: ratingFromScore(relativeStrength),
      score: relativeStrength,
      company: profile.company || payload.symbol || ticker,
      sector,
      industry: profile.industry || sector,
      overview: `${profile.company || payload.symbol || ticker} is mapped to the ${sector} sector with a ${theme} theme. The sector engine measures whether this stock is being supported by the broader peer group and theme momentum.`,
      keyTrends: [
        `${theme} demand remains tied to enterprise AI budgets, infrastructure buildout, and investor risk appetite.`,
        `Revenue growth is ${fmt.pct(sales.revenue_growth)} versus an industry benchmark of ${fmt.pct(sales.industry_revenue_growth)}.`,
        `Cash-flow spread versus the group is ${fmt.pct(sales.fcf_spread)}, which affects how much premium investors may pay for growth.`,
      ],
      peerComparison: peerRows(payload),
      sectorLeaders: sectorLeaderRows(payload),
      relativeSectorStrength: `${fmt.number(relativeStrength, 3)} / 1.000`,
      risks: risks.length ? risks : ["No major sector-specific risk flags were generated from the current data set."],
      opportunitySummary: sectorAgent.summary || `The sector setup is ${ratingFromScore(relativeStrength).toLowerCase()} based on theme strength, peer comparison, and growth quality.`,
      interpretation: `Trade-desk read: ${payload.symbol || ticker} should be sized against sector risk first. A high sector score supports momentum exposure; a weak score means stock-specific catalysts need to carry the trade.`,
    };
  }

  function getFundamentalAnalysis(ticker, context = {}) {
    const payload = resolvePayload(ticker, context);
    const profile = payload.profile || {};
    const valuation = payload.valuation || {};
    const fundamentals = payload.fundamentals || {};
    const sales = payload.sales_quality || {};
    const cash = payload.cash_flow || {};
    const fundamentalAgent = payload.agent_breakdown?.fundamental || {};
    const pe = Number(valuation.pe_ratio || 0);
    const industryPe = Number(valuation.industry_pe || 0);
    const valuationView = valuation.valuation_view || (pe > industryPe * 1.25 ? "overvalued" : pe > 0 && pe < industryPe * 0.85 ? "undervalued" : "fairly valued");
    const ratingInputs = [
      valuationRating(valuationView),
      Number(sales.growth_spread || 0) > 0 ? "Bullish" : "Neutral",
      Number(cash.free_cash_flow_margin ?? sales.fcf_margin ?? 0) < 0 ? "Bearish" : "Neutral",
      ratingFromScore(fundamentalAgent.score),
    ];
    const finalRating = ratingInputs.filter((item) => item === "Bullish").length > ratingInputs.filter((item) => item === "Bearish").length
      ? "Bullish"
      : ratingInputs.filter((item) => item === "Bearish").length > 1
        ? "Bearish"
        : "Neutral";

    return {
      ticker: payload.symbol || ticker,
      title: "Fundamental Analysis",
      rating: finalRating,
      company: profile.company || payload.symbol || ticker,
      metrics: {
        marketCap: fundamentals.market_cap_millions ? fmt.moneyMillions(fundamentals.market_cap_millions) : "N/A in current feed",
        revenueGrowth: fmt.pct(sales.revenue_growth),
        earnings: fundamentals.estimated_eps ? `${fmt.price(fundamentals.estimated_eps)} est. EPS` : "N/A",
        epsGrowth: fmt.pct(fundamentals.eps_growth),
        peRatio: fmt.multiple(valuation.pe_ratio),
        forwardPe: fmt.multiple(fundamentals.forward_pe),
        grossMargin: fmt.pct(sales.gross_margin ?? cash.gross_margin),
        operatingMargin: fmt.pct(sales.operating_margin ?? cash.operating_margin),
        debtLevel: fundamentals.debt_to_equity !== null && fundamentals.debt_to_equity !== undefined ? `${fmt.number(fundamentals.debt_to_equity, 2)} debt/equity` : "N/A",
        freeCashFlow: fundamentals.free_cash_flow_millions ? fmt.moneyMillions(fundamentals.free_cash_flow_millions) : "N/A",
      },
      valuationView: titleCase(valuationView),
      bullCase: [
        `Revenue growth of ${fmt.pct(sales.revenue_growth)} can support the multiple if it remains above the industry benchmark.`,
        `Gross margin of ${fmt.pct(sales.gross_margin)} gives the business room to scale if operating expenses normalize.`,
        fundamentalAgent.summary || "The fundamental engine sees a balanced setup from valuation, growth, and cash-flow inputs.",
      ],
      bearCase: [
        pe <= 0 ? "No positive P/E means valuation depends on future earnings rather than current profitability." : `P/E of ${fmt.multiple(pe)} must be justified against industry ${fmt.multiple(industryPe)}.`,
        Number(sales.fcf_margin ?? cash.free_cash_flow_margin ?? 0) < 0 ? "Negative free-cash-flow margin raises funding and dilution risk." : "Free-cash-flow quality still needs to stay resilient if growth slows.",
        `Debt level is ${fundamentals.debt_to_equity !== undefined ? fmt.number(fundamentals.debt_to_equity, 2) : "not available"}, which can matter if rates or financing conditions tighten.`,
      ],
      narrative: `Trade-desk read: fundamentals are ${finalRating.toLowerCase()}. The key question is whether growth and cash conversion are strong enough to justify the valuation view of ${titleCase(valuationView)}.`,
    };
  }

  function getTechnicalAnalysis(ticker, context = {}) {
    const payload = resolvePayload(ticker, context);
    const technical = payload.technical || {};
    const indicators = technical.indicators || {};
    const price = Number(technical.last_price || 0);
    const volatility = Math.max(Number(indicators.volatility || 0), 0.025);
    const support = [
      price ? price * (1 - volatility * 2.5) : null,
      Number(indicators.sma_long || 0) || null,
      price ? price * (1 - volatility * 4.5) : null,
    ].filter(Boolean);
    const resistance = [
      price ? price * (1 + volatility * 2.5) : null,
      Number(indicators.sma_short || 0) || null,
      price ? price * (1 + volatility * 4.5) : null,
    ].filter(Boolean);
    const rating = trendRating(technical.signal);
    const momentum = Number(indicators.momentum || 0);
    const rsi = Number(indicators.rsi || 0);
    const breakout = technical.signal === "BUY" && momentum > 0 ? "Breakout watch" : technical.signal === "SELL" && momentum < 0 ? "Breakdown risk" : "Range / confirmation needed";

    return {
      ticker: payload.symbol || ticker,
      title: "Technical Analysis",
      rating,
      currentPrice: fmt.price(price),
      dailyTrend: technical.signal || "N/A",
      weeklyTrend: momentum > 0.04 ? "Positive momentum" : momentum < -0.04 ? "Negative momentum" : "Neutral consolidation",
      supportLevels: support.map((value) => fmt.price(value)),
      resistanceLevels: resistance.map((value) => fmt.price(value)),
      movingAverages: {
        ma20: fmt.price(indicators.sma_short),
        ma50: fmt.price(indicators.sma_long),
        ma200: "N/A in current feed",
      },
      rsi: Number.isFinite(rsi) && rsi ? fmt.number(rsi, 1) : "N/A",
      macd: momentum > 0 ? "Positive momentum proxy" : momentum < 0 ? "Negative momentum proxy" : "Flat momentum proxy",
      volumeTrend: "N/A in current feed",
      breakoutSignal: breakout,
      shortTermSetup: technical.rationale || "No technical rationale available.",
      riskLevel: volatility > 0.05 || rsi > 75 ? "High" : volatility > 0.035 ? "Medium" : "Controlled",
      narrative: `Trade-desk read: the technical setup is ${rating.toLowerCase()}. Watch ${support.length ? support.map((value) => fmt.price(value)).join(" / ") : "support"} on weakness and ${resistance.length ? resistance.map((value) => fmt.price(value)).join(" / ") : "resistance"} on strength.`,
    };
  }

  function getFullStockDeskAnalysis(ticker, context = {}) {
    const payload = resolvePayload(ticker, context);
    const sector = getSectorAnalysis(ticker, { ...context, payload });
    const fundamental = getFundamentalAnalysis(ticker, { ...context, payload });
    const technical = getTechnicalAnalysis(ticker, { ...context, payload });
    const recommendation = payload.recommendation || {};
    const summary = payload.decision_summary || {};
    const scores = {
      sector: Number(payload.agent_breakdown?.sector?.score ?? sector.score ?? 0.5),
      fundamental: Number(payload.agent_breakdown?.fundamental?.score ?? 0.5),
      technical: Number(payload.agent_breakdown?.technical?.score ?? 0.5),
    };
    const overallScore = clamp(scores.sector * 0.2 + scores.fundamental * 0.35 + scores.technical * 0.45);
    const overallRating = ratingFromScore(overallScore);
    const riskCount = (summary.risks || payload.benchmark_assessment?.risk_flags || []).length;
    const watchlistActionValue = watchlistAction({
      overallRating,
      action: summary.action || recommendation.action,
      riskCount,
      optionAction: summary.option_action || payload.option_recommendation?.action,
    });

    return {
      ticker: payload.symbol || ticker,
      title: "Trade Desk Summary",
      sectorScore: scores.sector,
      fundamentalScore: scores.fundamental,
      technicalScore: scores.technical,
      overallScore,
      overallRating,
      suggestedWatchlistAction: watchlistActionValue,
      thesis: summary.thesis || recommendation.rationale || "No thesis available.",
      positives: summary.positives || [],
      risks: summary.risks || payload.benchmark_assessment?.risk_flags || [],
      nextChecks: summary.next_checks || [],
      disclaimer: "This is not financial advice. Use this for research only.",
      sector,
      fundamental,
      technical,
    };
  }

  global.StockAnalysisService = {
    getSectorAnalysis,
    getFundamentalAnalysis,
    getTechnicalAnalysis,
    getFullStockDeskAnalysis,
    __formatters: fmt,
  };
})(window);
