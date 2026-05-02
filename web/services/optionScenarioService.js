(function attachOptionScenarioService(global) {
  const CONTRACT_SIZE = 100;

  function num(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function money(value) {
    const amount = num(value, 0);
    const sign = amount < 0 ? "-" : "";
    return `${sign}$${Math.abs(amount).toLocaleString(undefined, {
      maximumFractionDigits: Math.abs(amount) >= 1000 ? 0 : 2,
      minimumFractionDigits: Math.abs(amount) >= 1000 ? 0 : 2,
    })}`;
  }

  function pct(value) {
    return `${(num(value, 0) * 100).toFixed(1)}%`;
  }

  function actionLabel(action) {
    return String(action || "NO_TRADE").replaceAll("_", " ");
  }

  function latestPrice(option, context = {}) {
    const payload = context.payload || {};
    const lastPrice = num(payload.technical?.last_price, 0);
    if (lastPrice > 0) return lastPrice;

    const strike = num(option?.strike, 0);
    if (!strike) return 0;
    if (option?.action === "CALL") return strike / 1.03;
    if (option?.action === "PUT") return strike / 0.97;
    return strike;
  }

  function estimatedPremium(option, context = {}) {
    const price = latestPrice(option, context);
    const strike = num(option?.strike, price);
    const days = Math.max(1, num(option?.expiry_days, 30));
    const volatility = clamp(num(context.payload?.technical?.indicators?.volatility, 0.035), 0.01, 0.18);
    const timeValue = price * volatility * Math.sqrt(days / 365) * 0.85;
    const moneynessGap = Math.abs(price - strike) * 0.2;
    return Math.max(0.05, Number((timeValue + moneynessGap).toFixed(2)));
  }

  function intrinsicValue(action, stockPrice, strike) {
    if (action === "CALL") return Math.max(0, stockPrice - strike);
    if (action === "PUT") return Math.max(0, strike - stockPrice);
    return 0;
  }

  function payoffRow(action, strike, premium, contracts, stockPrice) {
    const intrinsic = intrinsicValue(action, stockPrice, strike);
    const plPerShare = intrinsic - premium;
    const pl = plPerShare * CONTRACT_SIZE * contracts;
    const maxLoss = premium * CONTRACT_SIZE * contracts;
    return {
      stockPrice,
      intrinsic,
      plPerShare,
      pl,
      returnOnRisk: maxLoss > 0 ? pl / maxLoss : 0,
      status: pl > 0 ? "Profit" : pl < 0 ? "Loss" : "Break-even",
    };
  }

  function getOptionRationale(option, context = {}) {
    const payload = context.payload || {};
    const action = option?.action || "NO_TRADE";
    const recommendation = payload.recommendation || {};
    const summary = payload.decision_summary || {};
    const technical = payload.technical || {};
    const optionAgent = payload.agent_breakdown?.option || {};
    const technicalAgent = payload.agent_breakdown?.technical || {};
    const valuation = payload.valuation || {};

    const directionalRead =
      action === "CALL"
        ? "The model is expressing upside participation because the stock recommendation and technical read lean positive."
        : action === "PUT"
          ? "The model is expressing downside protection/speculation because the composite profile is weak or the technical trend is bearish."
          : "The model is avoiding directional premium because the edge is not strong enough versus decay and spread cost.";

    const drivers = [
      option?.rationale || optionAgent.summary || "No option-agent rationale was supplied.",
      recommendation.action ? `Equity action is ${actionLabel(recommendation.action)}.` : "",
      technical.signal ? `Technical signal is ${technical.signal} with score ${num(technical.score, 0).toFixed(3)}.` : technicalAgent.summary || "",
      valuation.valuation_view ? `Valuation view is ${valuation.valuation_view}; this affects how much upside premium should be paid.` : "",
    ].filter(Boolean);

    const assessmentChecklist = [
      "Check live bid/ask spread; wide spreads can erase expected edge before the stock even moves.",
      "Compare premium paid against break-even and expected move.",
      "Confirm open interest and volume so the contract can be exited.",
      "Review implied volatility; high IV means the stock can move correctly while the option still loses value.",
      "Size from max loss first, not from upside potential.",
    ];

    const riskNotes = [
      "Long options can lose 100% of premium paid.",
      "Time decay accelerates near expiration, especially if the stock stalls.",
      "The current app uses a model contract proxy unless live broker bid/ask data is entered.",
      ...(summary.risks || []),
    ];

    return {
      action,
      title: `${option?.symbol || "Selected"} ${actionLabel(action)} rationale`,
      directionalRead,
      drivers,
      assessmentChecklist,
      riskNotes,
    };
  }

  function buildOptionScenario(option, state = {}, context = {}) {
    const action = option?.action || "NO_TRADE";
    const currentPrice = latestPrice(option, context);
    const strike = num(option?.strike, currentPrice);
    const premium = Math.max(0, num(state.premium, estimatedPremium(option, context)));
    const contracts = Math.max(1, Math.round(num(state.contracts, 1)));
    const scenarioPrice = Math.max(0, num(state.scenarioPrice, currentPrice));
    const expiryDays = Math.max(1, Math.round(num(option?.expiry_days, 30)));
    const breakeven = action === "CALL" ? strike + premium : action === "PUT" ? strike - premium : null;
    const maxLoss = premium * CONTRACT_SIZE * contracts;
    const selected = payoffRow(action, strike, premium, contracts, scenarioPrice);
    const stressMoves = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2];
    const rows = stressMoves.map((move) => ({
      move,
      ...payoffRow(action, strike, premium, contracts, currentPrice * (1 + move)),
    }));

    return {
      symbol: option?.symbol || context.payload?.symbol || "",
      action,
      strike,
      expiryDays,
      currentPrice,
      scenarioPrice,
      premium,
      contracts,
      breakeven,
      maxLoss,
      maxProfit:
        action === "CALL"
          ? "Theoretical unlimited"
          : action === "PUT"
            ? money(Math.max(0, strike - premium) * CONTRACT_SIZE * contracts)
            : "No directional contract",
      selected,
      rows,
      inputs: {
        scenarioPrice: Number(scenarioPrice.toFixed(2)),
        premium: Number(premium.toFixed(2)),
        contracts,
      },
    };
  }

  function answerOptionQuestion(question, option, scenario, context = {}) {
    const text = String(question || "").toLowerCase();
    const rationale = getOptionRationale(option, context);
    const action = option?.action || "NO_TRADE";
    const breakevenText = scenario.breakeven == null ? "not available for a no-trade bias" : money(scenario.breakeven);

    if (!text.trim()) {
      return "Ask about the option bias, max risk, break-even, strike, expiry, premium, IV, liquidity, or how to assess call versus put quality.";
    }

    const asksRationale = text.includes("why") || text.includes("rationale") || text.includes("call") || text.includes("put");
    const asksRisk = text.includes("risk") || text.includes("loss") || text.includes("safe");

    if (asksRationale && asksRisk) {
      return `${rationale.directionalRead} Main inputs: ${rationale.drivers.join(" ")} Max premium risk is ${money(scenario.maxLoss)} for ${scenario.contracts} contract(s). Watch spread cost, IV crush, time decay, and whether the stock can clear break-even ${breakevenText}.`;
    }

    if (asksRationale) {
      return `${rationale.directionalRead} Main inputs: ${rationale.drivers.join(" ")} This is a research signal, not an order; live chain quality still decides whether the trade is usable.`;
    }

    if (asksRisk) {
      return `For this long ${actionLabel(action)} proxy, max premium risk is ${money(scenario.maxLoss)} for ${scenario.contracts} contract(s), assuming a premium of ${money(scenario.premium)}. The practical risks are spread cost, IV crush, time decay, and a stock move that does not clear break-even.`;
    }

    if (text.includes("break") || text.includes("even")) {
      return `Break-even is ${breakevenText}. For a CALL, the stock must finish above strike plus premium. For a PUT, it must finish below strike minus premium.`;
    }

    if (text.includes("premium") || text.includes("bid") || text.includes("ask") || text.includes("spread")) {
      return `Use your broker premium, not the model estimate, when deciding. A good contract usually has a tight bid/ask spread, enough volume/open interest, and premium that still leaves a realistic break-even before expiration.`;
    }

    if (text.includes("strike") || text.includes("expiry") || text.includes("expiration")) {
      return `The proxy uses strike ${money(scenario.strike)} and ${scenario.expiryDays} days. Assess whether that expiry gives the thesis enough time and whether the strike is close enough to participate without overpaying for extrinsic value.`;
    }

    if (text.includes("iv") || text.includes("volatility")) {
      return "High implied volatility can make both calls and puts expensive. If IV is elevated, compare long premium against defined-risk spreads or wait for a better entry because the stock can move correctly while the option loses value from IV compression.";
    }

    return `Assessment framework: confirm the directional thesis, check break-even ${breakevenText}, cap risk at ${money(scenario.maxLoss)}, verify liquidity, and avoid paying premium unless the expected move is larger than the option market is pricing.`;
  }

  global.OptionScenarioService = {
    buildOptionScenario,
    getOptionRationale,
    answerOptionQuestion,
    money,
    pct,
    actionLabel,
  };
})(window);
