const IS_HTTP_PAGE = /^https?:$/.test(location.protocol);
const API_PROTOCOL = IS_HTTP_PAGE ? location.protocol : "http:";
const API_HOST = location.hostname || "127.0.0.1";
const API_BASES = [
  ...(IS_HTTP_PAGE ? [location.origin] : []),
  `${API_PROTOCOL}//${API_HOST}:5051`,
  `${API_PROTOCOL}//${API_HOST}:8765`,
  `${API_PROTOCOL}//${API_HOST}:8766`,
].filter((value, index, values) => values.indexOf(value) === index);

const SCENARIOS = {
  custom: {
    title: "Custom Basket",
    copy: "Run the exact tickers in the symbol box.",
    symbols: null,
  },
  all: {
    title: "Full AI + Quantum Universe",
    copy: "Run every stock available in the local AI and quantum computing universe.",
    symbols: "ALL",
  },
  leaders: {
    title: "AI Leaders",
    copy: "Large-cap AI infrastructure, semiconductor, and software leaders.",
    symbols: ["NVDA", "MSFT", "GOOGL", "AMD", "AVGO", "ORCL", "CRM", "PLTR"],
  },
  "spec-ai": {
    title: "Speculative AI",
    copy: "Higher-beta AI application, voice AI, defense AI, and automation names.",
    symbols: ["PLTR", "SOUN", "AI", "BBAI", "FFAI", "HIMS", "PATH", "SNOW"],
  },
  quantum: {
    title: "Quantum Computing",
    copy: "Pure-play quantum, quantum security, and quantum-adjacent infrastructure.",
    symbols: ["RGTI", "IONQ", "QBTS", "QUBT", "ARQQ", "IBM"],
  },
  risk: {
    title: "Risk + Hedge Watch",
    copy: "Volatile names where put/call suitability should be checked with live liquidity.",
    symbols: ["RGTI", "SOUN", "BBAI", "AI", "FFAI", "QBTS", "QUBT", "HIMS"],
  },
};

const els = {
  form: document.querySelector("#analysis-form"),
  symbols: document.querySelector("#symbols"),
  apiKey: document.querySelector("#api-key"),
  top: document.querySelector("#top"),
  modeLabel: document.querySelector("#mode-label"),
  lastRun: document.querySelector("#last-run"),
  universeCount: document.querySelector("#universe-count"),
  feedStatus: document.querySelector("#feed-status"),
  scenarioTitle: document.querySelector("#scenario-title"),
  scenarioCopy: document.querySelector("#scenario-copy"),
  scenarioActions: document.querySelector("#scenario-actions"),
  bestAction: document.querySelector("#best-action"),
  optionsBias: document.querySelector("#options-bias"),
  riskFlags: document.querySelector("#risk-flags"),
  stockCount: document.querySelector("#stock-count"),
  stockTable: document.querySelector("#stock-table"),
  recommendationWorkbench: document.querySelector("#recommendation-workbench"),
  optionsList: document.querySelector("#options-list"),
  optionWorkbench: document.querySelector("#option-workbench"),
  deepAnalysis: document.querySelector("#deep-analysis"),
  stockSelector: document.querySelector("#stock-selector"),
  analysisTabs: document.querySelector("#analysis-tabs"),
  sectorAgent: document.querySelector("#sector-agent"),
  fundamentalAgent: document.querySelector("#fundamental-agent"),
  technicalAgent: document.querySelector("#technical-agent"),
};

let currentMode = "sample";
let currentScenario = "custom";
let universe = [];
let activeSymbol = null;
let activeAnalysisTab = "overview";
let activeOptionSymbol = null;
let currentRun = null;
let currentDeepPayload = null;
let optionScenarioState = {};
const optionChatBySymbol = new Map();
const recommendationChatBySymbol = new Map();

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function actionLabel(action) {
  return String(action || "--").replaceAll("_", " ");
}

function score(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "--";
}

function confidence(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "--";
}

function money(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `$${number.toLocaleString(undefined, {
    minimumFractionDigits: number >= 1000 ? 0 : 2,
    maximumFractionDigits: number >= 1000 ? 0 : 2,
  })}`;
}

function pct(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `${(number * 100).toFixed(1)}%`;
}

function compactErrorMessage(error) {
  return String(error?.message || error || "Unknown error").replace(/^Error:\s*/i, "");
}

function pill(action) {
  const text = actionLabel(action);
  const normalized = text.toLowerCase();
  const className = normalized.includes("buy") || normalized.includes("call")
    ? "good"
    : normalized.includes("reduce") || normalized.includes("sell") || normalized.includes("put")
      ? "bad"
      : "neutral";
  return `<span class="pill ${className}">${escapeHtml(text)}</span>`;
}

function RatingBadge(rating) {
  const normalized = String(rating || "Neutral");
  const className = normalized.toLowerCase().includes("bull")
    || normalized.toLowerCase().includes("call")
    ? "good"
    : normalized.toLowerCase().includes("bear")
      || normalized.toLowerCase().includes("put")
      ? "bad"
      : "neutral";
  return `<span class="rating-badge ${className}">${escapeHtml(normalized)}</span>`;
}

function MetricCard(label, value, detail = "", explanation = "") {
  return `
    <article class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value ?? "N/A")}</strong>
      ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
      ${explanation ? `<p>${escapeHtml(explanation)}</p>` : ""}
    </article>
  `;
}

function renderList(items) {
  if (!items || !items.length) return "<li>No items available.</li>";
  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function currentSymbols() {
  const scenario = SCENARIOS[currentScenario];
  if (scenario?.symbols === "ALL") return universe.map((item) => item.symbol);
  if (Array.isArray(scenario?.symbols)) return scenario.symbols;
  return els.symbols.value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

async function apiGet(path, params = {}) {
  const query = new URLSearchParams(params);
  let lastError = null;
  for (const base of API_BASES) {
    try {
      const url = `${base}${path}?${query.toString()}`;
      const response = await fetch(url);
      const payload = await response.json();
      if (!response.ok || payload.error) throw new Error(payload.error || response.statusText);
      return payload;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error("Stock API is unreachable.");
}

function setFeedStatus(type, message, detail = "") {
  els.feedStatus.className = `status-panel ${type || ""}`;
  els.feedStatus.innerHTML = `<strong>${escapeHtml(message)}</strong>${detail ? `<span>${escapeHtml(detail)}</span>` : ""}`;
}

function setBusy(isBusy) {
  const button = els.form.querySelector("button[type='submit']");
  button.disabled = isBusy;
  button.textContent = isBusy ? "Running..." : "Run Analysis";
}

function updateModeUI() {
  document.querySelectorAll(".feed-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === currentMode);
  });
  els.modeLabel.textContent = currentMode === "live"
    ? "Yahoo/Stooq Live"
    : currentMode === "alpha"
      ? "Alpha Vantage"
      : "Sample CSV";
}

function setScenario(name) {
  currentScenario = SCENARIOS[name] ? name : "custom";
  const scenario = SCENARIOS[currentScenario];
  els.scenarioTitle.textContent = scenario.title;
  els.scenarioCopy.textContent = scenario.copy;
  document.querySelectorAll("[data-scenario]").forEach((button) => {
    button.classList.toggle("active", button.dataset.scenario === currentScenario);
  });
  if (Array.isArray(scenario.symbols)) els.symbols.value = scenario.symbols.join(", ");
  if (scenario.symbols === "ALL") els.symbols.value = universe.map((item) => item.symbol).join(", ");
}

function runMetaWarnings(meta) {
  return (meta?.warnings || []).join(" ");
}

function providerRunDetails(meta) {
  const count = meta?.symbols?.length || 0;
  return `${count} symbols analyzed.`;
}

function renderRunSummary(stocks, options, meta) {
  const best = stocks[0];
  els.bestAction.textContent = best
    ? `${best.symbol} ${actionLabel(best.action)} (${score(best.composite_score)})`
    : "--";
  const call = options.filter((item) => item.action === "CALL").length;
  const put = options.filter((item) => item.action === "PUT").length;
  const wait = options.length - call - put;
  els.optionsBias.textContent = `${call} CALL / ${put} PUT / ${Math.max(0, wait)} WAIT`;
  const reducers = stocks.filter((item) => item.action === "REDUCE").length;
  const warnings = meta?.warnings?.length || 0;
  els.riskFlags.textContent = `${reducers + warnings} flagged`;
}

function renderStocks(stocks) {
  els.stockCount.textContent = `${stocks.length} picks`;
  if (!stocks.length) {
    els.stockTable.innerHTML = `<tr><td colspan="7">No matching stock recommendations.</td></tr>`;
    renderRecommendationWorkbench(null);
    return;
  }
  els.stockTable.innerHTML = stocks
    .map(
      (stock) => `
        <tr class="${stock.symbol === activeSymbol ? "active-row" : ""}">
          <td><button type="button" class="symbol-link" data-open-symbol="${escapeHtml(stock.symbol)}">${escapeHtml(stock.symbol)}</button><small>${escapeHtml(stock.company || "")}</small></td>
          <td>${escapeHtml(stock.theme || stock.sector || "--")}</td>
          <td>${pill(stock.action)}</td>
          <td>${score(stock.composite_score)}</td>
          <td>${confidence(stock.confidence)}</td>
          <td>${escapeHtml(stock.rationale || "--")}</td>
          <td><button type="button" class="open-btn" data-open-symbol="${escapeHtml(stock.symbol)}">Open</button></td>
        </tr>
      `,
    )
    .join("");
}

function currentStocks() {
  return currentRun?.actions?.stock_recommendation || [];
}

function stockForSymbol(symbol, stocks = currentStocks()) {
  return (stocks || []).find((stock) => stock.symbol === symbol);
}

function bounded(value, min, max) {
  return Math.max(min, Math.min(max, Number(value) || 0));
}

function inferReferencePrice(stock) {
  const payload = currentDeepPayload?.symbol === stock?.symbol ? currentDeepPayload : null;
  const technicalPrice = Number(payload?.technical?.last_price);
  if (Number.isFinite(technicalPrice) && technicalPrice > 0) return technicalPrice;

  const option = optionForSymbol(stock?.symbol);
  const strike = Number(option?.strike);
  if (Number.isFinite(strike) && strike > 0) {
    if (option.action === "CALL") return strike / 1.03;
    if (option.action === "PUT") return strike / 0.97;
    return strike;
  }
  return null;
}

function recommendationPricePlan(stock) {
  const payload = currentDeepPayload?.symbol === stock?.symbol ? currentDeepPayload : null;
  const indicators = payload?.technical?.indicators || {};
  const reference = inferReferencePrice(stock);
  const action = String(stock?.action || "HOLD").toUpperCase();
  const volatility = bounded(indicators.volatility, 0.018, 0.12);
  const smaLong = Number(indicators.sma_long);
  const smaShort = Number(indicators.sma_short);
  const support = reference ? Math.min(
    Number.isFinite(smaLong) && smaLong > 0 ? smaLong : reference,
    reference * (1 - Math.max(0.045, volatility * 2.6)),
  ) : null;
  const resistance = reference ? Math.max(
    Number.isFinite(smaShort) && smaShort > 0 ? smaShort : reference,
    reference * (1 + Math.max(0.045, volatility * 2.6)),
  ) : null;
  const buyZoneLow = reference ? reference * (1 - Math.max(0.015, volatility * 1.35)) : null;
  const buyZoneHigh = reference ? reference * (1 + Math.max(0.008, volatility * 0.55)) : null;
  const reduceCeiling = reference ? reference * (1 + Math.max(0.045, volatility * 2.6)) : null;
  const stop = reference && action.includes("BUY")
    ? Math.min(support, reference * (1 - Math.max(0.055, volatility * 3.2)))
    : action === "HOLD" && reference
      ? reference * (1 - Math.max(0.045, volatility * 2.3))
      : null;
  const riskCeiling = action === "REDUCE" && reference ? reduceCeiling : null;
  const upsideCheckpoint = reference && stop ? reference + (reference - stop) * 1.8 : resistance;

  let entryText = "Run deep analysis to calculate a reference price.";
  if (reference && action.includes("BUY")) {
    entryText = `${money(buyZoneLow)} to ${money(buyZoneHigh)} research entry zone. Avoid chasing far above this zone unless new data improves the score.`;
  } else if (reference && action === "HOLD") {
    entryText = `Watch near ${money(reference)}. Wait for either a breakout above ${money(resistance)} or a pullback toward ${money(support)}.`;
  } else if (reference && action === "REDUCE") {
    entryText = `Reduce/avoid bias near ${money(reference)}. Reassess only if price can reclaim ${money(riskCeiling)} with improving technicals.`;
  }

  const stopText = stop
    ? `${money(stop)} model invalidation line. A close below this level means the setup is not behaving as expected.`
    : riskCeiling
      ? `${money(riskCeiling)} upside risk ceiling for the bearish/reduce view. A sustained move above it weakens the reduce thesis.`
      : "No stop-loss proxy available until a reference price is available.";

  return {
    reference,
    entryText,
    stopText,
    stop,
    support,
    resistance,
    upsideCheckpoint,
    riskPerShare: reference && stop ? reference - stop : null,
    volatility,
  };
}

function recommendationThesis(stock) {
  if (!stock) return null;
  const payload = currentDeepPayload?.symbol === stock.symbol ? currentDeepPayload : null;
  const actions = currentRun?.actions || {};
  const sector = payload?.agent_breakdown?.sector || assessmentFor(stock.symbol, actions.sector_analysis);
  const fundamental = payload?.agent_breakdown?.fundamental || assessmentFor(stock.symbol, actions.fundamental_analysis);
  const technical = payload?.agent_breakdown?.technical || assessmentFor(stock.symbol, actions.technical_analysis);
  const valuation = payload?.valuation?.valuation_view;
  const sales = payload?.sales_quality || {};
  const option = optionForSymbol(stock.symbol);
  const pricePlan = recommendationPricePlan(stock);
  const scoreText = `Composite score ${score(stock.composite_score)} with ${confidence(stock.confidence)} model confidence.`;
  const priceText = pricePlan.reference ? `Reference price is ${money(pricePlan.reference)}.` : "Reference price is unavailable until quote data loads.";
  const strength = [
    sector?.summary || `Sector score ${score(sector?.score)}.`,
    fundamental?.summary || `Fundamental score ${score(fundamental?.score)}.`,
    technical?.summary || `Technical score ${score(technical?.score)}.`,
  ].filter(Boolean);
  const positives = [
    scoreText,
    stock.rationale || "",
    Number(sales.growth_spread) > 0 ? `Revenue growth spread is ${pct(sales.growth_spread)} above the industry benchmark.` : "",
    option ? `Options agent bias is ${actionLabel(option.action)} with ${confidence(option.confidence)} confidence.` : "",
  ].filter(Boolean);
  const watchouts = [
    valuation ? `Valuation view is ${valuation}; price discipline matters.` : "",
    ...(payload?.decision_summary?.risks || []),
    pricePlan.stopText,
  ].filter(Boolean);

  return {
    stock,
    option,
    pricePlan,
    strength,
    positives,
    watchouts,
    narrative: `${stock.symbol} is ranked ${actionLabel(stock.action)} because the model blends sector, fundamental, and technical signals into ${score(stock.composite_score)}. ${priceText} ${pricePlan.entryText}`,
  };
}

function comparableRecommendationRows(selectedSymbol) {
  return currentStocks()
    .filter((stock) => stock.symbol !== selectedSymbol)
    .slice(0, 6)
    .map((stock) => ({
      symbol: stock.symbol,
      company: stock.company || "",
      action: actionLabel(stock.action),
      score: score(stock.composite_score),
      confidence: confidence(stock.confidence),
      rationale: stock.rationale || "",
    }));
}

function RecommendationComparisonTable(rows) {
  if (!rows.length) return `<div class="mini-table"><table><tbody><tr><td>No comparable stocks in this run.</td></tr></tbody></table></div>`;
  return `
    <div class="mini-table recommendation-compare-table">
      <table>
        <thead><tr><th>Comparable</th><th>Action</th><th>Score</th><th>Confidence</th><th>Why it differs</th></tr></thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td><strong>${escapeHtml(row.symbol)}</strong><small>${escapeHtml(row.company)}</small></td>
              <td>${escapeHtml(row.action)}</td>
              <td>${escapeHtml(row.score)}</td>
              <td>${escapeHtml(row.confidence)}</td>
              <td>${escapeHtml(row.rationale)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function recommendationChatMessages(symbol) {
  if (!recommendationChatBySymbol.has(symbol)) {
    recommendationChatBySymbol.set(symbol, [
      {
        role: "assistant",
        text: "Ask why this stock is ranked here, what price zone to watch, where the stop-loss proxy is, or how it compares to peers.",
      },
    ]);
  }
  return recommendationChatBySymbol.get(symbol);
}

function answerRecommendationQuestion(question, thesis) {
  const text = String(question || "").toLowerCase();
  const stock = thesis.stock;
  const plan = thesis.pricePlan;
  if (!text.trim()) return "Ask about recommendation rationale, price zone, stop loss, peer comparison, or risk.";
  if (text.includes("why") || text.includes("rationale") || text.includes("good")) {
    return `${stock.symbol} is rated ${actionLabel(stock.action)} because ${thesis.positives.join(" ")} Key engine reads: ${thesis.strength.join(" ")}`;
  }
  if (text.includes("price") || text.includes("entry") || text.includes("buy")) {
    return `${plan.entryText} Reference price: ${plan.reference ? money(plan.reference) : "N/A"}. This is a research zone, not an order ticket.`;
  }
  if (text.includes("stop") || text.includes("loss") || text.includes("risk")) {
    return `${plan.stopText} Main watchouts: ${thesis.watchouts.join(" ")}`;
  }
  if (text.includes("compare") || text.includes("peer") || text.includes("other")) {
    const peers = comparableRecommendationRows(stock.symbol).map((row) => `${row.symbol}: ${row.action}, score ${row.score}`).join("; ");
    return peers || "No comparable rows are available in this run.";
  }
  return `${stock.symbol} trade-desk summary: ${thesis.narrative} Use the peer table, stop-loss proxy, and live quote validation before acting.`;
}

function renderRecommendationChat(messages) {
  return messages.map((message) => `
    <div class="chat-message ${message.role === "user" ? "user" : "assistant"}">
      <span>${message.role === "user" ? "You" : "Recommendation Bot"}</span>
      <p>${escapeHtml(message.text)}</p>
    </div>
  `).join("");
}

function renderRecommendationWorkbench(stock) {
  if (!els.recommendationWorkbench) return;
  if (!stock) {
    els.recommendationWorkbench.innerHTML = `<div class="empty-state">Select a stock recommendation to view trade-desk rationale.</div>`;
    return;
  }

  const thesis = recommendationThesis(stock);
  const plan = thesis.pricePlan;
  const messages = recommendationChatMessages(stock.symbol);
  els.recommendationWorkbench.innerHTML = `
    <section class="recommendation-lab" aria-label="Stock recommendation rationale">
      <div class="recommendation-hero">
        <div>
          <p class="eyebrow">Recommendation Rationale</p>
          <h3>${escapeHtml(stock.symbol)} ${pill(stock.action)}</h3>
          <p>${escapeHtml(thesis.narrative)}</p>
        </div>
        <div class="recommendation-scorebox">
          <span>Composite Score</span>
          <strong>${score(stock.composite_score)}</strong>
          <small>${confidence(stock.confidence)} confidence</small>
        </div>
      </div>

      <div class="metric-grid compact recommendation-metrics">
        ${MetricCard("Reference Price", plan.reference ? money(plan.reference) : "N/A", "Latest close / proxy")}
        ${MetricCard("Entry / Watch Zone", plan.entryText, "Research level")}
        ${MetricCard("Stop / Invalidation", plan.stopText, "Risk line")}
      </div>

      <div class="recommendation-rationale-grid">
        <article><h4>Why This Stock</h4><ul>${renderList(thesis.positives)}</ul></article>
        <article><h4>Risk Plan</h4><ul>${renderList(thesis.watchouts)}</ul></article>
        <article><h4>Execution Checks</h4><ul>${renderList([
          "Confirm live quote, spread, and volume before using any level.",
          plan.upsideCheckpoint ? `First upside checkpoint: ${money(plan.upsideCheckpoint)}.` : "Upside checkpoint needs price data.",
          plan.riskPerShare ? `Approximate model risk per share: ${money(plan.riskPerShare)}.` : "Risk per share unavailable until price data loads.",
          "This is not financial advice. Use it for research only.",
        ])}</ul></article>
      </div>

      <section class="recommendation-compare">
        <div>
          <p class="eyebrow">Comparable Stocks</p>
          <h3>How It Ranks Against This Basket</h3>
        </div>
        ${RecommendationComparisonTable(comparableRecommendationRows(stock.symbol))}
      </section>

      <section class="recommendation-chat" aria-label="Recommendation rationale chatbot">
        <div>
          <p class="eyebrow">Text Rationale</p>
          <h3>Ask the Recommendation Bot</h3>
          <p class="analysis-note">Local rule-based explainer. It summarizes why the stock was ranked, what price zone to watch, and what invalidates the setup.</p>
        </div>
        <div class="quick-questions" aria-label="Suggested recommendation questions">
          <button type="button" data-recommendation-question="Why is this a good recommendation?">Why this stock?</button>
          <button type="button" data-recommendation-question="At what price should I watch it?">Price zone?</button>
          <button type="button" data-recommendation-question="What is the stop loss and risk?">Stop loss?</button>
          <button type="button" data-recommendation-question="How does it compare to other stocks?">Compare peers?</button>
        </div>
        <div class="chat-log recommendation-chat-log" aria-live="polite">${renderRecommendationChat(messages)}</div>
        <form class="recommendation-chat-form" data-recommendation-chat-form>
          <input type="text" name="question" placeholder="Ask about why, price, stop loss, peer comparison, or risk" aria-label="Ask the recommendation bot" />
          <button type="submit" data-ask-recommendation-chat>Ask</button>
        </form>
      </section>
    </section>
  `;
}

function StockSelector(stocks) {
  if (!stocks.length) {
    els.stockSelector.innerHTML = `<span>Select a stock to view analysis.</span>`;
    return;
  }
  els.stockSelector.innerHTML = stocks
    .map(
      (stock) => `
        <button type="button" class="${stock.symbol === activeSymbol ? "active" : ""}" aria-pressed="${stock.symbol === activeSymbol ? "true" : "false"}" data-select-symbol="${escapeHtml(stock.symbol)}">
          ${escapeHtml(stock.symbol)}
          <small>${escapeHtml(actionLabel(stock.action))}</small>
        </button>
      `,
    )
    .join("");
}

function optionKey(option) {
  if (!option) return "none";
  return `${option.symbol}:${option.action}:${option.strike}:${option.expiry_days}`;
}

function currentOptions() {
  return currentRun?.actions?.option_recommendation || [];
}

function optionForSymbol(symbol, options = currentOptions()) {
  return (options || []).find((option) => option.symbol === symbol);
}

function setActiveOptionSymbol(symbol) {
  if (!symbol || activeOptionSymbol === symbol) return;
  activeOptionSymbol = symbol;
  optionScenarioState = {};
}

function ensureActiveOption(options) {
  if (!options.length) {
    activeOptionSymbol = null;
    optionScenarioState = {};
    return null;
  }
  const preferredSymbol =
    activeOptionSymbol && optionForSymbol(activeOptionSymbol, options)
      ? activeOptionSymbol
      : activeSymbol && optionForSymbol(activeSymbol, options)
        ? activeSymbol
        : options[0].symbol;
  setActiveOptionSymbol(preferredSymbol);
  return optionForSymbol(preferredSymbol, options) || options[0];
}

function optionContext(option) {
  const payload = currentDeepPayload?.symbol === option?.symbol ? currentDeepPayload : null;
  return { payload, run: currentRun };
}

function optionChatMessages(symbol) {
  if (!optionChatBySymbol.has(symbol)) {
    optionChatBySymbol.set(symbol, [
      {
        role: "assistant",
        text: "Ask why this is a call or put, where break-even sits, what the max risk is, or how to assess premium quality.",
      },
    ]);
  }
  return optionChatBySymbol.get(symbol);
}

function renderOptions(options) {
  if (!options.length) {
    els.optionsList.innerHTML = "No option recommendations.";
    renderOptionWorkbench(null);
    return;
  }
  const activeOption = ensureActiveOption(options);
  els.optionsList.innerHTML = options
    .map(
      (option) => `
        <button type="button" class="option-card ${option.symbol === activeOption?.symbol ? "active" : ""}" data-option-symbol="${escapeHtml(option.symbol)}" data-open-symbol="${escapeHtml(option.symbol)}" aria-pressed="${option.symbol === activeOption?.symbol ? "true" : "false"}">
          <span>${escapeHtml(option.symbol)} ${pill(option.action)}</span>
          <strong>$${Number(option.strike || 0).toFixed(2)} strike / ${escapeHtml(option.expiry_days || "--")}D</strong>
          <small>${escapeHtml(option.rationale || "")}</small>
        </button>
      `,
    )
    .join("");
  renderOptionWorkbench(activeOption);
}

function renderScenarioRows(rows) {
  if (!rows?.length) return `<tr><td colspan="5">No scenario rows available.</td></tr>`;
  return rows
    .map(
      (row) => `
        <tr class="${row.pl > 0 ? "positive-row" : row.pl < 0 ? "negative-row" : ""}">
          <td>${window.OptionScenarioService.pct(row.move)}</td>
          <td>${window.OptionScenarioService.money(row.stockPrice)}</td>
          <td>${window.OptionScenarioService.money(row.intrinsic)}</td>
          <td>${window.OptionScenarioService.money(row.pl)}</td>
          <td>${window.OptionScenarioService.pct(row.returnOnRisk)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderOptionChat(messages) {
  return messages
    .map(
      (message) => `
        <div class="chat-message ${message.role === "user" ? "user" : "assistant"}">
          <span>${message.role === "user" ? "You" : "Option Rationale Bot"}</span>
          <p>${escapeHtml(message.text)}</p>
        </div>
      `,
    )
    .join("");
}

function renderOptionWorkbench(option) {
  if (!option) {
    els.optionWorkbench.innerHTML = `<div class="empty-state">Select an option recommendation to run scenarios.</div>`;
    return;
  }
  if (!window.OptionScenarioService) {
    els.optionWorkbench.innerHTML = `<div class="provider-failure caution"><strong>Option scenario service unavailable.</strong><p>Refresh the app so the scenario engine can load.</p></div>`;
    return;
  }
  const context = optionContext(option);
  const scenario = window.OptionScenarioService.buildOptionScenario(option, optionScenarioState, context);
  const rationale = window.OptionScenarioService.getOptionRationale(option, context);
  const messages = optionChatMessages(optionKey(option));
  const breakEven = scenario.breakeven == null ? "N/A" : window.OptionScenarioService.money(scenario.breakeven);
  const selectedReturn = window.OptionScenarioService.pct(scenario.selected.returnOnRisk);
  els.optionWorkbench.innerHTML = `
    <section class="option-lab" aria-label="Option scenario analysis">
      <div class="option-lab-header">
        <div>
          <p class="eyebrow">Scenario Analysis</p>
          <h3>${escapeHtml(scenario.symbol)} ${RatingBadge(rationale.action)}</h3>
          <p>${escapeHtml(rationale.directionalRead)}</p>
        </div>
        <div class="option-risk-box">
          <span>Max Premium Risk</span>
          <strong>${window.OptionScenarioService.money(scenario.maxLoss)}</strong>
          <small>${escapeHtml(scenario.contracts)} contract(s), ${escapeHtml(scenario.expiryDays)}D proxy</small>
        </div>
      </div>
      <div class="metric-grid compact option-metrics">
        ${MetricCard("Strike", window.OptionScenarioService.money(scenario.strike), "Model proxy")}
        ${MetricCard("Break-even", breakEven, "At expiration")}
        ${MetricCard("Premium", window.OptionScenarioService.money(scenario.premium), "Use broker bid/ask when available")}
      </div>
      <form class="option-scenario-form" data-option-scenario-form>
        <label>Scenario price <input type="number" step="0.01" min="0" name="scenarioPrice" value="${escapeHtml(scenario.inputs.scenarioPrice)}" /></label>
        <label>Premium paid <input type="number" step="0.01" min="0" name="premium" value="${escapeHtml(scenario.inputs.premium)}" /></label>
        <label>Contracts <input type="number" step="1" min="1" name="contracts" value="${escapeHtml(scenario.inputs.contracts)}" /></label>
        <button type="button" data-run-option-scenario>Run Scenario</button>
      </form>
      <div class="option-result-strip">
        <article>
          <span>Scenario P/L</span>
          <strong class="${scenario.selected.pl >= 0 ? "positive-text" : "negative-text"}">${window.OptionScenarioService.money(scenario.selected.pl)}</strong>
          <small>${selectedReturn} return on premium risk</small>
        </article>
        <article>
          <span>Max Profit</span>
          <strong>${escapeHtml(scenario.maxProfit)}</strong>
          <small>Before spread, fees, and IV changes</small>
        </article>
      </div>
      <div class="mini-table option-table" role="region" aria-label="Option stress scenarios">
        <table>
          <thead><tr><th>Stock Move</th><th>Stock Price</th><th>Intrinsic</th><th>P/L</th><th>Return on Risk</th></tr></thead>
          <tbody>${renderScenarioRows(scenario.rows)}</tbody>
        </table>
      </div>
      <div class="option-rationale-grid">
        <article><h4>Why ${escapeHtml(window.OptionScenarioService.actionLabel(rationale.action))}</h4><ul>${renderList(rationale.drivers)}</ul></article>
        <article><h4>How to Assess It</h4><ul>${renderList(rationale.assessmentChecklist)}</ul></article>
        <article><h4>Risk Notes</h4><ul>${renderList(rationale.riskNotes)}</ul></article>
      </div>
      <section class="option-chat" aria-label="Option rationale chatbot">
        <div>
          <p class="eyebrow">Text Chatbot</p>
          <h3>Ask the Option Rationale Bot</h3>
          <p class="analysis-note">Local rule-based rationale bot. It explains the current option signal and scenario math; it does not place trades.</p>
        </div>
        <div class="quick-questions" aria-label="Suggested option questions">
          <button type="button" data-option-question="Why is this a call or put?">Why this bias?</button>
          <button type="button" data-option-question="What is my max risk?">Max risk?</button>
          <button type="button" data-option-question="How do I assess premium and bid ask spread?">Assess premium?</button>
        </div>
        <div class="chat-log" aria-live="polite">${renderOptionChat(messages)}</div>
        <form class="option-chat-form" data-option-chat-form>
          <input type="text" name="question" placeholder="Ask about rationale, break-even, risk, IV, strike, or expiry" aria-label="Ask the option rationale bot" />
          <button type="submit" data-ask-option-chat>Ask</button>
        </form>
      </section>
      <p class="option-disclaimer">This is not financial advice. Use this for research only. Always verify live bid/ask, open interest, implied volatility, liquidity, and max loss in your broker before trading.</p>
    </section>
  `;
}

function assessmentFor(symbol, items) {
  return (items || []).find((item) => item.symbol === symbol);
}

function renderAssessment(symbol, item, type) {
  if (!item) return `No ${type} assessment for ${escapeHtml(symbol)}.`;
  return `
    <div class="agent-result">
      <strong>${escapeHtml(symbol)} score ${score(item.score)}</strong>
      <p>${escapeHtml(item.summary || "")}</p>
    </div>
  `;
}

function topAssessment(items, label) {
  const sorted = [...(items || [])].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  const first = sorted[0];
  if (!first) return `Run analysis to inspect ${label}.`;
  return `
    <div class="agent-result">
      <strong>${escapeHtml(first.symbol)} leads at ${score(first.score)}</strong>
      <p>${escapeHtml(first.summary || "")}</p>
    </div>
  `;
}

function renderAgentCards(payload) {
  const actions = payload?.actions || {};
  const symbol = activeSymbol || actions.stock_recommendation?.[0]?.symbol;
  if (!symbol) {
    els.sectorAgent.innerHTML = topAssessment(actions.sector_analysis, "sector momentum");
    els.fundamentalAgent.innerHTML = topAssessment(actions.fundamental_analysis, "fundamentals");
    els.technicalAgent.innerHTML = topAssessment(actions.technical_analysis, "technicals");
    return;
  }
  els.sectorAgent.innerHTML = renderAssessment(symbol, assessmentFor(symbol, actions.sector_analysis), "sector");
  els.fundamentalAgent.innerHTML = renderAssessment(symbol, assessmentFor(symbol, actions.fundamental_analysis), "fundamental");
  els.technicalAgent.innerHTML = renderAssessment(symbol, assessmentFor(symbol, actions.technical_analysis), "technical");
}

function renderRun(payload) {
  currentRun = payload;
  const actions = payload.actions || {};
  const stocks = actions.stock_recommendation || [];
  const options = actions.option_recommendation || [];
  if (!activeSymbol && stocks[0]) activeSymbol = stocks[0].symbol;
  renderStocks(stocks);
  StockSelector(stocks);
  renderRecommendationWorkbench(stockForSymbol(activeSymbol, stocks) || stocks[0]);
  renderOptions(options);
  renderRunSummary(stocks, options, payload.meta);
  renderAgentCards(payload);
}

function AnalysisTabs(activeTab = activeAnalysisTab) {
  els.analysisTabs.querySelectorAll("[data-analysis-tab]").forEach((button) => {
    const active = button.dataset.analysisTab === activeTab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
    button.setAttribute("tabindex", active ? "0" : "-1");
  });
}

function AnalysisPanel({ title, rating, body }) {
  return `
    <section class="analysis-panel" role="tabpanel">
      <div class="analysis-panel-header">
        <div>
          <p class="eyebrow">Selected Stock</p>
          <h3>${escapeHtml(activeSymbol || "")} ${escapeHtml(title || "")}</h3>
        </div>
        ${rating ? RatingBadge(rating) : ""}
      </div>
      ${body}
    </section>
  `;
}

function PeerComparisonTable(rows, mode = "peers") {
  if (!rows || !rows.length) return `<div class="mini-table"><table><tbody><tr><td>No comparison rows available.</td></tr></tbody></table></div>`;
  const leaderMode = mode === "leaders";
  return `
    <div class="mini-table">
      <table>
        <thead>
          <tr>
            ${leaderMode
              ? "<th>Symbol</th><th>Action</th><th>Score</th><th>Rationale</th>"
              : "<th>Symbol</th><th>P/E</th><th>Sales</th><th>Growth</th><th>Action</th>"}
          </tr>
        </thead>
        <tbody>
          ${rows
            .map((row) => leaderMode
              ? `<tr><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.action)}</td><td>${escapeHtml(row.score)}</td><td>${escapeHtml(row.rationale)}</td></tr>`
              : `<tr><td>${escapeHtml(row.symbol)}</td><td>${escapeHtml(row.peRatio)}</td><td>${escapeHtml(row.sales)}</td><td>${escapeHtml(row.growth)}</td><td>${escapeHtml(row.action)}</td></tr>`)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function metricGridFromObject(metrics) {
  return `
    <div class="metric-grid analyst-metrics">
      ${Object.entries(metrics || {})
        .map(([label, value]) => MetricCard(label.replace(/([A-Z])/g, " $1").trim(), value))
        .join("")}
    </div>
  `;
}

function agentSignal(agent, fallbackLabel) {
  if (!agent) return "";
  return `
    <article class="engine-signal">
      <span>${escapeHtml(fallbackLabel)}</span>
      <strong>${escapeHtml(agent.verdict || "View")} · ${score(agent.score)}</strong>
      <p>${escapeHtml(agent.summary || "")}</p>
    </article>
  `;
}

function TradeDeskSummary(summary) {
  return `
    <section class="trade-summary">
      <div>
        <p class="eyebrow">Overall Rating</p>
        <h3>${escapeHtml(summary.ticker)} ${RatingBadge(summary.overallRating)}</h3>
        <p>${escapeHtml(summary.thesis)}</p>
      </div>
      <div class="summary-action">
        <span>Watchlist Action</span>
        <strong>${escapeHtml(summary.suggestedWatchlistAction)}</strong>
        <small>Overall score ${score(summary.overallScore)}</small>
      </div>
      <div class="metric-grid compact">
        ${MetricCard("Sector Score", score(summary.sectorScore), "Theme and group strength")}
        ${MetricCard("Fundamental Score", score(summary.fundamentalScore), "Valuation, growth, cash flow")}
        ${MetricCard("Technical Score", score(summary.technicalScore), "Trend and timing")}
      </div>
      <div class="deep-columns">
        <article><h4>Positive Drivers</h4><ul>${renderList(summary.positives)}</ul></article>
        <article><h4>Risk Flags</h4><ul>${renderList(summary.risks)}</ul></article>
        <article><h4>Next Checks</h4><ul>${renderList(summary.nextChecks)}</ul></article>
      </div>
      <div class="analysis-explainer">
        <strong>Disclaimer</strong>
        <p>${escapeHtml(summary.disclaimer)}</p>
      </div>
    </section>
  `;
}

function renderOverviewPanel(summary) {
  els.deepAnalysis.innerHTML = AnalysisPanel({
    title: "Overview",
    rating: summary.overallRating,
    body: `
      <div class="analysis-explainer">
        <strong>How to read this overview</strong>
        <p>The overview combines sector, fundamental, and technical engines into one trade-desk read for the selected stock. Use the tabs above to inspect each engine in detail.</p>
      </div>
      ${TradeDeskSummary(summary)}
      <div class="engine-signal-grid" aria-label="Background agent outputs">
        ${agentSignal(currentDeepPayload.agent_breakdown?.sector, "Sector Engine")}
        ${agentSignal(currentDeepPayload.agent_breakdown?.fundamental, "Fundamental Engine")}
        ${agentSignal(currentDeepPayload.agent_breakdown?.technical, "Technical Engine")}
        ${agentSignal(currentDeepPayload.agent_breakdown?.option, "Options Engine")}
      </div>
    `,
  });
}

function renderSectorPanel(data) {
  els.deepAnalysis.innerHTML = AnalysisPanel({
    title: data.title,
    rating: data.rating,
    body: `
      <div class="metric-grid analyst-metrics">
        ${MetricCard("Company Sector", data.sector, data.company)}
        ${MetricCard("Industry", data.industry, "Classification from current profile feed")}
        ${MetricCard("Relative Strength", data.relativeSectorStrength, "Sector score")}
        ${MetricCard("Final Sector Rating", data.rating, "Bullish / Neutral / Bearish")}
      </div>
      <section class="analyst-section"><h4>Sector Overview</h4><p>${escapeHtml(data.overview)}</p></section>
      <section class="analyst-section"><h4>Key Sector Trends</h4><ul>${renderList(data.keyTrends)}</ul></section>
      <section class="analyst-section"><h4>Peer Comparison</h4>${PeerComparisonTable(data.peerComparison)}</section>
      <section class="analyst-section"><h4>Sector Leaders</h4>${PeerComparisonTable(data.sectorLeaders, "leaders")}</section>
      <section class="analyst-section"><h4>Sector Risks</h4><ul>${renderList(data.risks)}</ul></section>
      <section class="analyst-section"><h4>Sector Opportunity Summary</h4><p>${escapeHtml(data.opportunitySummary)}</p><p>${escapeHtml(data.interpretation)}</p></section>
    `,
  });
}

function renderFundamentalPanel(data) {
  els.deepAnalysis.innerHTML = AnalysisPanel({
    title: data.title,
    rating: data.rating,
    body: `
      ${metricGridFromObject(data.metrics)}
      <section class="analyst-section"><h4>Valuation View</h4><p>${escapeHtml(data.valuationView)}. ${escapeHtml(data.narrative)}</p></section>
      <div class="deep-columns">
        <article><h4>Bull Case</h4><ul>${renderList(data.bullCase)}</ul></article>
        <article><h4>Bear Case</h4><ul>${renderList(data.bearCase)}</ul></article>
        <article><h4>Final Fundamental Rating</h4>${RatingBadge(data.rating)}<p>${escapeHtml(data.narrative)}</p></article>
      </div>
    `,
  });
}

function renderTechnicalPanel(data) {
  els.deepAnalysis.innerHTML = AnalysisPanel({
    title: data.title,
    rating: data.rating,
    body: `
      <div class="metric-grid analyst-metrics">
        ${MetricCard("Current Price", data.currentPrice, "Latest close / live proxy")}
        ${MetricCard("Daily Trend", data.dailyTrend, "Model signal")}
        ${MetricCard("Weekly Trend", data.weeklyTrend, "Momentum proxy")}
        ${MetricCard("Risk Level", data.riskLevel, "Volatility and RSI")}
      </div>
      <div class="table-pair">
        <section class="analyst-section"><h4>Support Levels</h4><ul>${renderList(data.supportLevels)}</ul></section>
        <section class="analyst-section"><h4>Resistance Levels</h4><ul>${renderList(data.resistanceLevels)}</ul></section>
      </div>
      <section class="analyst-section"><h4>Moving Averages</h4><ul><li>20D: ${escapeHtml(data.movingAverages.ma20)}</li><li>50D: ${escapeHtml(data.movingAverages.ma50)}</li><li>200D: ${escapeHtml(data.movingAverages.ma200)}</li></ul></section>
      <div class="metric-grid compact">
        ${MetricCard("RSI", data.rsi, "Overbought / oversold context")}
        ${MetricCard("MACD", data.macd, "Momentum proxy from current feed")}
        ${MetricCard("Volume Trend", data.volumeTrend, "If available from provider")}
        ${MetricCard("Breakout Signal", data.breakoutSignal, "Breakout / breakdown read")}
      </div>
      <section class="analyst-section"><h4>Short-Term Trade Setup</h4><p>${escapeHtml(data.shortTermSetup)}</p><p>${escapeHtml(data.narrative)}</p></section>
      <section class="analyst-section"><h4>Final Technical Rating</h4>${RatingBadge(data.rating)}</section>
    `,
  });
}

function renderTradeDeskPanel(summary) {
  els.deepAnalysis.innerHTML = AnalysisPanel({
    title: summary.title,
    rating: summary.overallRating,
    body: TradeDeskSummary(summary),
  });
}

function renderDeepAnalysis(payload) {
  currentDeepPayload = payload;
  if (!payload || payload.error) {
    els.deepAnalysis.innerHTML = `<div class="provider-failure"><strong>Deep analysis unavailable.</strong><p>${escapeHtml(payload?.error || "Select a stock to view analysis.")}</p></div>`;
    return;
  }
  activeSymbol = payload.symbol;
  if (optionForSymbol(activeSymbol)) setActiveOptionSymbol(activeSymbol);
  renderStocks(currentRun?.actions?.stock_recommendation || []);
  StockSelector(currentRun?.actions?.stock_recommendation || []);
  renderRecommendationWorkbench(stockForSymbol(activeSymbol));
  renderOptions(currentOptions());
  renderAgentCards(currentRun);
  AnalysisTabs();
  renderActiveAnalysis();
}

async function renderActiveAnalysis() {
  AnalysisTabs();
  if (!activeSymbol) {
    els.deepAnalysis.innerHTML = `<div class="empty-state">Select a stock to view analysis.</div>`;
    return;
  }
  if (!currentDeepPayload || currentDeepPayload.symbol !== activeSymbol) {
    els.deepAnalysis.innerHTML = `<div class="loading">Loading analysis for ${escapeHtml(activeSymbol)}...</div>`;
    return;
  }
  if (!window.StockAnalysisService) {
    els.deepAnalysis.innerHTML = `<div class="provider-failure"><strong>Analysis service unavailable.</strong><p>Refresh the app so the stock analysis service can load.</p></div>`;
    return;
  }
  try {
    const context = { payload: currentDeepPayload, run: currentRun };
    if (activeAnalysisTab === "sector") return renderSectorPanel(await window.StockAnalysisService.getSectorAnalysis(activeSymbol, context));
    if (activeAnalysisTab === "fundamental") return renderFundamentalPanel(await window.StockAnalysisService.getFundamentalAnalysis(activeSymbol, context));
    if (activeAnalysisTab === "technical") return renderTechnicalPanel(await window.StockAnalysisService.getTechnicalAnalysis(activeSymbol, context));
    const summary = await window.StockAnalysisService.getFullStockDeskAnalysis(activeSymbol, context);
    if (activeAnalysisTab === "trade-desk") return renderTradeDeskPanel(summary);
    return renderOverviewPanel(summary);
  } catch (error) {
    els.deepAnalysis.innerHTML = `<div class="provider-failure"><strong>Analysis render failed.</strong><p>${escapeHtml(compactErrorMessage(error))}</p></div>`;
  }
}

async function loadDeepAnalysis(symbol) {
  if (!symbol) return;
  activeSymbol = symbol;
  els.deepAnalysis.innerHTML = `<div class="loading">Loading analysis for ${escapeHtml(symbol)}...</div>`;
  renderStocks(currentRun?.actions?.stock_recommendation || []);
  StockSelector(currentRun?.actions?.stock_recommendation || []);
  renderRecommendationWorkbench(stockForSymbol(symbol));
  try {
    const mode = currentMode === "live" ? "live" : "sample";
    const payload = await apiGet("/api/deep-analysis", { symbol, mode });
    renderDeepAnalysis(payload);
  } catch (error) {
    renderDeepAnalysis({ symbol, error: compactErrorMessage(error) });
  }
}

async function runBackendAnalysis() {
  const symbols = currentSymbols().join(",");
  const mode = currentMode === "live" ? "live" : "sample";
  const modeName = currentMode === "live" ? "Live market analysis running." : currentMode === "alpha" ? "Alpha Vantage mode needs backend credentials; using sample-safe analysis." : "Sample analysis running.";
  setFeedStatus("loading", modeName, "Calling the stock multi-agent backend.");
  const payload = await apiGet("/api/analyze", {
    symbols,
    top: els.top.value || "40",
    mode,
  });
  renderRun(payload);
  els.lastRun.textContent = new Date().toLocaleTimeString();
  const warnings = runMetaWarnings(payload.meta);
  if (currentMode === "alpha") {
    setFeedStatus("warn", "Alpha Vantage direct client was not used in this restored local file.", "Use the backend/live feed or connect Alpha Vantage server-side for production.");
  } else if (warnings) {
    setFeedStatus("warn", `${payload.meta?.provider || "Provider"} completed with fallback warnings.`, warnings);
  } else {
    setFeedStatus("ok", `${payload.meta?.provider || "Stock feed"} connected.`, providerRunDetails(payload.meta));
  }
  const first = payload.actions?.stock_recommendation?.[0];
  if (first) await loadDeepAnalysis(first.symbol);
}

async function runAnalysis() {
  setBusy(true);
  updateModeUI();
  try {
    await runBackendAnalysis();
  } catch (error) {
    setFeedStatus("error", "Stock API failed.", compactErrorMessage(error));
    els.stockTable.innerHTML = `<tr><td colspan="7">Analysis failed: ${escapeHtml(compactErrorMessage(error))}</td></tr>`;
    renderRecommendationWorkbench(null);
    els.optionsList.innerHTML = "No option recommendations.";
    renderOptionWorkbench(null);
  } finally {
    setBusy(false);
  }
}

async function loadUniverse() {
  try {
    const payload = await apiGet("/api/universe");
    universe = payload.symbols || [];
    els.universeCount.textContent = universe.length || "--";
  } catch (error) {
    els.universeCount.textContent = "--";
    setFeedStatus("warn", "Universe endpoint is not reachable yet.", compactErrorMessage(error));
  }
}

function applyOptionScenarioForm(form) {
  if (!form) return;
  optionScenarioState = {
    scenarioPrice: form.querySelector("input[name='scenarioPrice']")?.value,
    premium: form.querySelector("input[name='premium']")?.value,
    contracts: form.querySelector("input[name='contracts']")?.value,
  };
  renderOptionWorkbench(optionForSymbol(activeOptionSymbol));
}

function submitOptionChatQuestion(form) {
  const option = optionForSymbol(activeOptionSymbol);
  if (!form || !option || !window.OptionScenarioService) return;
  const input = form.querySelector("input[name='question']");
  const question = input?.value?.trim();
  if (!question) return;
  const scenario = window.OptionScenarioService.buildOptionScenario(option, optionScenarioState, optionContext(option));
  const messages = optionChatMessages(optionKey(option));
  messages.push({ role: "user", text: question });
  messages.push({
    role: "assistant",
    text: window.OptionScenarioService.answerOptionQuestion(question, option, scenario, optionContext(option)),
  });
  input.value = "";
  renderOptionWorkbench(option);
}

els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  currentScenario = "custom";
  setScenario("custom");
  runAnalysis();
});

document.querySelectorAll(".feed-btn").forEach((button) => {
  button.addEventListener("click", () => {
    currentMode = button.dataset.mode;
    updateModeUI();
    runAnalysis();
  });
});

els.scenarioActions.addEventListener("click", (event) => {
  const button = event.target.closest("[data-scenario]");
  if (!button) return;
  setScenario(button.dataset.scenario);
  runAnalysis();
});

els.optionWorkbench.addEventListener("submit", (event) => {
  const chatForm = event.target.closest("[data-option-chat-form]");
  if (!chatForm) return;
  event.preventDefault();
  submitOptionChatQuestion(chatForm);
});

els.optionWorkbench.addEventListener("click", (event) => {
  const scenarioButton = event.target.closest("[data-run-option-scenario]");
  if (scenarioButton) {
    event.preventDefault();
    applyOptionScenarioForm(scenarioButton.closest("[data-option-scenario-form]"));
    return;
  }
  const chatButton = event.target.closest("[data-ask-option-chat]");
  if (chatButton) {
    event.preventDefault();
    submitOptionChatQuestion(chatButton.closest("[data-option-chat-form]"));
    return;
  }
  const questionButton = event.target.closest("[data-option-question]");
  if (!questionButton) return;
  const option = optionForSymbol(activeOptionSymbol);
  if (!option || !window.OptionScenarioService) return;
  const question = questionButton.dataset.optionQuestion;
  const scenario = window.OptionScenarioService.buildOptionScenario(option, optionScenarioState, optionContext(option));
  const messages = optionChatMessages(optionKey(option));
  messages.push({ role: "user", text: question });
  messages.push({
    role: "assistant",
    text: window.OptionScenarioService.answerOptionQuestion(question, option, scenario, optionContext(option)),
  });
  renderOptionWorkbench(option);
});

function submitRecommendationChatQuestion(form) {
  const stock = stockForSymbol(activeSymbol) || currentStocks()[0];
  if (!form || !stock) return;
  const input = form.querySelector("input[name='question']");
  const question = input?.value?.trim();
  if (!question) return;
  const thesis = recommendationThesis(stock);
  const messages = recommendationChatMessages(stock.symbol);
  messages.push({ role: "user", text: question });
  messages.push({ role: "assistant", text: answerRecommendationQuestion(question, thesis) });
  input.value = "";
  renderRecommendationWorkbench(stock);
}

els.recommendationWorkbench?.addEventListener("submit", (event) => {
  const chatForm = event.target.closest("[data-recommendation-chat-form]");
  if (!chatForm) return;
  event.preventDefault();
  submitRecommendationChatQuestion(chatForm);
});

els.recommendationWorkbench?.addEventListener("click", (event) => {
  const chatButton = event.target.closest("[data-ask-recommendation-chat]");
  if (chatButton) {
    event.preventDefault();
    submitRecommendationChatQuestion(chatButton.closest("[data-recommendation-chat-form]"));
    return;
  }
  const questionButton = event.target.closest("[data-recommendation-question]");
  if (!questionButton) return;
  const stock = stockForSymbol(activeSymbol) || currentStocks()[0];
  if (!stock) return;
  const thesis = recommendationThesis(stock);
  const question = questionButton.dataset.recommendationQuestion;
  const messages = recommendationChatMessages(stock.symbol);
  messages.push({ role: "user", text: question });
  messages.push({ role: "assistant", text: answerRecommendationQuestion(question, thesis) });
  renderRecommendationWorkbench(stock);
});

document.body.addEventListener("click", (event) => {
  const optionButton = event.target.closest("[data-option-symbol]");
  if (optionButton) {
    setActiveOptionSymbol(optionButton.dataset.optionSymbol);
    renderOptions(currentOptions());
    loadDeepAnalysis(optionButton.dataset.optionSymbol);
    return;
  }
  const selectButton = event.target.closest("[data-select-symbol]");
  if (selectButton) {
    loadDeepAnalysis(selectButton.dataset.selectSymbol);
    return;
  }
  const openButton = event.target.closest("[data-open-symbol]");
  if (!openButton) return;
  loadDeepAnalysis(openButton.dataset.openSymbol);
});

function setActiveAnalysisTab(tabName) {
  activeAnalysisTab = tabName || "overview";
  renderActiveAnalysis();
}

els.analysisTabs.addEventListener("click", (event) => {
  const tab = event.target.closest("[data-analysis-tab]");
  if (!tab) return;
  setActiveAnalysisTab(tab.dataset.analysisTab);
});

els.analysisTabs.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...els.analysisTabs.querySelectorAll("[data-analysis-tab]")];
  const currentIndex = tabs.findIndex((tab) => tab.dataset.analysisTab === activeAnalysisTab);
  let nextIndex = currentIndex < 0 ? 0 : currentIndex;
  if (event.key === "ArrowRight") nextIndex = (nextIndex + 1) % tabs.length;
  if (event.key === "ArrowLeft") nextIndex = (nextIndex - 1 + tabs.length) % tabs.length;
  if (event.key === "Home") nextIndex = 0;
  if (event.key === "End") nextIndex = tabs.length - 1;
  event.preventDefault();
  tabs[nextIndex]?.focus();
  setActiveAnalysisTab(tabs[nextIndex]?.dataset.analysisTab);
});

updateModeUI();
setScenario("custom");
loadUniverse().then(() => runAnalysis());

// ── Portfolio & Reports ──────────────────────────────────────────────────────

const SECTOR_COLORS = [
  "#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6",
  "#ec4899","#14b8a6","#f97316","#84cc16","#3b82f6","#a78bfa",
];

function dollarPnl(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "N/A";
  const prefix = n >= 0 ? "+" : "";
  return `${prefix}$${Math.abs(n).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})}`;
}

function pctPnl(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "N/A";
  const prefix = n >= 0 ? "+" : "";
  return `${prefix}${n.toFixed(2)}%`;
}

function strategyPill(strategy) {
  const text = String(strategy || "HOLD").replaceAll("_", " ");
  const normalized = text.toLowerCase();
  const className = normalized.includes("buy")
    ? "good"
    : normalized.includes("sell") || normalized.includes("trim")
      ? "bad"
      : "neutral";
  return `<span class="pill ${className}">${escapeHtml(text)}</span>`;
}

function renderPortfolio(data) {
  const tbody = document.querySelector("#portfolio-table");
  const summary = document.querySelector("#portfolio-summary");
  if (!data || !data.positions) {
    tbody.innerHTML = `<tr><td colspan="9">No portfolio data.</td></tr>`;
    return;
  }

  // summary bar
  document.querySelector("#port-total-value").textContent = money(data.total_value);
  document.querySelector("#port-cash").textContent = money(data.cash);
  const pnlEl = document.querySelector("#port-total-pnl");
  pnlEl.textContent = dollarPnl(data.total_pnl);
  pnlEl.className = data.total_pnl >= 0 ? "up" : "down";
  const pctEl = document.querySelector("#port-total-pct");
  pctEl.textContent = pctPnl(data.total_pnl_pct);
  pctEl.className = data.total_pnl_pct >= 0 ? "up" : "down";
  summary.style.display = "";

  // holdings table — each row is followed by a collapsible rationale row
  const rows = data.positions.map((p, i) => {
    const pnlClass = p.unrealized_pnl >= 0 ? "up" : "down";
    const composite = p.agent_composite ? p.agent_composite.composite_score.toFixed(2) : "--";
    const detailId = `port-detail-${i}`;
    const ac = p.agent_composite;
    const detail = `
      <div class="strategy-detail">
        <p class="strategy-verdict"><strong>${escapeHtml(p.strategy || "HOLD")}</strong> — ${escapeHtml(p.strategy_reason || "")}</p>
        ${ac ? `
        <div class="agent-breakdown">
          <div><span class="agent-label">Sector (${ac.sector_score.toFixed(2)})</span><p>${escapeHtml(ac.sector_summary)}</p></div>
          <div><span class="agent-label">Fundamental (${ac.fundamental_score.toFixed(2)})</span><p>${escapeHtml(ac.fundamental_summary)}</p></div>
          <div><span class="agent-label">Technical (${ac.technical_score.toFixed(2)})</span><p>${escapeHtml(ac.technical_summary)}</p></div>
        </div>` : `<p class="subtle">No agent data available for this symbol.</p>`}
        ${p.price_targets ? `
        <p class="price-signal"><span class="agent-label">Price Signal</span> ${escapeHtml((p.price_signal || "").replaceAll("_", " "))} ·
          1W ${money(p.price_targets["1_week"])} · 1M ${money(p.price_targets["1_month"])} · 3M ${money(p.price_targets["3_month"])}</p>` : ""}
      </div>`;
    return `<tr class="portfolio-row" data-detail-target="${detailId}" style="cursor:pointer">
      <td><strong>${escapeHtml(p.symbol)}</strong> <span class="subtle">▾</span></td>
      <td>${p.shares}</td>
      <td>${money(p.avg_cost)}</td>
      <td>${money(p.current_price)}</td>
      <td>${money(p.market_value)}</td>
      <td class="${pnlClass}">${dollarPnl(p.unrealized_pnl)}</td>
      <td class="${pnlClass}">${pctPnl(p.pnl_pct)}</td>
      <td>${composite}</td>
      <td>${strategyPill(p.strategy)}</td>
    </tr>
    <tr class="portfolio-detail-row" id="${detailId}" style="display:none"><td colspan="9">${detail}</td></tr>`;
  }).join("");
  tbody.innerHTML = rows || `<tr><td colspan="9">No positions.</td></tr>`;

  tbody.querySelectorAll(".portfolio-row").forEach(row => {
    row.addEventListener("click", () => {
      const target = document.getElementById(row.dataset.detailTarget);
      if (target) target.style.display = target.style.display === "none" ? "" : "none";
    });
  });

  // sector allocation bar
  const alloc = data.sector_allocation || {};
  const entries = Object.entries(alloc).sort((a, b) => b[1] - a[1]);
  const allocEl = document.querySelector("#portfolio-allocation");
  if (entries.length) {
    const segments = entries.map(([sector, pct], i) => {
      const color = SECTOR_COLORS[i % SECTOR_COLORS.length];
      return `<div class="allocation-segment" style="flex:${pct};background:${color}" title="${escapeHtml(sector)}: ${pct}%">${pct > 8 ? escapeHtml(sector.split(" ")[0]) : ""}</div>`;
    }).join("");
    allocEl.innerHTML = `<p class="report-section-title">Sector Allocation</p><div class="allocation-bar">${segments}</div>
      <div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.5rem">
        ${entries.map(([s, p], i) => `<span style="font-size:.75rem"><span style="display:inline-block;width:10px;height:10px;background:${SECTOR_COLORS[i % SECTOR_COLORS.length]};border-radius:2px;margin-right:3px"></span>${escapeHtml(s)} ${p}%</span>`).join("")}
      </div>`;
  } else {
    allocEl.innerHTML = "";
  }

  // risk flags
  const flagsEl = document.querySelector("#portfolio-flags");
  const flags = data.risk_flags || [];
  flagsEl.innerHTML = flags.length
    ? `<p class="report-section-title">Risk Flags</p>${flags.map(f => `<span class="flag-badge">⚠ ${escapeHtml(f)}</span>`).join("")}`
    : "";

  // suggestions
  const suggestEl = document.querySelector("#portfolio-suggestions");
  const suggs = data.rebalancing_suggestions || [];
  suggestEl.innerHTML = suggs.length
    ? `<p class="report-section-title">Rebalancing Suggestions</p>${suggs.map(s => `<div class="suggestion-card">${escapeHtml(s)}</div>`).join("")}`
    : "";
}

async function loadPortfolio() {
  const tbody = document.querySelector("#portfolio-table");
  tbody.innerHTML = `<tr><td colspan="9">Loading…</td></tr>`;
  try {
    const base = API_BASES[0] || `http://127.0.0.1:8765`;
    const res = await fetch(`${base}/api/portfolio?mode=${currentMode}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderPortfolio(data);
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9" style="color:var(--bad-fg, red)">${escapeHtml(compactErrorMessage(err))}</td></tr>`;
  }
}

async function addPortfolioPosition(symbol, shares, cost) {
  const statusEl = document.querySelector("#add-pos-status");
  statusEl.textContent = "Saving…";
  try {
    const base = API_BASES[0] || `http://127.0.0.1:8765`;
    const res = await fetch(`${base}/api/portfolio`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "add", symbol, shares: Number(shares), cost: Number(cost) }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    statusEl.textContent = "Saved!";
    setTimeout(() => { statusEl.textContent = ""; }, 2000);
    loadPortfolio();
  } catch (err) {
    statusEl.textContent = compactErrorMessage(err);
  }
}

function renderReport(data) {
  const out = document.querySelector("#report-output");
  if (!data || data.error) {
    out.innerHTML = `<p style="color:var(--bad-fg, red)">${escapeHtml(data?.error || "No data")}</p>`;
    return;
  }

  const period = data.period || "report";
  let html = `<p class="report-period-title">${escapeHtml(period.charAt(0).toUpperCase() + period.slice(1))} Brief — ${escapeHtml(data.generated || "")}</p>`;

  // top buys
  const topBuys = data.top_buys || [];
  if (topBuys.length) {
    html += `<p class="report-section-title">Top Buy Signals</p><div class="report-grid">`;
    html += topBuys.map(r => `<div class="report-card">${pill(r.action)}<strong>${escapeHtml(r.symbol)}</strong><small>Score: ${r.composite_score?.toFixed(3)}</small></div>`).join("");
    html += `</div>`;
  }

  // buy windows
  const bw = data.buy_windows || [];
  if (bw.length) {
    html += `<p class="report-section-title">Buy Windows (RSI oversold + positive momentum)</p><div class="report-grid">`;
    html += bw.map(r => `<div class="report-card"><strong>${escapeHtml(r.symbol)}</strong><small>RSI ${r.rsi} · ${escapeHtml(r.signal_note || "")}</small></div>`).join("");
    html += `</div>`;
  }

  // take profit
  const tp = data.take_profit_zones || [];
  if (tp.length) {
    html += `<p class="report-section-title">Take Profit Zones</p><div class="report-grid">`;
    html += tp.map(r => `<div class="report-card" style="border-color:#ef4444"><strong>${escapeHtml(r.symbol)}</strong><small>RSI ${r.rsi} · ${escapeHtml(r.signal_note || "")}</small></div>`).join("");
    html += `</div>`;
  }

  // reduce signals
  const reduces = data.reduce_signals || [];
  if (reduces.length) {
    html += `<p class="report-section-title">Reduce / Watch List</p><div class="report-grid">`;
    html += reduces.map(r => `<div class="report-card" style="border-color:#f59e0b"><strong>${escapeHtml(r.symbol)}</strong><small>Score: ${r.composite_score?.toFixed(3)}</small></div>`).join("");
    html += `</div>`;
  }

  // portfolio snapshot
  if (data.portfolio) {
    const p = data.portfolio;
    html += `<p class="report-section-title">Portfolio Snapshot</p>
      <div class="report-grid">
        <div class="report-card"><span class="subtle">Total Value</span><strong>${money(p.total_value)}</strong></div>
        <div class="report-card"><span class="subtle">Total P&L</span><strong class="${p.total_pnl >= 0 ? 'up' : 'down'}">${dollarPnl(p.total_pnl)}</strong></div>
        <div class="report-card"><span class="subtle">Return</span><strong class="${p.total_pnl_pct >= 0 ? 'up' : 'down'}">${pctPnl(p.total_pnl_pct)}</strong></div>
      </div>`;
    if (p.risk_flags?.length) {
      html += `<p class="report-section-title">Risk Flags</p>${p.risk_flags.map(f => `<span class="flag-badge">⚠ ${escapeHtml(f)}</span>`).join("")}`;
    }
  }

  // weekly extras
  if (data.sector_rotation?.length) {
    html += `<p class="report-section-title">Sector Rotation (Week)</p><div class="report-grid">`;
    html += data.sector_rotation.map(r => {
      const cls = r.week_pct >= 0 ? "up" : "down";
      return `<div class="report-card"><strong class="${cls}">${r.week_pct >= 0 ? "+" : ""}${r.week_pct}%</strong><small>${escapeHtml(r.sector)}</small></div>`;
    }).join("");
    html += `</div>`;
  }

  if (data.best_performers_week?.length) {
    html += `<p class="report-section-title">Best Performers (Week)</p><div class="report-grid">`;
    html += data.best_performers_week.map(r => `<div class="report-card"><strong>${escapeHtml(r.symbol)}</strong><small class="up">+${r.week_pct}%</small></div>`).join("");
    html += `</div>`;
  }

  // monthly extras
  if (data.suggested_allocation?.length) {
    html += `<p class="report-section-title">Suggested Allocation (Top 10)</p><div class="report-grid">`;
    html += data.suggested_allocation.map(r =>
      `<div class="report-card">${pill(r.action)}<strong>${escapeHtml(r.symbol)}</strong><small>${r.composite_score?.toFixed(3)} · ${r.target_weight_pct}%</small></div>`
    ).join("");
    html += `</div>`;
  }

  if (data.price_target_progress?.length) {
    html += `<p class="report-section-title">3-Month Price Targets (Top Upside)</p><div class="report-grid">`;
    html += data.price_target_progress.slice(0, 10).map(r =>
      `<div class="report-card"><strong>${escapeHtml(r.symbol)}</strong><small>Now ${money(r.last_price)} → Target ${money(r["3_month_target"])} <span class="up">+${r.upside_pct}%</span></small></div>`
    ).join("");
    html += `</div>`;
  }

  // download button
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  html += `<a href="${url}" download="${period}-report-${data.generated || 'report'}.json" class="primary report-download-btn">Download JSON</a>`;

  out.innerHTML = html;
}

async function loadReport(period) {
  const out = document.querySelector("#report-output");
  out.innerHTML = `<p class="subtle">Generating ${period} report…</p>`;
  try {
    const base = API_BASES[0] || `http://127.0.0.1:8765`;
    const res = await fetch(`${base}/api/report?period=${period}&mode=${currentMode}`);
    const data = await res.json();
    renderReport(data);
  } catch (err) {
    out.innerHTML = `<p style="color:var(--bad-fg, red)">${escapeHtml(compactErrorMessage(err))}</p>`;
  }
}

function gradeClass(grade) {
  switch (grade) {
    case "STRONG": return "good";
    case "MODERATE": return "neutral";
    case "WEAK": return "warn";
    default: return "bad";
  }
}

function renderIntrospection(data) {
  const out = document.querySelector("#introspect-output");
  if (!data || data.error) {
    out.innerHTML = `<p style="color:var(--bad-fg, red)">${escapeHtml(data?.error || "No data")}</p>`;
    return;
  }

  const scoreClass = gradeClass(data.grade);
  const recentHistory = (data.history || []).slice(-20).reverse();
  const truncatedNote = data.history && data.history.length > 20
    ? `<p class="subtle">Showing the most recent 20 of ${data.history.length} scored days.</p>`
    : "";

  const historyRows = recentHistory.map(h => {
    const hitClass = h.direction_hit ? "up" : "down";
    return `<tr>
      <td>${escapeHtml(h.date)}</td>
      <td>${money(h.predicted_close)}</td>
      <td>${money(h.actual_close)}</td>
      <td>${h.pct_error.toFixed(2)}%</td>
      <td>${escapeHtml(h.predicted_direction)} / ${escapeHtml(h.actual_direction)}</td>
      <td class="${hitClass}">${h.direction_hit ? "✓ hit" : "✗ miss"}</td>
    </tr>`;
  }).join("");

  out.innerHTML = `
    <div class="introspect-summary">
      <div class="introspect-score">
        <span class="pill ${scoreClass} introspect-grade">${escapeHtml(data.grade)}</span>
        <strong class="introspect-score-num">${data.accuracy_score}</strong>
        <span class="subtle">/ 100 accuracy score</span>
      </div>
      <div class="introspect-stats">
        <div><span>Direction Accuracy</span><strong>${data.direction_accuracy_pct}%</strong></div>
        <div><span>Avg Price Error</span><strong>${data.mape_pct}%</strong></div>
        <div><span>Median Error</span><strong>${data.median_error_pct}%</strong></div>
        <div><span>Days Scored</span><strong>${data.trials}</strong></div>
      </div>
      <p class="subtle">Backtested ${escapeHtml(data.days_covered || "")}. Each prediction used only data available as of that day — no lookahead.</p>
    </div>
    <div class="report-grid" style="margin-top:.75rem">
      <div class="report-card"><span class="subtle">Best Call</span><strong>${escapeHtml(data.best_call.date)}</strong><small>Predicted ${money(data.best_call.predicted_close)} vs actual ${money(data.best_call.actual_close)} (${data.best_call.pct_error.toFixed(2)}% off)</small></div>
      <div class="report-card" style="border-color:#ef4444"><span class="subtle">Worst Call</span><strong>${escapeHtml(data.worst_call.date)}</strong><small>Predicted ${money(data.worst_call.predicted_close)} vs actual ${money(data.worst_call.actual_close)} (${data.worst_call.pct_error.toFixed(2)}% off)</small></div>
    </div>
    ${truncatedNote}
    <div class="table-wrap" style="margin-top:.75rem">
      <table>
        <thead><tr><th>Date</th><th>Predicted</th><th>Actual</th><th>Error</th><th>Pred/Actual Dir</th><th>Direction</th></tr></thead>
        <tbody>${historyRows}</tbody>
      </table>
    </div>`;
}

async function runIntrospection(symbol, days) {
  const out = document.querySelector("#introspect-output");
  out.innerHTML = `<p class="subtle">Running walk-forward backtest for ${escapeHtml(symbol)}…</p>`;
  try {
    const base = API_BASES[0] || `http://127.0.0.1:8765`;
    const res = await fetch(`${base}/api/introspect?symbol=${encodeURIComponent(symbol)}&mode=${currentMode}&days=${days}`);
    const data = await res.json();
    renderIntrospection(data);
  } catch (err) {
    out.innerHTML = `<p style="color:var(--bad-fg, red)">${escapeHtml(compactErrorMessage(err))}</p>`;
  }
}

document.querySelector("#introspect-form")?.addEventListener("submit", (e) => {
  e.preventDefault();
  const symbol = document.querySelector("#introspect-symbol")?.value?.trim().toUpperCase();
  const days = document.querySelector("#introspect-days")?.value || "63";
  if (!symbol) return;
  runIntrospection(symbol, days);
});

// wire up portfolio
document.querySelector("#portfolio-refresh-btn")?.addEventListener("click", loadPortfolio);

document.querySelector("#add-position-form")?.addEventListener("submit", (e) => {
  e.preventDefault();
  const symbol = document.querySelector("#pos-symbol")?.value?.trim().toUpperCase();
  const shares = document.querySelector("#pos-shares")?.value;
  const cost = document.querySelector("#pos-cost")?.value;
  if (!symbol || !shares || !cost) return;
  addPortfolioPosition(symbol, shares, cost);
});

// wire up reports
document.querySelectorAll("[data-report-period]").forEach(btn => {
  btn.addEventListener("click", () => loadReport(btn.dataset.reportPeriod));
});

// auto-load portfolio on page ready
loadPortfolio();
