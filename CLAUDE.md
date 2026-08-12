# CLAUDE.md — Stock Agent AI

Guidance for Claude Code when working in this directory. This is a sub-project of the larger `Agentic AI` workspace (see the parent `CLAUDE.md` for how it relates to `brain/`, `vedic_astro/`, and `health-agent/`).

## What this is

A 5-agent stock research and analysis tool (trimmed from an original 7 — `research` and `risk` LLM agents were removed as redundant/off-thesis for an investing, not trading, use case; the deterministic risk pillar in `agents/recommendation.py` already covers risk), powered by DeepSeek (OpenAI-compatible API), with a single-page dark-themed dashboard at **http://localhost:5051**. It aggregates fundamentals, technicals, social sentiment, and quantitative signals, then has an LLM synthesize a final BUY/SELL/HOLD verdict.

**Status as of this writing (mid-upgrade):** the app is fully functional, but its final verdict's specific numbers (`entry_price`, `target_price`, `stop_loss`, `upside_pct`) are LLM-invented prose with no backtested grounding — see "Known limitation" below. A statistical-grounding upgrade is in progress; see `~/.claude/plans/memoized-dreaming-salamander.md` for the full plan and current phase status.

---

## Folder structure

```
stock_agent/
├── CLAUDE.md              ← this file
├── requirements.txt       ← added mid-upgrade; captures previously-implicit deps + scipy
├── start.sh                ← convenience launcher (has a known bug, see below)
├── .env                    ← DEEPSEEK_API_KEY=sk-... (gitignored, not committed)
├── .env.example             ← template for the above
├── data/                    ← currently empty; reserved for the SQLite recommendation-tracking DB (planned Phase 4)
│
├── tools/
│   ├── __init__.py          ← re-exports the public API (explicit __all__ list — follow this convention for new public functions)
│   └── market_data.py       ← all data fetching + signal computation (687 lines). See "tools/market_data.py" section below.
│
├── agents/
│   ├── __init__.py          ← re-exports analyze_stock, run_prediction_agent
│   ├── orchestrator.py      ← analyze_stock() — the single pipeline entry point, runs all 5 agents in sequence
│   └── stock_agents.py      ← the 5 agent functions + PREDICTION_SCHEMA (the exact site of the "LLM invention" gap — see below)
│
├── backtest/                ← NEW, added mid-upgrade (statistical grounding project)
│   ├── __init__.py          ← re-exports engine.py + strategies.py public API
│   ├── engine.py            ← vectorized backtest core: run_vectorized_backtest(), compute_performance_metrics(), deflated_sharpe_ratio()
│   ├── strategies.py        ← 7 backtestable strategies (SMA crossover, momentum, mean-reversion, stat-arb, trend-following, candlestick-filtered, RSI/MACD) + STRATEGY_REGISTRY
│   ├── risk.py              ← position sizing: kelly_fraction(), safe_kelly_fraction() (half-Kelly + 10% cap), volatility_target_scale(), correlation_aware_position_size()
│   └── run_backtest.py      ← CLI runner: `python3 backtest/run_backtest.py TICKER [strategy] [--period 5y]` — prints a strategy-comparison table (incl. a Kelly-sized position column) + a plain-English BUY/SELL/HOLD-lean takeaway for non-technical readers (honest by construction: distinguishes "a strategy had a historical edge" from "buy this stock", always caveats past-data/not-advice)
│
├── tests/                    ← NEW, added mid-upgrade (no test infrastructure existed before)
│   ├── test_backtest_engine.py   ← synthetic leak-free proof, drawdown unit test, dSR sanity bounds, real-data directional check
│   └── test_strategies.py        ← per-strategy validity checks + consistency check against market_data.py's live signal labels
│
└── web/
    ├── app.py                ← Flask server, port 5051. All API endpoints live here.
    └── static/
        └── index.html         ← the entire frontend: single HTML file, inline <style> and <script>, no build step, no framework
```

---

## `tools/market_data.py` — data & signals layer

All functions here are pure data-in/dict-or-DataFrame-out — no LLM calls happen in this file.

**Fetching (all via `yfinance` unless noted):**
- `fetch_price_history(ticker, period="6mo")` — OHLCV DataFrame
- `fetch_fundamentals(ticker)` — 32 keys: PE, PB, margins, growth, analyst targets, etc.
- `fetch_recent_news(ticker, max_items=8)`, `fetch_earnings(ticker)`, `fetch_analyst_ratings(ticker)`
- `fetch_reddit_sentiment(ticker)` — scrapes the **public Reddit JSON API** (no auth) across r/wallstreetbets, r/stocks, r/investing
- `fetch_stocktwits_sentiment(ticker)` — scrapes the **public StockTwits API**
- `fetch_web_forum_sentiment(ticker, company_name)` — DuckDuckGo web search

**Computed indicators — `compute_indicators(df) -> dict`:** 40+ keys via the `ta` library (optional import, guarded by a `TA_OK` flag — the app degrades gracefully if `ta` isn't installed). Includes `sma_50`/`sma_200`/`ma_cross` ("golden"/"death"), `rsi_14`/`rsi_signal`, `macd`/`macd_signal`/`macd_cross` ("bullish"/"bearish"), Bollinger Bands, ATR, ADX, OBV, `above_sma_200`.

**Algo signals — `compute_algo_signals(df, indicators) -> dict`:** real, correct quantitative math, but **only computes the LATEST value** (a snapshot dict for the live dashboard), not a historical series — this is why `backtest/strategies.py` recomputes the same formulas as full rolling Series rather than calling this function in a loop.
- `mean_reversion_zscore` — 20-day rolling Z-score of price, `mean_reversion_signal` at thresholds ±0.8 (BUY/SELL) and ±1.5 (STRONG_BUY/STRONG_SELL)
- `momentum_composite` — average of 1M/3M/6M (21/63/126-day) % returns, `momentum_signal` at thresholds ±3 and ±10
- `linreg_slope_pct_per_day`, `linreg_r_squared`, `linreg_signal` — 30-day least-squares trend fit
- `historical_volatility_20d`/`60d`, `vol_regime` (LOW/MEDIUM/HIGH), `vol_expanding`
- Monte Carlo GBM simulation (delegates to `_monte_carlo_simulation()`) — **1000 paths, 30 days, seeded** (`random.seed(42)`, reproducible), outputs 5th/25th/50th/75th/95th percentile prices. Output is a **price distribution only** — nothing downstream (until the backtest upgrade) checks whether this distribution has ever been predictive.
- Volume-price divergence, candlestick pattern detection (delegates to `_detect_candlestick_patterns()` — latest candle only, human-readable strings for the dashboard) and its new sibling `detect_candlestick_signal(df) -> pd.Series` (vectorized, whole-history, machine-readable ±1/0, added for backtesting — Bullish/Bearish Engulfing only, the least ambiguous 2-candle pattern)
- Composite `algo_score` (0-100) + `algo_direction`

**`compute_signal_summary(indicators) -> dict`:** a separate, simpler 0-100 technical score (RSI/MACD/MA/BB/volume/ADX voting) — this is the "Technical Signal Score" meter shown in the UI, distinct from `algo_score`.

---

## `agents/` — the 5-agent LLM pipeline

All 5 agents call DeepSeek via `openai.OpenAI(base_url="https://api.deepseek.com")` — no Anthropic dependency in the core path (Anthropic is a documented optional fallback elsewhere in the workspace, not wired into this file specifically). `run_research_agent()` and `run_risk_agent()` were removed (trimmed for latency): research's news/macro sentiment skewed trading- rather than investing-flavored, and risk's prose duplicated the deterministic risk pillar already computed in `agents/recommendation.py`/`backtest/pillars.py`.

1. `run_fundamentals_agent()` — valuation/growth/analyst consensus, free-text
2. `run_technical_agent()` — price action + indicators, free-text (mentions entry/exit levels in prose, not structured)
3. `run_social_agent()` — Reddit/StockTwits/forum mood, free-text
4. `run_algo_agent()` — interprets the quant signals + Monte Carlo output in prose
5. `run_prediction_agent()` — **the final synthesis call**. `PREDICTION_SCHEMA` (~line 343) defines the target JSON shape: `action`, `conviction`, `entry_price`, `target_price`, `stop_loss`, `upside_pct`, `downside_pct`, `risk_reward`, `scores`, plus prose fields (`summary`, `bull_case`, `bear_case`, `key_catalysts`).

**Known limitation (the reason for the backtest upgrade):** `run_prediction_agent()`'s prompt contains **zero numeric price data** beyond `current_price` — it hands the LLM four agents' prose summaries and asks it to invent specific dollar figures for entry/target/stop-loss "thinking in expected value, Sharpe ratios, and statistical edge," but nothing computes those; the LLM is inferring plausible-sounding numbers from text, not calculating them. This is being fixed by `backtest/` (Phases 0-2 done; Phase 5 will wrap `run_prediction_agent()`'s output with a new `agents/synthesis.py::ground_prediction()` that overrides these numeric fields with backtested values). **Do not present this system's current entry/target/stop-loss numbers as quantitatively grounded until Phase 5 lands.**

`agents/orchestrator.py::analyze_stock()` is the single pipeline entry point — calls all 5 agents in sequence with a `progress()` callback (used by the SSE endpoint), returns one large result dict.

---

## `backtest/` — statistical grounding layer (new, mid-upgrade)

See `~/.claude/plans/memoized-dreaming-salamander.md` for the full 6-phase plan. Summary of what exists so far:

- **`engine.py`**: `run_vectorized_backtest(prices, signal, transaction_cost_bps=10)` — point-in-time correctness enforced by a single internal `.shift(1)` (a signal computed using today's data can only be acted on starting tomorrow). `compute_performance_metrics()` returns Sharpe/Sortino/max-drawdown/Calmar/win-rate. `deflated_sharpe_ratio()` implements López de Prado's dSR (adjusts Sharpe for the number of strategy variants tried, guarding against "test enough parameter combinations and one looks good by luck"). All formulas verified directly against the source books (Hilpisch's *Python for Algorithmic Trading*, López de Prado's *Machine Learning for Asset Managers* — both in `~/Desktop/AI/Applied AI in Finance/`), including reproducing the books' own worked examples as correctness checks.
- **`strategies.py`**: 7 backtestable strategies, each `f(prices, **params) -> pd.Series` of `{-1,0,1}`. Deliberately reuses `market_data.py`'s existing signal formulas (same rolling windows, same thresholds) rather than reinventing them — verified via a consistency test that the strategy's signal agrees with the live dashboard's own label for the same date. Pairs trading is deliberately deferred (needs a second-ticker UX + cointegration testing).
- **`risk.py`**: `kelly_fraction()` (raw `f* = (μ-r)/σ²`, verified to reproduce Hilpisch's ~4.5 for the S&P), `safe_kelly_fraction()` (the one callers should use — half-Kelly + hard 10% cap, floors negative-edge strategies to 0% size), `volatility_target_scale()`, and `correlation_aware_position_size()` (shrinks redundant highly-correlated strategies so the same bet isn't counted twice). All verified in `tests/test_risk.py`.
- **Not yet built:** recommendation persistence/tracking (`data/store.py`, Phase 4), the actual wiring that makes `run_prediction_agent()`'s numbers grounded (`agents/synthesis.py::ground_prediction()`, Phase 5), and any new API/UI surface for backtest results.
- **Explicitly out of scope:** broker/live execution (Alpaca/IB/etc.) — this stays a decision-support tool, not an auto-trading system, per an explicit scoping decision.

Run tests: `python3 tests/test_backtest_engine.py` and `python3 tests/test_strategies.py`.

---

## `xsection/` — survivorship-safe cross-sectional ranking engine (M-F3)

Turns the single-ticker tool into a **point-in-time opportunity-discovery engine**: _"as of
date X, using only info available then, which stocks ranked highest/lowest, why, and how did
they perform afterward?"_ Free of survivorship, look-ahead, and hidden-substitution bias. Full
docs: [CROSS_SECTIONAL_RANKING.md](CROSS_SECTIONAL_RANKING.md) (+ `UNIVERSE_PROVIDER`,
`SECURITY_MASTER`, `FEATURE_DICTIONARY`, `SURVIVORSHIP_POLICY`, `DELISTING_POLICY`,
`RANKING_EVALUATION`, `CROSS_SECTIONAL_RANKING_REPORT`).

- **Identity is a permanent `security_id`, never the ticker** (`universe.py::SecurityMaster`) —
  ticker changes preserve identity, reused tickers never merge companies.
- **Membership is point-in-time** (`UniverseProvider.members(as_of)`); delisted names are ranked
  and evaluated, never dropped. Production survivorship needs a paid constituent dataset —
  `PaidUniverseProvider` is **BLOCKED** (raises `UNIVERSE_INCOMPLETE`, fabricates nothing). The
  keyless `reference-smallcap-demo` fixture is genuinely PIT but **synthetic** (labelled).
- **No LLM on the ranking path** (charter P4): ranks/factors/composite are deterministic
  arithmetic. Weights are **pre-registered** in `backtest/experiments.py::RANKING_CONFIGS`
  (immutable, content-hashed) — never tuned on the eval set; dSR deflates against the fixed count.
- Rankings freeze to an **immutable** `ranking_runs` table (triggers + content hash) in the
  existing `recommendations.db`; `decision_fingerprint` makes runs reproducible.
- API: `/api/universes`, `POST /api/rankings/run`, `/api/rankings/{id|latest|history}`,
  `POST /api/rankings/evaluate`. Dashboard: "Cross-Sectional Ranking" card (keyless).
- Tests: `python3 tests/test_xsection.py` (18 mission cases).

**Production data activation (M-F3B):** `xsection/providers/sharadar.py` is the
integration-ready survivorship-safe production adapter (permaticker identity, SF1 filed-date
fundamentals, ACTIONS delisting returns) — key-gated on `NASDAQ_DATA_LINK_API_KEY`, BLOCKED
(UNIVERSE_INCOMPLETE) without it, fabricates nothing. `xsection/providers/edgar_features.py`
supplies REAL point-in-time fundamentals from SEC EDGAR (filed-date governed, YTD→discrete-
quarter differencing) injected via `compute_features(..., fundamentals_fn=…)`; the synthetic
fixture path is unchanged (default). `xsection/health.py` (dataset health), `xsection/backfill.py`
(resumable/idempotent/checkpointed, gitignored workdir), `xsection/acceptance.py` (Part A real
features PROVEN, Part B survivorship replay BLOCKED-needs-license). Full status + operator
decision: [PRODUCTION_DATA_ACTIVATION_REPORT.md](PRODUCTION_DATA_ACTIVATION_REPORT.md). Tests:
`python3 tests/test_xsection_production.py` (40 offline checks).

---

## `web/app.py` — Flask server (port 5051)

- `GET /` — serves `static/index.html`
- `GET /api/status` — `{ok, key_set, key_preview, model}` — whether `DEEPSEEK_API_KEY` is loaded
- `POST /api/quick` — fast, **no LLM call**: indicators + algo signals + news + sparkline, <2s, works with no API key at all
- `POST /api/analyze/stream` — the full 5-agent pipeline via **Server-Sent Events**: emits `progress` events per agent stage (matching the UI's progress bar — data/social_data/fundamentals/technical/social/algo/prediction/grounding/recommendation), then a final `result` event, then `done`

`.env` is loaded from `stock_agent/.env` — tries `python-dotenv` first, falls back to manual line-by-line parsing if that package isn't installed (no hard dependency on `dotenv`).

**Known bug:** `start.sh` checks for `ANTHROPIC_API_KEY` and refuses to start without it, even though the actual app reads `DEEPSEEK_API_KEY` — this looks like a leftover from before DeepSeek support was wired in. The correct way to start the server today is `python3 web/app.py` directly (which does the correct `.env`-based DeepSeek key loading), not `./start.sh`.

**Other stray file noticed, not yet cleaned up:** `.sk-64ae6a766df1449aa362cec152b38b07` in the project root appears to be debris from a broken command at some point (contains the literal text `echo DEEPSEEK_API_KEY=`, not a real API key file) — worth deleting, but left alone since its origin wasn't investigated and it's unrelated to the current work.

---

## `web/static/index.html` — UI design system

Single 985-line file: no build step, no JS framework, inline `<style>` and `<script>`. If you add UI, match this file's existing conventions exactly rather than introducing a new pattern.

**Color palette** (CSS custom properties, dark theme throughout):
```css
--bg: #080c14        /* page background, near-black navy */
--bg2: #0d1220        /* slightly lighter panel background (header, search bar) */
--bg3: #111827        /* input/inner-panel background */
--border: #1e2d45     /* all card/input borders */
--accent: #00d4ff     /* primary cyan — logo, focus states, links, BULLISH pole of the signal meter */
--accent2: #7c3aed    /* purple — gradient partner to accent, badges */
--green: #00e676      /* BUY / bullish / positive everywhere */
--red: #ff1744        /* SELL / bearish / negative everywhere */
--yellow: #ffea00     /* HOLD / neutral-caution everywhere */
--text: #e0e6f0       /* primary text */
--muted: #6b7fa3      /* secondary/label text */
--card-bg: #0f1929    /* card background (slightly lighter than --bg) */
```
The BUY/SELL/HOLD traffic-light convention (green/red/yellow) is used consistently for: the verdict badge, price-item values, conviction badges, and the bull/bear signal list dots — don't introduce a different color mapping for similar concepts elsewhere.

**Typography:** `'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif` — a native-system-font stack, no webfont loading. Base size 14px. Headings and key numbers lean very heavy (`font-weight: 700-900`) — this is a data-dashboard aesthetic (bold numbers, muted labels), not a content/reading typeface choice.

**Layout system:** CSS Grid throughout, no Flexbox-based page layout (Flexbox is used only for small inline groups like the header/search bar). Standard content width `max-width: 1400px`, centered. Row-based grid classes, used in this exact order top-to-bottom in the dashboard:
1. `.top-row` (`320px 1fr` — a fixed-width verdict card beside a flexible price chart)
2. `.mid-row` (`1fr 1fr 1fr` — three equal columns)
3. `.bottom-row` (`1fr 1fr` — two equal columns) — **used twice**: once for Social Sentiment + Monte Carlo, again lower down for Bull/Bear Case + Latest News
4. `.full-row` (single column, full width) — used for the AI Agent Analysis tabs

All cards share the base `.card` style (`background: var(--card-bg)`, `border: 1px solid var(--border)`, `border-radius: 12px`, `padding: 18px`) with a `.card-title` (uppercase, 11px, `letter-spacing: 1px`, muted color, emoji prefix). The verdict card is the one exception with its own heavier style (`border-radius: 16px`, 2px border, a colored top accent bar via `::before` that switches gradient color based on BUY/SELL/HOLD).

**Page sections, in the order they appear:**
1. **Header** — logo (📈 icon + gradient-text "STOCK AGENT AI" wordmark) + 3 status badges (`● LIVE`, `7 AI Agents`, `Quant + Social`)
2. **Search bar** — ticker input (uppercase, letter-spaced, monospace-feeling large text) + Analyze button + 9 quick-pick ticker buttons (AAPL, MSFT, NVDA, TSLA, GOOGL, META, AMZN, SPY, BTC-USD)
3. **Progress bar** (hidden until an analysis starts) — 9 pill-shaped steps matching the SSE pipeline stages, each pulsing cyan while active and turning green when done
4. **Empty state** (shown before the first analysis) — headline + a 6-card feature grid describing the app's capabilities (News & Research, Social Sentiment, Quant Algo, Fundamentals, Technical Analysis, AI Prediction)
5. **Dashboard** (hidden until an analysis completes), top to bottom:
   - **Verdict card** — BUY/SELL/HOLD badge, company name, price, entry/target/stop-loss price grid, conviction badge, italic summary
   - **Price History** — canvas-based sparkline with low/ticker/high labels
   - **Technical Signal Score** — a needle gauge (red→yellow→green gradient track) + bull/bear signal counts + active-signals list
   - **Key Indicators** — 2-column grid of indicator name/value/signal tiles
   - **Algo Trading Signals** — the 0-100 quant composite score + a 2-column grid of individual algo signal tiles
   - **Social Sentiment** — Reddit mention count + top posts, StockTwits bull/bear bar
   - **Monte Carlo Simulation** — percentile price bars + detected candlestick patterns
   - **AI Agent Analysis** — 6 tabs (Research/Fundamentals/Technical/Risk/Social/Algo), one per agent's raw prose output
   - **Bull vs Bear Case** — two-column case boxes + a key-catalysts list
   - **Latest News** — headline list + a fundamentals summary table

**Interaction conventions:** all dynamic content is injected via `document.getElementById(...).innerHTML`/`.textContent` from a global `chartData`-style pattern (no framework, no virtual DOM) — if extending the UI, follow this same vanilla-JS DOM-manipulation style rather than introducing a framework.

---

## Running this project

```bash
cd stock_agent
echo "DEEPSEEK_API_KEY=sk-..." > .env    # get one at platform.deepseek.com, ~$0.01/analysis
pip3 install -r requirements.txt
python3 web/app.py                        # → http://localhost:5051
```

`/api/quick` works with no key at all (pure yfinance + math). The full 5-agent `/api/analyze/stream` pipeline needs the DeepSeek key.

Run the backtest test suite: `python3 tests/test_backtest_engine.py && python3 tests/test_strategies.py`.
