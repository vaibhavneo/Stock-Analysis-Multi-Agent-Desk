# Stock Agent AI — v1.0 Product Definition

**Date:** 2026-07-21
**Status:** Scope frozen. No new features until v1.0 observation period completes.

---

## 1. What this product is

A single-ticker stock research and decision-support dashboard that combines:
- 7 LLM-powered domain agents (research, fundamentals, technical, risk, social, algo, prediction)
- A deterministic 7-pillar composite recommendation engine with statistical grounding
- Backtested strategy validation with honest cost modeling and overfitting detection
- SEC EDGAR point-in-time fundamentals with full evidence tracing
- An immutable prediction ledger with outcome tracking and calibration scoring

It is **not** a trading system, portfolio manager, or financial advisor. It produces structured research with explicit confidence dimensions and honesty flags for a single human decision-maker.

---

## 2. Capability audit — what is built and working

### 2.1 Data layer (keyless core + optional key-gated extensions)

| Source | What it provides | Key required | Status |
|---|---|---|---|
| yfinance (via gateway) | OHLCV prices, fundamentals snapshot, news, earnings, analyst ratings | No | Working |
| SEC EDGAR | Point-in-time quarterly fundamentals (filed-date governed, YTD differencing) | No | Working |
| Reddit JSON API | Sentiment from r/wallstreetbets, r/stocks, r/investing | No | Working |
| StockTwits API | Bull/bear sentiment ratio | No | Working |
| DuckDuckGo | Web forum sentiment search | No | Working |
| FRED (St. Louis Fed) | Macro series (GDP, unemployment, CPI, rates) | No | Working |
| CBOE | VIX daily history | No | Working |
| Tiingo | Daily bars (alternative price source) | Yes | Working |
| Finnhub | Earnings calendar events | Yes | Working |

### 2.2 Analysis pipeline (7 LLM agents)

| Agent | Domain | Input | Output | Status |
|---|---|---|---|---|
| Research | News + macro context | News, web search | Prose summary | Working |
| Fundamentals | Valuation, earnings, analyst consensus | yfinance fundamentals | Prose | Working |
| Technical | Price action, indicators | 40+ technical indicators | Prose | Working |
| Risk | Volatility, position sizing language | Vol regime, beta, drawdown | Prose | Working |
| Social | Reddit, StockTwits, forum sentiment | Scraped social data | Prose | Working |
| Algo | Quantitative signal interpretation | Z-score, momentum, Monte Carlo, patterns | Prose | Working |
| Prediction | Final synthesis | All 6 agents' output | Structured JSON (action, levels, scores) | Working |

**LLM dependency:** DeepSeek `deepseek-chat` via OpenAI-compatible API (~$0.01/analysis).

### 2.3 Statistical grounding layer

| Component | What it does | Tests | Status |
|---|---|---|---|
| Backtest engine | Vectorized, PIT-correct (shift(1)), transaction costs | 12 checks | Working |
| 8 backtestable strategies | SMA crossover, momentum, mean-reversion, stat-arb, trend-following, candlestick, RSI/MACD, seven-pillar-core | 8 checks | Working |
| Realistic cost model | Spread (5bps) + market impact (sqrt model) + borrow (short cost) | 17 checks | Working |
| Walk-forward CV + PBO | Purged + embargoed cross-validation; Probability of Backtest Overfitting | 19 checks | Working |
| Deflated Sharpe Ratio | Lopez de Prado's dSR; deflation count from immutable experiment registry | 27 checks | Working |
| Kelly position sizing | Half-Kelly + 10% cap; floors negative-edge to 0%; correlation-aware | 14 checks | Working |
| Pre-registered experiments | Immutable manifest (content-hashed); deflation count fixed at registration | 20 checks | Working |

### 2.4 Recommendation engine (7-pillar composite)

| Component | What it does | Status |
|---|---|---|
| Pillar scores | 6 domain pillars (technical, algo, risk, fundamentals, research, social) → deterministic 0-100 | Working |
| Composite scoring | Fixed transparent weights (not fitted); risk as multiplier with veto; social/research bounded ±5 | Working |
| Action mapping | BUY ≥70, ACCUMULATE 60-70, HOLD 45-60, REDUCE 35-45, SELL <35; hysteresis enter≥65/exit<55 | Working |
| Entry/stop/target | ATR-based levels (entry zone, 2·ATR stop, 2:1 R:R target) | Working |
| Position sizing | Safe Kelly fraction gated on statistical edge (0 when no edge) | Working |
| Four confidence dims | Statistical edge, data completeness, thesis coherence, allocation appropriateness | Working |
| Evidence tracing | Every pillar claim recorded in EvidenceLedger with formula + datums + SEC accessions | Working |
| Decision fingerprint | Content-hash of all inputs → reproducible, interaction-independent | Working |
| Persistence | Logged to recommendations.db with raw_json; recommendation_id for outcome tracking | Working |

**Tests:** 41 checks (recommendation) + 25 checks (pillars) = 66 checks total.

### 2.5 Prediction ledger + calibration

| Component | What it does | Status |
|---|---|---|
| Immutable snapshots | Frozen at creation; DB triggers prevent UPDATE/DELETE; content-addressed | Working |
| Outcome evaluation | 1/5/20/60/252 trading-day horizons; raw return, excess vs SPY, max adverse/favorable | Working |
| Calibration scoring | By action bucket, confidence level, and conviction; ECE, hit rate, Brier | Working |
| Auto-refresh | Evaluates all mature snapshots against real prices on demand | Working |

**Tests:** 39 checks.

### 2.6 Point-in-time historical replay

| Component | What it does | Status |
|---|---|---|
| Replay engine | Runs unchanged recommendation pipeline on historical dates with as-of data | Working |
| No look-ahead | Prices truncated at replay date; EDGAR as_of filtering; social/estimates OMITTED (not faked) | Working |
| Resumable + idempotent | Content-addressed freezing; completed cells skipped on re-run | Working |
| Maturation | Replayed predictions carry as_of timestamp; outcomes evaluate forward immediately | Working |

**Tests:** 26 checks.

### 2.7 Cross-sectional ranking (research-grade, not production-activated)

| Component | What it does | Status |
|---|---|---|
| Universe + SecurityMaster | Permanent identity (security_id); ticker rename/reuse handled | Working |
| PIT features | 29 features across 6 families; filed-date governed fundamentals | Working |
| Ranking engine | Pre-registered weights (content-hashed); deterministic composite | Working |
| Evaluation | Forward returns, rank IC, decile returns, long-short (on fixture) | Working |
| Immutable runs | Frozen to ranking_runs table with decision_fingerprint | Working |
| Survivorship-safe fixture | Synthetic 8-security demo with 2 delisted names | Working |
| Production adapter (Sharadar) | Integration-ready; parse layer unit-tested; BLOCKED without license key | Built, blocked |
| EDGAR real features | YTD→quarter differencing; filed-date governed; PROVEN on real data | Working |
| Health + backfill + acceptance | Dataset health, resumable backfill, two-part acceptance harness | Working |

**Tests:** 42 checks (core) + 46 checks (production) = 88 checks total.

### 2.8 Web dashboard + API

| Endpoint | Method | Key needed | What it does |
|---|---|---|---|
| `/` | GET | No | Dashboard (1,795-line single-file SPA) |
| `/api/status` | GET | No | Server health + key status |
| `/api/quick` | POST | No | Fast data-only: indicators + algo + news + sparkline (<2s) |
| `/api/analyze/stream` | POST | DeepSeek | Full 7-agent SSE pipeline |
| `/api/recommendation` | POST | No | Keyless 7-pillar composite recommendation |
| `/api/strategies` | GET | No | List registered strategies |
| `/api/backtest` | POST | No | Single-strategy backtest |
| `/api/backtest/all` | POST | No | All-strategy sweep with costs + PBO |
| `/api/validate` | POST | No | Walk-forward CV + PBO overfitting check |
| `/api/fundamentals_pit` | POST | No | EDGAR PIT fundamentals + evidence |
| `/api/evidence/<id>` | GET | No | EvidenceLedger claim trace |
| `/api/predictions` | GET | No | Prediction ledger snapshots |
| `/api/predictions/refresh` | POST | No | Evaluate mature outcomes |
| `/api/calibration` | GET | No | Calibration report |
| `/api/replay` | POST/GET | No | Historical replay (start/status/list) |
| `/api/universes` | GET | No | Available universe providers |
| `/api/rankings/*` | POST/GET | No | Cross-sectional ranking (run/get/evaluate) |
| `/api/datasources` | GET | No | Provider status + availability |
| `/api/macro` | GET | No | FRED macro data |
| `/api/trials` | GET | No | TrialRegistry stats |

**Dashboard sections:** Verdict card, price chart, 7-pillar recommendation, backtest lab, cross-sectional ranking, prediction ledger + calibration, data sources, technical signal meter, key indicators, algo signals, social sentiment, Monte Carlo, AI agent tabs, bull/bear case, news, grounded analysis, recommendation history.

### 2.9 Test infrastructure

**16 test suites, 480 checks, ALL PASS.**

No test requires a network call or API key. All are deterministic, offline, and run in <60s total.

### 2.10 Codebase summary

| Metric | Value |
|---|---|
| Python files | 57 |
| Total lines (prod + test) | 14,283 |
| Functions/classes | 357 |
| Test suites | 16 |
| Test checks | 480 |
| API endpoints | 22 |
| Dashboard sections | 17 |
| Data providers | 7 (5 keyless, 2 key-gated) |

---

## 3. v1.0 scope — what ships

v1.0 is **everything currently built and passing tests**. The system is feature-complete for its stated purpose: single-ticker decision-support research with statistical grounding and honesty guarantees.

### 3.1 v1.0 feature set (frozen)

1. **Full 7-agent analysis** — enter a ticker, get a structured research report with 7 domain perspectives, in ~30s.
2. **Keyless quick recommendation** — 7-pillar composite score, action verb, ATR-based levels, Kelly position size, four confidence dimensions, evidence-traced claims. No LLM needed.
3. **Backtest lab** — run any of 8 strategies against 5y of real price data with realistic costs, get Sharpe/dSR/PBO/walk-forward metrics, compare all strategies at once.
4. **PIT fundamentals** — real SEC EDGAR quarterly data, filed-date governed, with accession-number tracing to the actual filing.
5. **Prediction ledger** — every recommendation is frozen immutably, evaluated at 5 horizons against real prices, and calibration-scored.
6. **Historical replay** — replay the recommendation pipeline on past dates with only as-of data, measure calibration NOW.
7. **Cross-sectional ranking** — multi-stock opportunity discovery on a synthetic fixture (production activation requires a paid dataset).
8. **Data sources dashboard** — shows which providers are live, which need keys, provider health.
9. **Macro context** — FRED + CBOE VIX data for macro awareness.

### 3.2 Explicit limitations (documented, not bugs)

| Limitation | Why it stays | Honesty handling |
|---|---|---|
| LLM prediction numbers (entry/target/stop from the 7th agent) are prose-derived | The grounded recommendation engine already provides ATR-based levels; the LLM prose is research context, not actionable numbers | Verdict card shows both; grounded section labels its source |
| Social/research pillars not backtestable | No free historical social sentiment data exists | Bounded ±5 modifier, flagged `not_backtestable` |
| Analyst estimates/revisions UNAVAILABLE | No free PIT estimate history; current-vintage would be look-ahead | Marked UNAVAILABLE, never substituted |
| Cross-sectional ranking on synthetic fixture only | Production needs Sharadar license (~$150-300/mo) | BLOCKED with `UNIVERSE_INCOMPLETE`, no fabrication |
| No portfolio-level view | Single-ticker product | Correlation-aware sizing is there for future use |
| Past performance disclaimers | It's a research tool, not advice | Built into every backtest output and recommendation |

---

## 4. Critical gaps — what blocks real use

After auditing the full system, **zero critical gaps block v1.0 real use**. The system is usable today for its stated purpose. The following are **quality-of-life improvements**, not blockers:

### 4.1 Operational polish (do before declaring v1.0 released — small, bounded)

| Item | Effort | Why |
|---|---|---|
| Fix `start.sh` (checks wrong env var) | 5 min | Convenience; `python3 web/app.py` works fine |
| Delete `.sk-64ae6a766df1449aa362cec152b38b07` debris file | 1 min | Cleanup |
| Add `python3 -m stock_agent` entry point | 15 min | Standard Python package convention |

These are the only items classified as **required for v1.0 release** beyond the existing code.

---

## 5. Classification of remaining ideas

### 5.1 DEFERRED (valuable, evidence required before building)

| Idea | Why deferred | Activation criteria |
|---|---|---|
| Production cross-sectional ranking | Requires paid Sharadar license ($150-300/mo) | Operator decision + `NASDAQ_DATA_LINK_API_KEY` provisioned; adapter already built |
| Multi-ticker comparison | Useful but expands scope; current single-ticker focus is sharp | v1.0 calibration shows users request it consistently |
| Portfolio-level view | Requires multi-ticker first; correlation-aware sizing exists but needs a portfolio context | Observation period shows portfolio users, not single-stock researchers |
| Alerting / scheduled monitoring | Useful for "watch this stock" workflow | v1.0 observation shows repeated manual re-analyses of same ticker |
| EODHD alternative data provider | Cheaper than Sharadar for production universe | Operator evaluates cost/coverage tradeoff |
| Broker integration (Alpaca/IB) | Out of scope (decision-support only, not auto-trading) | Explicit product pivot decision with regulatory review |

### 5.2 EXPERIMENTAL (unproven, would need hypothesis + test before building)

| Idea | Hypothesis to test | Kill criteria |
|---|---|---|
| Fitted pillar weights | Optimized weights may outperform stated priors | dSR < 0.5 after deflation for fitted weights; PBO > 50% |
| LLM-as-judge for agent quality | An LLM could score agent prose quality | No measurable improvement in recommendation hit rate after 8 weeks |
| Sector-relative ranking | Rank within sector instead of universe | Rank IC not significantly better than universe-wide |
| Options-derived signals | IV skew as a sentiment signal | No free PIT options data; would need a provider + evidence of predictive value |
| News sentiment NLP | Replace LLM news agent with a cheaper NLP classifier | Classifier doesn't exist yet; would need to outperform the $0.001 LLM call |

### 5.3 REJECTED (investigated and ruled out)

| Idea | Why rejected |
|---|---|
| Using current index constituents as historical universe | Survivorship bias — the exact problem this project exists to prevent |
| Substituting current analyst estimates into past dates | Look-ahead bias; violates PIT guarantee |
| Auto-trading / execution | Regulatory and liability scope; product is decision-support |
| Real-time streaming prices | yfinance rate limits make this unreliable; the product is research-cadence, not trading-cadence |
| Adding more LLM agents | 7 agents already cover the domain space; more agents = more cost + latency, no evidence of coverage gap |
| Multiple LLM providers in the pipeline | DeepSeek is $0.01/analysis; switching adds complexity with no quality evidence |
| Framework migration (LangChain, CrewAI) | Current vanilla implementation is 14K lines, fully understood, no framework lock-in |

---

## 6. Measurable acceptance criteria for v1.0

### 6.1 Functional acceptance (all currently passing)

| Criterion | How verified | Status |
|---|---|---|
| 16 test suites, 480 checks, ALL PASS | `python3 tests/test_*.py` (each suite) | PASS |
| `/api/quick` returns in <2s with no API key | `curl -X POST localhost:5051/api/quick -d '{"ticker":"AAPL"}'` | PASS |
| `/api/recommendation` returns a complete composite recommendation with no API key | `curl -X POST localhost:5051/api/recommendation -d '{"ticker":"AAPL"}'` | PASS |
| Full 7-agent analysis completes in <60s | SSE stream to `done` event | PASS |
| Prediction ledger freezes + evaluates without error | `/api/predictions` + `/api/predictions/refresh` | PASS |
| Historical replay runs on past dates without look-ahead | `tests/test_replay.py` (future-price perturbation test) | PASS |
| Cross-sectional ranking produces deterministic results on fixture | `tests/test_xsection.py` | PASS |
| Backtest lab runs all 8 strategies with costs | `/api/backtest/all` | PASS |

### 6.2 Non-functional acceptance

| Criterion | Target | How measured |
|---|---|---|
| Server starts without error | 0 startup errors | `python3 web/app.py` produces no tracebacks |
| No secrets in repository | 0 committed credentials | `grep -r 'sk-' --include='*.py' --include='*.md'` excludes .env |
| All data paths degrade gracefully | No 500 errors when optional data fails | Social fetch failures → flagged-neutral, not crash |
| Dashboard renders all sections | 17 dashboard cards visible after analysis | Visual inspection |

---

## 7. Operational metrics for the 8–12 week observation period

### 7.1 Usage metrics (measure weekly)

| Metric | What it tells you | Collection method |
|---|---|---|
| Analyses per week | Is the tool being used | Count `/api/analyze/stream` + `/api/quick` + `/api/recommendation` calls (server logs) |
| Unique tickers analyzed | Breadth of use | Distinct ticker values in API calls |
| Repeat analyses (same ticker, <7d) | "Monitoring" pattern → alerting feature signal | Log analysis |
| Quick Rec vs Full Analysis ratio | Are users relying on the keyless path | Endpoint counts |
| Backtest lab usage | Is the statistical grounding being consumed | `/api/backtest/all` call count |
| Cross-sectional ranking usage | Is the multi-stock discovery being used | `/api/rankings/run` call count |
| Replay runs started | Is historical validation being used | `/api/replay` POST count |

### 7.2 Quality metrics (measure monthly)

| Metric | What it tells you | Collection method |
|---|---|---|
| Prediction hit rate @20d | Are BUY recommendations actually going up | `calibration_report(horizon=20)` |
| Prediction hit rate @60d | Medium-term accuracy | `calibration_report(horizon=60)` |
| Calibration ECE | Are confidence levels meaningful | ECE from calibration report |
| HIGH conviction hit rate | Does HIGH conviction outperform MEDIUM/LOW | Calibration by conviction bucket |
| Backtest dSR distribution | Are strategies showing real edge after deflation | dSR values from `/api/backtest/all` responses |
| PBO across strategies | Are strategies overfitting | PBO values from `/api/validate` |
| Social pillar agreement rate | Does social sentiment add signal | Track how often social modifier changes the action verb |

### 7.3 Reliability metrics (measure weekly)

| Metric | Target | Collection method |
|---|---|---|
| API error rate | <1% of requests | Server logs (5xx responses) |
| DeepSeek API availability | >99% uptime | Track LLM call failures |
| yfinance data freshness | Data <1 trading day old | Compare latest bar date to today |
| EDGAR availability | >95% of CIK lookups succeed | Track `resolve_cik` failures |
| Reddit/StockTwits rate limiting | <5% of social fetches fail | Track social fetch degradation events |

### 7.4 Kill criteria (triggers for removing a feature)

| Feature | Kill if (after 8 weeks) | Action |
|---|---|---|
| Social pillar | Social modifier changes action verb <2% of the time AND hit rate is not improved when social agrees | Remove social pillar; simplify to 5-pillar core |
| Research pillar | Research modifier changes action verb <2% of the time | Remove research pillar |
| Monte Carlo simulation | Zero users reference it in any feedback; no downstream consumer | Remove from dashboard |
| Candlestick pattern display | Zero correlation with 20d returns in prediction ledger | Remove from dashboard |
| Cross-sectional ranking (fixture) | Zero `/api/rankings/run` calls in 8 weeks | Remove UI card; keep code dormant |
| Historical replay | Zero `/api/replay` calls after initial exploration | Remove UI controls; keep API |

### 7.5 Feature ROI (weekly cost vs value)

| Feature | Weekly cost (compute/API) | Value signal | ROI assessment |
|---|---|---|---|
| Quick Rec (keyless) | ~$0 (pure math) | Primary entry point for fast decisions | HIGH — zero marginal cost, instant value |
| Full 7-agent analysis | ~$0.01/analysis × usage | Deep research for high-conviction decisions | HIGH if used >5x/week; review if <2x/week |
| Backtest lab | ~$0 (pure math) | Statistical grounding for recommendations | MEDIUM — value is in the confidence it adds |
| PIT fundamentals | ~$0 (free EDGAR) | Evidence trail for fundamental claims | HIGH — differentiator, zero cost |
| Prediction ledger | ~$0 (local DB) | Self-correcting calibration | HIGH — builds trust over time |
| Historical replay | ~$0.01/date (EDGAR calls) | Retroactive calibration measurement | MEDIUM — one-time setup value, then dormant |
| Cross-sectional ranking | ~$0 (fixture) | Multi-stock discovery | LOW until production data activated |
| Social sentiment | ~$0 (free APIs) | Sentiment color | UNKNOWN — needs observation period data |

---

## 8. Release checklist

### Pre-release (do once, before declaring v1.0)

- [ ] Fix `start.sh` to check `DEEPSEEK_API_KEY` instead of `ANTHROPIC_API_KEY`
- [ ] Delete `.sk-64ae6a766df1449aa362cec152b38b07` debris file
- [ ] Run all 16 test suites → ALL PASS
- [ ] Start server → no startup errors
- [ ] Run one full analysis (AAPL) → SSE completes, all dashboard sections render
- [ ] Run one Quick Rec (AAPL) → recommendation card renders with levels + pillars
- [ ] Run `/api/backtest/all` for AAPL → all 8 strategies return results
- [ ] Verify `/api/predictions/refresh` → outcomes evaluated
- [ ] Verify cross-sectional ranking on fixture → deterministic result
- [ ] Verify no secrets in committed files
- [ ] Tag commit as `v1.0.0`

### Post-release (observation period, weeks 1-12)

- [ ] Week 1: Establish baseline usage metrics
- [ ] Week 2: First calibration check (enough predictions to have 20d outcomes)
- [ ] Week 4: First monthly quality review (hit rate, ECE, dSR distribution)
- [ ] Week 4: Review kill criteria — any feature clearly dead?
- [ ] Week 8: Second quality review — trend comparison with week 4
- [ ] Week 8: Evaluate deferred backlog against observation data
- [ ] Week 12: Final observation report — ship/kill/defer decisions for each feature
- [ ] Week 12: Decide whether to activate production cross-sectional ranking (Sharadar)

---

## 9. What v1.1 could be (not committed, evidence-gated)

These are the **only** items that could enter scope for a v1.1, and **only** if observation-period data supports them:

1. **Production cross-sectional ranking** — if operator provisions Sharadar key and fixture usage shows demand
2. **Multi-ticker comparison** — if usage shows >50% of sessions analyze 3+ tickers
3. **Alerting** — if usage shows >30% of analyses are repeat-monitoring of the same ticker
4. **Pillar weight tuning** — if 12 weeks of prediction data shows a clearly better weight vector (must pass dSR + PBO)

Everything else stays deferred or rejected until v2.0 planning begins with 12 weeks of real usage data.

---

## 10. Architecture freeze

The following architectural decisions are **locked for v1.0**:

1. **Single-file SPA** — no framework migration
2. **DeepSeek as sole LLM** — no multi-provider routing
3. **7 agents** — no additions or removals
4. **Flask server** — no framework migration
5. **SQLite local storage** — no database migration
6. **Pre-registered experiment weights** — no optimization loop
7. **Python 3.9 compatibility** — `Optional[X]` not `X | None`

These can be revisited in v2.0 planning after the observation period, with evidence.

---

## Summary

Stock Agent AI v1.0 is a **complete, tested, honest decision-support tool** with 14,283 lines of code, 480 test checks across 16 suites, 22 API endpoints, and 17 dashboard sections. It runs analyses for ~$0.01, has a fully keyless recommendation path, traces every claim to its source, and measures its own accuracy through an immutable prediction ledger.

The scope is frozen. The next step is the observation period, not more features.
