# TRADING DESK PLAN — from single-ticker analyzer to systematic desk on AIOS

Version 1.0 · 2026-07-16 · **PLAN ONLY — nothing below is implemented until a
milestone is approved.** Prereq reading: PROJECT_CHARTER.md (P4/P6/P7 govern
everything here), DOMAIN_PACKS.md §5.3 (Finance Pack spec), the honest-gap
assessment that motivated this plan (summarized in §1).

## 0. Thesis

We cannot replicate Renaissance's edge (30 years of proprietary tick data,
microsecond execution, capacity discipline). We CAN replicate their **method**:
many weak signals → brutal statistical honesty → kill most hypotheses →
neutralize factor exposure → let portfolio construction do the work. The
target outcome is a system whose most valuable output is a TRUSTWORTHY answer
to "do I have an edge?" — a genuine post-cost, out-of-sample Sharpe of 0.5–1.0
cross-sectionally would be a real result; most retail systems are negative
after costs and can't even measure that fact.

**AIOS is the chassis, deliberately:** signals/backtests/portfolio math are
deterministic skills (P4 — compute, never generate numbers); every hypothesis
runs through the evaluator-first loop (P6); dead ideas stay recorded, not
deleted (P7); the LLM agents are repositioned as the research assistant
(hypothesis generation with kill tests, filing/news parsing, explanation),
never the predictor.

## 1. What must be fixed before anything is added (measurement debt)

Verified against current code (2026-07-16):
1. **Survivorship bias** — yfinance = today's listed names only; every backtest inflated by an unknowable amount.
2. **Restated fundamentals** — yfinance serves current restatements; any fundamental signal is look-ahead.
3. **Cost fantasy** — flat 10bps; no spread, impact, or borrow. The 233-trade candlestick strategy is probably cost-negative and the engine can't say so.
4. **Undercounted trials** — dSR uses n_trials=7 per run; the true count is every hypothesis ever tried. Without a persistent trial registry, dSR is a comfortable lie.
5. **Single-name framing** — `analyze_stock(ticker)` asks a question no desk asks. The unit must become the cross-section.

Rule for this whole plan: **no new signal work lands until M-F1 and M-F2 are
green.** Building strategies on a biased measurement stack manufactures
convincing garbage.

## 2. Data & financial API plan (the full menu, tiered by cost)

Design principle: **a provider-agnostic Market Data Gateway** (M-F2) mirroring
the retrieval gateway — skills call `market_data(request)`, never a vendor SDK;
providers are registry entries with reliability weights, rate limits, and
provenance stamped on every response (P8-style seam: vendor names live in
config, not in skill code). yfinance becomes one provider among many, not the
foundation.

### Tier 0 — free, start immediately
| source | what it gives | why it matters |
|---|---|---|
| **SEC EDGAR** (full-text + XBRL Financial Statements API + submissions) | as-reported, point-in-time fundamentals BY FILING DATE; insider trades (Form 4); 13F holdings; 8-K events | fixes restated-fundamentals bias for FREE; the single highest-value addition |
| **Alpaca Markets** | free real-time IEX quotes, historical bars, **paper-trading API** | the execution/paper-track backbone (M-F6) without a brokerage commitment |
| **FRED** (St. Louis Fed) | rates, spreads, CPI, unemployment, recession indicators | macro regime features + risk-free rate done right |
| **FINRA** | bi-monthly short interest | crowding/squeeze features |
| **CBOE** | VIX family indices | regime detection, vol targeting input |
| **Tiingo (free tier)** | clean adjusted EOD, basic fundamentals, news | quality check against yfinance; generous free EOD |
| **Finnhub (free tier)** | earnings calendar, estimates, company news | event-driven features; earnings-date awareness for the coach |
| **GDELT** | global news event firehose | news-flow features at zero cost |
| **US Treasury / BLS / BEA** | yields, employment, GDP | macro features |
| **Senate/House financial disclosures** | congressional trades | niche alt-signal (scrape or QuiverQuant later) |
| existing: **Reddit/StockTwits scrapers, yfinance** | retained as providers with LOW reliability weights | already built; demoted, not deleted |

### Tier 1 — ~$40–80/mo, unlocks honest backtesting (recommended once M-F1 lands)
| source | what it gives | why it matters |
|---|---|---|
| **Nasdaq Data Link — Sharadar bundle** (SF1/SEP/TICKERS) | **point-in-time fundamentals + delisted companies + survivorship-free price history**, 20+ yrs | THE fix for biases #1/#2 at retail price; industry-standard cheap PIT |
| **Polygon.io (Starter)** | clean minute/EOD bars, splits/divs, ticker events, options chains (higher tier) | primary price provider with real SLAs |
| **EODHD** (alternative to above) | global EOD + delisted add-on | cheaper alternative if Sharadar is skipped |
| **FMP** | as-reported statements, transcripts, insider data | cheap redundancy/cross-check provider |

### Tier 2 — later, only if a proven edge justifies it
Options flow (ORATS/CBOE DataShop), intraday tick history (Polygon Advanced/
Databento), earnings-call NLP feeds (paid transcripts), IBKR live execution
(cheapest real trading + its paper TWS API), borrow-rate feeds. **Gate: do not
spend here until the paper track record (M-F6) shows a live edge.**

### Gateway contract (sketch — implemented in M-F2)
```
market_data.get(kind, symbols, start, end, as_of=None, provider=None)
  kind ∈ {bars, fundamentals_pit, corporate_actions, universe, macro,
          events, filings, short_interest, sentiment}
  → {data, provider, retrieved_at, as_of_honored: bool, provenance: {...}}
```
`as_of` is first-class: any request that cannot honor point-in-time semantics
says so explicitly (`as_of_honored: false`) — biased data is allowed only when
LABELLED (P7). Local cache layer (parquet per provider/kind/symbol) so
backtests never re-hit rate limits; cache is rebuildable (P2).

## 3. Milestones (each independently shippable; playbook format)

### M-F1 — Measurement honesty (FIRST; ~1–2 wk)
- **Objective**: the backtest stack tells the truth or says it can't.
- **Work**: (a) realistic cost model — spread estimate from price/volume,
  square-root impact term, per-side bps + borrow for shorts; (b) purged &
  embargoed walk-forward CV (López de Prado Ch.7 — companion to the dSR we
  already reproduce) + Probability of Backtest Overfitting; (c) **trial
  registry**: every hypothesis/parameter-set ever tested gets a persistent row
  (extends `data/store.py`); dSR reads its true n_trials from the registry;
  (d) bias labels: every backtest result carries `{survivorship_safe,
  pit_fundamentals, cost_model}` flags — false flags render prominently.
- **Ground truth for tests**: reproduce a known cost-drag example; synthetic
  strategy with injected overfitting must be caught by PBO; registry n_trials
  strictly non-decreasing.
- **DoD**: all existing strategy results re-reported under the new harness
  with honest flags (expect them to get worse — that is success, not failure).
- **Risks**: none external; pure engine work on the existing vectorized core.

### M-F2 — Market Data Gateway + Tier-0 providers (~1–2 wk, parallelizable with F1)
- **Objective**: provider-agnostic data layer with provenance; EDGAR + Alpaca
  + FRED + Tiingo/Finnhub free tiers wired; yfinance demoted to fallback.
- **Work**: gateway module in `stock_agent/tools/` (pattern-copy of
  `second_brain/gateway.py`), provider registry JSON (mirrors corpus registry:
  id/reliability/rate_limit/kinds), parquet cache, EDGAR XBRL
  point-in-time fundamentals extractor (companyfacts + filing dates).
- **DoD**: same request served by ≥2 providers cross-checks within tolerance;
  a fundamentals query with `as_of=2023-01-01` provably excludes later
  restatements (test against a known restatement case); rate limits respected.
- **Risks**: EDGAR XBRL mapping is fiddly (start with ~20 core concepts:
  revenue, NI, assets, equity, shares, OCF…); free-tier rate limits (cache
  aggressively, nightly batch pulls).

### M-F3 — Cross-sectional engine (~2 wk)
- **Objective**: the unit of analysis becomes the universe, not the ticker.
- **Work**: defined liquid universe (e.g. top ~500–1000 by dollar volume,
  point-in-time membership from Sharadar TICKERS or EDGAR listings); signal
  MATRIX (dates × names × signals) built from the existing 7 strategies
  refactored into cross-sectional ranks + new PIT-fundamental signals
  (value/quality/accruals) + short-interest/insider features; long-short
  decile backtests through the M-F1 harness; IC/rank-IC and decay reporting
  per signal.
- **DoD**: every legacy strategy reproduced as a matrix column; per-signal
  tearsheet (IC, decay, turnover, cost-adjusted deciles); at least one
  documented signal KILLED by the harness and recorded in the trial registry.
- **Depends**: M-F1 + M-F2 (Tier-1 Sharadar strongly recommended here — this
  is the milestone where the $40/mo starts paying for itself).

### M-F4 — Factor risk model + portfolio construction (~2 wk)
- **Objective**: separate alpha from factor exposure; output = a PORTFOLIO.
- **Work**: cross-sectional factor model (market/size/value/momentum/sector
  via regression — no vendor risk model needed at this scale); residualize
  signals against it; combine signals (equal-weight → IC-weighted); optimizer
  with constraints (beta≈0, sector caps, name caps, turnover budget, cost
  penalty) — start with rank-based sizing + constraints before convex
  optimization; vol targeting on the book (risk.py already has the pieces).
- **DoD**: any strategy's return decomposes into factor + residual; the
  "alpha" of a naive momentum tilt is shown to be mostly the momentum factor
  (the canonical honesty demo); portfolio backtest reports net-of-cost with
  factor attribution.

### M-F5 — Research loop on AIOS (Finance Pack, ~1 wk)
- **Objective**: the desk becomes an AIOS Domain Pack; the LLM agents become
  the research staff around the numeric core.
- **Work**: build `packs/finance/` per DOMAIN_PACKS.md §5.3 — skills wrapping
  (not rewriting) the engine: `backtest`, `signal_matrix`, `portfolio_construct`,
  `ground_prediction` (existing), plus agent-type `strategy_hypothesis`
  (kill-test-mandatory, feeding the trial registry) and `filing_analyst`
  (EDGAR 8-K/10-K → structured features, grounded quotes). Workflows:
  `strategy_research` (hypothesize → backtest-as-kill-test → registry →
  critic) and `daily_desk` (universe refresh → signals → portfolio → orders →
  ledger). Mission template `run-a-trading-strategy`; coach triggers: earnings
  upcoming in book, signal IC decay, drawdown breach, stale paper sweep.
  Research retrieval scoped to the `finance` corpus (187 chunks; grow it with
  the Tier-0 doc sources).
- **DoD**: pack passes conformance + zero-core-change tests; `strategy_research`
  workflow runs end-to-end producing a registry entry (alive or dead).

### M-F6 — Paper desk (~1–2 wk)
- **Objective**: a live, honest track record — the only evidence that matters.
- **Work**: Alpaca paper account wired through an `execution` skill (orders,
  fills, positions — PAPER ONLY, hard-coded guard); nightly scheduled
  `daily_desk` workflow (BackgroundRun/cron); the existing
  recommendations/outcomes store becomes the ledger with a scheduled outcome
  sweep (closes the long-standing open item); Mission Control panel reads the
  ledger (paper P&L, factor exposures, hit rate, slippage vs backtest).
- **DoD**: ≥20 consecutive trading days of automated paper operation; daily
  backtest-vs-paper slippage report; zero manual interventions required.
- **Risks**: scheduler reliability on a laptop (use launchd/cron + idempotent
  runs — the planner's resumability pattern applies directly).

### M-F7 — Live gate (explicitly OUT of scope until earned)
Real money requires ALL OF: ≥3 months paper, paper Sharpe > 0.5 net,
backtest/paper slippage < agreed band, max drawdown inside limit, and a
human decision recorded in decisions.md. Broker: IBKR or Alpaca live. This
milestone exists in the plan precisely so nothing drifts into it casually.

## 4. Order & spend summary

```
now ──▶ M-F1 honesty ──┬─▶ M-F3 cross-section ─▶ M-F4 portfolio ─▶ M-F5 pack ─▶ M-F6 paper ─▶ (gate) M-F7
        M-F2 data  ────┘
spend:  $0 (Tier 0) ──▶ +$40–80/mo at M-F3 (Sharadar/Polygon) ──▶ Tier 2 only after M-F6 proves edge
```

## 5. What we are NOT doing (recorded so it isn't relitigated)
- No HFT/intraday ambitions — data + latency costs are institutional; daily/
  weekly rebalance only.
- No LLM price prediction — LLMs stay in hypothesis/parsing/explanation roles (P4).
- No vendor risk model or execution-management platform at this scale.
- No live capital before the M-F7 gate conditions, no exceptions.
```
