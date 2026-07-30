# FIL — Financial Intelligence Layer (M1 + M2)

Status:
- **M1 shipped 2026-07-16** — gateway + EDGAR point-in-time fundamentals + EvidenceLedger + TrialRegistry.
- **M2a shipped 2026-07-17** — dSR reads the true trial count (loophole closed).
- **M2b shipped 2026-07-17** — `fetch_price_history` routed through the gateway (bars path off direct yfinance). Alpaca/FRED deferred by choice.
- **M-F1 remainder shipped 2026-07-17** — realistic `CostModel` (spread/impact/borrow) + `validation.py` (purged walk-forward CV + PBO). "Cost fantasy" bias closed for opt-in backtests.
- **Live wiring + web shipped 2026-07-17** — `agents/fundamentals_pit.py` (EDGAR ratios → EvidenceLedger claims) surfaced with the honest backtest, overfitting check, and evidence tracing at **`http://localhost:5051/fil`** (browser-verified). Existing `index.html` untouched.
- **7-Pillar Investing Strategy + Recommendation shipped 2026-07-17** — see §Seven-Pillar below.
- **Data sources expanded 2026-07-18** — FRED + CBOE (keyless macro) live; Tiingo + Finnhub key-gated; `/api/datasources` + Backtest Lab on the dashboard.
- **Recommendation hardening shipped 2026-07-18** — reproducible + interaction-independent + four confidence dimensions; see §Recommendation-Hardening below.
- **Prediction Ledger + Outcome Calibration shipped 2026-07-19** — see §Prediction-Ledger below.
- **Point-in-Time Historical Replay shipped 2026-07-19** — see §Historical-Replay below.

## Point-in-Time Historical Replay (2026-07-19)

`agents/replay.py` — replay the UNCHANGED recommendation pipeline on historical
dates using only as-of data, freeze the results through the immutable Prediction
Ledger, and evaluate outcomes — so calibration, agent attribution, and
confidence reliability can be measured NOW without waiting for live predictions
to mature. Rests on two facts that already held: build_recommendation is a pure,
reproducible function of its inputs, and EDGAR already supports `as_of`.

- **No look-ahead**: `pit_inputs(ticker, as_of)` truncates the price series at
  the replay date (a bar dated ≤D was knowable at D), recomputes
  technical/algo/volatility from the truncated series, fetches
  `analyze_fundamentals_pit(as_of=D)` (restatements filed after D excluded), and
  takes the 52-week range from the truncated series. Test-proven: tripling ALL
  future prices leaves the decision fingerprint byte-identical.
- **No synthesis of unavailable history**: social sentiment, analyst consensus,
  beta, and short interest have no free point-in-time history, so they are
  OMITTED and recorded as `missing inputs` (the pillars flag themselves) —
  never fabricated. FRED macro is current-vintage, recorded as an exclusion, and
  is not a recommendation input anyway.
- **Frozen dated D**: replayed predictions carry `created_at = as_of` and a
  `replay_run_id`, so the existing outcome engine evaluates them forward and
  they mature immediately. Snapshots record strategy version, data coverage,
  missing inputs, exclusions, and the run id (immutably, in `replay_meta`).
- **Runs**: `run_replay(config)` supports ticker lists, date ranges, sampling
  cadence (monthly/weekly/N-days), benchmarks, and is RESUMABLE — cells already
  done/skipped are not redone, and freezing is content-addressed so re-runs
  never duplicate. Delisted/missing symbols and insufficient-history dates are
  skipped with a reason. Run state lives in the existing DB (`replay_runs`,
  `replay_items`) — no new database.
- **Surfaces**: `POST /api/replay` (background), `GET /api/replay/<id>`
  (progress), `GET /api/replay` (list); `data/prediction_ledger.py` gained a
  `replay_run_id` column + a `source` filter (`all|live|replay`); the dashboard
  card gained replay controls, live progress, and an all/historical/live
  calibration toggle.
- **Tests** (`tests/test_replay.py`): look-ahead prevention, restated filings
  (as-of request), duplicate runs (idempotent), interrupted resume, delisted/
  missing symbols, deterministic reproduction, and no-synthesis of missing inputs.
- Live-verified: a real 16-date AAPL replay (2022–2023, monthly) froze
  point-in-time predictions using genuine as-of EDGAR fundamentals; matured NOW
  they show win rate 54% @60d, and — honestly — MEDIUM confidence predicted 90%
  but realized 54% (ECE 0.36), surfacing the model's own overconfidence.

## Prediction Ledger + Outcome Calibration (2026-07-19)

`data/prediction_ledger.py` — freeze every recommendation at creation, evaluate
it later against real prices, and measure whether the confidence dimensions and
pillar (agent) signals have historically been reliable. **Measurement only —
outcomes never tune strategies or change the recommendation gates.** Uses the
EXISTING recommendations.db (two new tables, no new database).

- **Immutable snapshots** (`prediction_snapshots`): frozen at creation with
  ticker, timestamp, price, action, horizon, expected return, the four
  confidence dimensions, pillar scores, gates, evidence ids (ledger claim ids),
  strategy version (experiment manifest hash), data timestamps, and decision
  fingerprint. Immutability is enforced **in the database** — BEFORE UPDATE /
  BEFORE DELETE triggers ABORT any edit or delete, so no code path can rewrite
  history. `content_hash` makes tampering evident; re-freezing an identical rec
  is idempotent (content-addressed id). Frozen automatically at the single
  persistence point (`log_composite_recommendation`).
- **Outcome evaluation** (`prediction_outcomes`) at **1/5/20/60/252 trading
  days**: raw return, benchmark-relative (vs SPY) excess, maximum adverse /
  favorable excursion, hit rate, Brier (where the edge-score probability
  exists), all pure price arithmetic (no LLM). Corporate actions handled by
  reading both endpoints from ONE adjusted series; missing market days snapped
  to the nearest trading day; horizons measured in index positions (holidays
  skipped). Deterministic → `refresh_outcomes()` is idempotent.
- **Calibration + attribution** (`calibration_report`): win rate, calibration
  error (ECE), and breakdowns by action, confidence bucket, regime, sector, and
  **pillar (agent) attribution** (outcome when a pillar was leaning ≥60) —
  "which agents & confidence levels have actually worked".
- **Refresh**: `POST /api/predictions/refresh` (manual) or
  `python3 data/prediction_ledger.py refresh [TICKER]` (schedulable, idempotent).
- **Dashboard**: "🧾 Prediction Ledger & Calibration" card — active vs matured
  counts, win rate, calibration, confidence-reliability table, agent-attribution
  table, and recent frozen predictions with their strategy-version·fingerprint.
- **Tests** (`tests/test_prediction_ledger.py`): immutability (UPDATE/DELETE
  blocked, tamper-evident), duplicate-refresh idempotence, missing market days,
  corporate actions (adjusted-series, no split artifact — contrasted against the
  raw-series bug), benchmark comparison arithmetic, reproducibility.

## Recommendation Hardening — reproducible & calibrated (2026-07-18)

The bug: the dSR denominator came from `ledger.effective_n_trials(ticker)`, a
per-ticker count that GREW with every backtest. So a recommendation changed
because the user clicked — a fresh ticker scored dSR≈1.0 (n=1), the same data
after a few Compare-All clicks scored dSR≈0.0 (n=16). Fixed:

- **`backtest/experiments.py` — immutable ExperimentRegistry.** 8 strategy
  variants pre-registered with their canonical params (validated to EQUAL the
  functions' shipped defaults, so params can't be tuned on eval data). The dSR
  deflation count is `deflation_n()` = |variants| = 8 — **fixed, interaction-
  and ticker-independent**, used by the recommendation, `/api/backtest`,
  `/api/backtest/all`, and synthesis. Executions are logged via
  `record_execution()` idempotently per (ticker, variant), so Compare-All /
  repeated calls never inflate anything. `manifest_hash` is recorded with each
  recommendation as an immutability tripwire.
- **Reproducibility contract.** `recommendation.decision_fingerprint(rec)`
  hashes exactly the decision-determining fields. Guarantee (test-enforced):
  identical input data ⇒ identical fingerprint, regardless of backtest count,
  action order, or ledger state. Live-verified: NVDA fingerprint unchanged
  across a Compare-All run.
- **Four separate calibrated confidence dimensions** (was one blended
  `conviction`): **thesis** (narrative; informed by pillars + optional LLM
  prose — GATES NOTHING), **data** (SEC/PIT coverage, freshness), **statistical
  edge** (the strict deterministic gate), **allocation** (follows the edge).
  `conviction` now == the statistical-edge level (proven, never narrative).
- **Statistical-edge gate.** HIGH statistical edge — and any non-zero Kelly —
  requires ALL of: purged+embargoed **walk-forward** (positive, consistent OOS),
  **PBO** < 0.5, **net-of-cost positive** return, **≥504-bar sample**, and
  **dSR ≥ 0.5** (against the fixed 8). Otherwise size is gated to **0**, shown
  honestly (`position_size_gated`, with `raw_kelly_pct` for transparency). A
  great story with no proven edge sizes to zero, by rule.
- **Fundamentals via the gateway, not yfinance.** The recommendation's
  fundamentals pillar is SEC-sourced only (EDGAR PIT via
  `analyze_fundamentals_pit`); `strict_fundamentals=True` forbids the legacy
  yfinance accounting fallback — a non-filer reports `fundamentals_source:
  unavailable` and drops data-confidence rather than substituting Yahoo numbers.
  Test-proven: the fundamentals score is invariant to garbage yfinance inputs.
- **Not changed on purpose:** no new strategies/agents; no parameter tuning; FRED
  stays non-PIT (current-vintage, `pit_capable=[]`); LLM prose never touches a
  gate. Live NVDA now: BUY / composite 87 / thesis HIGH · data HIGH · **edge
  MEDIUM · allocation NONE** → **size 0% (gated)**, dSR 0.00 (n=8 fixed).

## Seven-Pillar Composite Investing Strategy + Recommendation (2026-07-17)

Each analysis agent's domain became a deterministic 0-100 pillar
(`backtest/pillars.py`); the prediction agent is the combiner, not a pillar.
Long-only; verbs BUY/ACCUMULATE/HOLD/REDUCE/SELL; enter ≥65 / exit <55
hysteresis on a weekly cadence. The **backtestable core** (technical + algo ×
risk multiplier) is registered as `seven_pillar_core` (8th strategy) and runs
through the full honest stack; **fundamentals/research/social are
tracked-forward modifiers** (±5 pts, flagged) because free data has no history
for them. `agents/recommendation.py` builds the structured recommendation
(deterministic conviction gates, 2×ATR investing levels, `time_horizon_days`
vol-regime map — fixing the always-NULL column) with EvidenceLedger claims for
every pillar + composite (new `derived_metric` datum kind, `as_of_honored=False`
stated). Persisted with `grounding_strategy="seven_pillar_composite"` so
`get_historical_hit_rate` tracks the engine's real forward record. Surfaces:
keyless `POST /api/recommendation`, `result["recommendation"]` in the 7-agent
pipeline, and the Recommendation card on the main dashboard (Quick Rec button
= no-LLM path).

**Honest limitations:** social/research pillars cannot be backtested from free
data (bounded modifiers, flagged everywhere — "backtested composite" means the
core only); the tested core excludes the fundamentals term (PIT not in the
backtest path); survivorship ✗ inherited; weights/thresholds/horizon map are
fixed stated priors, deliberately NOT optimized (a bad core backtest is
reported as a dead trial, never tuned away; any variant = one registered
trial); single-ticker absolute scores, not cross-sectional ranks (M-F3);
snapshot claims carry `as_of_honored=False` (provenance-honest, not
time-reconstructable, unlike EDGAR claims).

Executes `TRADING_DESK_PLAN.md` M-F2 (data) and the evidence half of M-F1
(honesty). Session prompt: `FIL_MISSION_PROMPT.md`.

## Why this exists

The 7-agent pipeline reads `yfinance` directly. That is fine for a live
dashboard and fatal for research, because of one thing: **yfinance serves the
CURRENT view of history**. Ask it for 2019 revenue and it hands you today's
restated figure — a number nobody had in 2019. Backtest on that and you are
trading on knowledge from the future, and the backtest will look wonderful.

FIL makes that specific mistake structurally impossible for new code:

| Bias (TRADING_DESK_PLAN §1) | M1 status |
|---|---|
| Restated fundamentals (look-ahead) | **fixed** (M1) — EDGAR `filed` dates + `as_of` cut |
| Undercounted dSR trials | **fixed** (M2a) — TrialRegistry wired into dSR via `effective_n_trials()`; strategy-shopping loophole closed. (M1 shipped the *table* but no producer, so `n_trials()` returned 0 — a worse lie than the old constant. M2a added the producers + backfill + engine guard.) |
| Unsupported numeric claims | **fixed** (M1) — `record_claim()` refuses empty evidence |
| Survivorship | *open* — needs delisted history (M-F3, Sharadar) |
| Cost fantasy | *open* — M-F1 backtester work |

## The one idea

Every number that enters the system is a **Datum**: a value plus the answer to
*"when could I first have known this?"* — the `available_at` field.

`available_at` is NOT `period_end`. Apple's FY2023 revenue has period_end
2023-09-30 but was unknowable until the 10-K was **filed** 2023-11-03. Using
period_end as the decision timestamp is the classic look-ahead bug; the schema
makes `available_at` mandatory so a datum cannot exist without it.

## Contract

```python
from financial_data import get, get_concept, get_bars_df

res = get("fundamentals_pit", "AAPL", as_of="2024-01-01", concepts=["revenue"])
if not res["as_of_honored"]:
    ...     # NO point-in-time guarantee. Do not backtest on this.

df = get_bars_df("AAPL", period="1y")   # OHLCV DataFrame; provenance in df.attrs
```

`get(kind, symbols, start=, end=, as_of=, provider=, concepts=, collapse_restatements=True)`

`get_bars_df(symbol, period=, start=, end=, as_of=, provider=)` — bars as an
OHLCV DataFrame (the shape the indicator/backtest code consumes) reconstructed
from the same Datums `get("bars", ...)` returns, so it flows through the gateway
(provider from the registry, not a hardcoded vendor). Provenance —
`provider`/`reliability`/`as_of_honored`/`caveats` — lives in `df.attrs`, the
right granularity for a bars frame. Empty result → empty DataFrame (not a
raise), so the live dashboard degrades to "no data". This is what
`tools/market_data.fetch_price_history` calls (M2b).

**kinds**: `bars` · `fundamentals_pit` · `corporate_actions` · `universe` ·
`macro` · `events` · `filings` · `short_interest` · `sentiment`
(only `bars` + `fundamentals_pit` have providers today; the rest raise
`NoProviderError` — loudly, because "no provider" and "no such data" are
different claims and conflating them is how a silent zero becomes a decision.)

Returns:
```json
{"kind":"fundamentals_pit","symbols":["AAPL"],"data":[{...Datum...}],"n":147,
 "provider":"sec-edgar","providers_tried":["sec-edgar"],"reliability":1.0,
 "as_of":"2024-01-01","as_of_honored":true,"excluded_future_datums":84,
 "retrieved_at":"...","unavailable":[],"warnings":[],"caveats":[]}
```

### `as_of_honored` is a promise, not a courtesy flag

True only when **all three** hold: the caller passed `as_of`, the serving
provider declares `pit_capable` for that kind, and the cut actually ran. Absent
any one, it is `false` and the result carries no PIT guarantee. A backtester
that ignores this flag is choosing look-ahead with its eyes open.
`excluded_future_datums` reports how many datums the cut removed — evidence the
filter ran, not a claim that it did.

### The Datum

```json
{"kind":"fundamentals_pit","symbol":"AAPL","concept":"revenue",
 "value":383285000000,"unit":"USD","period_start":"2022-09-25",
 "period_end":"2023-09-30","available_at":"2023-11-03",
 "retrieved_at":"2026-07-16T...","confidence":1.0,"status":"actual",
 "source":{"provider":"sec-edgar","document":"0000320193-23-000106",
           "ref":"us-gaap:Revenues","url":"https://www.sec.gov/..."},
 "datum_id":"a1b2c3d4e5f60718"}
```
`datum_id` is a **content hash**, not a counter: the same fact fetched twice
collapses to one ledger row, so evidence links stay stable across re-runs and a
report is still reproducible months later. `status` separates `actual` from
`estimate` — mixing them silently is how a backtest trades on analyst knowledge.

### The restatement rule (the whole trick)

1. `filter_pit()` — drop everything with `available_at > as_of`
2. `latest_by_period()` — of the SURVIVORS, keep the latest filed per period

**Order is load-bearing.** Collapsing first and filtering second leaks tomorrow's
revision into yesterday's decision. `test_financial_data.py` asserts the wrong
order provably fails, so a future refactor cannot quietly reverse it.

## Provider registry (`financial_data/registry.json`)

Providers are **data, not code** — the same P8 seam as `capability.json`. Vendor
names appear in the registry and in `providers/*` only; `gateway.py` imports by
the `module` path from the registry entry and contains no vendor name. Adding a
provider = one JSON entry + one module with `fetch()`. No caller changes.

| provider | kinds | pit_capable | reliability | why |
|---|---|---|---|---|
| `sec-edgar` | fundamentals_pit, filings | ✅ | 1.0 | `filed` is first-class; restatements append rather than overwrite → true PIT, free |
| `yfinance` | bars | ✅ (availability only) | 0.6 | incumbent, **demoted not deleted**; low score reflects schema drift, not price error |

Providers **never** filter on `as_of` themselves — the gateway owns that
guarantee so it holds identically for every provider, present and future.

**yfinance's honest caveat** (declared in the registry, surfaced in `warnings`):
split/dividend adjustments are applied *retroactively*. A bar's **availability**
is PIT-correct (a bar dated D was knowable at D's close), but its **value** is
restated by later corporate actions. Availability guarantee: holds. Value
stability: does not. Both facts travel with the data.

## EDGAR setup (required — free, no key)

SEC's fair-access policy **rejects any User-Agent without a real contact email**
with a hard `403` (verified empirically 2026-07-16 — a browser-style UA fails
too). Add to `stock_agent/.env` (gitignored):

```
SEC_USER_AGENT=Your Name your@email.com
```

Absent/invalid config raises `NotConfiguredError`, deliberately distinct from
`ProviderError`. During M1 this test skipped with "network unavailable" while
the real cause was the 403 — a config bug wearing an environment bug's clothes.
The types are separate so that can never recur silently.

~20 core concepts are mapped (`edgar.CONCEPT_MAP`), each to an ordered list of
us-gaap tags (companies tag the same quantity differently and change tags between
years). The tag actually used is recorded in `source.ref`.

## EvidenceLedger + TrialRegistry (`data/ledger.py`)

Additive to the existing SQLite file: `recommendations`/`outcomes` are
**untouched** (a live coach reader depends on them — memory/log.md C25.1).

```python
from data import ledger
cid = ledger.record_claim("net_margin", 0.2531, "net_income / revenue",
                          {"net_income": ni_datum, "revenue": rev_datum},
                          symbol="AAPL", as_of="2024-01-01", as_of_honored=True)
ledger.explain(cid)     # -> formula + every datum + filing accession + freshness
```

**`record_claim()` raises on empty evidence.** That refusal is the feature: "no
unsupported numeric claims" becomes a property of the system rather than an
instruction a model may ignore under pressure. An LLM cannot argue past a
constraint. `get_concept()` returns the *Datum*, not a bare float, so a caller
physically cannot cite a fact it never fetched.

**TrialRegistry** — every hypothesis ever backtested, alive or dead:
```python
ledger.record_trial("SMA cross beats buy&hold", {"fast":10,"slow":50}, "dead",
                    family="sma", sharpe=-0.2)
ledger.n_trials(family="sma")   # -> the TRUE n for Deflated Sharpe
ledger.trial_stats()            # -> {"total":.., "kill_rate":..}
```
The old `n_trials=7` constant understated reality by orders of magnitude, making
dSR a comfortable lie. Dead ideas are the *point* — they are the denominator.
`trial_id` is a hash of the experiment (not the moment), so re-running an
identical test does not inflate the count and decay dSR.

Tests write to a throwaway DB via `ledger.set_db_path()`. This is not hygiene
theater: test fixtures in the real registry would permanently corrupt dSR's
denominator — the same pollution that already made the platform's own
`metrics.jsonl` success rate meaningless (PLATFORM_IMPROVEMENT_REPORT M-Q1b).

## Verify

```bash
cd stock_agent && python3 tests/test_financial_data.py     # 78 checks, ALL PASS
```
Groups 1–7 are offline/deterministic; 8–9 hit live EDGAR and **skip** (never
fail) without network or config, reporting *which*. A guarantee that only holds
when sec.gov is reachable is not a guarantee.

Real end-to-end trace from the live suite:
```
trace: claim 5706f1a9e7150ce6 = 0.2531 via 'net_income / revenue'
  net_income       96,995,000,000 USD  filed 2023-11-03  accn 0000320193-23-000106
  revenue         383,285,000,000 USD  filed 2023-11-03  accn 0000320193-23-000106
```

## Known gaps (honest, tracked)

1. **`tools/market_data.py` — bars MIGRATED (M2b); fundamentals/news NOT.**
   `fetch_price_history` now routes through `get_bars_df` (yfinance provider,
   behavior bit-identical — verified max diff 0.0 on OHLCV vs the old direct
   call). Still on direct yfinance: `fetch_fundamentals`, `fetch_recent_news`,
   `fetch_earnings`, `fetch_analyst_ratings`. Those are the OLD yfinance
   fundamentals path; the point-in-time replacement is EDGAR (M1), not yet wired
   into the live pipeline. The boundary test pins BOTH facts (bars migrated,
   fundamentals not) so code and this doc cannot silently diverge. **Fundamentals
   migration to EDGAR is a later milestone.**
2. **Cache is JSON, not parquet** as the plan said. Parquet needs `pyarrow`; a
   heavyweight dep to store a few MB of filings is a bad trade in a stdlib-first
   repo. Deliberate deviation, recorded rather than silently swapped.
3. **`filings` kind declared but not implemented** — raises honestly.
4. **Only `bars` + `fundamentals_pit` have providers.** Alpaca/FRED/FINRA are
   registry entries away (M2).
5. **No consumer yet.** M1 built the layer; nothing in the 7-agent path reads it.
   That is intentional — wiring is M2/M3, and doing it here would have made this
   commit non-revertible.
6. **`as_of` intraday granularity**: EDGAR gives filing *dates*, not times, so a
   same-day decision treats the filing as available. Documented in
   `schemas.visible_at`; matters only for intraday strategies (out of scope).

## Rollback

Single commit, all-new files except two additive edits:
```bash
git revert <commit>        # complete removal; nothing else depends on FIL
rm -rf stock_agent/financial_data/.cache    # untracked cache, if desired
```
- New: `financial_data/**`, `data/ledger.py`, `tests/test_financial_data.py`, `FIL.md`
- Modified (additive only): `data/store.py` (+1 `DB_PATH` alias),
  `.gitignore` (+cache/parquet), `.env.example` (+SEC_USER_AGENT doc)
- **No existing behavior changed.** The 7-agent pipeline, web UI, and
  backtest suites are untouched and independently verified green.
- The ledger's tables are additive; `recommendations`/`outcomes` are not
  modified, so reverting cannot harm the coach's reader.

## Next (M2 proposal)

Migrate `tools/market_data.py` bars→gateway (closes gap 1) · add Alpaca + FRED
provider entries · wire `n_trials()` into `backtest/engine.py`'s dSR call
(closes the last measurement lie) · then M-F1's cost model + purged walk-forward CV.
