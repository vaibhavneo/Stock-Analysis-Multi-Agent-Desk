# Production Data Activation Report (M-F3B)

**Date:** 2026-07-20 · **Scope:** activate the M-F3 cross-sectional ranking engine on
real, survivorship-safe, point-in-time data.

## Verdict

| Capability | Status | Evidence |
|---|---|---|
| **Real PIT feature pipeline** | ✅ **PROVEN** on real data | Part A below — real EDGAR filings + corp-action-adjusted prices, 93.1% feature coverage, 6.96% missing-data rate |
| **Survivorship-safe production ranking** | ⛔ **BLOCKED — needs license** | Part B below — requires the Sharadar historical-constituent dataset; the engine STOPS with `UNIVERSE_INCOMPLETE` rather than fake it |
| **Integration-ready provider adapter** | ✅ delivered | `xsection/providers/sharadar.py` — parse layer unit-tested offline; only the network fetch is key-gated |

Per the mission's explicit stop-condition — *"Stop if the required licensed dataset is
unavailable; produce an integration-ready provider adapter and operator decision report
instead"* — this is that report. Everything that can be real without a license **is** real
and proven; the one licensed input (historical index membership) is cleanly blocked, and
**no current-constituent or synthetic substitute is passed off as a historical universe.**

---

## What was built

| Component | File | What it does |
|---|---|---|
| Production universe provider | `xsection/providers/sharadar.py` | Survivorship-safe Sharadar adapter: `permaticker` permanent identity, ticker history, membership/delisting windows, SF1 filed-date fundamentals, ACTIONS-based delisting returns. Key-gated; `UNIVERSE_INCOMPLETE` without the license. |
| Real PIT feature adapters | `xsection/providers/edgar_features.py` | `edgar_fundamentals` (real SEC filings, filed-date governed, YTD→discrete-quarter differencing) + `WatchlistUniverseProvider` (real prices+fundamentals, explicitly **not** survivorship-safe). |
| Feature injection | `xsection/features.py`, `xsection/ranking.py` | `compute_features(..., fundamentals_fn=…)` — synthetic default unchanged; a provider injects real EDGAR fundamentals. Division None-guarded for real (partial) filings. |
| Dataset health report | `xsection/health.py` | Membership / delisting / ticker-mapping / feature coverage, stale & conflicting records, excluded reasons. `BLOCKED` cleanly on a licensed provider. |
| Reproducible backfill | `xsection/backfill.py` | Configurable universe/range/cadence; resumable, idempotent (stable content hash), checkpointed, rate-limit aware; writes only to a **gitignored** working dir. |
| Acceptance harness | `xsection/acceptance.py` | Part A real-feature validation (runnable) + Part B survivorship replay (STOPS without license). |
| Tests | `tests/test_xsection_production.py` | 40 offline/deterministic checks; the existing `test_xsection.py` and all other suites stay green. |

---

## Part A — Real PIT feature pipeline (PROVEN, keyless)

Ran the real pipeline on a labelled operator watchlist (`AAPL, MSFT, JNJ, XOM, KO`) as of
**2020-06-30**, benchmark SPY, using real corporate-action-adjusted prices (yfinance via the
FIL gateway, `auto_adjust=True`) and real SEC EDGAR fundamentals (filed-date governed).

```
n_securities            : 5
feature_coverage_mean   : 0.931
missing_data_rate       : 6.96%
data_quality status mix : {ELIGIBLE: 5}
prices                  : 754 corp-action-adjusted daily bars each
```

**Concrete point-in-time evidence** — each fundamental traces to a real SEC accession, filed
on or before the ranking date (a filing filed later is invisible):

| Ticker | revenue_growth_yoy | period_end | filed | SEC accession |
|---|---|---|---|---|
| AAPL | +8.37% | 2020-03-28 | 2020-05-01 | 0000320193-20-000052 |
| MSFT | +3.87% | 2020-03-31 | 2020-04-29 | 0001564590-20-019706 |
| JNJ | +0.63% | 2020-03-29 | 2020-04-29 | 0000200406-20-000035 |
| KO | −13.96% | 2020-03-27 | 2020-04-24 | 0000021344-20-000014 |

(The Coca-Cola −14% is the real Q1-2020 COVID revenue hit — not a synthetic artifact.) The
YTD→discrete-quarter differencing was validated against Apple's actual 10-Q: fiscal Q2-2020
revenue **$58.313B**, gross profit **$22.37B**, net income **$11.249B** — all matching the
filing, recovered as `H1 cumulative − Q1`.

**Explicitly unavailable (never faked):** analyst estimates, revisions, sentiment history,
macro vintages — there is no free point-in-time history for these, so they are marked
`UNAVAILABLE`/missing, per charter P4 and the no-look-ahead rule.

> ⚠️ A watchlist is **not** a universe. Part A validates that the *feature pipeline* is real
> and point-in-time; it makes **no** survivorship or edge claim, so rank-IC / decile / long-
> short are deliberately withheld here — those belong to Part B.

---

## Part B — Survivorship-safe historical ranking replay (BLOCKED)

The edge metrics the mission asks for — universe size, delisted members included, factor
coverage, **rank IC, top/bottom decile returns, long-short net, turnover/costs, missing-data
rate on the real universe** — are only meaningful (and only honest) on a survivorship-safe
universe that contains the names which later delisted, merged, or went bankrupt. That
requires a licensed historical-constituent dataset.

```
survivorship_ranking_acceptance("sharadar", …) ->
  status  : UNIVERSE_INCOMPLETE
  runnable: False
  reason  : production survivorship safety is BLOCKED — the Sharadar
            historical-constituent dataset is not configured
            (set NASDAQ_DATA_LINK_API_KEY). No membership is fabricated.
```

The engine STOPS here by design. Using today's index members for a 2018–2020 replay would
delete every company that failed — the exact survivorship bias this project exists to prevent.

### Operator decision required

To activate production survivorship-safe rankings, provision **one** licensed dataset:

| Option | Provides | Env var | Indicative cost |
|---|---|---|---|
| **Sharadar** (Nasdaq Data Link) — recommended, adapter built | SEP prices (`closeadj`), SF1 fundamentals (`datekey`=filed), TICKERS (permaticker + membership + delisting), ACTIONS | `NASDAQ_DATA_LINK_API_KEY` | ~\$150–300/mo (personal tier) |
| EODHD | Historical constituents + delistings + fundamentals | `EODHD_API_KEY` | ~\$60–100/mo |

Only the operator can authorize a paid subscription (entering paid credentials is outside an
agent's authority). No key is committed; the adapter reads it from the environment or
`stock_agent/.env` (gitignored).

### Activation steps (once a key is provisioned)

```bash
echo "NASDAQ_DATA_LINK_API_KEY=..." >> stock_agent/.env
pip install nasdaq-data-link
python3 -c "from xsection.health import dataset_health; from xsection.universe import get_provider; \
            import json; print(json.dumps(dataset_health(get_provider('sharadar'), \
            ['2018-06-29','2019-06-28','2020-06-30']), indent=2))"   # should now be OK, not BLOCKED
python3 xsection/acceptance.py                                        # Part B should now run + report edge metrics
```

The same code path that returns `UNIVERSE_INCOMPLETE` today produces a real survivorship-safe
ranking once the key is present — nothing else changes.

---

## Constraint compliance

| Constraint | How it is honored |
|---|---|
| Do not optimize weights | Weights untouched — still the pre-registered `RANKING_CONFIGS['xsec-v1']` (immutable, content-hashed). |
| Do not add factors or agents | No new factors; no new agents. Same factor set as M-F3. |
| Do not use current constituents historically | `PaidUniverseProvider`/Sharadar STOP without a license; there is no "today's constituents" fallback. |
| Do not substitute current estimates into past dates | Estimates/revisions remain `UNAVAILABLE`; fundamentals are filed-date filtered (`datekey`/`available_at ≤ as_of`). |
| Do not claim production survivorship safety without evidence | Production ranking labelled **BLOCKED**; Part A is explicitly non-survivorship-safe. |
| Do not alter immutable ranking/prediction records | No writes to `ranking_runs`/prediction ledger; backfill writes only to a gitignored working dir. |
| Preserve current APIs and synthetic tests | All existing endpoints unchanged; `test_xsection.py` + 15 other suites green; synthetic fixture ranking byte-identical (fundamentals_fn default = synthetic). |
| Stop if licensed dataset unavailable; ship adapter + report | This report + the integration-ready `SharadarUniverseProvider`. |

## Verification

```
python3 tests/test_xsection_production.py     # 40 offline checks — ALL PASS
python3 tests/test_xsection.py                # M-F3 suite — ALL PASS
# + all 14 other stock suites, docs/validate_repo_consistency.py,
#   docs/validate_api_reference.py — ALL PASS
```

## Rollback

Additive change: new files under `xsection/providers/`, `xsection/{health,backfill,acceptance}.py`,
`tests/test_xsection_production.py`, this report; plus a backward-compatible
`fundamentals_fn` parameter on `compute_features` (defaults to synthetic) and its pass-through
in `run_ranking`. `git revert <commit>` removes it all; the synthetic engine is unaffected.
