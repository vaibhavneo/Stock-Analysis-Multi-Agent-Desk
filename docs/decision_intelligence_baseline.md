# Decision Intelligence — Phase 0 Baseline and Audit

Measured against the code at commit `ea5592d` (branch `seven-agent-desk`), not against
documentation. Every claim below was verified by reading the source or by executing it.

- Test suite at baseline: **416 passed, 0 failed** (`python3 -m pytest tests/ -q`, 34s).
- Live briefs captured for AAPL, NVDA, PFE, INTC on 2026-09-19 (see "Observed output").
- Evidence ledger at baseline: 1,537 frozen predictions, 298 matured at 20d, 3,606
  outcome rows carrying MAE/MFE, 86 distinct tickers, 119 quarantined.

---

## 1. Current architecture

```
tools/market_data.py        raw fetch: yfinance OHLCV, fundamentals, earnings,
                            analyst ratings, Reddit / StockTwits / web-forum sentiment
financial_data/             PIT-honoring gateway: EDGAR, FRED, CBOE(VIX), Tiingo,
                            Finnhub(events) — provider registry + disk cache
        ↓
backtest/pillars.py         6 deterministic pillar scores + composite
backtest/engine.py          vectorized backtest, Sharpe, deflated Sharpe
backtest/validation.py      walk-forward CV (purge+embargo), PBO
backtest/risk.py            ATR, half-Kelly (capped), vol targeting
        ↓
agents/recommendation.py    build_recommendation() — THE numeric authority
                            action verb, composite, 4 confidence dimensions,
                            statistical-edge gate, Kelly sizing, ATR levels,
                            horizon, evidence claim ids, honesty flags
        ↓
agents/decision_synthesis.py  synthesize_decision() — structured report
agents/decision_brief.py      build_decision_brief() — the 20-second brief (UI)
agents/portfolio_brief.py     portfolio-level version
        ↓
web/app.py                  Flask, 40 routes
web/static/index.html       3,274-line single-file frontend, 5 tabs
```

Parallel, and **not** connected to the Decision Brief:

```
intelligence/regime.py               market-wide regime (SPY trend, VIX, QQQ/SPY)
intelligence/historical_context.py   multi-horizon returns/vol/drawdown,
                                     support & resistance at 20D / 6M / 1Y
intelligence/analog_engine.py        historical analog matching
intelligence/prediction_engine.py    per-horizon p_up + bull/base/bear price ranges
intelligence/risk_engine.py          downside, R:R, drawdown risk, cost-basis math
intelligence/evidence_synthesis.py   reliability-weighted ledger + 4 named contradictions
intelligence/calibration.py          isotonic calibration, gated on purged CV
data/prediction_ledger.py            immutable frozen predictions + outcomes (MAE/MFE)
```

## 2. Current decision pipeline — exactly how the brief is produced

`POST /api/decision-brief` → `agents/decision_brief.py::build_decision_brief()`:

1. Guard on five required keys (`current_price`, `action`, `composite`, `confidence`,
   `levels`) → `INSUFFICIENT_EVIDENCE` if any is missing.
2. Build `position_context` from `avg_cost` (+ optional `shares`) if supplied.
3. Call `synthesize_decision(..., owns_position=False)` **once**, with the six
   `intelligence/` arguments left at their `None` defaults.
4. `_tier(rec)` → `positive` / `neutral` / `negative`.
5. `_refine_action(tier, branch)` → the 6-verb vocabulary, per ownership branch.
6. `_apply_position_awareness()` — fixed ±20% / −15% unrealized-P&L thresholds.
7. Assemble: one-line verdict, 3 insights, action plan, upgrade/downgrade triggers,
   evidence strip, better-ranked alternatives, fingerprint.

## 3. Current decision fields, traced to source

| Displayed field | Source | Deterministic? |
|---|---|---|
| `final_action` | `_tier` + `_refine_action` over `rec.action`, `composite`, `risk_veto`, statistical-edge level | Yes |
| `one_line_verdict` | f-string over the same | Yes |
| `composite_score` | `backtest/pillars.py` weighted sum × risk multiplier | Yes |
| `current_price` | `compute_indicators` last close | Yes |
| `action_plan.entry_low/high` | `rec.levels.entry_zone_*` = `p − 0.5·ATR`, `p + 0.25·ATR` | Yes (but see §6) |
| `action_plan.max_exposure_pct` | half-Kelly, capped 10%, **gated to 0** unless edge HIGH | Yes |
| `action_plan.time_horizon_days` | map from `vol_regime`: HIGH→45, MEDIUM→91, LOW→126 | Yes |
| `decisive_insights` | priority list over veto / edge / xsec / calibration / gating / consensus | Yes |
| `triggers.upgrade/downgrade` | dSR, xsec percentile, risk veto, max drawdown | Yes |
| `evidence_status.*` | `rec.confidence.*.level`, xsec, calibration `n` | Yes |
| `alternatives` | cross-sectional ranking rows above this ticker | Yes |
| `warnings` | `honesty_flags` + calibration + veto + gating | Yes |
| Scenario prose (`bull_case`/`bear_case`) in `/api/decision` | `rec.thesis.*` — **LLM-authored** | **No** |
| `thesis.key_catalysts` | **LLM-authored free text** | **No** |

So: the Decision Brief itself is fully deterministic. The only LLM-authored content
that can reach a user is `thesis` prose, and today it reaches `/api/decision`'s
`scenarios`, **not** the Decision Brief (which never reads `shared["scenarios"]`
except for `thesis_breakers[0]` as a possible third insight).

## 4. Where the final decision classification is made

Three places, and they can disagree:

1. `backtest/pillars.py::action_for()` — composite bands → BUY ≥70, ACCUMULATE ≥60,
   HOLD ≥45, REDUCE ≥35, SELL below.
2. `agents/decision_synthesis.py::_determine_action()` — BUY/HOLD/SELL/AVOID.
3. `agents/decision_brief.py::_tier()` + `_refine_action()` — the 6-verb vocabulary
   actually shown to the user.

## 5. Observed contradictions (live, 2026-09-19)

These are not hypotheticals — they are from the captured runs.

1. **AAPL: the recommendation card says `BUY`, the Decision Brief says `WATCH`.**
   `rec.action = BUY` (composite 71.8 ≥ 70 band) but `_tier()` demands statistical
   edge ∈ {MEDIUM, HIGH} and AAPL's is LOW, so the tier drops to neutral. Both are
   on screen at once with no reconciliation.
2. **INTC: `ACCUMULATE` + "Build the position gradually (partial size)" next to
   `max_exposure_note = "Unsupported — sizing withheld until the statistical edge is
   proven."`** The instruction says build a position; the sizing line says no size is
   supported. Nothing states how to act on both.
3. **The entry zone always contains the current price, by construction.**
   `[p − 0.5·ATR, p + 0.25·ATR]` — INTC printed 105.80–110.00 around a 107.5 last
   close. "Is now a good entry?" is therefore always answered *yes* whenever an entry
   zone is shown at all. It is ATR geometry, not an entry decision.
4. **`expected_return_pct` is a restatement of the stop distance, not a forecast.**
   `target = p + 2·max(2·ATR, 2%·p)` fixes reward:risk at exactly 2:1 for every
   ticker, so AAPL's "+9.16%" and INTC's "+20.61%" differ only by volatility. A
   positive target here carries no directional information whatsoever.
5. **PFE: composite 70.2 ("screens acceptably") with backtest Sharpe −0.977 and max
   drawdown 53.2%.** The brief's own backtest interpretation for this case reads "the
   core strategy has been unprofitable historically — no edge detected", but the
   composite is presented beside it without the conflict being named.
6. **dSR is 0.00 on all four tickers while statistical edge reads MEDIUM on two.**
   Honest (MEDIUM = net-cost-positive + walk-forward positive, dSR not yet convincing)
   but the words "MEDIUM edge" next to "dSR 0.00" invite exactly the misreading the
   safeguards exist to prevent.
7. **Ownership is presented as two parallel branches, never as two different
   problems.** `if_owned` / `if_not_owned` are the same tier re-worded. There is no
   add / trim / exit analysis, no averaging-down guard, no cost-basis-aware exit.

## 6. Fields calculated but never consumed

Verified by grepping every non-test consumer.

| Produced | Location | Consumed by |
|---|---|---|
| `rec["hit_rate"]` | `recommendation.py:211` | **nothing** |
| `rec["raw_kelly_pct"]` | `recommendation.py:197` | only its own fingerprint |
| `prediction_outcomes.mae_pct` / `mfe_pct` | ledger `_upsert_outcome` | **tests only** — 3,606 rows of real maximum-adverse / maximum-favourable excursion data, never read by any decision or report |
| `financial_data` `events` kind (Finnhub earnings calendar) | `providers/finnhub_events.py` | **zero callers anywhere** |
| `fetch_earnings()` / `fetch_analyst_ratings()` | `tools/market_data.py` | fed to LLM prompts in `orchestrator.py`; **never returned** in `analyze_stock()`'s result dict, so they cannot reach the brief |
| `intelligence/` outputs: `regime`, `historical_context`, `analog`, `forecast`, `risk_profile`, `evidence_ledger` | `intelligence/orchestration.py` | **only** `/api/intelligence`. `build_decision_brief` calls `synthesize_decision` with all six left `None` |
| `report["conviction"]`, `strongest_evidence`, `what_would_change_the_call` | `decision_synthesis.py` | `/api/intelligence` UI panel only; all degrade to `None`/generic on the brief's path because `evidence` is `None` there |
| `support_resistance` at 20D / 6M / 1Y | `historical_context.py` | `risk_engine` + `prediction_engine` only — never the brief's price levels |
| `rec["lanes"]` | `pillars.py:289` | LLM prompt grounding only |

**The single largest gap: the Decision Brief does not consume the intelligence layer
at all.** Contradiction detection, market regime, per-horizon forecasts, support and
resistance, and cost-basis risk are all computed and all unreachable from the brief.

## 7. Missing data / capabilities

- **No catalyst timeline reaches any decision surface.** Finnhub is key-gated and
  `FINNHUB_API_KEY` is unset; `fetch_earnings()` (yfinance) works but is dropped.
- **No position context beyond `avg_cost` and `shares`.** No portfolio size, no
  weight, no risk budget — so no position sizing in dollars can be honest.
- **No add/trim/exit analysis.** "Averaging down" is currently *encouraged* by
  `_apply_position_awareness`: at −15% on a positive signal it says the thesis
  "supports adding — this pulls your average cost down", with no check on whether the
  thesis strengthened or only the price fell.
- **No horizon separation.** Technical (days), algo momentum (1–12M) and fundamentals
  (quarters) are blended into one composite with fixed weights.
- **No decision journal of the *decision*.** The prediction ledger freezes the
  recommendation (action, pillars, gates); it does not record entry/add/reduce/exit
  conditions or invalidation, so those can never be graded.
- **No consistency checker.** Nothing detects the contradictions in §5.

## 8. Existing tests (baseline)

416 tests across 56 files. Relevant coverage: `test_decision_brief.py` (335 lines),
`test_decision_synthesis.py` (337), `test_decision_synthesis_intelligence.py`,
`test_evidence_synthesis.py`, `test_risk_engine.py`, `test_prediction_engine.py`,
`test_prediction_ledger*.py`, `test_trial_honesty.py`, `test_missing_stale_data.py`,
`test_pillar_grounding.py`, `test_intelligence_prompt_grounding.py`.

No mutation testing exists in this repo at baseline.

## 9. What Phase 1+ must preserve

The research engine is sound and stays untouched in its authority:

- `build_recommendation()` remains the only producer of composite, action verb,
  statistical-edge level, Kelly size and ATR levels.
- The statistical gate (walk-forward + embargo + purge + PBO + net-cost + min-sample)
  remains the only thing that unlocks non-zero sizing.
- `honesty_flags`, the dSR bar, the quarantine rules and the immutable snapshot
  triggers are not relaxed.
- The Decision Intelligence layer sits **above** these as a pure consumer, exactly as
  `decision_brief.py` already does — it may reorganize, separate and explain, but it
  may not invent a number.
