# Mission Prompt — Stock Agent Financial Intelligence Layer (FIL)

Reusable session prompt. Derived from `stock_agent/TRADING_DESK_PLAN.md`
(M-F1/M-F2); supersedes the earlier draft prompt. Paste the block below to
start an implementation session.

---

Mission: Build the Stock Agent Financial Intelligence Layer (executes
TRADING_DESK_PLAN.md milestones M-F1 + M-F2).

READ FIRST, in order: START_HERE.md → PROJECT_CHARTER.md (P1–P10) →
stock_agent/TRADING_DESK_PLAN.md → stock_agent/CLAUDE.md →
docs/API_REFERENCE.md §gateway (the retrieval gateway is the pattern to copy)
→ second_brain/gateway.py + memory/corpora/registry.json (provider-registry
precedent) → stock_agent/data/store.py (tables you will extend, incl. the
recommendations/outcomes ledger a live coach reader depends on).

Goal: every analysis is point-in-time correct, evidence-backed, historically
testable, and calibrated — and the measurement stack tells the truth or says
it can't. The four documented biases this kills (TRADING_DESK_PLAN §1):
survivorship, restated fundamentals, cost fantasy, undercounted trials.

GOVERNING RULE (from the plan, non-negotiable): no new signal or strategy
work lands until the honesty layer (M-F1) and data layer (M-F2) are green.

Implement one independently revertible milestone at a time:

1. **FinancialDataGateway** (M-F2) — provider-agnostic, pattern-copied from
   second_brain/gateway.py: providers are registry entries (JSON: id,
   reliability, rate_limit, kinds served) — vendor names live in that config,
   NEVER in skill/agent code (P8 seam; capability.json precedent).
   Contract: `market_data.get(kind, symbols, start, end, as_of=None,
   provider=None)` with kind ∈ {bars, fundamentals_pit, corporate_actions,
   universe, macro, events, filings, short_interest, sentiment}. `as_of` is
   first-class: every response carries `as_of_honored: bool` — biased data is
   permitted only when labelled (P7). Local parquet cache per
   provider/kind/symbol (rebuildable, never committed — P2). Existing
   yfinance/Reddit/StockTwits code becomes low-reliability providers behind
   the gateway; demoted, not deleted.
2. **SEC EDGAR/XBRL as primary fundamentals provider** — companyfacts +
   submissions APIs; as-reported values keyed by FILING date (that IS the
   point-in-time fix, free). Start with ~20 core concepts (revenue, NI,
   assets, equity, shares, OCF…). Tier-0 companions when cheap to add behind
   the same gateway: Alpaca (bars), FRED (macro/risk-free), FINRA (short
   interest), Tiingo/Finnhub free tiers (EOD cross-check, earnings calendar).
   Tier-1 (Sharadar PIT + delisted, ~$40/mo) is a LATER human decision —
   design the universe/fundamentals kinds so it drops in as one more provider.
3. **Timestamped normalization** — statements, filings, prices, estimates,
   macro, news, social → schemas where every datum carries: source (provider
   + document ref), period_end, available_at (filing/publication time — the
   anti-look-ahead field), retrieved_at, confidence, estimate|actual status.
4. **EvidenceLedger** — extends stock_agent/data/store.py (SQLite, same
   conventions; PRESERVE the existing recommendations/outcomes tables — a
   live coach reader depends on them). Every numeric claim in a report links
   to {datum ids, formula, computed_at}. Companion: a **TrialRegistry** table
   — every hypothesis/parameter-set ever backtested gets a permanent row, and
   the Deflated Sharpe's n_trials reads from it (kills the n_trials=7 lie;
   dead ideas stay recorded — P7, RenTec method).
5. **Point-in-time backtester** (M-F1) — extends the existing vectorized
   engine, does not replace it: (a) prohibits any datum with
   available_at > decision time (test with a known restatement case);
   (b) realistic costs — spread estimate from price/volume, square-root
   impact, per-side bps, borrow for shorts; (c) purged + embargoed
   walk-forward CV (López de Prado Ch.7 — companion to the dSR already
   reproduced at 3.255≈3.26) + Probability of Backtest Overfitting;
   (d) metrics: forward returns, benchmark-relative, drawdown (∈[0,1],
   asserted — L4), Brier score + calibration curve for probabilistic calls;
   (e) every result stamped {survivorship_safe, pit_fundamentals, cost_model}
   — false flags render prominently, never hidden.
6. **DataQualityAgent + InvestmentCommitteeAgent** — as AIOS skills via the
   Skill SDK (templates/skill/, full profile, validator VALID + quality ≥85).
   DataQualityAgent: deterministic driver (cross-provider tolerance checks,
   staleness, gap detection). InvestmentCommitteeAgent: agent-type,
   adapter-gated, schema-enforced verdict that must cite EvidenceLedger ids —
   an unsupported numeric claim is a schema failure, not a style issue.
7. **Model-comparison scenarios replacing single-run Monte Carlo** — GBM vs
   bootstrap vs regime-conditioned paths, shown side-by-side with each
   model's HISTORICAL calibration (from the backtester's Brier/calibration
   output). No single-model probability presented as fact.

Constraints (all enforced, not aspirational):
- Preserve the seven existing specialist agents; reposition them as research
  staff around the numeric core — LLMs never produce numbers (P4).
- Reuse AIOS runtime/mission/workflow/evaluator/memory/event contracts;
  mission-governed dispatch via mission.run_agent where applicable (D20/P15).
  No architectural change outside the Stock Agent integration boundary.
- No direct provider calls from agents or routes — gateway only (mirror the
  P9 rule: a grep for provider SDK imports outside the gateway must be empty,
  and add that grep as a test).
- No look-ahead: available_at governs everything; add a synthetic
  future-leak test (the proven leak-test pattern in tests/test_backtest_engine.py).
- No unsupported numeric claims; unavailable data is labelled
  ("unavailable", as_of_honored:false) — NEVER fabricated fallbacks.
- No live trading, no order execution (paper comes later as M-F6; live is
  gated at M-F7 — do not build toward it here).
- Do not commit API keys, DBs, caches, parquet, or downloaded market data
  (extend .gitignore first). Tier-0 free APIs only; any spend is a human
  decision recorded in decisions.md.
- Python 3.9 (Optional[...], not X|None in runtime-evaluated signatures);
  tests are standalone scripts (exit 1 / "ALL PASS"), idempotent, with ≥1
  negative-path and ≥1 bounds check each.

End state: a stock report can reproduce every calculation from the
EvidenceLedger, show data freshness and conflicting evidence per claim,
replay the recommendation as-of any historical date with no leaked data, and
report calibrated confidence with the honesty flags visible.

START WITH MILESTONE 1 ONLY: FinancialDataGateway + EDGAR fundamentals
provider + EvidenceLedger/TrialRegistry schemas (items 1, 2, 4).
Definition of Done for this session:
- gateway serves bars (Alpaca or yfinance-fallback) AND fundamentals_pit
  (EDGAR) with full provenance stamps; one cross-provider tolerance check;
- an `as_of` fundamentals query provably excludes a later restatement
  (test against a real known case);
- EvidenceLedger + TrialRegistry tables created alongside (not replacing)
  the existing store tables; one end-to-end demo: a computed metric traced
  claim → datum ids → EDGAR accession number;
- new tests green AND the full §V checklist green; .gitignore updated;
  docs: stock_agent/FIL.md (gateway contract, provider registry, schemas,
  rollback: `git revert` of the single commit — all new files, two existing
  files touched at most: store.py additive, .gitignore).
Then STOP after one revertible commit and a concise report: what shipped,
evidence (test output paths), honest gaps, and the next milestone proposal.
