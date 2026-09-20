# Decision Intelligence — Phase 29 Literal Audit

Every requirement in the commissioning brief, with its code location, its test,
the evidence it was verified by, and a status.

**PASS is not granted for code existing.** It requires the feature to be
demonstrably *consumed downstream* — either a test asserts the output changes
when the feature's input changes (a mutation test), or a live run is shown
producing it. Where that could not be demonstrated, the status is PARTIAL and
the gap is named.

Baseline: 416 tests. After: **510 tests, 0 failures.**

---

## Phase 0 — Baseline and audit

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Inspect the complete repository | — | — | 33,582 LOC across 130 files enumerated | PASS |
| Identify agents, orchestration, scoring, statistics, backtesting, data, risk, calibration, brief, UI, APIs, persistence, tests | `docs/decision_intelligence_baseline.md` §1 | — | Architecture diagram traced from source | PASS |
| Identify exactly how the Decision Brief is produced | baseline §2 | — | 7-step trace of `build_decision_brief()` | PASS |
| Trace every displayed field to its source | baseline §3 | — | 14-row field table | PASS |
| Identify deterministic vs LLM-generated | baseline §3 | `test_llm_prose_cannot_change_the_decision` | Only `thesis` prose is LLM-authored, and it reaches `/api/decision`, not the brief | PASS |
| Identify where the final classification is made | baseline §4 | — | Three sites named: `action_for()`, `_determine_action()`, `_tier()`+`_refine_action()` | PASS |
| Identify contradictions between displayed metrics | baseline §5 | `tests/test_decision_consistency.py` | 7 live contradictions captured from AAPL/NVDA/PFE/INTC on 2026-09-19 | PASS |
| Do not trust documentation | — | — | Every claim read from source or executed | PASS |
| `docs/decision_intelligence_baseline.md` | the file | — | 193 lines | PASS |
| Dead / unused analytical outputs | baseline §6 | — | 9 named, incl. 3,606 unread MAE/MFE rows and the entire `intelligence/` layer | PASS |

## Phase 1 — Preserve the existing research engine

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Do NOT replace the existing agents | — | full suite | `agents/`, `backtest/`, `intelligence/`, `xsection/` unmodified except the one rule fix in §24 | PASS |
| New system sits ABOVE them | `decision/__init__.py` | `test_decision_intelligence.py` | Package docstring states the pure-consumer rule; `build_decision_intelligence()` takes existing outputs as arguments | PASS |
| LLM is NOT the source of the numerical decision | `decision/engine.py` `authority`, `decision/narrative.py` | `test_llm_prose_cannot_change_the_decision`, `test_a_lying_model_is_rejected_not_shown` | Fingerprint identical with and without prose | PASS |

## Phase 2 — Decision evidence object

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Normalized evidence with source/category/observation/as-of/horizon/direction/strength/reliability/validation/quality/provenance/relevance | `decision/evidence.py::DecisionEvidence` | `test_26`, `test_27`, `test_29` | 14-field dataclass with closed vocabularies validated in `__post_init__` | PASS |
| Contradiction relationship | `decision/conflict.py` | `test_20_contradictory_agents` | Conflicts computed over the normalized items rather than stored per item | PASS |
| Reuse existing infrastructure | `evidence.py` docstring, `analyze_conflicts(pillar_contradictions=…)` | `test_03` | `intelligence/evidence_synthesis.detect_contradictions` is passed through, not reimplemented | PASS |

## Phase 3 — Separate four different things

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| THESIS separate | `decision/thesis.py::build_thesis` | `test_01` | Direction/strength/net weight, no sizing or edge claim | PASS |
| STATISTICAL EDGE separate | `decision/thesis.py::build_statistical_edge` | `test_24` | NOT_DIRECTIONAL by construction | PASS |
| SCENARIO separate | `decision/scenarios.py` | `test_no_probability_without_calibration` | Conditionals with named invalidation | PASS |
| POSITION DECISION separate | `decision/state.py`, `decision/position.py` | `test_09`, `test_11` | Two independently-computed branches | PASS |
| UI must never imply strong thesis = proven edge | `relate_thesis_and_edge()`, UI "Thesis vs statistical edge" panel | `test_01` | `THESIS_WITHOUT_EDGE` relation rendered as its own bordered block | PASS |
| bearish thesis ≠ automatic short | `decision/state.py` `AVOID_NEW_POSITION` reason, `statistical_honesty.one_sided` | `test_02`, `test_30` | "not a short recommendation — this system evaluates long exposure only" | PASS |
| positive target ≠ automatic buy | `decision/entry.py` | `test_08` | Entry status is computed from location and reward:risk, never from the target | PASS |

## Phase 4 — Real decision state

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| The ten states | `decision/state.py::STATE_MEANING` | `test_decision_intelligence.py` | All ten defined with plain-English meanings | PASS |
| Do NOT force a direction when evidence is insufficient | `decide_if_not_owned` → `NO_TRADE` / `CONFLICTED` | `test_20` | `CONFLICTED` returned at agreement ≤ 55% | PASS |
| IF OWNED and IF NOT OWNED evaluated separately | `decide_if_owned` / `decide_if_not_owned` | `test_09`, `test_10`, `OWNERSHIP_BRANCHES_IDENTICAL` check | Different inputs: the owned branch additionally reads cost basis, risk budget and the add analysis | PASS |
| Do not assume ownership | `build_position_context` | `test_11`, `test_position_is_never_inferred` | `POSITION_CONTEXT_NOT_PROVIDED`, `owns: None` | PASS |
| Say POSITION CONTEXT NOT PROVIDED | same | `test_11` | Exact string in `status` and in `ownership_note` | PASS |

## Phase 5 — Scenario engine

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| BULL / BASE / BEAR minimum | `decision/scenarios.py::build_scenarios` | `test_no_probability_without_calibration` | Always three; live INTC run produced five | PASS |
| CATALYST case where supported | same | `test_18_catalyst_absent` | Emitted only when a dated event exists in the window | PASS |
| FAILURE case where supported | same | `test_24` | Emitted only when a real failure mode is measurable | PASS |
| Each carries thesis/evidence/direction/horizon/catalysts/risks/invalidation/conditions/levels/uncertainty | same | `test_no_probability_without_calibration` | All ten keys present on every scenario | PASS |
| DO NOT invent probabilities | `_probability_for()` | `test_no_probability_without_calibration`, `UNCALIBRATED_PROBABILITY` check | `probability` is None unless calibrated AND n ≥ 30 | PASS |
| "Probability not calibrated" must be displayed | UI scenario row | — | Live INTC: every scenario rendered "Probability not calibrated" | PASS |
| Never convert an LLM opinion into a probability | `decision/narrative.py::validate_narrative` | `test_uncalibrated_probability_is_caught` | Narrative is never parsed for numbers; a stated probability is rejected | PASS |

## Phase 6 — Conditional price paths

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Scenario ladder | `decision/levels.py::build_level_map` | `test_mutation_support_level_moves_invalidation` | Live INTC: 9 clustered levels from 18 candidates | PASS |
| Only display levels supported by actual data | `_add()` skips None/NaN/non-positive; `DERIVED_ONLY` status | `test_25_missing_price_history` | Returns a refusal, not a fallback band | PASS |
| Level sources: support/resistance, MAs, volatility bands, ATR, historical distribution, prior swings, model-derived | `SOURCE_CONFIDENCE`, `LEVEL_BASIS` | `test_25` | 19 sources across 4 basis classes | PASS |
| Volume profile, gap levels, options-derived | — | — | **Not implemented.** No options feed is configured and no volume-profile computation exists; fabricating them was the alternative | NOT_APPLICABLE |
| Every level has level/source/timeframe/confidence/rationale | `_cluster()` output | `test_25` | Plus `basis`, `distance_pct`, `n_sources` | PASS |
| Never manufacture precise prices | `LEVEL_BASIS` + `DERIVED_ONLY` | `test_25` | A ladder of only `CURRENT_PRICE_DERIVED` levels is refused outright | PASS |

## Phase 7 — Entry engine

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Is the current price an attractive entry? | `decision/entry.py::assess_entry` | `test_06`, `test_07`, `test_08` | Seven states; live runs produced ATTRACTIVE, ACCEPTABLE, WAIT and HIGH_RISK across four tickers | PASS |
| If not, what would make it attractive? | `conditions` | `test_08` | Conditions A (pullback), B (confirmation), C (volatility) with measurable sub-conditions | PASS |
| AVOID conditions | `avoid_conditions` | `test_17_catalyst_approaching` | Thesis invalidation, volatility regime, imminent binary event | PASS |
| Conditions, not guaranteed outcomes | `entry_plan.why_no_band`, `note` | `test_08` | No band is emitted for a non-supported entry | PASS |

## Phase 8 — Adding to a position

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Distinguish averaging down / up / pyramiding / pullback / catalyst-confirmation | `analyze_add()` `add_kind` | `test_14` | Five kinds, derived from position state and entry read | PASS |
| Do not automatically recommend averaging down | `cheaper_not_better` | `test_14` | `DO_NOT_ADD` with the headline "CHEAPER, NOT BETTER" | PASS |
| All ten questions evaluated | `checks` | `test_16` | Ten rows, each with an answer or an explicit UNKNOWN | PASS |
| PRICE FALLING ALONE IS NEVER A REASON TO ADD | `rule` field | `test_14` | Asserted verbatim | PASS |
| Distinguish CHEAPER from BETTER | `cheaper_not_better`, `thesis_delta` | `test_14`, `test_16` | `PRICE_MOVEMENT_ONLY` vs `NEW_INFORMATION` | PASS |
| Has the thesis strengthened *since*? | `prior_thesis` from `journal.latest_decision()` | `test_journal_writes_and_reads_back` | Answerable only once a prior decision is journaled; UNKNOWN on first sight, and it says so | PASS |

## Phase 9 — Position-management playbook

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| HOLD / ADD / REDUCE / EXIT / INVALIDATION / CATALYST WATCH / REVIEW DATE | `decision/playbook.py::build_playbook` | `test_30` | All seven produced | PASS |
| WHY: evidence for and against | `why` block | `test_30` | Drawn from `thesis.supporting_evidence` / `opposing_evidence` | PASS |
| DO NOT ADD IF | `do_not_add_conditions` | `test_14` | Populated from the add analysis's blockers | PASS |
| Deterministic where possible | `deterministic` flag per condition | `test_30` | Each condition carries `measurable_as` and a `deterministic` boolean | PASS |

## Phase 10 — Cost basis / position context

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Incorporate shares, avg cost, value, weight, unrealized P/L | `build_position_context` | `test_09`, `test_12`, `test_13` | All computed when supplied | PASS |
| Distance to cost basis, to invalidation, concentration, incremental risk, break-even | `build_position_context` + `build_risk_budget` | `test_12` | Recovery math is asymmetric (−37.5% → +60.0% verified) | PASS |
| Never infer these values | same | `test_11`, `test_invalid_position_values_are_dropped_not_guessed` | Unparseable or negative input yields no position, not a guess | PASS |
| Say POSITION CONTEXT INSUFFICIENT | `POSITION_CONTEXT_INSUFFICIENT` status | `test_11` | Distinct from NOT_PROVIDED | PASS |
| UI makes the missing information obvious | Risk + Adding panels | — | Live: "POSITION CONTEXT NOT PROVIDED" rendered as the section subtitle | PASS |

## Phase 11 — Risk budget

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| risk_to_stop, portfolio_risk_if_stopped, incremental risk, concentration, vol-adjusted exposure | `build_risk_budget` | `test_09` | Live INTC with a $50k portfolio: 2.41% portfolio risk against a 2.00% budget → within_risk_budget False | PASS |
| Correlation | — | — | **Not implemented here.** `agents/portfolio_brief.py` already computes cross-position correlation; duplicating it per-security would create a second, divergent implementation | NOT_APPLICABLE |
| Do not recommend a dollar amount without a supplied budget | `sizing.dollar_sizing` | `test_11` | `POSITION_SIZING_NOT_COMPUTABLE` | PASS |
| Say POSITION SIZING NOT COMPUTABLE | same | `test_11` | Exact string | PASS |

## Phase 12 — Contradiction engine

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| EVIDENCE AGREEMENT / CONFLICT / QUALITY | `decision/conflict.py::analyze_conflicts` | `test_20` | Three blocks in one result | PASS |
| WHAT/WHY/WHICH HORIZON/WHICH SOURCE/HOW RELIABLE/DOES IT CHANGE THE DECISION | per-conflict keys | `test_20` | All six answered on every conflict | PASS |
| Do NOT simply average contradictory scores | `_thesis_vs_edge_conflict`, `conviction` capping | `test_mutation_conflict_moves_the_evidence_confidence` | A contradiction lowers the evidence confidence dimension rather than being netted out | PASS |
| Reconcile the screenshot's own case (technical 60 / algo bearish / dSR 0.00 / calibration insufficient) | `decision/consistency.py` | `test_composite_verb_versus_state_is_surfaced` | The composite-vs-state asymmetry is emitted as a named INFO issue | PASS |

## Phase 13 — Horizon-aware synthesis

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Every signal carries a horizon | `PILLAR_PROFILE`, per-source assignment | `test_19` | Each `DecisionEvidence` has a `horizon` | PASS |
| Separate SHORT / MEDIUM / LONG | `decision/horizons.py::synthesize_by_horizon` | `test_19` | Three bands, each with its own lean and weight | PASS |
| Do not mix without adjustment | `synthesize_by_horizon` | `test_mutation_horizon_conflict_is_reported` | `ALL`-horizon items are deliberately not distributed into the bands, so nothing is counted three times | PASS |
| State which horizon drives the decision | `dominant_horizon`, `dominant_reason` | `test_19` | Live INTC: "short-term driven, 0.80 of 1.53 total weight" | PASS |

## Phase 14 — Catalyst engine

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Earnings, guidance, dividends, analyst changes | `decision/catalysts.py` | `test_17`, `test_18` | Live INTC: earnings 2026-10-22 with a $0.37–$0.48 consensus span, 5 recent analyst actions | PASS |
| Product / regulatory / macro / sector events | — | — | **Not implemented.** No provider supplies dated events of these kinds; inventing them is what the "no speculation" rule forbids | NOT_APPLICABLE |
| date / event / relevance / bull / bear / uncertainty | per-event keys | `test_17` | All six present | PASS |
| Do not speculate about unknown events | `build_catalyst_timeline` | `test_18` | Only provider-dated events are emitted; NO_EVENTS is distinguished from UNAVAILABLE | PASS |
| Answer "what am I waiting for?" | `waiting_for` | `test_17` | Prefers the next *material* event over the next calendar date | PASS |

## Phase 15 — What would change the mind

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| THESIS CONFIRMATION and INVALIDATION as first-class | `decision/scenarios.py::build_mind_changers` | `test_30` | Both lists, each item carrying `measurable_as` and `currently` | PASS |
| The user knows exactly what to monitor | `decision/playbook.py::build_monitoring_plan` | `test_17` | Five checks with cadence and state-change trigger | PASS |

## Phase 16 — Decision confidence

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Decompose into data / thesis / statistical / validation / calibration / scenario / decision | `decision/confidence.py::decompose_confidence` | `test_05`, `test_21`, `test_28` | Seven dimensions | PASS |
| Prevent data quality being mistaken for predictive evidence | `CAPPING_DIMENSIONS` + `_min_level` | `test_05`, `CONFIDENCE_WITHOUT_CALIBRATION` check | Headline can never exceed the weakest decision-critical dimension | PASS |

## Phase 17 — Statistical honesty

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Do not convert weak backtests into confidence | `build_statistical_edge` reads the gate verbatim | `test_24` | `demonstrated` is the gate's own `level == "HIGH"` | PASS |
| Do not hide multiple-testing correction | `statistical_honesty.multiple_testing` | `test_23` | dSR, n_trials and the fixed-variant rule are displayed | PASS |
| Do not hide one-sided results | `statistical_honesty.one_sided` | `test_22` | "This system evaluates long exposure only" | PASS |
| Do not hide insufficient samples | `sample_size` | `test_21` | Observations against the 504-bar minimum | PASS |
| Do not call uncalibrated outputs probabilities | `_probability_for` | `test_no_probability_without_calibration` | Enforced in code, checked again by `consistency` | PASS |
| Do not call historical return alpha | `forward.py` `safeguard` | `test_forward_summary_refuses_to_claim_an_edge_on_a_thin_sample` | The word "alpha" is asserted absent | PASS |
| Do not call a thesis a demonstrated edge | `relate_thesis_and_edge` | `test_01` | `THESIS_WITHOUT_EDGE` | PASS |
| Do not turn LLM explanation into statistical evidence | `validate_narrative` | `test_claiming_a_proven_edge_is_caught` | "proven edge" is rejected unless negated | PASS |
| If dSR is 0.00 the UI must not imply a proven edge | UI honesty strip | `test_23` | dSR chip renders red against its bar | PASS |

## Phase 18 — Decision quality

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| NOT another 0-100 score | `decision/quality.py` | `test_18`, `test_28` | Eight components, each status + explanation + provenance | PASS |
| DATA/EVIDENCE/THESIS/EDGE/RISK/CALIBRATION/CATALYST/POSITION_CONTEXT | `COMPONENTS` | `test_28` | All eight | PASS |

## Phase 19 — LLM synthesis layer

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| LLM receives the structured object | `build_narrative_prompt` | `test_prompt_contains_no_secrets` | The user message is the compacted object plus the fourteen questions | PASS |
| Its job is to explain, not decide | `SYSTEM_PROMPT`, `decision/engine.py` `authority` | `test_llm_prose_cannot_change_the_decision` | Narrative is generated *after* the object is complete and fingerprinted | PASS |
| All fourteen questions | `QUESTIONS` | `test_an_honest_model_is_accepted` | Live DeepSeek run answered all fourteen with field citations | PASS |
| Must cite the structured evidence | prompt instruction | — | Live output cites `[thesis.statement]`, `[statistical_edge.verdict]`, etc. | PASS |
| Never invent prices/probabilities/catalysts/metrics/results/significance/signals | `validate_narrative` | `test_fabricated_price_level_is_caught`, `test_a_lying_model_is_rejected_not_shown` | A live first attempt was rejected and retried; the second validated all 86 numbers | PASS |

## Phase 20 — Rebuild the Decision Brief UI

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| CURRENT STATE / WHY / THESIS / POSITION PLAYBOOK / ENTRY MAP / SCENARIO TREE / WHAT CHANGES MY MIND / CATALYSTS / RISK / EVIDENCE CONFLICT / STATISTICAL HONESTY / MONITORING PLAN / FULL AGENT EVIDENCE | `web/static/index.html` `_paintDecisionIntelligence` | `tests/test_decision_intelligence_api.py` | All sixteen sections rendered | PASS |
| One evidence snapshot → one synthesis object → multiple views | `_build_decision_intelligence` in `web/app.py` | `test_repeat_requests_are_served_from_cache` | Single engine call per request; sections read the same object | PASS |

## Phase 21 — Do not destroy existing agent tabs

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Keep Fundamentals / Technical / Social / Algo | `web/static/index.html` tab bar | — | Unmodified; only the fifth tab was replaced | PASS |
| The brief must not merely repeat the first sentence of every agent | `decision/engine.py` | `test_decision_intelligence.py` | The view contains no per-agent prose at all; it renders normalized evidence and derived state | PASS |

## Phase 22 — Decision journal

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Record ticker/timestamp/price/state/thesis/entry/add/reduce/exit/invalidation/catalysts/evidence/model version/data version/calibration state | `decision/journal.py` | `test_journal_writes_and_reads_back` | 38-column table, all fields captured | PASS |
| Do not rewrite historical decisions when the model changes | BEFORE UPDATE / BEFORE DELETE triggers | `test_journal_rows_are_immutable`, `test_model_change_does_not_rewrite_history` | The database itself rejects both verbs | PASS |

## Phase 23 — Forward validation

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Track decision/timestamp/return/MFE/MAE/time to target/time to invalidation/volatility/catalyst outcome | `decision/forward.py` | `test_forward_evaluation_computes_excursions` | All but catalyst outcome, which has no automated resolver | PARTIAL |
| Do NOT call this demonstrated edge until requirements are satisfied | `summarize_forward_validation` | `test_forward_summary_refuses_to_claim_an_edge_on_a_thin_sample` | `NO_DEMONSTRATED_EDGE` below 30 independent decisions | PASS |
| Existing research safeguards remain authoritative | `safeguard` field | same | Asserted verbatim | PASS |

## Phase 24 — Backtest position-management rules

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| buy immediately / wait for pullback / confirmation / staged / add-on-confirmation / reduce-on-deterioration / exit-on-invalidation | `backtest/position_rules.py::RULES` | `test_every_rule_produces_a_bounded_long_only_position` | Seven rules | PASS |
| Walk-forward, purging, embargo | `evaluate_position_rules` via `walk_forward_cv` | `test_full_evaluation_reports_honestly` | 5 folds, 50-bar purge, 2% embargo | PASS |
| Multiple-testing correction | deflated Sharpe at n_trials=7 + PBO across the rule set | `test_full_evaluation_reports_honestly` | Both computed | PASS |
| Out-of-sample evaluation | walk-forward mean OOS Sharpe per rule | same | Reported per rule | PASS |
| Realistic transaction costs and slippage | `DEFAULT_COST_MODEL` (spread + √impact + borrow) | same | Costs charged and reported | PASS |
| Survivorship safeguards | `method.survivorship` | same | Disclosed as *not* survivorship-safe rather than claimed | PASS |
| If no strategy demonstrates edge, report NO_DEMONSTRATED_EDGE | `verdict` | `test_no_demonstrated_edge_is_the_honest_default` | **10-ticker sweep: NO_DEMONSTRATED_EDGE.** `confirmation_entry` beat the baseline on 6/10 names but with mean dSR 0.00 | PASS |

## Phase 25 — Adversarial testing

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| All 30 named scenarios | `tests/test_decision_intelligence.py` | `test_01`…`test_30` | 30 scenario tests | PASS |
| MUTATE source evidence and verify the brief changes where the evidence is decision-relevant | `_mutate()` helper | 10 mutation tests | Pillars→thesis, gate→sizing, veto→state, support→invalidation, calibration→confidence, position→headline, catalyst→avoid conditions, conflict→confidence, volatility→cadence, horizons→conflicts | PASS |
| Prevent "computed but never consulted" | same | same | Each mutation asserts the output actually moves | PASS |

## Phase 26 — Consistency checks

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| target < price while labelled upside | `TARGET_BELOW_PRICE` | `test_bull_target_below_the_price_is_an_error` | ERROR | PASS |
| BUY while the statistical gate says no edge | `ACTION_WITHOUT_EDGE` | `test_add_recommendation_without_edge_is_an_error` | ERROR | PASS |
| sizing > 0 while the gate says 0% | `SIZING_GATE_VIOLATED` | `test_sizing_gate_violation_is_an_error` | ERROR | PASS |
| confidence HIGH while calibration INSUFFICIENT | `CONFIDENCE_WITHOUT_CALIBRATION` | `test_high_confidence_with_insufficient_calibration_is_an_error` | ERROR | PASS |
| entry = current price while the engine says WAIT | `ENTRY_AT_PRICE_WHILE_WAITING` | `test_entry_band_at_the_quote_while_waiting_is_an_error` | ERROR | PASS |
| Explicit warnings | UI consistency banner | `test_every_issue_names_the_fields_involved` | Rendered above the fold, before any other section | PASS |

## Phase 27 — Security / data provenance

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Secrets never enter prompts | `_compact()` | `test_prompt_contains_no_secrets` | Whitelisted keys only | PASS |
| API keys never appear in narratives | `validate_narrative` + `_compact` | `test_no_secret_reaches_the_response` | Asserted against the raw HTTP body | PASS |
| Source timestamps remain available | `DecisionEvidence.as_of`, `data_asof` | `test_29_stale_data` | Carried per item | PASS |
| Stale data is labelled | `data_quality == "STALE"` | `test_29_stale_data` | Surfaced as an evidence-quality issue | PASS |
| External data attributed | `provenance.module` | `test_decision_intelligence.py` | Every item names its producing module | PASS |
| Deterministic calculations remain deterministic | `_fingerprint` | `test_repeatability_same_inputs_same_fingerprint` | Identical inputs → identical fingerprint | PASS |
| LLM cannot modify source data | `generate_narrative` runs after the object is built | `test_llm_prose_cannot_change_the_decision` | Narrative is a sibling key, never an input | PASS |
| Historical decisions retain their original snapshot | journal triggers | `test_model_change_does_not_rewrite_history` | Immutable | PASS |

## Phase 28 — Performance

| Requirement | Code location | Test | Evidence | Status |
|---|---|---|---|---|
| Avoid duplicate API calls / historical calculations / repeated agent execution | `_build_decision_intelligence` gathers once | `test_repeat_requests_are_served_from_cache` | One recommendation build per request | PASS |
| Avoid unnecessary LLM calls | `narrative` is opt-in | `test_narrative_requests_are_never_cached` | No LLM call unless the user presses Generate | PASS |
| Cache deterministic evidence | `_DI_CACHE`, 300s TTL | `test_repeat_requests_are_served_from_cache`, `test_different_position_context_is_a_different_cache_entry` | Cache key includes the position context | PASS |
| One snapshot → one object → many views | `decision/engine.py` | `test_decision_intelligence.py` | Sections take the already-built evidence list as an argument | PASS |
| The page should not become slow | `deep` flag | — | The universe-wide cross-sectional ranking and the strategy-library race are off by default (~80s → ~10s, matching the existing `/api/intelligence` measurement) | PASS |

## Phase 30 — Final testing

| Requirement | Evidence | Status |
|---|---|---|
| All existing tests | 416 → 416 still passing, none modified in substance | PASS |
| All new tests | 94 new | PASS |
| Mutation tests | 10, all asserting the output moves | PASS |
| Integration tests | `tests/test_decision_intelligence_api.py` (11) | PASS |
| UI/API tests | same | PASS |
| Deterministic repeatability | `test_repeatability_same_inputs_same_fingerprint` | PASS |
| Performance tests | `test_repeat_requests_are_served_from_cache` | PASS |
| Security tests | `test_prompt_contains_no_secrets`, `test_no_secret_reaches_the_response`, `test_no_order_placement_route_exists` | PASS |
| Before/after comparison on real tickers | AAPL, NVDA, PFE, INTC captured before and after | PASS |

---

## Known limitations, stated plainly

1. **Options-derived and volume-profile levels are absent.** No options feed is
   configured. The level ladder is therefore built from price history,
   indicators and volatility only.
2. **Catalysts cover earnings, dividends and analyst actions only.** Product,
   regulatory and macro events have no dated provider here, and inventing them
   is precisely what the no-speculation rule forbids.
3. **Forward validation has one decision per ticker-day and a thin journal.**
   The framework is complete and tested; the *sample* is not there yet, which
   is why it reports NO_DEMONSTRATED_EDGE rather than a result.
4. **Catalyst-outcome tracking is not automated.** Whether an earnings result
   confirmed or refuted a thesis needs a resolver this system does not have.
5. **`thesis_delta` is UNKNOWN on a ticker's first decision.** "Has the thesis
   strengthened?" needs a prior journaled decision; it says UNKNOWN rather than
   guessing.
6. **Per-security correlation is not computed.** `agents/portfolio_brief.py`
   already does this at the portfolio level; a second implementation would
   eventually disagree with the first.
7. **The 30-decision minimum for forward inference is a stated prior**, chosen
   to match the existing calibration floor, not derived from a power analysis.
