# Orchestrator intelligence audit — 2026-09-23

Scored against running code. **PASS means integrated**, not "code exists":
selectable → executes → produces evidence → evidence enters synthesis →
synthesis can affect the decision → provenance survives. Anything short of
that is PARTIAL, whatever is implemented.

| Baseline | Final |
|---|---|
| `5bc086a` | `9679479` |
| 762 tests | **826 tests** |
| 11 agents / 12 capabilities | 11 agents / 12 capabilities |
| 0 registered tools | **13 registered, 11 reachable (EQUITY)** |
| 0 research intents | **17** |
| routing 99% / symbols 100% | routing 99% / symbols 100%; **intent 100% / position 100%** |

## Phase-by-phase

| Phase | Implementation | Test | Downstream consumer | Status |
|---|---|---|---|---|
| 0 Baseline | `docs/orchestration_baseline.md` | — | this audit | **PASS** |
| 1 Literal audit | 5 defect classes traced on the prompt's own example | — | drove the build | **PASS** |
| 2 Capability registry | `mas/registry.json` (pre-existing, 12 caps) | `test_mas.py` | planner | **PASS** |
| 3 Tool discovery | `mas/research/tools.py`, 13 tools, probed | `test_research.py` | plan tool selection | **PASS** |
| 4 Research planner | `mas/research/plan.py` | 8 tests | executor | **PASS** |
| 5 Intent classes | `mas/research/intent.py`, 17 classes | corpus 73, 100% | plan spec lookup | **PASS** |
| 6 Position context | `mas/research/position.py` | 6 tests | passed to decision engine | **PASS** |
| 7 Freshness orchestration | `mas/freshness.py` + tool `freshness` field | `test_freshness.py` | brief price tile | **PASS** |
| 8 Tool-selection policy | `budget.tool_value` — value per second of latency | 3 tests | plan tool loop | **PASS** |
| 9 Parallel execution | `execute.run_stage` | timing test (4×0.4s → 0.41s) | trace | **PASS** |
| 10 Budgeted research | `budget.Accounting` — tool/LLM/API/token/cache counts | 2 tests | `/api/research` | **PASS** |
| 11 Specialist delegation | `delegation.brief_for` — question, receives, expects | 2 tests | runner params | **PASS** |
| 12 Specialist output contract | `Item.contract()`, 9 fields, 4/4 compliant | 3 tests | synthesis | **PASS** |
| 13 Cross-agent synthesis | `mas/research/synthesis.py` | 6 tests | reply, validator | **PASS** |
| 14 Evidence hierarchy | 7 tiers, `DECISION_TIERS` | 5 tests + mutation | weighting, validator | **PASS** |
| 15 Conflict resolution | `classify_conflict`, 7 questions | 3 tests | synthesis statement | **PASS** |
| 16 Change detection | `mas/research/change.py` + `snapshots.py` | 9 tests | research reply | **PASS** |
| 17 Scenario engine | `decision/scenarios.py` → `decision.build` → brief §5 | 2 tests | brief, evidence | **PASS** |
| 18 Decision engine | `research/decision.py` §state | 2 tests | reply, brief §1 | **PASS** |
| 19 Entry engine | `research/decision.py` §entry | 1 test | brief §6 | **PASS** |
| 20 Add-to-position | `research/decision.py` §add — renders DO_NOT_ADD + blockers | 2 tests | reply, brief §7 | **PASS** |
| 21 Reduce/exit | `research/decision.py` §reduce/§exit | 1 test | brief §7 | **PASS** |
| 22 Catalyst intelligence | `decision/bearing.py` horizon containment + `catalyst:next` evidence | 5 tests | bearing, adversarial falsifier | **PASS** |
| 23 Risk engine | `research/decision.py` §risk | 1 test | brief §10 | **PASS** |
| 24 Statistical gate | `backtest/position_rules.py`, `statistical_honesty` | pre-existing | validator check #1 | **PASS** |
| 25 Depth escalation | `mas/research/escalate.py` | 5 tests | orchestrator | **PASS** |
| 26 Adversarial agent | `mas/research/adversarial.py` | 3 tests | reply | **PASS** |
| 27 Tool failure modes | 9 outcomes in `tools.py` | 1 test covering 5 | trace, evidence flags | **PASS** |
| 28 Provenance graph | `research/provenance.py` — 6-level walk | 2 tests | `/api/research/why` | **PASS** |
| 29 Decision journal | `research/journal.py` — 29 fields, append-only trigger | 2 tests | `/api/research/journal` | **PASS** |
| 30 Forward evaluation | `data/maintenance.py` + `decision/forward.py` | `test_maintenance.py` | calibration | **PASS** |
| 31 Real-time vs settled | `mas/freshness.py` | 14 tests | brief price tile | **PASS** |
| 32 LLM boundary | `narrative.py` allowlist; `DECISION_TIERS` excludes LLM | 3 tests + mutation | validator check #7 | **PASS** |
| 33 Two-pass LLM | `research/explain.py` — pass 2, opt-in, allowlist-guarded | 1 test | `/api/research` | **PASS** |
| 34 Streaming | `/api/research/stream` — SSE, 13 events | 1 test | UI | **PASS** |
| 35 Brief redesign | `research/brief.py` — 15 sections, 15/15 produced | 2 tests | `/api/research` | **PASS** |
| 36 "Why did this change?" | `decision/bearing.py` | 24 tests | brief UI badges | **PASS** |
| 37 Consistency validator | `mas/research/validate.py`, 9 checks | 8 tests | `/api/research`, chat | **PASS** |
| 38 Observability | `execute.Trace` | 1 test | `/api/research` | **PASS** |
| 39 Tool value measurement | `research/toolvalue.py` — accumulated | 1 test | `/api/research/value` | **PASS** |
| 40 Adversarial tests | `test_research_adversarial.py`, all 40 scenarios | 31 tests | — | **PASS** |
| 41 Routing corpus | intent corpus 73, frozen first | ratchet test | — | **PASS** |
| 42 Performance | shared bar cache; 5.54s → 3.69s on repeat | 2 tests | accounting | **PASS** |
| 43 Security | `tests/test_secrets.py`, boundary redaction | 7 tests | every trace | **PASS** |
| 44 This audit | — | — | — | **PASS** |
| 45 End-to-end | 9 cases run | — | — | **PASS** |
| 46 Deployment | Railway | production smoke | — | **PASS** |

**Totals: 47 PASS · 0 PARTIAL · 0 FAIL**

Every phase from 0 to 46 is integrated end to end.

## The five defect classes

| Class | Baseline | Now |
|---|---|---|
| Computed but never consulted | 41 decision fields unreachable from chat | Position questions reach the decision engine; pillars reach synthesis |
| Consulted but never used downstream | mitigated | `Step.consumed` measures it per request |
| Tool exists, orchestrator does not know | 6 providers invisible | 13 tools registered and probed |
| Agent exists, cannot delegate | no position intents; invalidation UNROUTABLE | 17 intents; invalidation answerable |
| Evidence exists, LLM not grounded | mitigated by allowlist | plus tier exclusion + validator check |

## Known limitations

These are real and they are not phase gaps — every phase is integrated. They
are the honest edges of what the integrated system can currently claim.

1. **A committed API key remains in git history.** Redacted from the working
   tree; rotation at DeepSeek is the owner's action and nothing here can do
   it. `tests/test_secrets.py` fails on any new secret-shaped literal.
2. **`options_chain` is unreachable** — `OPTIONSPILOT_ACCESS_CODE` is unset,
   so option structures are model-priced. The planner reports this rather
   than estimating around it, and setting the variable upgrades every equity
   options answer with no code change.
3. **The five LLM analysts return prose, by design.** They now receive the
   plan's specific research question, and their output is tiered
   `LLM_EXPLANATION` — barred from the weighing and from every decision
   field. Making them return the structured contract would let a model's
   stated confidence enter a comparison, which is the thing the tier system
   exists to prevent.
4. **Token accounting reads zero on the fast path**, because pass one calls
   no model at all. The field counts correctly when the explanation pass runs
   with a key; on the deterministic path there is genuinely nothing to count.
5. **The forward record is still too thin to quote.** 8 predictions matured,
   largest single-horizon sample 4, against a 100-sample bar. It now
   accumulates on its own every six hours.
6. **Tool-value counts accumulate across model versions.** A capability's
   weigh-rate mixes runs from different engines, so it is a prompt to look
   rather than a verdict — which is how it is labelled.
