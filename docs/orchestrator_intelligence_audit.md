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
| 8 Tool-selection policy | `plan.build_plan` tool loop | 2 tests | execution | **PARTIAL** — selects by reliability and reachability; does not yet trade latency against value |
| 9 Parallel execution | `execute.run_stage` | timing test (4×0.4s → 0.41s) | trace | **PASS** |
| 10 Budgeted research | `Trace.over_budget`, per-depth budgets | 1 test | optional-step skip | **PARTIAL** — budget skips optional steps; no token/LLM-call accounting |
| 11 Specialist delegation | `plan.capabilities` derived from evidence | 2 tests | executor | **PARTIAL** — chooses WHICH and WHY; does not yet send a per-specialist question |
| 12 Specialist output contract | `evidence.Item` (14 fields) | 5 tests | synthesis | **PARTIAL** — the contract exists and adapters are normalised into it; the 5 LLM analysts still return prose |
| 13 Cross-agent synthesis | `mas/research/synthesis.py` | 6 tests | reply, validator | **PASS** |
| 14 Evidence hierarchy | 7 tiers, `DECISION_TIERS` | 5 tests + mutation | weighting, validator | **PASS** |
| 15 Conflict resolution | `classify_conflict`, 7 questions | 3 tests | synthesis statement | **PASS** |
| 16 Change detection | — | — | — | **FAIL** — not built |
| 17 Scenario engine | `decision/scenarios.py` (pre-existing) | pre-existing | brief; emitted as FORECAST evidence | **PARTIAL** — reached, not rebuilt |
| 18 Decision engine | `decision/state.py` (pre-existing, 10 states) | pre-existing | brief | **PARTIAL** — reachable from research now; research does not yet emit its own state |
| 19 Entry engine | `decision/entry.py` (pre-existing) | pre-existing | brief | **PARTIAL** |
| 20 Add-to-position | `decision/position.py::analyze_add` + `cheaper_not_better` | pre-existing | brief; now REACHED by ADD_TO_POSITION intent | **PARTIAL** — reachable; research reply does not yet render its verdict |
| 21 Reduce/exit | `decision/playbook.py` | pre-existing | brief | **PARTIAL** |
| 22 Catalyst intelligence | `decision/bearing.py` horizon containment + `catalyst:next` evidence | 5 tests | bearing, adversarial falsifier | **PASS** |
| 23 Risk engine | `decision/position.py::build_risk_budget` | pre-existing | brief, evidence | **PARTIAL** |
| 24 Statistical gate | `backtest/position_rules.py`, `statistical_honesty` | pre-existing | validator check #1 | **PASS** |
| 25 Depth escalation | `mas/research/escalate.py` | 5 tests | orchestrator | **PASS** |
| 26 Adversarial agent | `mas/research/adversarial.py` | 3 tests | reply | **PASS** |
| 27 Tool failure modes | 9 outcomes in `tools.py` | 1 test covering 5 | trace, evidence flags | **PASS** |
| 28 Provenance graph | `Item.provenance` + `Ledger` | 2 tests | evidence output | **PARTIAL** — every item traces to capability and tool; no graph walk / "Why?" UI |
| 29 Decision journal | `decision/journal.py` (pre-existing, append-only) | pre-existing | brief | **PARTIAL** — journals the brief, not the research plan |
| 30 Forward evaluation | `data/maintenance.py` + `decision/forward.py` | `test_maintenance.py` | calibration | **PASS** |
| 31 Real-time vs settled | `mas/freshness.py` | 14 tests | brief price tile | **PASS** |
| 32 LLM boundary | `narrative.py` allowlist; `DECISION_TIERS` excludes LLM | 3 tests + mutation | validator check #7 | **PASS** |
| 33 Two-pass LLM | research is deterministic; explanation separate | — | `/api/research` | **PARTIAL** — separation holds because pass 1 uses no LLM at all |
| 34 Streaming | `/api/analyze/stream` (pre-existing) | — | UI | **PARTIAL** — research pipeline is fast (0.7–5.5s) and does not stream |
| 35 Brief redesign | `mas/research/reply.py` sections | — | chat | **PARTIAL** — research reply has 6 sections; the full 15-section brief is unchanged |
| 36 "Why did this change?" | `decision/bearing.py` | 24 tests | brief UI badges | **PASS** |
| 37 Consistency validator | `mas/research/validate.py`, 9 checks | 8 tests | `/api/research`, chat | **PASS** |
| 38 Observability | `execute.Trace` | 1 test | `/api/research` | **PASS** |
| 39 Tool value measurement | `Step.consumed`, `Trace.summary` | 1 test | trace | **PARTIAL** — produced-vs-consumed measured per request; not aggregated over time |
| 40 Adversarial tests | 57 in `test_research.py` | — | — | **PARTIAL** — ~28 of the 40 listed scenarios |
| 41 Routing corpus | intent corpus 73, frozen first | ratchet test | — | **PASS** |
| 42 Performance | parallel stages, measured | timing test | — | **PARTIAL** — no cache-hit metric |
| 43 Security | `tests/test_secrets.py`, boundary redaction | 7 tests | every trace | **PASS** |
| 44 This audit | — | — | — | **PASS** |
| 45 End-to-end | 9 cases run | — | — | **PASS** |
| 46 Deployment | Railway | production smoke | — | **PASS** |

**Totals: 22 PASS · 22 PARTIAL · 1 FAIL · 1 N/A**

## The five defect classes

| Class | Baseline | Now |
|---|---|---|
| Computed but never consulted | 41 decision fields unreachable from chat | Position questions reach the decision engine; pillars reach synthesis |
| Consulted but never used downstream | mitigated | `Step.consumed` measures it per request |
| Tool exists, orchestrator does not know | 6 providers invisible | 13 tools registered and probed |
| Agent exists, cannot delegate | no position intents; invalidation UNROUTABLE | 17 intents; invalidation answerable |
| Evidence exists, LLM not grounded | mitigated by allowlist | plus tier exclusion + validator check |

## Known limitations

1. **Change detection (Phase 16) is not built.** Nothing compares this
   research result with a prior snapshot. The decision journal stores briefs,
   so the data exists to build it.
2. **Specialists are selected but not interrogated.** The plan says which
   capability and why; it does not yet send each one a specific research
   question, so Phase 11's "evaluate whether the recent fundamental
   deterioration changes the medium-term thesis" is not yet what a specialist
   receives.
3. **The 5 LLM analysts still return prose**, not the structured contract of
   Phase 12. They are explicit-only and never feed a number, so this is a
   capability gap rather than a correctness one.
4. **The research reply does not yet render the add/reduce/exit verdicts.**
   `analyze_add` is now reached; its verdict is not yet surfaced in the
   research answer's own words.
5. **A committed API key remains in git history.** Redacted from the working
   tree; rotation is the owner's action.
6. **`options_chain` is unreachable** — `OPTIONSPILOT_ACCESS_CODE` unset. The
   planner reports this rather than estimating around it.
