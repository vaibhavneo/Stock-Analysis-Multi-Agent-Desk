# Orchestration baseline — 2026-09-23

Recorded before any change, from **running code**, not from prior docs.

| | |
|---|---|
| HEAD | `5bc086ac2e1751ad985a9a86dfb1992cc25aea60` |
| Branch | `seven-agent-desk` |
| Working tree | clean (excluding `financial_data/.cache`) |
| Tests | **762 passing**, 72 files |
| Routing corpus | 110/111 = **99%**; symbols 111/111 = **100%** |
| Agents / capabilities | **11 / 12** |
| Data providers | 6 (`sec-edgar`, `yfinance`, `tiingo`, `fred`, `cboe`, `finnhub`) |
| Scheduler | `grade_outcomes`, runs=6, failures=0 |
| Production | `agentic-ai-production-aea7` — 7 routes 200, zero error lines |

## Entry points, as they actually exist

| Route | Path taken |
|---|---|
| `POST /api/chat` | `mas.converse.turn` → `intent.parse` → `policy.decide` → `run._run_capability` → adapter → `reply.compose` |
| `POST /api/decision-intelligence` | `web.app._build_decision_intelligence` → ~14 direct computes → `decision.engine.build_decision_intelligence` |
| `POST /api/mas/ask` | `mas.core.ask` → `core_read` → `policy` → `plan` → `execute` → `synthesis` |
| `POST /api/analyze/stream` | `agents.orchestrator.analyze_stock` — the 5 LLM analysts, SSE |

**These are four separate pipelines.** They share adapters but not a planner.

## The trace the prompt asked for

> "Should I add to my IONQ position?"

```
USER → /api/chat → intent.parse
         kind=QUERY  capabilities=['equity_research']  symbols=['I', 'IONQ']
     → policy.decide (CHAT implies nothing)
     → research adapter → mas.core.core_read   ← FAST pillar read only
     → reply._say_research
RESULT: "NVDA — ACCUMULATE (composite 61.4). On the numbers I'd add to it."
```

There is no planner between the question and the capability. Intent maps
1:1 to a capability, and the capability runs a fixed computation.

## The five defect classes

### 1. COMPUTED BUT NEVER CONSULTED — severe, on the prompt's own example

A chat question about **adding to a position** returns a composite score and a
volatility number. The following are computed by the decision engine and are
**unreachable from the conversational surface** — 41 fields:

`add_analysis` · `entry` · `entry_plan` · `playbook` · `position_context` ·
`mind_changers` · `monitoring` · `scenarios` · `risk_budget` ·
`statistical_honesty` · `statistical_edge` · `thesis` · `thesis_edge_relation` ·
`catalysts` · `conflict` · `confidence` · `consistency` · `decision_state` ·
`horizon_plans` · `horizon_read` · `level_map` · `quality` · `sizing` ·
`track_record` · `bearing` · `freshness` · `options` · `evidence` (+13 more)

The reply says *"On the numbers I'd add to it"* while `add_analysis` — which
exists specifically to answer that question, and carries the
`cheaper_not_better` guard against averaging down — was never called.

### 2. CONSULTED BUT NEVER USED DOWNSTREAM — mitigated

`add_analysis` is consumed by the UI, `narrative.py` and `playbook.py`. No
large instance found in the brief path.

### 3. TOOL EXISTS BUT ORCHESTRATOR DOES NOT KNOW IT EXISTS — confirmed

`mas/registry.json` registers **agents**. `financial_data/registry.json`
registers **providers**. Nothing connects them. Agents call
`from financial_data import get_bars_df` directly, so the orchestrator has no
visibility into, and no choice over, data acquisition. Six providers —
including SEC EDGAR filings, FRED macro, CBOE volatility and the Finnhub
earnings calendar — are invisible to planning.

### 4. AGENT EXISTS BUT ORCHESTRATOR CANNOT DELEGATE TO IT — confirmed

Measured on the prompt's own cases:

| Question | Routed to | Should reach |
|---|---|---|
| "Should I add to my IONQ position?" | `equity_research` | add engine + position context |
| "IONQ dropped 10%. Should I average down?" | `equity_research` | the averaging-down guard |
| "Should I reduce my position?" | `equity_research` | reduce/exit engine |
| "What could invalidate the thesis?" | **UNROUTABLE** | `mind_changers` (exists) |

There is no `ADD_TO_POSITION`, `REDUCE_POSITION` or `EXIT_POSITION` intent.

### 5. EVIDENCE EXISTS BUT LLM SUMMARY IS NOT GROUNDED — mitigated

`decision/narrative.py` builds a numeric allowlist from the structured object
and rejects any number the object does not contain. This class is handled.

## Additional live defect found during the trace

**The pronoun "I" resolves as a ticker.**

```
"Should I add to my IONQ position?"        → symbols ['I', 'IONQ']
"I own 44 shares at $197.80. Should I add?" → symbols ['I']          ← only 'I'
```

`AMBIGUOUS` in `mas/converse/symbols.py` contains `IF`, `IN`, `IS`, `IT`,
`ITS` — but not `I`. The lowercase scan excludes single letters; the
**uppercase** scan does not. So the second question analyses ticker `I`
instead of asking which stock is meant. This is the wrong-symbol failure the
guard exists to prevent, reached through the one spelling it omits.

**The position context is also discarded entirely**: "44 shares at $197.80"
is parsed out of nothing and reaches no engine.

## What is genuinely missing (the build target)

1. No research **plan** representation — intent maps straight to a capability.
2. No **tool layer** — data acquisition is hidden inside agents.
3. No **intent taxonomy** beyond capability names.
4. No **position-context extraction** from language.
5. No **evidence hierarchy** — FACT vs FORECAST vs LLM_EXPLANATION.
6. No **cross-agent synthesis** beyond capability-keyed prose blocks.
7. No **escalation** — one depth, always.
8. No **adversarial** pass.
9. No **execution trace** of what was used vs merely available.
10. No **consistency validator** over the final answer.
