# The Multi-Agent Desk

Stock Agent is the **primary orchestrator**. Specialists answer to it.

```
                    ┌────────────────────────────────────┐
   any symbol ─────▶│  ORCHESTRATOR  (mas/)              │
   any asset class  │  classify → plan → run → synthesize │
                    └───────────────┬────────────────────┘
                                    │  the plan is DATA, and does no I/O
        ┌───────────────────────────┼──────────────────────────┐
        ▼                           ▼                          ▼
┌─────────────────┐      ┌────────────────────┐     ┌─────────────────────┐
│  options_pilot  │      │    derivatives     │     │    cross_asset      │
│  remote · HTTP  │      │   local · model    │     │  local · regression │
│  TRADED_PRICE   │      │      MODEL         │     │ PRICE_HISTORY_DERIV │
│  ~31 equities   │      │  every underlying  │     │  every underlying   │
│  priority 10    │      │    priority 50     │     │    priority 20      │
└─────────────────┘      └────────────────────┘     └─────────────────────┘
┌─────────────────┐ ┌──────────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐
│    research     │ │  backtester  │ │  regime  │ │ catalysts │ │portfolio │
│  the desk's own │ │ beats hold?  │ │ market   │ │ what is   │ │ weights, │
│  read, class-   │ │              │ │ context  │ │ scheduled │ │ what to  │
│  aware          │ │              │ │          │ │           │ │ trim     │
└─────────────────┘ └──────────────┘ └──────────┘ └───────────┘ └──────────┘
                    ┌──────────────┐
                    │ track_record │  the only MEASURED_LIVE basis here:
                    │ what it got  │  outcomes frozen before they were known,
                    │ right, graded│  which is the one claim that cannot be
                    └──────────────┘  back-fitted
        │                                                      │
        └────────── findings return as DecisionEvidence ───────┘
```

## Why asset class comes first

Every analytic in this repo was written for a US equity and assumed one
silently. Routing a crypto symbol into that math does not fail — it returns
confident, wrong numbers. Three measured failures:

| Assumption | Where | What it did |
|---|---|---|
| `sqrt(252)` annualization | `market_data`, `xsection`, `catalysts`, +9 | BTC vol read **34.54%** where the truth is **41.57%** |
| `_pillar` → neutral `50.0`, still weighted | `backtest/pillars.py` | 10 points of fabricated fundamentals in every crypto composite |
| Vol bands `40/20` | `compute_algo_signals` | every coin permanently HIGH (firing the risk veto), every currency permanently LOW |

On identical BTC inputs the equity mask returned **HOLD**; the corrected mask
returns **ACCUMULATE**. The flag on the pillar never reached the arithmetic.

`mas/asset_class.py` states each class once — calendar, capability mask,
volatility bands, options venue — and everything downstream reads it.
`classify()` does **no I/O**: it reads the symbol's shape, and consults
caller-supplied metadata only to separate EQUITY from ETF. Shape wins over a
vendor field that disagrees, so one bad `quoteType` cannot put a coin back on
the 252-day calendar. `BRK-B` stays an equity — a single-letter suffix is a
share class, not a quote currency.

### The regimes, side by side

| | Vol | Regime | Calendar |
|---|---|---|---|
| ETH-USD | 47.16% | MEDIUM | 365d |
| NVDA | 46.45% | **HIGH** | 252d |

Near-identical numbers, opposite regimes — correctly. NVDA's HIGH feeds the
risk veto; ETH's MEDIUM does not.

## The roster is data

`mas/registry.json`. Adding a specialist is an entry plus a module; the
planner, executor and UI do not change. Every capability must declare what it
**writes** — OptionsPilot's pipeline trigger freezes ideas into its own
journal on another service, so it is excluded from planning unconditionally
and stays a thing a user presses.

## The plan is data, and does no I/O

Routing is the part of a multi-agent system most likely to be wrong and least
likely to be noticed, because a mis-routed request still returns a confident
answer — just from the wrong specialist. A plan built without a socket can be
asserted against for every asset class with no HTTP mocking; one test enforces
that by replacing `socket.socket` with a raising stub.

It also lets the UI show the routing **before** anything runs, including who
is *not* being asked and why. A specialist that was never consulted and one
that declined are different facts about an answer.

## Four outcomes, never collapsed

`OK` · `UNAVAILABLE` (exists, cannot answer this request) · `SKIPPED` (nobody
asked) · `ERROR` (it raised). A decline without a reason raises at
construction — it would be indistinguishable from a step that never ran.

Attempts run in priority order until one answers, so a real chain always beats
a model. A fallback that fires is **recorded as a fallback**: an answer from
the model engine because the chain was down is materially weaker, and the
reader has to be able to tell which they got.

## The local derivatives engine

Black-Scholes-Merton with greeks, implied-vol solving (Newton with bisection
fallback), twelve structures in OptionsPilot's own vocabulary, exact
piecewise-linear breakevens, risk-neutral probability of profit.

Validated against published values, put-call parity to `1e-9`, and **finite
differences of its own price function** — delta, gamma, vega and theta each
checked against the numerical derivative they claim to be, which is what
catches the 100× vega and 365× theta unit errors that still look plausible.

What it refuses to do:

- **Report implied volatility.** It reads no chain. Relabelling realized vol
  as market expectation would look exactly like signal. It declines and says
  what realized vol is instead.
- **Build an unbounded-loss structure.** Rejected from the leg set alone,
  before pricing. (Only the call side can run away; the downside is bounded
  because the underlying cannot go below zero.)
- **Treat a HOLD as a range view.** "No directional view" and "I expect it to
  sit still" are different claims, and only the second justifies selling
  premium. A condor pays only inside its band.

Strikes are sized to the **expected move**, not price magnitude. A grid from
magnitude alone broke on low-volatility underlyings: EUR/USD's 45-day
one-sigma move is 0.015 against a 0.025 grid, so ±1σ and ±2σ rounded to the
same level and the structure was dropped silently. Now `expected_move / 3`,
snapped to an increment a chain would list.

## The return path

The forward half — hand a direction down, get structures back — is a hand-off.
What makes this a system is that a specialist's finding can **change** the
orchestrator's read. Findings return as `DecisionEvidence`, the same shape the
seven pillars produce, so they flow through conflict analysis and confidence
rather than sitting in a panel nothing reads.

> Bitcoin explains **82%** of ETH's variance. So an ACCUMULATE read on ETH
> raises a stated tension: sizing it as an independent idea would double an
> exposure already held.

Emitted `NOT_DIRECTIONAL`, for the same reason the risk pillar is — moving
with bitcoin is not a claim about going up.

Correlations join on **shared consecutive dates**. BTC has 184 bars over six
months where the S&P has 127; zipping them correlates one asset's Monday with
the other's previous Thursday. The overlap is reported (BTC/SPX: 249, not 365).

## The conversational front door

The ticker box asks one question in one way. `POST /api/chat` asks any of
them, and the routing is **deterministic** — no LLM, no key, milliseconds.

That is a choice, not a shortcut. An LLM can invent a ticker, and a
hallucinated symbol yields a complete, confident analysis of a company nobody
asked about. It cannot be graded against a frozen corpus without paying for it
and accepting variance in the one component whose failures are silent. The
analysis path here is keyless by design, so making the *front door* need a key
would be the strictest dependency in the system. And the reasoning agents run
at a 180-second timeout — right for them, unusable for a chat turn.

**The corpus was written and frozen before the router existed.** 104
utterances, phrased the way people speak, deliberately not derived from the
keyword lists. A router graded on questions written from its own vocabulary
cannot fail, and the coverage gap is exactly what such a test cannot see.

| | Routing | Symbols |
|---|---|---|
| First measured run | **93%** | 96% |
| After closing the five gaps it exposed | **99%** | **100%** |

The one remaining miss is the case the corpus itself flags as genuinely
ambiguous — *"is the call on AAPL earnings already priced in"* — and firing
options on every bare "call" would misroute *"how did your last call turn out"*.

### The wrong symbol is worse than none

`ALL`, `IT`, `ON` and `A` are real US tickers and ordinary English words. They
resolve only from a `$` prefix, a company cue, or being the whole message.
Two real bugs came out of building that guard:

- an unanchored lower-case scan chopped `report` into `repor` + `t` and
  resolved **T** (AT&T) from inside a word
- requiring capitals threw away `nvda vs amd`, which then *blocked* the
  capability that had correctly fired — a routing success that looked like a
  routing failure

### Five outcomes, not one

Routed · small talk · a capability question · a question with no subject · one
that could not be read. Each needs a different thing said back.

*"What's the best stock to buy"* routes **nowhere**, on purpose. Production
found the sharp edge here: asked after a turn about NVDA, it inherited NVDA
and was answered as research on it — the desk picking a name, which is the one
thing the guard exists to prevent, defeated by the context it was supposed to
be independent of. The guard now fires on context-supplied symbols too.

### What carries between turns

The **subject**, and nothing else. Re-answering from a cached read would serve
yesterday's price as today's advice with nothing on screen to say so. So when
a follow-up asks only for options, the direction is **re-derived** rather than
remembered — without that, *"and the options?"* after an ACCUMULATE returned
range structures, which is a different trade than the one just argued for.

## The orchestrator decides what to call

Two callers used to ask for a fixed set every time, so *"how does AAPL look"*
paid for an options analysis nobody wanted — and on a name with no directional
view it **volunteered range structures the desk had never argued for**.
Suggesting a premium-selling trade to someone who asked about a share price is
not thoroughness; it is answering a question nobody posed.

`mas/policy.py` decides per request, against a measured latency budget, and
records a reason for **every skip as well as every run** — a capability that
silently did not run is indistinguishable from one that ran and found nothing.

**Explicit always runs.** Ask for options on a coin with no venue and no view
and they are still priced. A layer that decides it knows better than a stated
request is not autonomy, it is refusal — so autonomy applies only to
capabilities *implied* by the shape of the request, never to ones that were
named. Four tests exist only to hold that line.

| Depth | Implies | Used by |
|---|---|---|
| `CHAT` | nothing | the conversation — the router found what was asked |
| `BRIEF` | options | "analyse this name" |
| `FULL` | options, benchmark | the multi-asset panel |

Measured in production:

| Request | Agents called |
|---|---|
| "how does AAPL look" | **research** |
| "how does AAPL look **and show me the options**" | research, options_pilot, derivatives |
| AAPL brief (bullish equity) | options run |
| EUR/USD brief (bullish, no venue) | options **NOT_RUN**, with the reason |

The UI renders `NOT_RUN` as a judgement with a *"price them anyway"* button,
never as a failure.

## Scheduled upkeep

Production held **55 frozen predictions and 0 graded ones**. Every part of the
learning loop existed; nothing on the deployed service ever ran the grader —
the only thing that did was a LaunchAgent on a laptop. A loop with no clock
does not fail loudly, it just never produces evidence.

`data/maintenance.py` grades every six hours. Only grading: the heartbeat's
other steps cost ~15 minutes of a web dyno, and production already accumulates
snapshots from live use.

It is started at **import**, not under `__main__` — gunicorn imports the module
and never runs that block, which is exactly how a scheduler looks wired up and
never runs in the only environment that needs it.

Three bugs found building it, all of the silent kind:

- **Timezone skew.** SQLite's `strftime('%s', …)` parses a naive ISO string as
  UTC while `datetime.now()` writes local time. On any host not set to UTC a
  just-finished job reads as finished *in the future*, so it is never due
  again — the scheduler starts, logs nothing, and never runs. Comparisons now
  use float epochs Python produced.
- **Lock contention.** Concurrent claims raised `database is locked` on 2 of 8
  threads, because the schema script ran on every connection —
  `executescript()` issues an implicit COMMIT and takes a write lock. Schema
  now runs once per database per process, and a locked database reads as *not
  claimed*, which is what it means: the only other writer is another worker
  claiming the same job. 12 racing workers → 1 claim, 0 errors.
- **The suite was starting it.** Tests import `web.app`, so every run spawned a
  thread fetching price history and writing to the *real* ledger while the
  tests pointed it at temp files.

Verified firing on its own in production: claimed 07:18:51, finished 07:18:59,
55 snapshots evaluated, 330 outcome rows, 20 matured, 0 errors — and `runs: 1`,
so only one of the two workers took it.

## API

| Route | Does |
|---|---|
| `GET /api/mas/agents` | the roster, including declared writes |
| `GET /api/mas/plan?symbol=` | routing only — **no I/O**, instant |
| `POST /api/mas/ask` | classify, read, run specialists, synthesize |
| `POST /api/mas/price` | price one named structure on any underlying |
| `POST /api/chat` | the conversational front door — any question, any asset |
| `GET /api/maintenance` | what the scheduler has actually done |
| `POST /api/maintenance/run` | force one job now (still takes the lock) |

All read-only. Nothing here places an order, connects to a broker for
execution, or writes to another service.

## Verified in production

All six classes: `AAPL` `SPY` `BTC-USD` `EURUSD=X` `^GSPC` `GC=F`.

## Known gaps

- **`OPTIONSPILOT_ACCESS_CODE` is unset** in this service's Railway
  environment, so the chain specialist declines and everything falls back to
  the model engine — visibly, in the trace. Setting it upgrades every equity
  answer from MODEL to TRADED_PRICE with no code change.
- **Volatility is realized, not implied**, everywhere the model engine
  answers. This is the single largest source of error in anything it returns,
  and is stated on every result.
- **No funding/carry model.** `q` defaults to 0: the dividend yield for an
  equity, the foreign rate for a currency (Garman-Kohlhagen), the funding rate
  for a coin. A perpetual's funding can run several percent annualized and
  would move a long-dated price.
- **European exercise, constant volatility.** No early exercise, no smile.
