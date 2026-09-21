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

## API

| Route | Does |
|---|---|
| `GET /api/mas/agents` | the roster, including declared writes |
| `GET /api/mas/plan?symbol=` | routing only — **no I/O**, instant |
| `POST /api/mas/ask` | classify, read, run specialists, synthesize |
| `POST /api/mas/price` | price one named structure on any underlying |

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
