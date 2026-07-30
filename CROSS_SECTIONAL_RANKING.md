# Cross-Sectional Ranking Engine (M-F3)

Turns the single-ticker tool into a **point-in-time opportunity-discovery engine**:
_"As of date X, using only securities and information available then, which stocks
ranked highest/lowest, why, and how did they perform afterward?"_

It is built to be free of the three biases that flatter almost every naive screen:

| Bias | How it is prevented |
|---|---|
| **Survivorship** | Membership is read from point-in-time windows; delisted names are ranked and evaluated, never dropped. See [SURVIVORSHIP_POLICY.md](SURVIVORSHIP_POLICY.md). |
| **Look-ahead** | Features are computed only from bars/filings with `available_at ≤ as_of`; estimates with no PIT history are marked `UNAVAILABLE`, never back-filled. |
| **Hidden substitution** | Identity is a permanent `security_id`, never the ticker. A ticker change preserves identity; a reused ticker never merges two companies. See [SECURITY_MASTER.md](SECURITY_MASTER.md). |

## Charter compliance

- **P4 (LLMs never produce/alter numbers):** every rank, factor score, and composite is
  deterministic arithmetic in `xsection/`. No LLM is on the ranking path. Prose may _explain_
  a ranking; it can never _change_ one.
- **Pre-registered weights (no tuning on the eval set):** the composite weights live in
  `backtest/experiments.py::RANKING_CONFIGS` (immutable, content-hashed). The engine reads
  the hash into every run; evaluation deflates against the fixed config count. Re-tuning
  weights on the evaluation window is impossible without a new registered config.
- **Reproducibility:** a `decision_fingerprint` is derived from the ranks alone; re-running the
  same `as_of` + universe + config yields the same fingerprint and the same frozen `ranking_run_id`.

## Pipeline

```
UniverseProvider.members(as_of)          # PIT membership (xsection/universe.py)
   → SecurityMaster.resolve/ticker_as_of # permanent identity
   → compute_features(member, as_of, …)  # PIT feature store (xsection/features.py)
   → data-quality gate                    # coverage/confidence/status/flags
   → normalize (winsor→percentile/robust-z, sector-relative)  # xsection/normalize.py
   → factor + composite scores            # pre-registered weights (xsection/factors.py)
   → ranks + screens + labels             # xsection/ranking.py
   → freeze to immutable ranking_runs     # content hash + triggers (recommendations.db)
   → evaluate forward (delisting-inclusive)  # xsection/evaluate.py
```

## Modules

| File | Responsibility |
|---|---|
| `xsection/universe.py` | `SecurityMaster`, `UniverseProvider`/`FixtureUniverseProvider`/`PaidUniverseProvider`, synthetic prices |
| `xsection/features.py` | PIT feature computation + provenance + data-quality status |
| `xsection/normalize.py` | robust cross-sectional transforms (winsorize, percentile rank, robust z, sector-relative) |
| `xsection/factors.py` | factor scores + risk penalty/veto + composite |
| `xsection/ranking.py` | orchestration + immutable Ranking Ledger + screens + labels |
| `xsection/evaluate.py` | delisting-inclusive forward evaluation (IC, long-short net of costs, turnover, dSR) |

## API (all keyless on the reference universe)

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/universes` | list universes + survivorship status |
| POST | `/api/rankings/run` | rank an `as_of` date (freezes immutably) |
| GET | `/api/rankings/<id>` | fetch a frozen ranking |
| GET | `/api/rankings/latest` | most recent ranking |
| GET | `/api/rankings/history` | list frozen runs |
| POST | `/api/rankings/evaluate` | delisting-inclusive historical evaluation over a schedule |

## Honesty labels on every ranking

`PIT_SAFE`, `SURVIVORSHIP_SAFE`, `CURRENT_CONSTITUENTS_ONLY`, `DELISTING_HANDLING`,
`ESTIMATE_HISTORY`, `DATA_SOURCE`. On the reference universe `SURVIVORSHIP_SAFE=true`
holds **over the fixture's coverage only** — it is synthetic data proving the mechanics.
Production survivorship safety requires a paid historical-constituent dataset and is
**BLOCKED** until one is configured (see [UNIVERSE_PROVIDER.md](UNIVERSE_PROVIDER.md)).

See also: [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md),
[DELISTING_POLICY.md](DELISTING_POLICY.md), [RANKING_EVALUATION.md](RANKING_EVALUATION.md),
[CROSS_SECTIONAL_RANKING_REPORT.md](CROSS_SECTIONAL_RANKING_REPORT.md).
