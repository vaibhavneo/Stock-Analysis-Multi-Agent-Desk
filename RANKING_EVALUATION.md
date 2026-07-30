# Ranking Evaluation (delisting-inclusive)

Answers _"did the ranks actually work?"_ without the two biases that flatter every naive study.
Implemented in `xsection/evaluate.py`.

## No survivorship, no look-ahead

- **Delisting-inclusive:** a name that delists inside the horizon is measured to its last
  tradable price, then the documented terminal return is applied ([DELISTING_POLICY.md](DELISTING_POLICY.md)).
  It is counted in every long/short leg; each horizon reports `n_delisted_included`.
- **Look-ahead-free:** forward returns use only prices **after** the ranking date; the ranks
  themselves were built point-in-time upstream.

## Metrics

**Per ranking, per horizon** (1/5/20/60/252 trading days):
- top-decile return, bottom-decile return
- long-short spread (gross **and** net of realistic costs)
- benchmark-relative and **sector-neutral** long-short (returns demeaned within sector first)
- **rank information coefficient** (Spearman of composite vs forward return)
- top-decile hit rate vs benchmark

**Across a schedule of dates** (`evaluate_schedule`):
- IC time series + **IC information ratio** (mean/std)
- cost-aware long-short return series → annualized Sharpe + **deflated Sharpe (dSR)**
- **turnover** (1 − overlap of consecutive top sets)

## dSR is deflated against the pre-registered config count

The dSR uses `n_trials = len(RANKING_CONFIGS)` from `backtest/experiments.py` — the number of
**pre-registered** ranking configs, fixed and content-hashed. Weights are never tuned on the
evaluation window; if they were, the trial count would have to rise and the dSR would fall
accordingly. This closes the "try enough weightings, one looks good by luck" loophole.

## Reading results on the reference universe

The reference universe has **synthetic, essentially random** prices, so the honest expectation
is **mean rank IC ≈ 0** and no durable edge. Seeing ~0 there is the correct, non-cherry-picked
result — it demonstrates the evaluation isn't manufacturing signal. Real edge can only be
assessed once a real historical-constituent + price feed is wired via `PaidUniverseProvider`
([UNIVERSE_PROVIDER.md](UNIVERSE_PROVIDER.md)).
