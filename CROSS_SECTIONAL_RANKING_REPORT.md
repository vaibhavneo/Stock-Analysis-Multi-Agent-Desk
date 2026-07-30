# Cross-Sectional Ranking — Evidence Report (M-F3)

Live evidence that the engine is survivorship-safe, point-in-time, identity-stable,
reproducible, and delisting-inclusive. All numbers below are reproduced from the keyless
reference universe; re-run the commands to regenerate them.

## 1. Survivorship: the universe changes with the date

`members(as_of)` over `reference-smallcap-demo`:

```
2020-06-01: ALPH BETA DLTA EPSI GAMA IOTA KAPB LMBD MUFN THET XILO ZETA   (12)
2023-06-01: ALPH BETA DLTA EPSI GAMA IOTA KPPA NUMT      THET XILO ZETA   (11)
```

- `LMBD` (SEC0010, bankrupt 2020-09-15) and `MUFN` (SEC0011, acquired 2022-03-31) are present
  in 2020 and **correctly gone** by 2023 — but their history stays reachable for evaluation.
- `NUMT` (SEC0012, added to the index 2021-04-01) is **absent in 2020**, present in 2023 — no
  look-ahead into a not-yet-added name.
- `KAPB → KPPA` reflects the 2021 ticker change of the **same** SEC0009.

## 2. Identity: ticker changes preserve identity; reused tickers never merge

```
resolve KAPB @2020 -> SEC0009      resolve KPPA @2023 -> SEC0009   (same company, renamed)
resolve XILO @2021 -> SEC0013      resolve XILO @2024 -> SEC0014   (reused ticker, NOT merged)
```

The 2023-06-01 ranking (screenshot in the dashboard) shows `XILO` resolving to **SEC0014**, the
company that reused the ticker — never grafted onto the delisted SEC0013's history.

## 3. Reproducibility + immutability

A ranking's `decision_fingerprint` is derived from the ranks alone. Re-running the same
`as_of` + universe + config reproduces the fingerprint and returns the **same frozen**
`ranking_run_id`; `UPDATE`/`DELETE` on `ranking_runs` are blocked by SQLite triggers.

```
as_of 2023-06-01 · config xsec-v1 (hash 5c489f2c) · fingerprint 369bef9e · frozen ✓
```

## 4. Delisting-inclusive evaluation (schedule, 2019-06 … 2024-01, monthly, 20d horizon)

```
dates_evaluated : 54
mean rank IC    : 0.0183      (≈ 0 — honest: synthetic prices have no real edge)
IC info ratio   : 0.0527
long-short net  : 2.628%      (net of 10bps/leg costs)
L/S ann. Sharpe : 0.508
deflated Sharpe : vs n_trials = 1 pre-registered config (fixed)
avg turnover    : 0.377
```

At the 252-day horizon, the bankrupt `LMBD` is **included** with a ~−100% contribution
(`n_delisted_included ≥ 1`) — the loss a survivorship-biased study would silently delete. The
near-zero IC is the correct, non-cherry-picked result on random reference data.

## 5. Honesty labels emitted on every ranking

`PIT_SAFE=true`, `SURVIVORSHIP_SAFE=true` (over fixture coverage only),
`CURRENT_CONSTITUENTS_ONLY=false`, `DELISTING_HANDLING=included; conservative delisting return`,
`ESTIMATE_HISTORY=UNAVAILABLE (no PIT estimate history in free data)`,
`DATA_SOURCE=reference_fixture (synthetic) — production needs paid constituent data`.

## 6. What is BLOCKED (and why that is the honest answer)

Production survivorship safety requires a paid historical-constituent dataset (Sharadar/EODHD).
`PaidUniverseProvider` raises `UNIVERSE_INCOMPLETE` until a key is configured and **fabricates
no membership**. See [SURVIVORSHIP_POLICY.md](SURVIVORSHIP_POLICY.md) and
[UNIVERSE_PROVIDER.md](UNIVERSE_PROVIDER.md).

## Reproduce

```bash
python3 tests/test_xsection.py                      # all 18 mission cases
python3 web/app.py                                  # dashboard → localhost:5051 (Cross-Sectional Ranking card)
curl -s -X POST localhost:5051/api/rankings/run     -H 'Content-Type: application/json' -d '{"as_of":"2023-06-01"}'
curl -s -X POST localhost:5051/api/rankings/evaluate -H 'Content-Type: application/json' \
     -d '{"start":"2019-06-01","end":"2024-01-01","cadence":"M","horizon":20,"cost_bps":10}'
```
