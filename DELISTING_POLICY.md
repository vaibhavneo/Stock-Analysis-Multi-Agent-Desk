# Delisting Policy

Delisted securities are **never dropped** from a ranking or its evaluation. Dropping them is
survivorship bias by omission — it deletes exactly the losses a real investor bore. Instead,
each delisting resolves to a documented, conservative terminal return.

## Terminal-return treatment

Applied in `xsection/evaluate.py::forward_return` when a security's price series ends inside
the evaluation horizon and a delisting is on record:

| Delisting reason | Terminal return | Rationale |
|---|---|---|
| Bankruptcy / liquidation | **−100%** | Equity is wiped out; this is the loss a survivorship study hides. |
| Going-private / hard delist | Conservative haircut (e.g. −35%) | Illiquid stub, uncertain recovery — err toward the loss. |
| Merger / acquisition (cash) | Return to the **buyout price** | The realized cash outcome, not a guess. |

The forward return is computed to the **last tradable price**, then the terminal return is
applied for the remainder of the horizon:

```
total = (1 + return_to_last_price) * (1 + delist_return_pct/100) − 1
```

The observation is marked `matured=True` (its fate is realized) and `delisted=True`, so
evaluation counts it in both the long and short legs and reports `n_delisted_included`.

## In the reference fixture

| Security | Event | `delist_return_pct` |
|---|---|---|
| `SEC0010` (Lambda Motors, `LMBD`) | Bankruptcy 2020-09-15 | −100 |
| `SEC0011` (Mu Financial, `MUFN`) | Acquired 2022-03-31 | buyout price 58.0 |
| `SEC0013` (Xi Logistics, `XILO`) | Delisted 2021-12-31 | −35 |

`tests/test_xsection.py` asserts that at the 252-day horizon the bankrupt name is **included**
with a ~−100% contribution, and that a cost-aware long-short spread still nets out lower than
gross — the losses are not quietly discarded.

## Principle

A conservative, documented number beats an optimistic omission. When the true recovery is
unknown, the policy rounds toward the loss, so the engine cannot flatter itself by forgetting
the companies that failed.
