# SecurityMaster — identity layer

Ranking across time is only honest if a company keeps a **stable identity** even when its
ticker changes, and if a **reused ticker** is never mistaken for the same company.
`SecurityMaster` (in `xsection/universe.py`) enforces this.

## The rule

- The primary key is a **permanent `security_id`** (e.g. `SEC0009`). It is assigned once and
  never changes, even across ticker changes, corporate actions, or delisting.
- A **ticker is only valid within an effective-date range.** `resolve(ticker, as_of)` returns
  the `security_id` that held that ticker on that date; `ticker_as_of(security_id, as_of)`
  returns the ticker a security traded under on that date.
- Ranks, ledgers, and evaluation join on `security_id`, never on ticker.

## Two edge cases it must get right

Both are encoded in the reference fixture and covered by `tests/test_xsection.py`:

1. **Ticker change preserves identity.** `SEC0009` traded as `KAPB` through 2020 and as
   `KPPA` from 2021. `resolve("KAPB", "2020-06-01")` and `resolve("KPPA", "2023-06-01")` both
   return `SEC0009`. Its price history and any prior rankings stay attached to the one identity.

2. **Reused ticker never merges companies.** `XILO` belonged to `SEC0013` (Xi Logistics,
   delisted 2021-12-31) and was later reused by `SEC0014` (Xi Innovations, from 2023-01-03).
   `resolve("XILO", "2021-06-01") == "SEC0013"` while `resolve("XILO", "2024-06-01") ==
   "SEC0014"`. The two are never collapsed into one time series — doing so would fabricate a
   continuous history that no real security had.

## Why this matters for bias

Substituting on ticker is a quiet form of look-ahead / survivorship error: it can graft a
successful company's later history onto a failed one that happened to share a symbol, or lose
a renamed company from a historical universe entirely. Anchoring on `security_id` makes both
mistakes structurally impossible.
