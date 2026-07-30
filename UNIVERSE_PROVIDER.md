# UniverseProvider

The abstraction that answers **"which securities were investable as of date X?"** without
survivorship or look-ahead bias. Defined in `xsection/universe.py`.

## Interface

```python
class UniverseProvider:
    def universe_id(self) -> str
    def coverage(self) -> dict            # {start, end} or {status:"blocked_needs_dataset"}
    def benchmark_id(self) -> Optional[str]
    def members(self, as_of: str) -> List[dict]   # PIT membership; raises UniverseIncomplete outside coverage
    def prices(self, security_id, start, end) -> pd.Series
    def delisting_return_pct(self, security_id) -> Optional[float]
    def survivorship_safe(self) -> bool
```

`members(as_of)` returns only securities whose membership window contains `as_of`
**and** whose `first_tradable ≤ as_of` — a name added to the index later does not appear
earlier, and a name that has delisted before `as_of` is not returned (but its history is
still reachable for evaluation). Outside the provider's coverage it raises
`UniverseIncomplete` rather than guessing — an incomplete answer is never dressed up as a
complete one.

## Implementations

### `FixtureUniverseProvider` — keyless reference universe (`reference-smallcap-demo`)
Reads `xsection/fixtures/reference_universe.json`. **Genuinely survivorship-safe over its
coverage** (2019-01-02 … 2025-01-01): membership windows, ticker changes, additions, and
delistings are all encoded point-in-time. Prices are deterministic synthetic series
(`price_model` per security) — **NOT real market data**. It exists to prove the ranking
mechanics end-to-end without a paid feed, and every output is labelled `DATA_SOURCE =
reference_fixture (synthetic)`.

### `PaidUniverseProvider` — production interface (BLOCKED by default)
Interface for a real historical-constituent dataset (Sharadar SF1/SEP/TICKERS or EODHD).
Without the dataset key it raises:

```
UNIVERSE_INCOMPLETE: production survivorship safety BLOCKED — the {dataset}
historical-constituent dataset is not configured (set NASDAQ_DATA_LINK_API_KEY).
No membership is fabricated.
```

This is the deliberate, honest failure mode: the code will **never** substitute today's
index constituents for a historical date, because doing so silently reintroduces
survivorship bias. See [SURVIVORSHIP_POLICY.md](SURVIVORSHIP_POLICY.md).

## Why not just use current constituents?

Because "the stocks in the index today" is exactly the set that _survived_. Ranking a 2020
date against 2025's membership would omit every company that went to zero, was acquired, or
delisted — the precise losers a real 2020 investor was exposed to. That is survivorship bias,
and it inflates every backtested edge. This engine refuses to do it.
