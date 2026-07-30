# Feature Dictionary (`xsec-features-v1`)

Every ranking feature is computed **point-in-time** in `xsection/features.py` and carries full
provenance: `feature_name, raw_value, normalized_value, available_at, period_end, source,
evidence_id, data_quality, missingness_reason, feature_version, formula`. A feature only enters
a ranking if its `available_at ≤ as_of`. `MIN_BARS = 130` trading days of history are required
before a name is `ELIGIBLE`.

`higher_is_better` sets orientation during normalization (valuation ratios and all risk metrics
are inverted so a higher normalized value always means "more attractive"). `sector_relative`
means the feature is percentile-ranked **within its sector**, not against the whole tape.

## Factors and their features

Factor score = mean of its **present** normalized features (missing features are dropped, not
zero-filled — so a name is never penalized for a data gap it can't control). Composite =
weighted sum of present factors, renormalized, minus `risk_penalty × risk_score`.

### quality
| Feature | higher_is_better | sector-relative |
|---|---|---|
| `gross_margin` | ✓ | ✓ |
| `operating_margin` | ✓ | ✓ |
| `fcf_margin` | ✓ | ✓ |
| `cash_conversion` | ✓ | — |
| `dilution` | ✗ (fewer new shares better) | — |

### growth
| `revenue_growth_yoy` | ✓ | ✓ |

### valuation
| Feature | higher_is_better | sector-relative |
|---|---|---|
| `price_to_sales` | ✗ (cheaper better) | ✓ |
| `fcf_yield` | ✓ | ✓ |
| `earnings_yield` | ✓ | ✓ |

### momentum
| Feature | higher_is_better | sector-relative |
|---|---|---|
| `mom_3m`, `mom_6m` | ✓ | — |
| `mom_12_1` (12-month, skip most recent month) | ✓ | — |
| `mom_voladj_6m` (vol-adjusted) | ✓ | — |
| `rs_vs_bench_6m` (relative strength vs benchmark) | ✓ | — |
| `trend_persistence` | ✓ | — |

### revisions
| `earnings_revision` | ✓ | — | — **usually `UNAVAILABLE`**: no PIT analyst-estimate history exists in free data. It is marked `source="unavailable"`, `missingness_reason="no_pit_estimate_history"`, and **never** back-filled with current estimates (that would be look-ahead). |

## Risk features (penalty, not an alpha factor)

`RISK_FEATURES` feed a `risk_score ∈ [0,1]`; risk can only **shrink** conviction
(`composite − risk_penalty × risk_score`) or trigger a **veto** (top `risk_veto_percentile`
names capped), never add a positive alpha weight — enforced by `validate_ranking_config`.

`realized_vol`, `downside_vol`, `max_drawdown`, `beta`, `leverage`, `gap_risk`.

## Fundamentals source

In the reference universe, fundamentals come from `synthetic_fundamentals()`: quarterly, with
`filed_date = period_end + 45 days`, and only periods with `filed_date ≤ as_of` are visible —
a filing lag that mirrors real EDGAR reporting and prevents peeking at not-yet-filed numbers.
In production these come through `FinancialDataGateway` / EDGAR `as_of` fundamentals with the
same filed-date discipline.

## Data-quality status values

`ELIGIBLE`, `PARTIAL_DATA`, `STALE_DATA`, `INSUFFICIENT_HISTORY`, `ILLIQUID`,
`DELISTED_BUT_VALID_HISTORICALLY` (kept for historical rankings/eval), `EXCLUDED`. Each ranking
row reports its `coverage`, `data_confidence`, `status`, and `flags`.
