# Survivorship Policy

**Claim discipline:** the engine claims `SURVIVORSHIP_SAFE=true` on a ranking **only** when the
active `UniverseProvider` can prove point-in-time membership over the requested date. It never
claims safety it cannot demonstrate.

## What "survivorship-safe" requires

A ranking as of date X is survivorship-safe iff its universe is exactly the set of securities
that were **investable on X** — including the ones that later delisted, went bankrupt, were
acquired, or were renamed. Concretely the provider must supply:

1. Historical membership windows (`member_from` / `member_to` per security).
2. `first_tradable` dates (so index _additions_ don't appear before they existed).
3. Delisting dates + a documented delisting return (so removals are handled, not dropped —
   see [DELISTING_POLICY.md](DELISTING_POLICY.md)).
4. Stable identity across ticker changes (see [SECURITY_MASTER.md](SECURITY_MASTER.md)).

## Status of each provider

| Provider | Survivorship-safe | Notes |
|---|---|---|
| `reference-smallcap-demo` (fixture) | **Yes, over its coverage** | Synthetic data. Genuinely PIT membership + delisting, but NOT real market membership or prices. For mechanism proof only. |
| `sharadar` / EODHD (`PaidUniverseProvider`) | **BLOCKED** | Requires a paid historical-constituent dataset key. Until configured, it raises `UNIVERSE_INCOMPLETE` and produces nothing. |

## The forbidden shortcut

Using **today's index constituents** for a historical ranking date is prohibited. It is the
single most common way survivorship bias enters a study: the current membership is, by
definition, the set that survived, so every company that failed is silently missing. The
`PaidUniverseProvider` deliberately fails loudly rather than fall back to current constituents.

## For production

To make production rankings genuinely survivorship-safe, configure a historical-constituent
dataset (e.g. `NASDAQ_DATA_LINK_API_KEY` for Sharadar) and implement the membership +
delisting reads in `PaidUniverseProvider`. Until then, production survivorship safety is
**BLOCKED** and every UI/label says so — no synthetic or current-constituent data is passed
off as real historical membership.
