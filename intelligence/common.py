"""
Shared helpers for the intelligence/ package.

Every module here follows the same honesty pattern backtest/pillars.py
::_pillar() established (a value paired with an honest confidence and
explanatory flags) so "never fabricate, always degrade explicitly on
missing/stale data" (item 9) is consistent - but each module's own output is
a compound, multi-field dict (a regime has trend/volatility/stance, a
forecast has one entry per horizon, etc.), so each carries its own bare
confidence/flags keys at the top level rather than every individual field
being wrapped through a shared scalar helper. is_stale() below is the one
piece of that pattern genuinely factored out, since staleness math is real
logic worth writing once (see intelligence/orchestration.py's price-history
staleness check, the one place a raw data timestamp - not a derived score -
needs checking against "how old is too old").
"""
from __future__ import annotations

from datetime import datetime, timezone


def is_stale(as_of_iso: str, max_age_days: int) -> bool:
    """True if a timestamp (ISO date or datetime string) is older than
    max_age_days relative to now. Never raises on a malformed or missing
    timestamp - treats it as stale, the conservative, honest default,
    rather than silently treating unparseable input as fresh.
    """
    if not as_of_iso:
        return True
    try:
        dt = datetime.fromisoformat(str(as_of_iso).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - dt).days > max_age_days
    except (ValueError, TypeError):
        return True
