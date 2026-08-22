"""
Shared helpers for the intelligence/ package.

Mirrors backtest/pillars.py::_pillar()'s honesty pattern (a value paired with
an honest confidence and explanatory flags) without being coupled to pillar
scoring specifically - every new module in this package uses these so
"never fabricate, always degrade explicitly on missing/stale data" (item 9)
is consistent, not reinvented per module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def confidence_flagged(value: Any, confidence: float, flags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Wrap a computed value with an honest confidence in [0,1] and any flags
    explaining why confidence isn't 1.0 (missing inputs, stale data, thin
    sample size, etc.).

    `value` may be None when there's genuinely nothing to report - the
    caller must still supply a confidence (0.0 for "nothing computed") and
    must never fabricate a plausible-looking number instead of admitting
    that.
    """
    return {
        "value": value,
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "flags": list(flags or []),
    }


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
