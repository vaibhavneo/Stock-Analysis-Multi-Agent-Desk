"""
Verification for intelligence/common.py's shared honesty-pattern helpers.

Run: python3 tests/test_intelligence_common.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.common import confidence_flagged, is_stale

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_confidence_flagged_shape():
    r = confidence_flagged(42.0, 0.8, ["some_flag"])
    check("value passes through", r["value"] == 42.0)
    check("confidence rounds to 2dp", r["confidence"] == 0.8)
    check("flags list passes through", r["flags"] == ["some_flag"])


def test_confidence_flagged_clamps_range():
    r_hi = confidence_flagged(1, 5.0)
    r_lo = confidence_flagged(1, -3.0)
    check("confidence above 1.0 clamps to 1.0", r_hi["confidence"] == 1.0)
    check("confidence below 0.0 clamps to 0.0", r_lo["confidence"] == 0.0)


def test_confidence_flagged_defaults_flags_to_empty_list():
    r = confidence_flagged(None, 0.0)
    check("value may be None (no fabrication)", r["value"] is None)
    check("flags defaults to an empty list, not None", r["flags"] == [])


def test_is_stale_recent_is_not_stale():
    recent = datetime.now(timezone.utc).isoformat()
    check("a timestamp from right now is not stale", is_stale(recent, max_age_days=5) is False)


def test_is_stale_old_is_stale():
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    check("a 400-day-old timestamp is stale with a 5-day max age", is_stale(old, max_age_days=5) is True)


def test_is_stale_handles_naive_datetime():
    naive = datetime.now().isoformat()  # no tzinfo
    check("a naive (no-tz) recent timestamp is not stale", is_stale(naive, max_age_days=5) is False)


def test_is_stale_never_raises_on_garbage():
    for garbage in (None, "", "not-a-date", 12345, "2024-13-99"):
        try:
            result = is_stale(garbage, max_age_days=5)
            check(f"never raises on {garbage!r}", True)
            check(f"{garbage!r} treated as stale (conservative default)", result is True)
        except Exception as e:
            check(f"never raises on {garbage!r}", False, f"raised {e!r}")


if __name__ == "__main__":
    test_confidence_flagged_shape()
    test_confidence_flagged_clamps_range()
    test_confidence_flagged_defaults_flags_to_empty_list()
    test_is_stale_recent_is_not_stale()
    test_is_stale_old_is_stale()
    test_is_stale_handles_naive_datetime()
    test_is_stale_never_raises_on_garbage()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — intelligence/common.py: honest by construction")
