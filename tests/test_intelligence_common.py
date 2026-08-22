"""
Verification for intelligence/common.py's is_stale() helper.

Run: python3 tests/test_intelligence_common.py

(This file used to also cover confidence_flagged(), a scalar-wrapping
helper that was removed: an audit of every new symbol from this session
found it had zero callers anywhere, including within the intelligence/
package itself - every module's actual output turned out to be a compound,
multi-field dict rather than a single wrapped scalar, so the helper never
fit any real use case. Rather than leave tested-but-unused code with a
docstring falsely claiming universal adoption, it was deleted; see
intelligence/common.py's module docstring for the corrected account. This
file's history is worth keeping in mind as a caution: a "shared helper"
built ahead of a concrete call site is a guess, and this session's own guess
was wrong.)
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.common import is_stale

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_is_stale_recent_is_not_stale():
    recent = datetime.now(timezone.utc).isoformat()
    check("a timestamp from right now is not stale", is_stale(recent, max_age_days=5) is False)


def test_is_stale_old_is_stale():
    old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    check("a 400-day-old timestamp is stale with a 5-day max age", is_stale(old, max_age_days=5) is True)


def test_is_stale_handles_naive_datetime():
    naive = datetime.now().isoformat()  # no tzinfo
    check("a naive (no-tz) recent timestamp is not stale", is_stale(naive, max_age_days=5) is False)


def test_is_stale_boundary_is_strictly_greater_than():
    exactly_at_limit = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    check("a timestamp exactly at max_age_days is NOT stale (boundary is strictly >, not >=)",
          is_stale(exactly_at_limit, max_age_days=5) is False)
    just_over = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
    check("one day past the boundary IS stale", is_stale(just_over, max_age_days=5) is True)


def test_is_stale_never_raises_on_garbage():
    for garbage in (None, "", "not-a-date", 12345, "2024-13-99"):
        try:
            result = is_stale(garbage, max_age_days=5)
            check(f"never raises on {garbage!r}", True)
            check(f"{garbage!r} treated as stale (conservative default)", result is True)
        except Exception as e:
            check(f"never raises on {garbage!r}", False, f"raised {e!r}")


if __name__ == "__main__":
    test_is_stale_recent_is_not_stale()
    test_is_stale_old_is_stale()
    test_is_stale_handles_naive_datetime()
    test_is_stale_boundary_is_strictly_greater_than()
    test_is_stale_never_raises_on_garbage()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — intelligence/common.py: honest by construction")
