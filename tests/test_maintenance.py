#!/usr/bin/env python3
"""The scheduler that keeps the forward record accumulating.

Production held 55 frozen predictions and 0 graded ones. Every part of the
learning loop existed; nothing on the deployed service ever closed it, because
the only thing that ran the grader was a LaunchAgent on a laptop. A loop with
no clock does not fail loudly — it just never produces evidence.

Two properties carry the weight here. Claiming must be atomic, because gunicorn
runs two workers and both start a scheduler. And the scheduler must NOT run
under pytest: the suite imports web.app, which starts it at import, so without
a guard every test run spawns a thread that fetches price history and writes to
the real ledger while the tests point it at temp files.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import maintenance as m
from data import prediction_ledger as pl

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def _tmp_ledger():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    pl.set_db_path(path)
    return path


# ── Claiming ──────────────────────────────────────────────────────────────

def test_a_claim_is_exclusive_until_it_finishes():
    _tmp_ledger()
    check("first claim wins", m.claim("grade_outcomes", 3600) is True)
    check("second is refused while held", m.claim("grade_outcomes", 3600) is False)
    m.finish("grade_outcomes", "OK", "done")
    check("still refused before the interval", m.claim("grade_outcomes", 3600) is False)
    check("granted once due", m.claim("grade_outcomes", 0) is True)


def test_a_dead_claimer_does_not_lock_the_job_forever():
    """A worker killed mid-run leaves claimed_at set and finished_at NULL.
    Without a staleness rule the job never runs again and the record silently
    stops accumulating."""
    _tmp_ledger()
    check("claimed", m.claim("grade_outcomes", 3600) is True)
    check("not reclaimable while fresh",
          m.claim("grade_outcomes", 3600, stale_claim_sec=3600) is False)
    check("reclaimable once the claim is stale",
          m.claim("grade_outcomes", 3600, stale_claim_sec=0) is True)


def test_exactly_one_of_many_racing_workers_claims():
    """The property that makes two gunicorn workers safe.

    Also asserts nothing RAISES. The first version of this raised
    "database is locked" on two of eight threads, because the schema script
    ran on every connection — executescript() issues an implicit COMMIT and
    takes a write lock, so every reader became a writer contending on a table
    that already existed."""
    _tmp_ledger()
    m._schema_applied.clear()
    won, errs = [], []
    barrier = threading.Barrier(12)

    def worker():
        barrier.wait()
        try:
            if m.claim("grade_outcomes", 3600):
                won.append(1)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("exactly one worker claimed", len(won) == 1, f"{len(won)} claimed")
    check("and nothing raised", errs == [], errs)


def test_a_locked_database_reads_as_not_claimed_not_as_an_error():
    """Contention IS the answer: the only other writer is another worker
    claiming this same job. Raising would turn correct behaviour into a
    logged failure."""
    import sqlite3
    _tmp_ledger()
    orig_conn = m._conn

    def locked():
        raise sqlite3.OperationalError("database is locked")

    m._conn = locked
    try:
        check("returns False rather than raising",
              m.claim("grade_outcomes", 0) is False)
    finally:
        m._conn = orig_conn


def test_timestamps_are_compared_as_epochs_not_parsed_strings():
    """SQLite's strftime('%s', ...) reads a naive ISO string as UTC while
    datetime.now() writes LOCAL time. On any host not set to UTC that skew
    makes a just-finished job look finished in the future, so it is never due
    again — the scheduler starts, logs nothing, and never runs."""
    _tmp_ledger()
    m.claim("grade_outcomes", 0)
    m.finish("grade_outcomes", "OK", "x")
    row = m.status()[0]
    check("an epoch column exists", "finished_epoch" in row.keys(), list(row.keys()))
    check("and it is a real timestamp, not a parsed string",
          abs(float(row["finished_epoch"]) - time.time()) < 60,
          row["finished_epoch"])
    check("the human column is kept too", bool(row["finished_at"]))


def test_failures_are_counted_not_swallowed():
    _tmp_ledger()
    m.claim("grade_outcomes", 0)
    m.finish("grade_outcomes", "FAILED", "boom")
    row = m.status()[0]
    check("failure recorded", row["status"] == "FAILED", row["status"])
    check("failure counted", row["failures"] == 1, row["failures"])
    check("and the reason is kept", "boom" in (row["detail"] or ""))


def test_run_job_reports_not_due_rather_than_pretending_it_ran():
    _tmp_ledger()
    calls = []
    m.JOBS["_probe"] = lambda: (calls.append(1), {"ok": True})[1]
    try:
        first = m.run_job("_probe", min_interval_sec=3600)
        second = m.run_job("_probe", min_interval_sec=3600)
        check("first ran", first["ran"] is True, first)
        check("second did not", second["ran"] is False, second)
        check("and says why", "not due" in second["reason"], second)
        check("the job body ran once", len(calls) == 1, len(calls))
    finally:
        m.JOBS.pop("_probe", None)


def test_a_forced_run_still_takes_the_lock():
    """Forcing must not bypass the lock, or a manual run can race the
    scheduled one and both grade the same bars at once."""
    _tmp_ledger()
    m.JOBS["_probe"] = lambda: {"ok": True}
    try:
        m.run_job("_probe", min_interval_sec=3600)
        forced = m.run_job("_probe", force=True)
        check("forced run ran", forced["ran"] is True, forced)
        row = [r for r in m.status() if r["job"] == "_probe"][0]
        check("both runs counted", row["runs"] == 2, row["runs"])
    finally:
        m.JOBS.pop("_probe", None)


def test_a_raising_job_is_recorded_and_does_not_escape():
    _tmp_ledger()

    def boom():
        raise RuntimeError("grading exploded")

    m.JOBS["_probe"] = boom
    try:
        out = m.run_job("_probe", force=True)
        check("no exception escaped", out["status"] == "FAILED", out)
        check("the message survives", "grading exploded" in out["error"], out)
    finally:
        m.JOBS.pop("_probe", None)


# ── The guard that keeps the suite clean ──────────────────────────────────

def test_the_scheduler_never_starts_under_pytest():
    check("start() refuses under pytest", m.start() is False)
    check("no thread was created", m._thread is None or not m._thread.is_alive())


def test_the_scheduler_can_be_disabled_by_environment():
    old = os.environ.get("MAINTENANCE_SCHEDULER")
    os.environ["MAINTENANCE_SCHEDULER"] = "0"
    try:
        check("env disables it", m.start() is False)
    finally:
        if old is None:
            os.environ.pop("MAINTENANCE_SCHEDULER", None)
        else:
            os.environ["MAINTENANCE_SCHEDULER"] = old


def test_grading_is_the_only_scheduled_job():
    """Freezing new predictions over the watchlist costs ~15 minutes of a web
    dyno. Production already accumulates snapshots from live analyses; grading
    was the half with no clock."""
    check("one job", set(m.JOBS) == {"grade_outcomes"}, sorted(m.JOBS))


def test_the_scheduler_shares_the_ledger_database():
    """If it opened its own connection path, a test redirecting the ledger
    would not redirect the scheduler, and it would write to the real store."""
    path = _tmp_ledger()
    m.claim("grade_outcomes", 0)
    import sqlite3
    conn = sqlite3.connect(path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        check("the table lives in the ledger db", "maintenance_runs" in names, names)
    finally:
        conn.close()


# ── The endpoints ─────────────────────────────────────────────────────────

def test_status_endpoint_reports_whether_it_is_enabled():
    from web.app import app
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.get("/api/maintenance")
    check("200", r.status_code == 200)
    d = r.get_json()
    check("enabled flag present", "scheduler_enabled" in d, list(d))
    check("interval reported", d.get("interval_sec", 0) > 0, d.get("interval_sec"))


def test_run_endpoint_rejects_an_unknown_job():
    from web.app import app
    app.config["TESTING"] = True
    r = app.test_client().post("/api/maintenance/run", json={"job": "rm -rf"})
    check("400", r.status_code == 400, r.status_code)
    check("and lists what it does know", "grade_outcomes" in r.get_json()["jobs"])


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
