"""Scheduled upkeep, so the forward record maintains itself.

THE GAP THIS CLOSES
-------------------
Production held 55 frozen predictions and 0 graded ones. Every capability for
learning existed — snapshots are frozen on every live analysis, outcomes are
computable from price history, calibration reads them — but nothing on the
deployed service ever ran the grader. The only thing that did was a LaunchAgent
on a laptop, so the deployed desk could freeze predictions forever and never
find out whether any of them were right.

That is not a missing feature. It is a loop with no clock, and it is the exact
shape of failure `agents/heartbeat.py` was written for on the laptop side:
"accumulation requires something to run whether or not a person opens the
dashboard."

WHY GRADING ONLY
----------------
The heartbeat's four steps are GRADE, FIT, FORECAST, FREEZE. Only the first
belongs here. Freezing new predictions over a 96-name watchlist costs ~15
minutes of a web dyno's time, and production already accumulates snapshots
naturally — every live analysis freezes one. Grading is the half that was
missing, it is pure arithmetic over price history, and it is cheap.

TWO WORKERS, ONE JOB
--------------------
gunicorn runs 2 workers, and both import this module, so both would start a
scheduler and both would grade. Claiming is therefore done with a single
conditional UPDATE inside SQLite, which serializes writes: the worker whose
UPDATE reports a changed row owns the run, and the other finds nothing to
claim. Double-grading would not corrupt anything — `refresh_outcomes` is
idempotent by construction — but it would double the work and make the logs
lie about how often the job runs.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

GRADE_OUTCOMES = "grade_outcomes"

# Outcomes change only when new daily bars arrive, so more often than daily
# buys nothing. Six hours gives a few chances to catch a day whose first
# attempt failed, without ever grading the same bars repeatedly for value.
DEFAULT_INTERVAL_SEC = 6 * 60 * 60

# How long a claimed-but-unfinished run is assumed live before another worker
# may take it. A worker killed mid-run must not lock the job out forever.
STALE_CLAIM_SEC = 30 * 60

# First run is delayed so it never competes with a cold start serving traffic.
STARTUP_DELAY_SEC = 90

# Timestamps are stored TWICE on purpose: an ISO string for a human reading the
# table, and a float epoch for every comparison. SQLite's strftime('%s', ...)
# parses a naive ISO string as UTC, while datetime.now() writes local time — so
# on any host not set to UTC the arithmetic is skewed by the offset, and a job
# whose finished_at reads as "in the future" is never due again. The scheduler
# would appear to start, log nothing, and quietly never run. Comparing floats
# that Python produced removes the conversion entirely.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS maintenance_runs (
    job            TEXT PRIMARY KEY,
    claimed_at     TEXT,
    claimed_epoch  REAL,
    finished_at    TEXT,
    finished_epoch REAL,
    status         TEXT,
    detail         TEXT,
    runs           INTEGER NOT NULL DEFAULT 0,
    failures       INTEGER NOT NULL DEFAULT 0
);
"""

_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


_schema_applied: set = set()


def _conn():
    """Shares the ledger's database and its path override, so a test that
    redirects the ledger redirects this too — a scheduler writing to the real
    store while the tests write to a temp file is a slow, invisible poisoning
    of the calibration record.

    The schema runs ONCE per database per process. executescript() issues an
    implicit COMMIT and takes a write lock, so re-running it on every
    connection turned every read into a writer and made concurrent claims
    collide on a table that already existed.
    """
    from data import prediction_ledger as pl
    conn = pl._conn()
    key = pl._db()
    if key not in _schema_applied:
        conn.executescript(_SCHEMA)
        conn.commit()
        _schema_applied.add(key)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def claim(job: str, min_interval_sec: float,
          stale_claim_sec: float = STALE_CLAIM_SEC) -> bool:
    """Atomically take ownership of `job`, or return False.

    One conditional UPDATE. SQLite serializes writers, so exactly one of two
    racing workers sees rowcount 1.

    A locked database is reported as NOT CLAIMED rather than raised, because
    that is what it means: another writer holds the lock, and the only writer
    here is another worker claiming this same job. Raising would turn correct
    contention into a logged error, and a caller that treats an error as a
    failed cycle would skip a run it never needed to make.
    """
    import sqlite3
    try:
        conn = _conn()
    except sqlite3.OperationalError:
        return False
    try:
        conn.execute(
            "INSERT OR IGNORE INTO maintenance_runs (job, runs) VALUES (?, 0)",
            (job,))
        conn.commit()
        now = time.time()
        cutoff = now - float(min_interval_sec)
        stale = now - float(stale_claim_sec)
        cur = conn.execute(
            """UPDATE maintenance_runs
                  SET claimed_at = ?, claimed_epoch = ?, status = 'RUNNING'
                WHERE job = ?
                  AND (
                        -- never run
                        claimed_epoch IS NULL
                        -- finished long enough ago to be due again
                     OR (finished_epoch IS NOT NULL AND finished_epoch <= ?)
                        -- claimed but the claimer evidently died
                     OR (finished_epoch IS NULL AND claimed_epoch < ?)
                  )""",
            (_now(), now, job, cutoff, stale))
        conn.commit()
        return cur.rowcount == 1
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            return False
        raise
    finally:
        conn.close()


def finish(job: str, status: str, detail: str) -> None:
    conn = _conn()
    try:
        conn.execute(
            """UPDATE maintenance_runs
                  SET finished_at = ?, finished_epoch = ?, status = ?, detail = ?,
                      runs = runs + 1,
                      failures = failures + CASE WHEN ?='FAILED' THEN 1 ELSE 0 END
                WHERE job = ?""",
            (_now(), time.time(), status, detail[:1000], status, job))
        conn.commit()
    finally:
        conn.close()


def status() -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM maintenance_runs ORDER BY job")]
    finally:
        conn.close()


def grade_outcomes() -> Dict[str, Any]:
    """Mature every frozen prediction whose horizon has elapsed. Idempotent."""
    from data import prediction_ledger as pl
    res = pl.refresh_outcomes()
    return {k: v for k, v in (res or {}).items() if k != "errors"} | {
        "n_errors": len((res or {}).get("errors") or [])}


JOBS: Dict[str, Callable[[], Dict[str, Any]]] = {
    GRADE_OUTCOMES: grade_outcomes,
}


def run_job(job: str, min_interval_sec: float = 0.0,
            force: bool = False) -> Dict[str, Any]:
    """Claim and run one job. Returns what happened, including 'not due'."""
    fn = JOBS.get(job)
    if fn is None:
        return {"job": job, "ran": False, "reason": f"no such job: {job}"}
    # force still goes through claim() with a zero interval, so a forced run
    # still takes the lock rather than racing the scheduled one.
    if not claim(job, 0.0 if force else min_interval_sec,
                 stale_claim_sec=0.0 if force else STALE_CLAIM_SEC):
        return {"job": job, "ran": False,
                "reason": "not due, or another worker holds it"}
    t0 = time.time()
    try:
        detail = fn()
        out = {"job": job, "ran": True, "status": "OK",
               "elapsed_ms": int((time.time() - t0) * 1000), **detail}
        finish(job, "OK", str(detail))
        print(f"[maintenance] {job} OK in {out['elapsed_ms']}ms: {detail}", flush=True)
        return out
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        finish(job, "FAILED", msg)
        print(f"[maintenance] {job} FAILED: {msg}\n{traceback.format_exc()}",
              flush=True)
        return {"job": job, "ran": True, "status": "FAILED", "error": msg}


def _loop(interval_sec: float, tick_sec: float) -> None:
    time.sleep(STARTUP_DELAY_SEC)
    while True:
        try:
            for job in JOBS:
                run_job(job, min_interval_sec=interval_sec)
        except Exception:
            # A scheduler that can die is worse than one that logs and retries:
            # its absence is silent, and the record simply stops accumulating.
            print("[maintenance] loop error\n" + traceback.format_exc(), flush=True)
        time.sleep(tick_sec)


def start(interval_sec: Optional[float] = None,
          tick_sec: float = 600.0) -> bool:
    """Start the scheduler thread once per process.

    Off in tests and anywhere MAINTENANCE_SCHEDULER=0: a background thread that
    fetches price history is not something a test suite should be racing with.
    """
    if os.getenv("MAINTENANCE_SCHEDULER", "1") == "0":
        return False
    # Never under pytest. The suite imports web.app, which starts this at
    # import — so without the guard every test run spawns a thread that
    # fetches price history and writes to the REAL ledger while the tests are
    # pointing it at temp files. Slow, racy, and a quiet poisoning of the
    # calibration record that nothing would report.
    import sys
    if "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST"):
        return False
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        interval = float(interval_sec if interval_sec is not None
                         else os.getenv("MAINTENANCE_INTERVAL_SEC",
                                        DEFAULT_INTERVAL_SEC))
        _thread = threading.Thread(
            target=_loop, args=(interval, tick_sec),
            name="maintenance", daemon=True)
        _thread.start()
        print(f"[maintenance] scheduler started; interval {interval:.0f}s",
              flush=True)
        return True
