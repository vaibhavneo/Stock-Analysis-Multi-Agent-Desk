#!/usr/bin/env python3
"""Tests for the two Phase-2 operational additions: the flywheel health report
and the verified ledger backup.

Neither may influence the strategy. The health report is monitoring, and the
backup only copies. The properties worth guarding are that the report counts
EVIDENCE rather than raw rows, and that a backup is verified rather than
assumed - a backup nobody has restored is a hypothesis.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import prediction_ledger as pl

_TMP = os.path.join(tempfile.mkdtemp(prefix="health-test-"), "ledger.db")

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _tmpdb(fn):
    def wrapper(*a, **k):
        prev = pl._db_override
        pl.set_db_path(_TMP)
        try:
            return fn(*a, **k)
        finally:
            pl.set_db_path(prev)
    wrapper.__name__ = fn.__name__
    return wrapper


def _rec(ticker, price=100.0, probs=None, pillars=None):
    return {"ticker": ticker, "current_price": price, "action": "BUY",
            "generated_at": "2025-01-02T00:00:00",
            "confidence": {"statistical_edge": {"level": "HIGH", "score": 0.8}},
            "pillars": pillars if pillars is not None else
                       {"technical": {"score": 70}, "algo": {"score": 65},
                        "fundamentals": {"score": 60}},
            "horizon_probabilities": probs or {1: 0.6}}


# ── health report ──────────────────────────────────────────────────────────

@_tmpdb
def test_report_counts_evidence_not_raw_rows():
    """A synthetic row is a row but not evidence. If the report counted raw
    rows, contamination would look like healthy growth."""
    from agents.flywheel_health import health_report
    before = health_report(day="2025-01-02")
    b_ev = before["today"]["evidence_rows"] if "error" not in before else 0
    b_q = before["today"]["quarantined_rows"] if "error" not in before else 0

    # Unique ticker+price: freeze_prediction is content-addressed, so reusing a
    # ticker another test already froze with identical content is a no-op.
    pl.freeze_prediction(_rec("COST", price=321.55))   # genuine -> evidence
    pl.freeze_prediction(_rec("TEST", price=321.55))   # synthetic -> quarantined

    rep = health_report(day="2025-01-02")
    check("report produced", "error" not in rep, str(rep.get("error")))
    # Delta, not absolute: this module's temp ledger is shared across tests.
    check("exactly ONE new evidence row (the synthetic one excluded)",
          rep["today"]["evidence_rows"] - b_ev == 1,
          f"{b_ev} -> {rep['today']['evidence_rows']}")
    check("the synthetic row is counted as quarantined instead",
          rep["today"]["quarantined_rows"] - b_q == 1,
          f"{b_q} -> {rep['today']['quarantined_rows']}")


@_tmpdb
def test_integrity_percentages_reflect_real_gaps():
    """A row missing core pillars must drag the percentage down, or the report
    would report health it has not verified."""
    from agents.flywheel_health import health_report
    pl.freeze_prediction(_rec("MSFT"))
    pl.freeze_prediction(_rec("NVDA", pillars={"technical": {"score": 70}}))  # incomplete
    rep = health_report(day="2025-01-02")
    i = rep["integrity"]
    check("prices all valid", i["prices_valid_pct"] == 100.0, str(i))
    check("pillar completeness is below 100 when a row is incomplete",
          i["pillars_complete_pct"] is not None and i["pillars_complete_pct"] < 100.0,
          str(i["pillars_complete_pct"]))


@_tmpdb
def test_report_never_raises_on_a_broken_ledger():
    """Monitoring that crashes tells you nothing on the day you need it."""
    from agents.flywheel_health import format_report, health_report
    prev = pl._db_override
    pl.set_db_path("/nonexistent/dir/does/not/exist.db")
    try:
        rep = health_report()
        check("returns a dict rather than raising", isinstance(rep, dict))
        check("renders without raising", isinstance(format_report(rep), str))
    finally:
        pl.set_db_path(prev)


@_tmpdb
def test_formatted_report_states_it_is_read_only():
    from agents.flywheel_health import format_report, health_report
    pl.freeze_prediction(_rec("AMD"))
    text = format_report(health_report(day="2025-01-02"))
    check("has a title", "DAILY FLYWHEEL HEALTH" in text)
    check("shows calibration readiness by independent windows",
          "independent windows" in text, text[:200])
    check("states it does not influence scoring",
          "never influences scoring" in text)


# ── backup ─────────────────────────────────────────────────────────────────

@_tmpdb
def test_backup_is_verified_by_reopening_the_written_file():
    """The verification must read the file that was written, not the handle it
    was written through - that is the difference between a backup and a hope."""
    from data.backup import backup_ledger
    pl.freeze_prediction(_rec("AAPL"))
    dest = tempfile.mkdtemp(prefix="bk-")
    b = backup_ledger(dest_dir=dest, source_path=_TMP)
    check("backup succeeded", b["ok"] is True, str(b.get("error")))
    check("it was verified", b["verified"] is True)
    check("integrity_check passed", b.get("integrity_check") == "ok")
    check("counts match the source", b["source_counts"] == b["reopened_counts"],
          f"{b.get('source_counts')} vs {b.get('reopened_counts')}")
    check("the file exists on disk", os.path.exists(b["path"]))


@_tmpdb
def test_backup_is_restorable_and_carries_the_evidence():
    """Restore-check: the copy must actually contain the rows."""
    from data.backup import backup_ledger, verify_backup
    pl.freeze_prediction(_rec("MSFT"))
    dest = tempfile.mkdtemp(prefix="bk-")
    b = backup_ledger(dest_dir=dest, source_path=_TMP)
    v = verify_backup(b["path"])
    check("restore check passes", v["ok"] is True, str(v))
    check("snapshots present in the restored copy",
          v["counts"]["prediction_snapshots"] >= 1, str(v["counts"]))
    conn = sqlite3.connect(b["path"])
    try:
        tickers = {r[0] for r in conn.execute("SELECT ticker FROM prediction_snapshots")}
    finally:
        conn.close()
    check("a known ticker survived the round trip", "MSFT" in tickers, str(tickers))


@_tmpdb
def test_retention_prunes_old_backups_only():
    from data.backup import backup_ledger, list_backups
    pl.freeze_prediction(_rec("AMD"))
    dest = tempfile.mkdtemp(prefix="bk-")
    marker = os.path.join(dest, "do-not-touch.txt")
    open(marker, "w").write("unrelated file")
    for _ in range(4):
        backup_ledger(dest_dir=dest, keep=2, source_path=_TMP)
    kept = [f for f in os.listdir(dest) if f.startswith("ledger-") and f.endswith(".db")]
    strays = [f for f in os.listdir(dest) if f.endswith("-wal") or f.endswith("-shm")]
    check("retention keeps only the newest N", len(kept) <= 2, str(kept))
    check("no WAL/SHM strays left beside backups", strays == [], str(strays))
    check("unrelated files are never removed", os.path.exists(marker))


@_tmpdb
def test_backup_failure_is_reported_not_raised():
    """A failed backup must be loud in the report but must not take down the
    heartbeat that produced the evidence."""
    from data.backup import backup_ledger
    b = backup_ledger(source_path="/nonexistent/ledger.db")
    check("returns a result rather than raising", isinstance(b, dict))
    check("ok is False", b["ok"] is False)
    check("reason is reported", "error" in b, str(b))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
