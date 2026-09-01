#!/usr/bin/env python3
"""Regression tests for the production-evidence invariant.

    Calibration and evaluation may consume ONLY genuine market snapshots.

Test runs had written 44 synthetic "TEST" snapshots into the production ledger.
They were inert — a fake ticker has no prices, so nothing ever matured — but
"it did not corrupt anything yet" is not a guarantee. These tests make the
exclusion structural.

Two mechanisms, tested separately because either alone would be insufficient:
  PREVENTION  a synthetic ticker is quarantined the moment it is frozen
  EXCLUSION   every production read filters quarantined ids out

And one property that must survive both: the frozen snapshot row is never
modified or deleted. Immutability is what makes the track record evidence, so
the cleanup must not be a hack around it.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import prediction_ledger as pl

# Throwaway DB — these tests write, and must never touch the real ledger.
_TMP = os.path.join(tempfile.mkdtemp(prefix="quarantine-test-"), "ledger.db")

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
    """Point the ledger at this module's throwaway DB for ONE test, then restore.

    Function-scoped on purpose. Setting the path at module scope (as this file
    originally did) mutates a process-wide global the moment the module is
    IMPORTED, so under pytest every test collected afterwards silently inherits
    the temp database. Production is unaffected - it runs in its own process -
    but a later test reading the ledger without setting its own path would read
    the wrong database, which is the kind of bug that hides for months.
    """
    def wrapper(*a, **k):
        prev = pl._db_override
        pl.set_db_path(_TMP)
        try:
            return fn(*a, **k)
        finally:
            pl.set_db_path(prev)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper



def _rec(ticker, price=100.0, probs=None):
    return {
        "ticker": ticker, "current_price": price, "action": "BUY",
        "generated_at": "2025-01-02T00:00:00", "data_asof": "2025-01-02",
        "confidence": {"statistical_edge": {"level": "HIGH", "score": 0.8}},
        "pillars": {"technical": {"score": 70}},
        "horizon_probabilities": probs or {1: 0.6, 5: 0.58},
    }


# ── Classification ─────────────────────────────────────────────────────────

@_tmpdb
def test_synthetic_tickers_are_recognised():
    for t in ("TEST", "test", " Test ", "EXPL", "AAA", "FOO", "DUMMY"):
        check(f"{t!r} classified synthetic", pl.is_synthetic_ticker(t))
    check("empty ticker is not a market observation", pl.is_synthetic_ticker(""))
    check("None is not a market observation", pl.is_synthetic_ticker(None))


@_tmpdb
def test_real_single_letter_symbols_are_not_quarantined():
    """T is AT&T and A is Agilent. A prefix or length rule would silently
    delete genuine evidence — matching must stay exact."""
    for t in ("T", "A", "F", "AAPL", "MSFT", "NVDA", "BRK.B", "AMD"):
        check(f"{t!r} treated as genuine", not pl.is_synthetic_ticker(t))


# ── Prevention: synthetic data cannot enter production evidence ────────────

@_tmpdb
def test_freezing_a_synthetic_ticker_auto_quarantines_it():
    sid = pl.freeze_prediction(_rec("TEST"))
    check("snapshot was still written (auditable, not silently dropped)", sid is not None)
    conn = pl._conn()
    try:
        row = conn.execute("SELECT 1 FROM snapshot_quarantine WHERE snapshot_id=?",
                           (sid,)).fetchone()
        stored = conn.execute("SELECT 1 FROM prediction_snapshots WHERE snapshot_id=?",
                              (sid,)).fetchone()
    finally:
        conn.close()
    check("it was quarantined at write time", row is not None)
    check("the snapshot row itself still exists", stored is not None)


@_tmpdb
def test_freezing_a_real_ticker_is_not_quarantined():
    sid = pl.freeze_prediction(_rec("AAPL"))
    conn = pl._conn()
    try:
        row = conn.execute("SELECT 1 FROM snapshot_quarantine WHERE snapshot_id=?",
                           (sid,)).fetchone()
    finally:
        conn.close()
    check("genuine evidence is left alone", row is None)


# ── Exclusion: production reads never see quarantined rows ─────────────────

def _seed_matured(ticker, probs, correct):
    """Freeze a snapshot and attach a matured outcome, as the real pipeline would."""
    sid = pl.freeze_prediction(_rec(ticker, probs=probs))
    conn = pl._conn()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO prediction_outcomes
               (snapshot_id, horizon_days, evaluated_at, as_of_date, matured,
                price_at_horizon, raw_return_pct, direction_correct)
               VALUES (?,?,?,?,1,?,?,?)""",
            (sid, 1, "2025-01-03T00:00:00", "2025-01-02", 101.0, 1.0, correct))
        conn.commit()
    finally:
        conn.close()
    return sid


@_tmpdb
def test_calibration_consumes_only_genuine_snapshots():
    """THE invariant. Synthetic rows with matured outcomes must not reach the
    calibration training set."""
    from intelligence.calibration import _pairs_for_horizon

    _seed_matured("MSFT", {1: 0.62}, 1)
    _seed_matured("NVDA", {1: 0.58}, 0)
    before = len(_pairs_for_horizon(1))

    _seed_matured("TEST", {1: 0.99}, 1)      # synthetic, fully matured
    _seed_matured("AAA", {1: 0.01}, 0)
    after = _pairs_for_horizon(1)

    check("synthetic rows do not enter the training set",
          len(after) == before, f"{before} -> {len(after)}")
    check("the extreme synthetic probabilities are absent",
          all(abs(p - 0.99) > 1e-9 and abs(p - 0.01) > 1e-9 for p, _, _ in after))


@_tmpdb
def test_source_where_excludes_quarantine_for_every_source():
    """'all' must mean all GENUINE evidence, not everything in the table."""
    for source in ("all", "live", "replay", "historical"):
        frag = pl._source_where(source)
        check(f"source={source!r} excludes quarantine",
              "snapshot_quarantine" in frag, frag)


@_tmpdb
def test_calibration_report_excludes_quarantined():
    rep = pl.calibration_report(horizon=1, source="all")
    n = (rep.get("overall") or {}).get("n")
    conn = pl._conn()
    try:
        raw = conn.execute("SELECT COUNT(*) FROM prediction_outcomes WHERE horizon_days=1 AND matured=1").fetchone()[0]
    finally:
        conn.close()
    check("report counts fewer rows than the raw table (quarantine applied)",
          n is None or n < raw, f"report n={n} raw={raw}")


# ── Immutability must survive the cleanup ──────────────────────────────────

@_tmpdb
def test_quarantine_never_mutates_or_deletes_a_snapshot():
    """The cleanup must not hack around immutability - that property is what
    makes the track record evidence rather than a story."""
    sid = pl.freeze_prediction(_rec("AMD"))
    conn = pl._conn()
    try:
        before = conn.execute("SELECT content_hash, frozen_json FROM prediction_snapshots "
                              "WHERE snapshot_id=?", (sid,)).fetchone()
    finally:
        conn.close()

    pl.quarantine_snapshot(sid, "manual_test")

    conn = pl._conn()
    try:
        after = conn.execute("SELECT content_hash, frozen_json FROM prediction_snapshots "
                             "WHERE snapshot_id=?", (sid,)).fetchone()
        # And prove the DB itself still refuses mutation.
        blocked_update = blocked_delete = False
        try:
            conn.execute("UPDATE prediction_snapshots SET action='SELL' WHERE snapshot_id=?", (sid,))
        except Exception:
            blocked_update = True
        try:
            conn.execute("DELETE FROM prediction_snapshots WHERE snapshot_id=?", (sid,))
        except Exception:
            blocked_delete = True
    finally:
        conn.close()

    check("frozen row is byte-identical after quarantine",
          before["content_hash"] == after["content_hash"]
          and before["frozen_json"] == after["frozen_json"])
    check("UPDATE is still refused by the database", blocked_update)
    check("DELETE is still refused by the database", blocked_delete)


@_tmpdb
def test_quarantine_is_idempotent():
    sid = pl.freeze_prediction(_rec("JPM"))
    pl.quarantine_snapshot(sid, "r1")
    pl.quarantine_snapshot(sid, "r2")
    conn = pl._conn()
    try:
        n = conn.execute("SELECT COUNT(*) FROM snapshot_quarantine WHERE snapshot_id=?",
                         (sid,)).fetchone()[0]
    finally:
        conn.close()
    check("re-quarantining does not duplicate", n == 1, str(n))


@_tmpdb
def test_quarantine_synthetic_sweep_reports_and_is_repeatable():
    dry = pl.quarantine_synthetic(dry_run=True)
    check("dry run changes nothing", dry["newly_quarantined"] == 0)
    check("dry run still reports what it found", dry["synthetic_found"] > 0)
    live = pl.quarantine_synthetic()
    again = pl.quarantine_synthetic()
    check("second sweep finds nothing new", again["newly_quarantined"] == 0,
          str(again))
    check("summary reports the reason", "synthetic_ticker" in
          (pl.quarantine_summary().get("by_reason") or {}))


# ── ONE canonical ledger ───────────────────────────────────────────────────

@_tmpdb
def test_canonical_is_the_default():
    """An unset LEDGER_ROLE must mean canonical. A deployment that forgets the
    variable should keep recording evidence, not silently stop."""
    import os
    prev = os.environ.pop("LEDGER_ROLE", None)
    try:
        check("default role is canonical", pl.ledger_role() == "canonical")
        check("is_canonical_ledger agrees", pl.is_canonical_ledger() is True)
        sid = pl.freeze_prediction(_rec("AAPL"))
        conn = pl._conn()
        try:
            q = conn.execute("SELECT 1 FROM snapshot_quarantine WHERE snapshot_id=?",
                             (sid,)).fetchone()
        finally:
            conn.close()
        check("canonical rows are NOT quarantined", q is None)
    finally:
        if prev is not None:
            os.environ["LEDGER_ROLE"] = prev


@_tmpdb
def test_secondary_deployment_never_becomes_evidence():
    """THE Phase-2 invariant: exactly one canonical ledger. A secondary
    deployment stays functional and records what it did, but its rows must not
    enter the population calibration reads - otherwise two partial histories
    compete and neither is complete."""
    import os
    from intelligence.calibration import _pairs_for_horizon

    prev = os.environ.get("LEDGER_ROLE")
    os.environ["LEDGER_ROLE"] = "secondary"
    try:
        check("role reports secondary", pl.ledger_role() == "secondary")
        check("is_canonical_ledger is False", pl.is_canonical_ledger() is False)

        before = len(_pairs_for_horizon(1))
        sid = _seed_matured("MSFT", {1: 0.77}, 1)     # real ticker, matured
        after = _pairs_for_horizon(1)

        conn = pl._conn()
        try:
            row = conn.execute("SELECT reason FROM snapshot_quarantine WHERE snapshot_id=?",
                               (sid,)).fetchone()
            listed = conn.execute("SELECT 1 FROM prediction_snapshots WHERE snapshot_id=?",
                                  (sid,)).fetchone()
        finally:
            conn.close()

        check("secondary row was quarantined at write time", row is not None)
        check("reason names the origin",
              row is not None and "non_canonical_origin" in row["reason"],
              str(row["reason"]) if row else "")
        check("the snapshot still EXISTS (auditable, not dropped)", listed is not None)
        check("it did NOT enter the calibration population",
              len(after) == before, f"{before} -> {len(after)}")
        check("its probability is absent from the training set",
              all(abs(p - 0.77) > 1e-9 for p, _, _ in after))
    finally:
        if prev is None:
            os.environ.pop("LEDGER_ROLE", None)
        else:
            os.environ["LEDGER_ROLE"] = prev


@_tmpdb
def test_secondary_rows_stay_visible_to_the_ui():
    """list_snapshots is deliberately unfiltered, so a secondary deployment can
    still show its own history even though it is not evidence."""
    import os
    prev = os.environ.get("LEDGER_ROLE")
    os.environ["LEDGER_ROLE"] = "secondary"
    try:
        pl.freeze_prediction(_rec("NVDA"))
        visible = [s for s in pl.list_snapshots(ticker="NVDA")]
        check("secondary rows remain listable for the UI", len(visible) >= 1)
    finally:
        if prev is None:
            os.environ.pop("LEDGER_ROLE", None)
        else:
            os.environ["LEDGER_ROLE"] = prev


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
