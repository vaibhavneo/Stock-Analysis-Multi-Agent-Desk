#!/usr/bin/env python3
"""Decision journal + forward validation (Phases 22, 23).

Every test writes to a throwaway database. A test suite that can reach the
canonical evidence ledger is one bad default away from polluting the only
record of what the system actually said.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from decision import forward, journal
from test_decision_intelligence import build   # the same synthetic fixtures

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


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    journal.set_db_path(path)
    return path


def test_journal_writes_and_reads_back():
    path = _tmp_db()
    try:
        d = build()
        d["ticker"] = "JRNLA"
        did = journal.journal_decision(d)
        check("a decision id is returned", bool(did))
        frozen = journal.get_decision(did)
        check("round-trips", frozen["headline_state"] == d["decision_state"]["headline_state"])
        check("conditions are captured", isinstance(frozen["exit_conditions"], list))
        check("invalidation is captured", frozen["invalidation"] is not None)
        check("evidence snapshot is captured", len(frozen["evidence_snapshot"]) > 0)
        check("model version recorded", bool(frozen["model_version"]))
    finally:
        journal.set_db_path(None)
        os.unlink(path)


def test_journal_is_idempotent():
    path = _tmp_db()
    try:
        d = build()
        d["ticker"] = "JRNLB"
        a = journal.journal_decision(d)
        b = journal.journal_decision(d)
        check("same id for identical input", a == b)
        check("only one row", journal.summary()["total_decisions"] == 1)
    finally:
        journal.set_db_path(None)
        os.unlink(path)


def test_journal_rows_are_immutable():
    """A journal you can edit is a story, not evidence."""
    path = _tmp_db()
    try:
        d = build()
        d["ticker"] = "JRNLC"
        journal.journal_decision(d)
        conn = sqlite3.connect(path)
        updated = deleted = False
        try:
            conn.execute("UPDATE decision_journal SET headline_state='X'")
            updated = True
        except sqlite3.DatabaseError:
            pass
        try:
            conn.execute("DELETE FROM decision_journal")
            deleted = True
        except sqlite3.DatabaseError:
            pass
        conn.close()
        check("UPDATE is rejected by the database", not updated)
        check("DELETE is rejected by the database", not deleted)
    finally:
        journal.set_db_path(None)
        os.unlink(path)


def test_model_change_does_not_rewrite_history():
    path = _tmp_db()
    try:
        d = build()
        d["ticker"] = "JRNLD"
        first = journal.journal_decision(d, model_version="v1")
        d2 = build()
        d2["ticker"] = "JRNLD"
        d2["decision_state"]["headline_state"] = "REDUCE"
        second = journal.journal_decision(d2, model_version="v2")
        check("a new row is written", first != second)
        check("the original still says what it said",
              journal.get_decision(first)["headline_state"] != "REDUCE")
        check("both are retained", journal.summary()["total_decisions"] == 2)
    finally:
        journal.set_db_path(None)
        os.unlink(path)


def _synthetic_prices(n=300, start=100.0, drift=0.0005, seed=3):
    import numpy as np
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.02, n)
    px = start * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.Series(px, index=idx)


def test_forward_evaluation_computes_excursions():
    prices = _synthetic_prices()
    frozen = {"ticker": "FWDA", "created_at": str(prices.index[0])[:19],
              "headline_state": "HOLD",
              "invalidation": [{"measurable_as": "daily close < 50.0"}],
              "confirmation": [{"measurable_as": "daily close > 500.0"}],
              "entry_plan": {"entry_low": 95.0}}
    res = forward.evaluate_decision(frozen, prices)
    h = res["horizons"][20]
    check("matured", h["matured"] is True)
    check("MFE is at least the realized return", h["mfe_pct"] >= h["raw_return_pct"] - 1e-9)
    check("MAE is at most the realized return", h["mae_pct"] <= h["raw_return_pct"] + 1e-9)
    check("HOLD graded against its stated rule", h["state_was_correct"] is True)
    check("the rule is reported alongside the verdict", "invalidation" in h["grading_rule"])


def test_forward_grades_states_by_their_own_rule():
    # Deterministic, strictly falling: the grading rule is what is under test,
    # not whether a random walk happened to end lower.
    import numpy as np
    idx = pd.bdate_range("2025-01-01", periods=300)
    prices = pd.Series(100.0 * np.exp(np.linspace(0, -0.5, 300)), index=idx)
    base = {"ticker": "FWDB", "created_at": str(prices.index[0])[:19],
            "invalidation": [{"measurable_as": "daily close < 50.0"}],
            "confirmation": [], "entry_plan": {}}
    avoid = forward.evaluate_decision({**base, "headline_state": "AVOID_NEW_POSITION"}, prices)
    check("AVOID is correct when price fell",
          avoid["horizons"][20]["state_was_correct"] is True)
    conflicted = forward.evaluate_decision({**base, "headline_state": "CONFLICTED"}, prices)
    check("CONFLICTED is not gradeable",
          conflicted["horizons"][20]["state_was_correct"] is None)


def test_forward_summary_refuses_to_claim_an_edge_on_a_thin_sample():
    path = _tmp_db()
    try:
        d = build()
        d["ticker"] = "FWDC"
        journal.journal_decision(d)
        s = forward.summarize_forward_validation(20)
        check("verdict is NO_DEMONSTRATED_EDGE", s["verdict"] == "NO_DEMONSTRATED_EDGE")
        check("the safeguard is stated", "authoritative" in s["safeguard"])
        check("never uses the word alpha", "alpha" not in s["statement"].lower())
    finally:
        journal.set_db_path(None)
        os.unlink(path)


def test_refresh_outcomes_writes_only_the_outcome_table():
    path = _tmp_db()
    try:
        prices = _synthetic_prices()
        d = build()
        d["ticker"] = "FWDD"
        d["generated_at"] = str(prices.index[0])[:19]
        did = journal.journal_decision(d)
        df = pd.DataFrame({"Close": prices})
        res = forward.refresh_outcomes("FWDD", fetch_fn=lambda t: df)
        check("outcomes were written", res["matured"] > 0)
        check("the journal row is untouched",
              journal.get_decision(did)["headline_state"] == d["decision_state"]["headline_state"])
        # And re-running must not fail on the immutable table.
        again = forward.refresh_outcomes("FWDD", fetch_fn=lambda t: df)
        check("re-evaluation is idempotent", again["matured"] == res["matured"])
    finally:
        journal.set_db_path(None)
        os.unlink(path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
