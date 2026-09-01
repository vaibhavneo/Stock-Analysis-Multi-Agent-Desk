#!/usr/bin/env python3
"""The ledger must be able to live on a mounted volume.

Railway's container filesystem is wiped on every deploy. The prediction ledger
is the accumulated forecast/outcome evidence the entire calibration layer is
built on, so if it lives on that disk it silently resets to zero on each
release - the flywheel would appear to run and never accumulate anything.

The volume also cannot be mounted over data/, because that package holds
store.py, ledger.py and prediction_ledger.py; a mount there shadows them and
breaks every import. Hence an explicit path override.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO)

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")


def _probe(env_value):
    """Resolve the DB path in a FRESH interpreter - the modules read the
    environment at import time, so an in-process test would measure nothing."""
    env = dict(os.environ)
    env.pop("STOCK_AGENT_DB_PATH", None)
    if env_value is not None:
        env["STOCK_AGENT_DB_PATH"] = env_value
    code = (
        "import sys; sys.path.insert(0,'.')\n"
        "from data import store, ledger, prediction_ledger as pl\n"
        "print(store.DB_PATH); print(ledger._db()); print(pl._db())\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=180)
    return [l.strip() for l in out.stdout.strip().splitlines()[-3:]]


def test_default_is_unchanged_without_the_env_var():
    paths = _probe(None)
    check("all three modules resolve a path", len(paths) == 3, str(paths))
    check("default lives beside the data package",
          all(p.endswith("data/recommendations.db") for p in paths), str(paths))


def test_override_redirects_every_consumer():
    """store, ledger and prediction_ledger must agree. ledger.py imports
    DB_PATH by value at module load, so a partial override would split the
    database in two without any error."""
    target = os.path.join(tempfile.mkdtemp(prefix="vol-"), "recommendations.db")
    paths = _probe(target)
    check("store honours the override", paths[0] == target, paths[0])
    check("ledger honours the override", paths[1] == target, paths[1])
    check("prediction_ledger honours the override", paths[2] == target, paths[2])
    check("no consumer is left pointing at the repo copy",
          not any(p.endswith("data/recommendations.db") for p in paths), str(paths))


def test_missing_parent_directory_is_created():
    """A freshly attached volume is an empty mount point, and sqlite3 will not
    create intermediate directories."""
    base = tempfile.mkdtemp(prefix="vol-")
    target = os.path.join(base, "nested", "deeper", "recommendations.db")
    paths = _probe(target)
    check("path resolves through a non-existent parent", paths[0] == target, paths[0])
    check("parent directory was created", os.path.isdir(os.path.dirname(target)))


def test_blank_env_var_falls_back_to_the_default():
    """An empty variable is a common deploy misconfiguration; it must not
    resolve to a database literally named '' at the filesystem root."""
    paths = _probe("")
    check("blank override falls back to the repo default",
          all(p.endswith("data/recommendations.db") for p in paths), str(paths))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
