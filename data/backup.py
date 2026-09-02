"""
Ledger backup — the Mac is the canonical evidence store, so it is also the
single point of failure.

WHY NOT `cp`
    The ledger runs in WAL mode. A plain file copy can catch the database
    mid-transaction, or copy the main file without its -wal sibling, producing
    a backup that opens fine and is quietly missing the most recent writes —
    the worst kind of backup, because it looks like it worked. sqlite3's online
    backup API takes a consistent snapshot of a live database instead, and
    needs no downtime.

WHY VERIFY EVERY TIME
    A backup nobody has restored is a hypothesis, not a backup. Each one is
    reopened and its row counts compared against the source before it is
    accepted, so a silently corrupt file is caught the day it is written rather
    than the day it is needed.

This module only copies and checks. It never writes to the source database, and
nothing here can alter evidence.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Tables whose row counts must match for a backup to be considered good.
VERIFY_TABLES = ("prediction_snapshots", "prediction_outcomes", "snapshot_quarantine")

DEFAULT_KEEP = 14          # ~3 trading weeks of dailies


def _counts(conn: sqlite3.Connection) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in VERIFY_TABLES:
        try:
            out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.Error:
            out[t] = -1        # table absent -> mismatch, surfaced by the caller
    return out


def backup_ledger(dest_dir: Optional[str] = None, keep: int = DEFAULT_KEEP,
                  source_path: Optional[str] = None) -> Dict[str, Any]:
    """Take a verified, timestamped snapshot of the ledger.

    Returns a result dict; never raises. A failed backup must be loud in the
    report but must not take down the heartbeat that produced the evidence —
    losing today's forecast to protect yesterday's copy is the wrong trade.
    """
    from data import prediction_ledger as pl

    result: Dict[str, Any] = {"ok": False, "path": None, "verified": False}
    try:
        src = Path(source_path or pl._db())
        if not src.exists():
            result["error"] = f"source_missing: {src}"
            return result

        out_dir = Path(dest_dir) if dest_dir else src.parent / "backups"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        dest = out_dir / f"ledger-{stamp}.db"

        # Online backup: consistent snapshot of a live WAL database.
        s = sqlite3.connect(str(src))
        try:
            d = sqlite3.connect(str(dest))
            try:
                s.backup(d)
                src_counts = _counts(s)
                dst_counts = _counts(d)
                # A backup must be ONE portable file. sqlite3.backup copies the
                # source's WAL journal mode, which leaves -wal/-shm siblings
                # next to the copy; moving the .db alone would then silently
                # lose the most recent writes. DELETE mode folds everything
                # into the single file.
                d.execute("PRAGMA journal_mode=DELETE").fetchone()
                d.commit()
            finally:
                d.close()
        finally:
            s.close()

        result.update(path=str(dest), source=str(src),
                      source_counts=src_counts, backup_counts=dst_counts,
                      bytes=dest.stat().st_size if dest.exists() else 0)

        # VERIFY by reopening the file that was actually written, not the
        # handle it was written through.
        v = sqlite3.connect(str(dest))
        try:
            reopened = _counts(v)
            integrity = v.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            v.close()
        # Opening a database can recreate -wal/-shm siblings. Remove them so
        # the artefact on disk is ONE portable file: a backup you can copy
        # elsewhere with a single `cp` and still restore completely.
        for sib in (Path(str(dest) + "-wal"), Path(str(dest) + "-shm")):
            try:
                if sib.exists():
                    sib.unlink()
            except OSError:
                pass

        result["reopened_counts"] = reopened
        result["integrity_check"] = integrity

        result["verified"] = (reopened == src_counts and integrity == "ok")
        result["ok"] = result["verified"]
        if not result["verified"]:
            result["error"] = "verification_failed"
            return result

        result["pruned"] = _prune(out_dir, keep)
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def _prune(out_dir: Path, keep: int) -> List[str]:
    """Keep the newest `keep` backups. Only ever removes files this module
    created (the ledger-*.db name pattern), never the live database."""
    if keep <= 0:
        return []
    files = sorted(out_dir.glob("ledger-*.db"), key=lambda p: p.name, reverse=True)
    removed = []
    for old in files[keep:]:
        for target in (old, Path(str(old) + "-wal"), Path(str(old) + "-shm")):
            try:
                if target.exists():
                    target.unlink()
                    removed.append(target.name)
            except OSError:
                continue
    return removed


def verify_backup(path: str) -> Dict[str, Any]:
    """Restore-check an existing backup: open it, run integrity_check, count
    rows. This is what makes a backup a backup rather than a hope."""
    out: Dict[str, Any] = {"path": path, "ok": False}
    try:
        p = Path(path)
        if not p.exists():
            out["error"] = "missing"
            return out
        c = sqlite3.connect(str(p))
        try:
            out["integrity_check"] = c.execute("PRAGMA integrity_check").fetchone()[0]
            out["counts"] = _counts(c)
        finally:
            c.close()
        out["bytes"] = p.stat().st_size
        out["ok"] = (out["integrity_check"] == "ok"
                     and out["counts"].get("prediction_snapshots", -1) >= 0)
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def list_backups(dest_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    from data import prediction_ledger as pl
    d = Path(dest_dir) if dest_dir else Path(pl._db()).parent / "backups"
    if not d.exists():
        return []
    return [{"name": p.name, "bytes": p.stat().st_size,
             "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
            for p in sorted(d.glob("ledger-*.db"), reverse=True)]
