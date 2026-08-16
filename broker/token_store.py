"""
broker.token_store — durable OAuth token persistence.

A single-user, single-token-set JSON file, matching this app's existing
single-tenant posture (data/store.py's `recommendations` table has no
user_id column either). Path resolves via BROKER_TOKEN_DIR so the same code
works two ways with zero branching:

  - Locally: falls back to <repo_root>/data/broker_tokens.json, exactly how
    data/recommendations.db already lives there without any volume.
  - On Railway: BROKER_TOKEN_DIR=/data points at the persistent volume
    mounted for this purpose (see M0) — required there, since the
    Agentic-AI service's local filesystem is wiped on every redeploy and a
    refresh token that doesn't survive that means re-authenticating via a
    real Robinhood browser click-through after every deploy.

Never cache tokens in a process-global — this app runs gunicorn --workers 2,
and two workers each holding their own in-memory copy would race
independently on refresh/rotation. Every read/write goes to disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

TOKEN_FILENAME = "broker_tokens.json"


def _token_path() -> Path:
    token_dir = os.environ.get("BROKER_TOKEN_DIR")
    if token_dir:
        base = Path(token_dir)
    else:
        base = Path(__file__).resolve().parents[1] / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base / TOKEN_FILENAME


def load() -> Optional[dict]:
    """{"access_token","refresh_token","expires_at","token_type","obtained_at"}
    or None if never connected / disconnected."""
    path = _token_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save(tokens: dict) -> None:
    """Overwrites the whole file. Callers building `tokens` for a refresh
    response must merge with the previous refresh_token themselves when the
    server didn't return a new one (some OAuth 2.1 servers rotate it on
    every use, some don't — see broker/oauth.py::ensure_fresh_access_token)."""
    path = _token_path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tokens, indent=2))
    tmp.replace(path)   # atomic on POSIX — a crash mid-write can't corrupt the real file


def clear() -> None:
    path = _token_path()
    if path.exists():
        path.unlink()
