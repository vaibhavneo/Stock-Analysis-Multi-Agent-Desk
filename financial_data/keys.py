"""
FIL — provider API-key resolution (env first, then stock_agent/.env).

One shared loader so every key-gated provider reports its absence the same
way: a typed NotConfiguredError naming the exact variable to set, never a
silent empty result and never a fabricated fallback. The same convention
web/app.py uses for DEEPSEEK_API_KEY and edgar.py uses for SEC_USER_AGENT.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class NotConfiguredError(RuntimeError):
    """A provider needs a key/setting the user has not supplied. Distinct from
    network failure on purpose: 'you have not configured this' and 'the host is
    down' demand opposite responses, and conflating them is how a config bug
    hides as an environment bug forever (learned the hard way with SEC's 403)."""


def get_key(var: str, required_by: str) -> str:
    val = _lookup(var)
    if not val:
        raise NotConfiguredError(
            f"{var} is not set (required by the {required_by} provider). "
            f"Get a free key and add to stock_agent/.env:  {var}=...")
    return val


def has_key(var: str) -> bool:
    return bool(_lookup(var))


def _lookup(var: str) -> Optional[str]:
    val = os.environ.get(var)
    if val:
        return val.strip()
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{var}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None
