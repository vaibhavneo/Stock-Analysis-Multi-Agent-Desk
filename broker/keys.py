"""
broker — static config resolution (env first, then stock_agent/.env).

Mirrors financial_data/keys.py's _lookup() convention exactly, so
ROBINHOOD_CLIENT_ID reports its absence the same way DEEPSEEK_API_KEY and
SEC_USER_AGENT already do. Kept as its own ~30-line copy rather than an
import from financial_data — a broker/ package reaching into a market-data
package for one lookup helper would blur a package boundary that costs
nothing to keep clean.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class NotConfiguredError(RuntimeError):
    """A required broker setting (e.g. ROBINHOOD_CLIENT_ID, which only exists
    after Dynamic Client Registration has been run once — see
    broker/register_client.py) is not set. Distinct from a network/auth
    failure on purpose, same reasoning as financial_data/keys.py."""


def get_key(var: str, required_by: str) -> str:
    val = _lookup(var)
    if not val:
        raise NotConfiguredError(
            f"{var} is not set (required by {required_by}). "
            f"Add it to stock_agent/.env:  {var}=...")
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
