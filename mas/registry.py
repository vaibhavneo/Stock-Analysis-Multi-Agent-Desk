"""Reading and validating the sub-agent roster.

Validation runs at load, not at call. A roster that names a capability no
adapter implements, or an asset class the taxonomy does not define, is a
routing bug that would otherwise surface as a mysterious SKIPPED weeks later.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .asset_class import ALL_CLASSES

_PATH = os.path.join(os.path.dirname(__file__), "registry.json")

# The closed capability vocabulary. Both sides of a link must name a thing the
# same way or the synthesis silently treats one idea as two.
CAPABILITIES = (
    "option_structures",      # candidate multi-leg structures for a view
    "option_pricing",         # price + greeks for a specified structure
    "implied_volatility",     # what the market charges for volatility
    "benchmark_relation",     # beta/correlation to the RIGHT benchmark
)


class RegistryError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load(refresh: bool = False) -> Dict[str, Any]:
    with open(_PATH) as f:
        reg = json.load(f)
    _validate(reg)
    return reg


def _validate(reg: Dict[str, Any]) -> None:
    agents = reg.get("agents") or {}
    if not agents:
        raise RegistryError("registry declares no agents")
    for aid, a in agents.items():
        if a.get("id") != aid:
            raise RegistryError(f"{aid}: id field does not match its key")
        caps = a.get("capabilities") or []
        if not caps:
            raise RegistryError(f"{aid}: declares no capabilities")
        for c in caps:
            if c not in CAPABILITIES:
                raise RegistryError(
                    f"{aid}: unknown capability {c!r}; known: {CAPABILITIES}")
        for cls in (a.get("asset_classes") or []):
            if cls not in ALL_CLASSES:
                raise RegistryError(f"{aid}: unknown asset class {cls!r}")
        if not a.get("module"):
            raise RegistryError(f"{aid}: no module")
        if a.get("transport") not in ("local", "http"):
            raise RegistryError(f"{aid}: transport must be local or http")
        # Every capability must have an explicit writes entry (null = reads
        # only). Silence here is how a side effect gets forgotten.
        writes = a.get("writes")
        if writes is None:
            raise RegistryError(f"{aid}: must declare a writes map (use {{}} if none)")


def agents(capability: Optional[str] = None,
           asset_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Agents offering `capability` for `asset_class`, best first.

    Ordered by declared priority, so a real chain (priority 10) is always
    preferred to a model (priority 50) where both can answer.
    """
    out = []
    for a in load()["agents"].values():
        if capability and capability not in (a.get("capabilities") or []):
            continue
        if asset_class and asset_class not in (a.get("asset_classes") or []):
            continue
        out.append(a)
    return sorted(out, key=lambda a: (a.get("priority", 100), a["id"]))


def get(agent_id: str) -> Dict[str, Any]:
    a = load()["agents"].get(agent_id)
    if not a:
        raise RegistryError(f"no such agent: {agent_id}")
    return a


def writes_for(agent_id: str, capability: str) -> Optional[str]:
    """What this capability mutates, or None if it only reads."""
    return (get(agent_id).get("writes") or {}).get(capability)


def load_adapter(agent_id: str):
    import importlib
    return importlib.import_module(get(agent_id)["module"])
