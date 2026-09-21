"""The one interface every sub-agent implements.

WHY A CONTRACT AND NOT JUST FUNCTION CALLS
------------------------------------------
Because the interesting case is failure. A specialist that is down, rate-
limited, out of universe, or missing a credential is the NORMAL case here —
OptionsPilot covers ~31 names out of every symbol a user might type, so it
declines far more often than it answers. An interface where declining is a
first-class, explained outcome keeps those declines in the trace where they
can be read. An interface where declining is an exception loses them.

So every adapter returns an AgentResult, always. `run()` must not raise: if it
does, the executor records the exception as an ERROR result rather than
letting one specialist take down the synthesis. That is the same
`available` + `reason` shape `optionspilot/client.py` already uses, promoted
to a rule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# Outcomes. The distinction between them is the point.
OK = "OK"                    # the agent answered
UNAVAILABLE = "UNAVAILABLE"  # the agent exists but cannot answer THIS request
SKIPPED = "SKIPPED"          # the planner never called it, and why
ERROR = "ERROR"              # the agent raised; a bug, not a decline

TERMINAL = (OK, UNAVAILABLE, SKIPPED, ERROR)


@dataclass(frozen=True)
class AgentRequest:
    symbol: str
    asset_class: str
    capability: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    agent_id: str
    capability: str
    status: str
    data: Optional[Dict[str, Any]] = None
    reason: str = ""
    price_basis: Optional[str] = None
    elapsed_ms: Optional[int] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in TERMINAL:
            raise ValueError(f"status must be one of {TERMINAL}, got {self.status!r}")
        if self.status != OK and not self.reason:
            raise ValueError(
                f"{self.agent_id}.{self.capability} returned {self.status} with no "
                "reason. A decline without a reason is indistinguishable from a "
                "step that never ran.")

    @property
    def ok(self) -> bool:
        return self.status == OK

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ok(agent_id: str, capability: str, data: Dict[str, Any],
       price_basis: Optional[str] = None, **provenance) -> AgentResult:
    return AgentResult(agent_id=agent_id, capability=capability, status=OK,
                       data=data, price_basis=price_basis, provenance=provenance)


def unavailable(agent_id: str, capability: str, reason: str) -> AgentResult:
    return AgentResult(agent_id=agent_id, capability=capability,
                       status=UNAVAILABLE, reason=reason)


def skipped(agent_id: str, capability: str, reason: str) -> AgentResult:
    return AgentResult(agent_id=agent_id, capability=capability,
                       status=SKIPPED, reason=reason)


def error(agent_id: str, capability: str, exc: BaseException) -> AgentResult:
    return AgentResult(agent_id=agent_id, capability=capability, status=ERROR,
                       reason=f"{type(exc).__name__}: {exc}")


class timer:
    """Wall-clock for one step. Recorded on every result, including failures —
    a specialist that is slow is a different problem from one that is broken,
    and only the timing separates them."""

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = int((time.time() - self._t0) * 1000)
        return False
