"""Plan -> traced execution.

ISOLATION IS THE WHOLE JOB
--------------------------
One specialist being down must degrade its own section and nothing else. That
is why every adapter returns an AgentResult instead of raising, why this
module catches anything that escapes anyway, and why each step carries its own
timeout from the registry rather than sharing one budget.

The trace records every attempt, including the ones that declined and the ones
that were never made. `decision/` learned this the hard way: a routing miss
that leaves no trace is indistinguishable from a pipeline that never ran, and
that is the exact class of bug a stage tracer exists to catch.
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from . import registry
from .contract import (AgentRequest, AgentResult, OK, UNAVAILABLE, SKIPPED,
                       ERROR, error, skipped, unavailable)

MAX_WORKERS = 4


def _run_one(agent_id: str, request: AgentRequest, timeout_sec: float) -> AgentResult:
    """One attempt, isolated. Returns a result for every outcome."""
    t0 = time.time()
    try:
        mod = registry.load_adapter(agent_id)
    except Exception as e:
        r = error(agent_id, request.capability, e)
        r.elapsed_ms = int((time.time() - t0) * 1000)
        return r

    # A DAEMON THREAD, deliberately, not a pooled one. ThreadPoolExecutor's
    # context manager calls shutdown(wait=True) on exit, so `fut.result(
    # timeout=...)` bounds when the RESULT is read but not when the caller is
    # released — a specialist sleeping 5s still held the request for 5s
    # despite a 0.2s timeout. A daemon thread can simply be abandoned: it
    # cannot block interpreter exit, and there is no pool for orphaned
    # attempts to exhaust.
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["res"] = mod.run(request)
        except BaseException as e:            # noqa: BLE001 - isolation is the job
            box["err"] = e

    th = threading.Thread(target=_target, daemon=True,
                          name=f"mas-{agent_id}-{request.capability}")
    th.start()
    th.join(timeout_sec)

    if th.is_alive():
        res = unavailable(agent_id, request.capability,
                          f"no answer within {timeout_sec:g}s")
    elif "err" in box:
        res = error(agent_id, request.capability, box["err"])
    else:
        res = box.get("res")
        if res is None:
            res = error(agent_id, request.capability,
                        TypeError("adapter returned None, not an AgentResult"))

    if not isinstance(res, AgentResult):
        res = error(agent_id, request.capability,
                    TypeError(f"adapter returned {type(res).__name__}, "
                              "not an AgentResult"))
    res.elapsed_ms = int((time.time() - t0) * 1000)
    return res


def _run_step(step: Dict[str, Any], symbol: str, asset_class: str) -> Dict[str, Any]:
    """Attempts in priority order until one answers.

    A fallback that fires is recorded as a fallback, not silently swapped in:
    an answer from the model engine because the real chain was down is a
    materially weaker answer than one from the chain, and the reader has to be
    able to see which they got.
    """
    cap = step["capability"]
    attempts: List[Dict[str, Any]] = []
    chosen: Optional[AgentResult] = None

    for i, a in enumerate(step["attempts"]):
        if chosen is not None:
            attempts.append({
                **skipped(a["agent_id"], cap,
                          f"not needed: {chosen.agent_id} already answered"
                          ).to_dict(),
                "price_basis": a.get("price_basis")})
            continue
        req = AgentRequest(symbol=symbol, asset_class=asset_class,
                           capability=cap, params=dict(step.get("params") or {}))
        res = _run_one(a["agent_id"], req, float(a.get("timeout_sec", 30)))
        attempts.append(res.to_dict())
        if res.ok:
            chosen = res

    return {
        "capability": cap,
        "answered_by": chosen.agent_id if chosen else None,
        "price_basis": chosen.price_basis if chosen else None,
        "fell_back": bool(chosen and step["attempts"]
                          and chosen.agent_id != step["attempts"][0]["agent_id"]),
        "result": chosen.to_dict() if chosen else None,
        "attempts": attempts,
        "unanswered_reason": (None if chosen else
                              "; ".join(a.get("reason") or "" for a in attempts
                                        if a.get("reason")) or "no agent answered"),
    }


def execute(plan: Dict[str, Any], parallel: bool = True) -> Dict[str, Any]:
    """Run a plan. Never raises."""
    t0 = time.time()
    symbol = plan["symbol"]
    asset_class = plan["asset_class"]
    steps = plan.get("steps") or []

    if parallel and len(steps) > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(steps))) as ex:
            futures = [ex.submit(_run_step, s, symbol, asset_class) for s in steps]
            outcomes = [f.result() for f in futures]
    else:
        outcomes = [_run_step(s, symbol, asset_class) for s in steps]

    by_capability = {o["capability"]: o for o in outcomes}

    # Capabilities that were requested and never reached an agent at all.
    for d in plan.get("declined", []):
        cap = d["capability"]
        if cap not in by_capability:
            by_capability[cap] = {
                "capability": cap, "answered_by": None, "price_basis": None,
                "fell_back": False, "result": None, "attempts": [],
                "unanswered_reason": d["reason"]}

    answered = [c for c, o in by_capability.items() if o["answered_by"]]
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "classification": plan.get("classification"),
        "view": plan.get("view"),
        "by_capability": by_capability,
        "answered": sorted(answered),
        "unanswered": sorted(c for c in by_capability if c not in answered),
        "elapsed_ms": int((time.time() - t0) * 1000),
        "trace": [
            {"capability": o["capability"],
             "answered_by": o["answered_by"],
             "fell_back": o["fell_back"],
             "attempts": [{"agent_id": a.get("agent_id"), "status": a.get("status"),
                           "elapsed_ms": a.get("elapsed_ms"),
                           "reason": a.get("reason")} for a in o["attempts"]],
             "reason": o["unanswered_reason"]}
            for o in by_capability.values()],
    }


def run_symbol(symbol: str, **kwargs) -> Dict[str, Any]:
    """Plan and execute in one call."""
    from .plan import build_plan
    plan = build_plan(symbol, **kwargs)
    out = execute(plan)
    out["plan"] = plan
    return out
