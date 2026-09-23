"""Run a plan: in parallel where independent, inside a budget, fully traced.

THREE THINGS THIS RECORDS THAT NOTHING PREVIOUSLY DID
-----------------------------------------------------
1. WHAT RAN vs WHAT EXISTS. A capability that was never asked and one that
   was asked and declined are different facts about an answer, and only the
   trace separates them.

2. WHAT WAS CONSUMED vs WHAT WAS PRODUCED. A tool called a hundred times that
   never changes a decision is not obviously useful. That cannot be measured
   without recording consumption, so consumption is recorded — and nothing is
   removed on the strength of it without measuring first.

3. WHY EACH OUTCOME. `tools.OUTCOMES` has nine members because "returned
   nothing", "returned stale data", "timed out" and "is not reachable in this
   deployment" mean four different things to a reader, and collapsing them
   into a falsy value is how a gap becomes invisible.

ISOLATION
---------
A specialist that raises must degrade its own section and nothing else. Each
attempt runs on an abandonable daemon thread rather than a pooled one:
ThreadPoolExecutor's context manager calls shutdown(wait=True), so a slow
worker holds the caller even when the future has already timed out.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from . import tools as T


@dataclass
class Step:
    """One capability invocation and everything that became of it."""
    capability: str
    agent_id: Optional[str] = None
    outcome: str = T.NOT_REQUESTED
    purpose: str = ""
    started_at: Optional[float] = None
    elapsed_ms: Optional[int] = None
    records: int = 0
    reason: str = ""
    consumed: bool = False
    changed_synthesis: bool = False
    changed_decision: bool = False
    data: Any = field(default=None, repr=False)

    def to_dict(self, include_data: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not include_data:
            d.pop("data", None)
        return d


class Trace:
    def __init__(self, budget_ms: int):
        self.steps: List[Step] = []
        self.budget_ms = budget_ms
        self.spent_ms = 0
        self.started = time.time()

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        self.spent_ms += step.elapsed_ms or 0
        return step

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self.started) * 1000)

    def over_budget(self) -> bool:
        return self.elapsed_ms > self.budget_ms

    def by_capability(self) -> Dict[str, Step]:
        return {s.capability: s for s in self.steps}

    def summary(self) -> Dict[str, Any]:
        ok = [s for s in self.steps if s.outcome == T.SUCCESS]
        used = [s for s in ok if s.consumed]
        weighed = [s for s in ok if s.changed_synthesis]
        # READ and WEIGHED are different facts. Every item synthesis
        # categorises is read; only the directional ones enter the weighing.
        # Reporting the second as though it were the first said "1 of 4 used"
        # beside a ledger holding ten decision-usable items.
        return {
            "n_steps": len(self.steps),
            "n_success": len(ok),
            "n_consumed": len(used),
            "n_weighed": len(weighed),
            "n_produced_but_unconsumed": len(ok) - len(used),
            "unconsumed": [s.capability for s in ok if not s.consumed],
            "outcomes": {s.capability: s.outcome for s in self.steps},
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "within_budget": not self.over_budget(),
            "statement": (
                f"{len(ok)} of {len(self.steps)} steps produced evidence; "
                f"{len(used)} were read by synthesis and {len(weighed)} "
                f"entered the directional weighing. "
                f"{self.elapsed_ms}ms of a {self.budget_ms}ms budget."),
        }


# Values that must never reach a trace, an evidence item or a reply. An
# adapter that puts a credential into its own exception message — "auth failed
# with sk-..." — is the realistic way a key reaches a user-visible trace, and
# no amount of care inside adapters makes that impossible. Redaction therefore
# happens at the boundary every message crosses.
_SECRET_ENV = ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPTIONSPILOT_ACCESS_CODE",
               "TIINGO_API_KEY", "FINNHUB_API_KEY", "OPENAI_API_KEY")
_SECRET_SHAPES = (
    r"sk-[A-Za-z0-9\-_]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}",
)


def redact(text: str) -> str:
    """Strip anything credential-shaped, and any live credential value."""
    import os as _os
    import re as _re
    out = str(text or "")
    for name in _SECRET_ENV:
        val = _os.environ.get(name)
        if val and len(val) >= 8 and val in out:
            out = out.replace(val, f"[{name} redacted]")
    for shape in _SECRET_SHAPES:
        out = _re.sub(shape, "[redacted]", out)
    return out


def _bounded(fn: Callable[[], Any], timeout_sec: float) -> Dict[str, Any]:
    """Run `fn` on an abandonable thread. Returns outcome + value/error."""
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as e:                  # isolation is the job
            box["error"] = e

    th = threading.Thread(target=_target, daemon=True, name="research-step")
    t0 = time.time()
    th.start()
    th.join(timeout_sec)
    elapsed = int((time.time() - t0) * 1000)

    if th.is_alive():
        return {"outcome": T.TIMEOUT, "elapsed_ms": elapsed,
                "reason": f"no answer within {timeout_sec:g}s"}
    if "error" in box:
        e = box["error"]
        msg = redact(f"{type(e).__name__}: {e}")
        low = msg.lower()
        if "rate" in low and "limit" in low:
            return {"outcome": T.RATE_LIMITED, "elapsed_ms": elapsed, "reason": msg}
        return {"outcome": T.ERROR, "elapsed_ms": elapsed, "reason": msg}
    return {"outcome": T.SUCCESS, "elapsed_ms": elapsed, "value": box.get("value")}


def _classify_result(value: Any) -> Dict[str, Any]:
    """Turn an adapter result into one of the nine outcomes.

    An adapter that declines is UNAVAILABLE, not an error, and an adapter that
    answers with nothing is EMPTY — which is a finding, not a failure.
    """
    if value is None:
        return {"outcome": T.EMPTY, "reason": "the step returned nothing",
                "records": 0}
    status = getattr(value, "status", None) or (
        value.get("status") if isinstance(value, dict) else None)
    reason = getattr(value, "reason", "") or (
        value.get("reason", "") if isinstance(value, dict) else "")
    data = getattr(value, "data", None)
    if data is None and isinstance(value, dict):
        data = value.get("data", value)

    if status in ("UNAVAILABLE", "SKIPPED"):
        return {"outcome": T.UNAVAILABLE, "reason": reason or "declined",
                "records": 0}
    if status == "ERROR":
        return {"outcome": T.ERROR, "reason": reason or "raised", "records": 0}
    n = len(data) if isinstance(data, (list, dict)) else (1 if data else 0)
    if not n:
        return {"outcome": T.EMPTY, "reason": reason or "no records",
                "records": 0}
    return {"outcome": T.SUCCESS, "reason": "", "records": n, "value": data}


def run_stage(capabilities: List[Dict[str, Any]],
              runner: Callable[[str], Any],
              trace: Trace,
              timeout_sec: float = 30.0,
              on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None
              ) -> Dict[str, Step]:
    """One parallel stage. Independent capabilities are genuinely concurrent.

    Serialising independent work is the difference between a four-second
    answer and a twelve-second one, and none of these capabilities reads
    another's output — the dependency that does exist (an event date narrowing
    a horizon) is resolved after this stage, not inside it.
    """
    results: Dict[str, Step] = {}
    threads: List[threading.Thread] = []
    lock = threading.Lock()

    def _one(entry: Dict[str, Any]):
        cap = entry["capability"]
        step = Step(capability=cap, purpose=entry.get("why", ""),
                    started_at=time.time())
        if trace.over_budget() and entry.get("necessity") != "required":
            step.outcome = T.NOT_REQUESTED
            step.reason = (f"skipped: the {trace.budget_ms}ms budget was "
                           f"already spent and this was optional")
            with lock:
                results[cap] = step
            return
        out = _bounded(lambda: runner(cap), timeout_sec)
        step.elapsed_ms = out["elapsed_ms"]
        if out["outcome"] != T.SUCCESS:
            step.outcome = out["outcome"]
            step.reason = redact(out["reason"])
        else:
            cls = _classify_result(out.get("value"))
            step.outcome = cls["outcome"]
            step.reason = redact(cls.get("reason", ""))
            step.records = cls.get("records", 0)
            step.data = cls.get("value")
        with lock:
            results[cap] = step
        if on_progress:
            on_progress("step", {"capability": cap, "outcome": step.outcome,
                                 "elapsed_ms": step.elapsed_ms,
                                 "records": step.records,
                                 "reason": step.reason})

    for entry in capabilities:
        th = threading.Thread(target=_one, args=(entry,), daemon=True)
        threads.append(th)
        th.start()
    for th in threads:
        th.join(timeout_sec + 2)

    for cap, step in results.items():
        trace.add(step)
    return results


def execute(plan, runner: Callable[[str], Any],
            timeout_sec: float = 30.0,
            on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None
            ) -> Dict[str, Any]:
    """Run every stage of a plan. Never raises."""
    trace = Trace(budget_ms=getattr(plan, "budget_ms", 6000))
    steps: Dict[str, Step] = {}

    if not getattr(plan, "answerable", True):
        return {"steps": {}, "trace": trace,
                "summary": {**trace.summary(),
                            "statement": "The plan was not answerable as "
                                         "asked, so nothing was run."}}

    by_cap = {c["capability"]: c for c in getattr(plan, "capabilities", [])}
    for stage in getattr(plan, "stages", []) or []:
        entries = [by_cap[c] for c in stage if c in by_cap]
        if not entries:
            continue
        steps.update(run_stage(entries, runner, trace, timeout_sec,
                               on_progress=on_progress))

    return {"steps": steps, "trace": trace, "summary": trace.summary()}
