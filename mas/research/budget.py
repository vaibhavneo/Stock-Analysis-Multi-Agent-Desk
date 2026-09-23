"""What a research request cost, and a cache so it costs less next time.

WHY COUNT
---------
A budget that only bounds wall-clock cannot answer "why was that request
expensive" or "did the second specialist re-fetch what the first already
had". Both are cost questions with concrete answers, and neither is visible
from a stopwatch.

So the accounting counts what is actually spent — tool calls, LLM calls,
external API calls, tokens, cache hits and misses — and the cache removes the
commonest waste: two specialists independently pulling the same daily bars for
the same symbol inside one request.

THE CACHE IS DELIBERATELY SHORT-LIVED
-------------------------------------
Sixty seconds. Long enough to deduplicate within a request and across the
handful that follow it; short enough that a settled bar series published
mid-session is picked up on the next question rather than an hour later. A
long cache on price data trades a real correctness property for a saving
nobody asked for.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_TTL_SEC = 60.0
MAX_ENTRIES = 256


@dataclass
class Accounting:
    """What one request spent."""
    tool_calls: int = 0
    llm_calls: int = 0
    api_calls: int = 0
    tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    by_tool: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    budget_ms: int = 0

    def record(self, tool: str, *, llm: bool = False, external: bool = False,
               tokens: int = 0) -> None:
        self.tool_calls += 1
        self.by_tool[tool] = self.by_tool.get(tool, 0) + 1
        if llm:
            self.llm_calls += 1
        if external:
            self.api_calls += 1
        self.tokens += int(tokens or 0)

    @property
    def cache_hit_rate(self) -> Optional[float]:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 3) if total else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["cache_hit_rate"] = self.cache_hit_rate
        d["statement"] = (
            f"{self.tool_calls} tool call(s), {self.llm_calls} LLM call(s), "
            f"{self.api_calls} external API call(s)"
            + (f", {self.tokens:,} tokens" if self.tokens else "")
            + (f"; cache {self.cache_hits}/{self.cache_hits + self.cache_misses} "
               f"({self.cache_hit_rate:.0%})" if self.cache_hit_rate is not None
               else "; cache not used")
            + f"; {self.elapsed_ms}ms of a {self.budget_ms}ms budget.")
        return d


class _Cache:
    """Shared, short-lived, and it counts its own hits.

    A cache whose hit rate is unknown cannot be justified or tuned — it can
    only be believed in.
    """

    def __init__(self, ttl_sec: float = DEFAULT_TTL_SEC):
        self.ttl = ttl_sec
        self._d: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, fn: Callable[[], Any],
                       acct: Optional[Accounting] = None) -> Any:
        now = time.time()
        with self._lock:
            hit = self._d.get(key)
            if hit and (now - hit[0]) < self.ttl:
                self.hits += 1
                if acct:
                    acct.cache_hits += 1
                return hit[1]
        value = fn()
        with self._lock:
            if len(self._d) >= MAX_ENTRIES:
                oldest = sorted(self._d.items(), key=lambda kv: kv[1][0])
                for k, _ in oldest[:len(self._d) // 4 or 1]:
                    self._d.pop(k, None)
            self._d[key] = (now, value)
            self.misses += 1
            if acct:
                acct.cache_misses += 1
        return value

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {"entries": len(self._d), "hits": self.hits,
                "misses": self.misses, "ttl_sec": self.ttl,
                "hit_rate": round(self.hits / total, 3) if total else None}

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0


BARS = _Cache()


def cached_bars(symbol: str, period: str = "1y",
                acct: Optional[Accounting] = None):
    """Daily bars, deduplicated across specialists inside a request.

    Two specialists pulling the same series for the same symbol is the
    commonest duplicated work here, and it was happening on every
    multi-capability plan.
    """
    def _fetch():
        from financial_data import get_bars_df
        if acct:
            acct.record("bars_daily", external=True)
        return get_bars_df(symbol, period=period)
    return BARS.get_or_compute(f"bars:{symbol.upper()}:{period}", _fetch, acct)


# ── Tool selection value ──────────────────────────────────────────────────

# A required evidence kind is worth waiting for; an optional one competes on
# value per millisecond. Without this the planner picked the most reliable
# source regardless of what it cost to wait for.
NECESSITY_WEIGHT = {"required": 4.0, "optional": 1.0}


def tool_value(reliability: float, est_ms: int, necessity: str = "optional",
               freshness_rank: float = 1.0) -> float:
    """Expected information per second of latency.

    Reliability and necessity are the numerator because they are what the
    answer gains; latency is the denominator because it is what the user
    pays. A 2500ms source at 0.9 reliability loses to a 120ms source at 0.55
    for OPTIONAL evidence and beats it for REQUIRED evidence — which is the
    trade the old selector could not express.
    """
    seconds = max(est_ms, 1) / 1000.0
    return round((float(reliability) * NECESSITY_WEIGHT.get(necessity, 1.0)
                  * float(freshness_rank)) / seconds, 4)
