#!/usr/bin/env python3
"""The orchestrator deciding what is worth calling.

Before this, two callers asked for a fixed set every time, so someone asking
"how does AAPL look" paid for an options analysis they did not want — and got
range structures volunteered on a name the desk had no directional view on.
Suggesting a premium-selling trade to someone who asked about a share price is
not thoroughness; it is answering a question nobody posed.

The rule that outranks everything: EXPLICIT ALWAYS RUNS. A layer that decides
it knows better than a stated request is not autonomy, it is refusal. These
tests exist mostly to hold that line, because every other rule here is a
judgement call and that one is not.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas import policy, registry
from mas.policy import decide, explain, CHAT, BRIEF, FULL

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


# ── Explicit is sovereign ─────────────────────────────────────────────────

def test_an_explicit_request_always_runs():
    """Every case where an implied request would have been skipped."""
    cases = [
        (["option_structures"], "CRYPTO", None),      # no venue, no view
        (["option_structures"], "EQUITY", None),      # no directional view
        (["event_calendar"], "CRYPTO", "BULLISH"),    # class has no earnings
        (["portfolio_review"], "EQUITY", None),       # no holdings supplied
    ]
    for caps, cls, direction in cases:
        r = decide(caps, cls, depth=CHAT, direction=direction)
        check(f"{caps[0]} on {cls} still runs", caps[0] in r.run, r.to_dict())
        d = next(x for x in r.decisions if x.capability == caps[0])
        check("marked explicit", d.basis == "EXPLICIT", d)
        check("and the reason says so", "asked" in d.reason, d.reason)


def test_explicit_work_is_never_budgeted_away():
    """The user is waiting for what they asked for, on purpose."""
    r = decide(["equity_research", "option_structures", "strategy_backtest",
                "portfolio_review"], "EQUITY", depth=CHAT, budget_ms=1)
    for cap in ("equity_research", "option_structures", "strategy_backtest",
                "portfolio_review"):
        check(f"{cap} survived a 1ms budget", cap in r.run, r.run)


def test_an_unregistered_capability_is_refused_with_a_reason():
    r = decide(["make_me_rich"], "EQUITY", depth=CHAT)
    check("not run", "make_me_rich" not in r.run)
    d = next(x for x in r.decisions if x.capability == "make_me_rich")
    check("and named as unregistered", "not a registered" in d.reason, d.reason)


# ── Chat implies nothing ──────────────────────────────────────────────────

def test_a_question_about_the_stock_calls_only_the_stock_specialist():
    """The behaviour this module was built for."""
    r = decide(["equity_research"], "EQUITY", depth=CHAT, direction="BULLISH")
    check("only research runs", r.run == ["equity_research"], r.run)
    check("options were not even considered",
          not any(d.capability == "option_structures" for d in r.decisions),
          [d.capability for d in r.decisions])


def test_chat_depth_implies_nothing_at_all():
    check("CHAT implies nothing", policy.IMPLIED_BY_DEPTH[CHAT] == (),
          policy.IMPLIED_BY_DEPTH[CHAT])


# ── Implied work is judged ────────────────────────────────────────────────

def test_options_are_not_volunteered_without_a_directional_view():
    r = decide([], "EQUITY", depth=BRIEF, direction=None)
    check("not run", "option_structures" not in r.run, r.run)
    d = next(x for x in r.decisions if x.capability == "option_structures")
    check("the reason names the real problem",
          "no directional view" in d.reason, d.reason)
    check("and does not read as a refusal",
          "did not ask" in d.reason.lower(), d.reason)


def test_options_are_volunteered_when_there_is_a_view_to_express():
    for direction in ("BULLISH", "BEARISH"):
        r = decide([], "EQUITY", depth=BRIEF, direction=direction)
        check(f"{direction} runs options", "option_structures" in r.run, r.run)


def test_options_are_not_volunteered_where_no_venue_exists():
    r = decide([], "CRYPTO", depth=BRIEF, direction="BULLISH")
    check("not volunteered on a coin", "option_structures" not in r.run, r.run)
    d = next(x for x in r.decisions if x.capability == "option_structures")
    check("the reason names the venue", "venue" in d.reason, d.reason)
    check("and points at the explicit route",
          "explicitly" in d.reason, d.reason)


def test_a_portfolio_review_is_not_implied_without_holdings():
    r = decide([], "EQUITY", depth=FULL, direction="BULLISH", have_holdings=False)
    check("not in the run set", "portfolio_review" not in r.run, r.run)


# ── Budget ────────────────────────────────────────────────────────────────

def test_implied_work_respects_the_budget_and_says_when_it_did_not_fit():
    r = decide([], "EQUITY", depth=FULL, direction="BULLISH", budget_ms=1)
    skipped = [d for d in r.skipped() if d.basis == "IMPLIED"]
    check("something was skipped", skipped, r.to_dict())
    check("at least one names the budget",
          any("budget" in d.reason for d in skipped),
          [d.reason for d in skipped])


def test_a_generous_budget_runs_the_implied_set():
    r = decide([], "EQUITY", depth=FULL, direction="BULLISH", budget_ms=100_000)
    check("both implied capabilities ran",
          {"option_structures", "benchmark_relation"} <= set(r.run), r.run)


def test_the_budget_uses_the_cheapest_agent_that_could_serve():
    """Budgeting on the preferred (expensive) agent would skip work the
    system would in fact have done quickly via its fallback."""
    cheapest = policy._est_ms("option_structures", "EQUITY")
    offers = registry.agents("option_structures", "EQUITY")
    check("more than one agent offers it", len(offers) > 1, len(offers))
    check("the estimate is the cheapest, not the preferred",
          cheapest == min(int(a["est_ms"]) for a in offers),
          [(a["id"], a["est_ms"]) for a in offers])


def test_every_agent_declares_a_measured_cost():
    for a in registry.load()["agents"].values():
        check(f"{a['id']} declares est_ms", isinstance(a.get("est_ms"), int)
              and a["est_ms"] > 0, a.get("est_ms"))


# ── Every decision is explained ───────────────────────────────────────────

def test_every_decision_carries_a_reason():
    """A capability that silently did not run is indistinguishable from one
    that ran and found nothing."""
    for depth in (CHAT, BRIEF, FULL):
        for direction in (None, "BULLISH"):
            for cls in ("EQUITY", "CRYPTO", "FX"):
                r = decide(["equity_research"], cls, depth=depth,
                           direction=direction)
                for d in r.decisions:
                    check(f"{d.capability} has a reason", bool(d.reason.strip()),
                          d)


def test_decisions_are_readable_as_sentences():
    lines = explain(decide([], "EQUITY", depth=BRIEF, direction=None))
    check("something is explained", lines, lines)
    check("skips read as 'Not running'",
          any(l.startswith("Not running") for l in lines), lines)


def test_the_result_serialises_for_the_api():
    d = decide(["equity_research"], "EQUITY", depth=BRIEF, direction="BULLISH").to_dict()
    import json
    json.dumps(d)
    check("run list present", isinstance(d["run"], list))
    check("decisions present", isinstance(d["decisions"], list) and d["decisions"])
    check("budget reported", d["budget_ms"] > 0)


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
