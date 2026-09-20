#!/usr/bin/env python3
"""Plain language and per-horizon plans.

Two things are under test here, and the first is unusual: **readability is
asserted, not judged.** `decision/plain.has_jargon()` is the same rule the
renderer uses, so "make it readable" becomes something a test can fail on
rather than a matter of taste. Any `ALL_CAPS_SNAKE` token, `[field.path]`
citation or `source:identifier` that reaches a string a person reads is a
failure.

The second is that nothing was removed when the view was reorganized. The
technical detail was collapsed, not deleted, and `test_nothing_was_deleted`
holds the whole engine to that.
"""
from __future__ import annotations

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision.horizon_plan import HORIZON_ORDER, STANCE_TEXT
from decision.plain import build_plain_summary, has_jargon, plain_source
from test_decision_intelligence import build, mock_rec

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


def _rendered_strings(decision):
    """Every string a person actually reads. Deliberately NOT the whole object:
    the structured layer keeps its machine vocabulary (stance codes, level
    bases, evidence source ids) because downstream code branches on it. The
    contract is that none of it reaches rendered prose."""
    out = []
    p = decision.get("plain") or {}
    out.append(json.dumps(p))
    for plan in (decision.get("horizon_plans") or {}).get("plans", []):
        out += [plan["label"], plan["stance_text"], plan["why"]]
        out += plan.get("warnings", [])
        out += [r["label"] for r in plan.get("action_rows", [])]
        ref = plan.get("reference_levels")
        if ref:
            out.append(ref["note"])
    out.append((decision.get("horizon_plans") or {}).get("note", ""))
    return out


# ══════════════════════════════════════════════════════════════════════════
# Readability
# ══════════════════════════════════════════════════════════════════════════

def test_no_jargon_reaches_the_reader():
    for kw in ({}, {"position": {"avg_cost": 80.0, "shares": 50, "portfolio_value": 100000}}):
        d = build(**kw)
        for s in _rendered_strings(d):
            leaks = has_jargon(s)
            check(f"no jargon in rendered text ({s[:40]!r})", not leaks, leaks)


def test_jargon_detector_actually_detects():
    """A guard that never fires is not a guard."""
    check("catches a snake-case token", has_jargon("state is NO_DEMONSTRATED_EDGE"))
    check("catches a field citation", has_jargon("see [thesis.statement]"))
    check("catches a source identifier", has_jargon("driven by pillar:technical"))
    check("passes clean prose", not has_jargon("Hold what you own; nothing argues for a change."))


def test_every_evidence_source_has_a_plain_translation():
    d = build(position={"avg_cost": 80.0, "shares": 50, "portfolio_value": 100000})
    for e in d["evidence"]:
        text = plain_source(e["source"])
        check(f"{e['source']} translates", not has_jargon(text), text)
        check(f"{e['source']} is not returned verbatim", text != e["source"], text)


def test_plain_summary_is_always_present():
    d = build()
    p = d["plain"]
    check("available", p["available"] is True)
    check("has a headline", bool(p["call"]["headline"]))
    check("answers both ownership cases",
          bool(p["call"]["if_you_own_it"]) and bool(p["call"]["if_you_dont"]))
    check("has reasons", len(p["why"]) >= 1)
    check("has three scenarios", len(p["scenarios"]) == 3)
    check("carries the honesty line", bool(p["honesty"]))


def test_plain_summary_survives_an_unusable_decision():
    from decision.engine import build_decision_intelligence
    d = build_decision_intelligence("TEST", {"current_price": None})
    p = build_plain_summary(d)
    check("marked unavailable", p["available"] is False)
    check("still says something", bool(p["headline"]))


def test_no_probability_is_asserted_in_plain_text():
    d = build()
    for s in d["plain"]["scenarios"]:
        check(f"{s['name']} states no unearned probability",
              "not measured" in s["probability_text"].lower()
              or "no reliable probability" in s["probability_text"].lower()
              or "chance" in s["probability_text"])


def test_the_honesty_line_never_disappears():
    """It is the one caution that stays visible however much is collapsed."""
    d = build()
    check("names the missing edge",
          "not proven" in d["plain"]["honesty"] or "cleared" in d["plain"]["honesty"])
    check("is a full sentence", d["plain"]["honesty"].strip().endswith("."))


# ══════════════════════════════════════════════════════════════════════════
# Per-horizon plans
# ══════════════════════════════════════════════════════════════════════════

def test_three_horizons_always_present():
    d = build()
    plans = d["horizon_plans"]["plans"]
    check("three plans", len(plans) == 3)
    check("in order", [p["horizon"] for p in plans] == list(HORIZON_ORDER))
    for p in plans:
        check(f"{p['horizon']} has a stance", p["stance"] in STANCE_TEXT)
        check(f"{p['horizon']} explains itself", bool(p["why"]))


def test_horizons_can_differ_from_each_other():
    """If all three always agreed, per-horizon plans would be three copies of
    one answer wearing different numbers — exactly the fake depth this is
    meant to avoid."""
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["technical"] = {"score": 95.0, "confidence": 1.0, "backtestable": True, "flags": []}
    pillars["fundamentals"] = {"score": 8.0, "confidence": 1.0, "backtestable": False, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    stances = {p["stance"] for p in d["horizon_plans"]["plans"]}
    check("a split chart-vs-fundamentals read yields different stances",
          len(stances) >= 2, stances)


def test_a_bearish_horizon_never_shows_a_buy():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["fundamentals"] = {"score": 8.0, "confidence": 1.0, "backtestable": False, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    for p in d["horizon_plans"]["plans"]:
        if p["stance"] not in ("TRIM_INTO_STRENGTH", "STAY_OUT"):
            continue
        labels = " ".join(r["label"].lower() for r in p["action_rows"])
        check(f"{p['horizon']} offers no buy row", "buy" not in labels and "add" not in labels,
              labels)
        check(f"{p['horizon']} says where to get out", "get out" in labels, labels)


def test_a_horizon_with_no_view_offers_no_plan():
    """Showing a buy area under 'not enough evidence' reads as a
    recommendation."""
    # Strip BOTH sources of long-horizon evidence: the fundamentals pillar and
    # the 1Y relative-strength reading. Removing only the pillar leaves the
    # benchmark comparison behind, which is itself a long-horizon view.
    from test_decision_intelligence import mock_hist
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["fundamentals"] = {"score": 50.0, "confidence": 0.0, "backtestable": False, "flags": []}
    hist = mock_hist(100.0)
    hist["relative_performance"].pop("1Y", None)
    d = build(rec=mock_rec(pillars=pillars), historical_context=hist)
    empty = [p for p in d["horizon_plans"]["plans"] if p["stance"] == "NOT_ENOUGH_EVIDENCE"]
    check("at least one horizon has no view", len(empty) >= 1)
    for p in empty:
        check(f"{p['horizon']} has no action rows", p["action_rows"] == [])
        check(f"{p['horizon']} has no buy zone", p["buy_zone"] is None)
        check(f"{p['horizon']} still offers reference levels",
              p["reference_levels"] is not None)


def test_thin_evidence_is_declared():
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["fundamentals"] = {"score": 8.0, "confidence": 1.0, "backtestable": False, "flags": []}
    d = build(rec=mock_rec(pillars=pillars))
    thin = [p for p in d["horizon_plans"]["plans"] if p["evidence_thin"]]
    check("a single-item horizon exists in this fixture", len(thin) >= 1)
    for p in thin:
        check(f"{p['horizon']} warns about its own depth",
              any("single reading" in w for w in p["warnings"]), p["warnings"])


def test_sizing_is_the_same_across_every_horizon():
    """The statistical gate is a claim about the method, not about a holding
    period. A longer horizon must not become a third way to imply size is
    available."""
    d = build()
    statements = {p["how_much"] for p in d["horizon_plans"]["plans"]}
    gated = {p["sizing_gated"] for p in d["horizon_plans"]["plans"]}
    check("one sizing statement for all horizons", len(statements) == 1, statements)
    check("one gate verdict for all horizons", len(gated) == 1, gated)
    check("matches the engine's sizing", d["sizing"]["statement"] in statements)


def test_exit_sits_below_the_buy_area():
    d = build()
    for p in d["horizon_plans"]["plans"]:
        if not p["buy_zone"] or not p["exit_below"]:
            continue
        check(f"{p['horizon']} exit is below its buy area",
              p["exit_below"]["price"] < p["buy_zone"]["price"],
              f"{p['exit_below']['price']} vs {p['buy_zone']['price']}")


def test_levels_carry_whether_the_market_traded_there():
    d = build()
    for p in d["horizon_plans"]["plans"]:
        for row in p["action_rows"]:
            if row["zone"]:
                check(f"{p['horizon']}/{row['label']} declares its basis",
                      "traded_here_before" in row["zone"])


def test_horizon_stances_are_in_the_fingerprint():
    """They are decision content, so two runs that disagree about them are
    different decisions."""
    a = build()
    b = build()
    check("identical inputs, identical fingerprint",
          a["decision_fingerprint"] == b["decision_fingerprint"])
    pillars = copy.deepcopy(mock_rec()["pillars"])
    pillars["fundamentals"] = {"score": 8.0, "confidence": 1.0, "backtestable": False, "flags": []}
    c = build(rec=mock_rec(pillars=pillars))
    check("a changed horizon read changes the fingerprint",
          c["decision_fingerprint"] != a["decision_fingerprint"])


# ══════════════════════════════════════════════════════════════════════════
# Nothing was deleted
# ══════════════════════════════════════════════════════════════════════════

def test_nothing_was_deleted():
    """The technical detail was collapsed in the UI, not removed from the
    payload. Every section the previous view rendered is still produced."""
    d = build(position={"avg_cost": 80.0, "shares": 50, "portfolio_value": 100000})
    for key in ("decision_state", "thesis", "statistical_edge", "thesis_edge_relation",
                "evidence", "horizon_read", "conflict", "level_map", "entry", "entry_plan",
                "position_context", "risk_budget", "add_analysis", "sizing", "playbook",
                "scenarios", "mind_changers", "catalysts", "monitoring", "confidence",
                "quality", "statistical_honesty", "consistency", "authority"):
        check(f"{key} still produced", key in d and d[key] is not None)
    check("plus the new plain summary", "plain" in d)
    check("plus the new horizon plans", "horizon_plans" in d)


def test_the_plain_layer_cannot_change_the_decision():
    """It renders the finished object; it is not an input to it."""
    d = build()
    before = d["decision_fingerprint"]
    d["plain"] = {"available": False, "headline": "tampered"}
    from decision.engine import _fingerprint
    check("fingerprint ignores the plain layer", _fingerprint(d) == before)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
