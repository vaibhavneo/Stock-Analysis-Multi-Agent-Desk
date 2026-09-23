#!/usr/bin/env python3
"""What each section of the working actually changed.

"Show the working" had grown into panels that each display a lot of data and
none of which say what it did. The catalyst calendar reported an earnings date
56 days out without saying that lands outside the next-few-weeks plan and
inside the other two — leaving the reader to do the arithmetic that decides
which plan it belongs to.

A panel that shows data without saying what it changed asks the reader to do
the synthesis the engine was supposed to do. These tests hold two things: the
bearing must be DERIVED from the data (not a fixed sentence per section), and
"this changed nothing" must be a statable outcome — it is information, not an
absence of it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decision import bearing as B
from decision.bearing import (build_bearing, CHANGED, NO_EFFECT, LIMITED,
                              UNKNOWN, UNLOCKED_BY)
from decision.horizon_plan import HORIZON_SPECS

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


# ── Catalysts: the case that motivated this ───────────────────────────────

def test_an_event_is_placed_inside_and_outside_each_horizon():
    """56 days is past the 21-day plan and inside the 126- and 252-day ones."""
    out = B.catalysts_bearing({"status": "OK", "next_event": {
        "event": "Quarterly earnings report", "days_away": 56}})
    check("it moved something", out["effect"] == CHANGED, out)
    joined = " ".join([out["statement"]] + out["detail"]).lower()
    check("names the horizons it lands inside", "three to six months" in joined, joined)
    check("and the one it does not", "next few weeks" in joined, joined)
    check("the count is right", "2 of the 3" in out["statement"], out["statement"])


def test_an_event_past_every_horizon_bears_on_nothing():
    longest = HORIZON_SPECS["LONG"]["days"]
    out = B.catalysts_bearing({"status": "OK", "next_event": {
        "event": "Annual meeting", "days_away": longest + 40}})
    check("no effect", out["effect"] == NO_EFFECT, out)
    check("and says it is past every horizon",
          "past every horizon" in out["statement"], out["statement"])


def test_an_event_inside_the_shortest_horizon_is_inside_all_of_them():
    out = B.catalysts_bearing({"status": "OK", "next_event": {
        "event": "Earnings", "days_away": 3}})
    check("changed", out["effect"] == CHANGED, out)
    check("all three", "3 of the 3" in out["statement"], out["statement"])


def test_no_calendar_is_reported_as_absent_data_not_a_quiet_week():
    out = B.catalysts_bearing({"status": "UNAVAILABLE", "waiting_for": "no provider"})
    check("no effect", out["effect"] == NO_EFFECT)
    check("the distinction is stated",
          "absence of data" in out["statement"], out["statement"])


def test_a_dated_event_without_a_date_is_unknown_not_assumed():
    out = B.catalysts_bearing({"status": "OK",
                               "next_event": {"event": "Something"}})
    check("unknown", out["effect"] == UNKNOWN, out)


# ── Risk ──────────────────────────────────────────────────────────────────

def test_missing_position_inputs_are_named_with_what_they_would_unlock():
    out = B.risk_bearing(
        {"invalidation_level": 219.36, "invalidation_basis": "PRICE_HISTORY_DERIVED",
         "inputs_missing": ["avg_cost", "shares", "portfolio_value"]}, {})
    check("constrained", out["effect"] == LIMITED, out)
    joined = " ".join(out["detail"])
    for word in ("average cost", "share count", "portfolio value"):
        check(f"{word} is explained", word in joined, joined)
    check("the 'not computable' chip is explained",
          "not computable" in joined, joined)


def test_the_unlock_sentences_are_whole_sentences():
    """The first version stored noun phrases and prefixed them with 'no ',
    producing 'Without avg cost: no your position's gain or loss' — a template
    seam showing through in the text meant to explain a gap clearly."""
    for field, text in UNLOCKED_BY.items():
        check(f"{field} starts capitalised", text[:1].isupper(), text)
        check(f"{field} ends in a full stop", text.rstrip().endswith("."), text)
        check(f"{field} reads as a sentence", " no your " not in text.lower(), text)


def test_the_invalidation_basis_is_translated_not_echoed():
    derived = B.risk_bearing({"invalidation_level": 100.0,
                              "invalidation_basis": "CURRENT_PRICE_DERIVED",
                              "inputs_missing": []}, {})
    check("a derived basis is called out",
          "not a level the market has defended" in " ".join(derived["detail"]),
          derived["detail"])
    traded = B.risk_bearing({"invalidation_level": 100.0,
                             "invalidation_basis": "TRADED_PRICE",
                             "inputs_missing": []}, {})
    check("a traded basis reads differently",
          "actually traded at" in " ".join(traded["detail"]), traded["detail"])


def test_a_complete_position_is_not_reported_as_constrained():
    out = B.risk_bearing({"invalidation_level": 100.0,
                          "invalidation_basis": "TRADED_PRICE",
                          "inputs_missing": []}, {"status": "PROVIDED"})
    check("changed, not limited", out["effect"] == CHANGED, out)


# ── Levels ────────────────────────────────────────────────────────────────

def test_levels_report_how_many_are_real_traded_prices():
    lm = {"supports": [{"basis": "TRADED_PRICE"}, {"basis": "MODEL"}],
          "resistances": [{"basis": "TRADED_PRICE"}, {"basis": "CURRENT_PRICE_DERIVED"}]}
    out = B.levels_bearing(lm)
    check("2 of 4", "2 of 4" in out["statement"], out["statement"])


def test_levels_with_nothing_traded_are_flagged_as_the_weaker_claim():
    lm = {"supports": [{"basis": "CURRENT_PRICE_DERIVED"}],
          "resistances": [{"basis": "MODEL"}]}
    out = B.levels_bearing(lm)
    check("constrained", out["effect"] == LIMITED, out)
    check("and says why it is weaker",
          "weaker claim" in out["statement"], out["statement"])


# ── Confidence ────────────────────────────────────────────────────────────

def test_a_capped_confidence_names_the_dimension_without_repeating_the_level():
    """'Confidence is NONE because of statistical (NONE)' is circular, and it
    buries the useful half — WHICH dimension did the capping."""
    out = B.confidence_bearing({"decision_confidence": "NONE",
                                "capped_by": ["statistical (NONE)"]})
    check("constrained", out["effect"] == LIMITED, out)
    check("the dimension is named", "statistical" in out["statement"])
    check("the level is not repeated back at itself",
          "NONE because" not in out["statement"], out["statement"])


def test_an_uncapped_confidence_says_nothing_held_it_down():
    out = B.confidence_bearing({"decision_confidence": "HIGH", "capped_by": []})
    check("no effect", out["effect"] == NO_EFFECT, out)


# ── Conflict, scenarios, consistency ──────────────────────────────────────

def test_conflicts_that_change_nothing_are_still_reported():
    out = B.conflict_bearing({"n_conflicts": 3, "n_decision_changing": 0})
    check("no effect", out["effect"] == NO_EFFECT, out)
    check("but recorded so agreement is not mistaken for unanimity",
          "unanimity" in out["statement"], out["statement"])


def test_a_decision_changing_conflict_is_marked_as_such():
    out = B.conflict_bearing({"n_conflicts": 3, "n_decision_changing": 1})
    check("changed", out["effect"] == CHANGED, out)


def test_scenarios_without_probability_say_so_plainly():
    out = B.scenarios_bearing({"scenarios": [1, 2, 3],
                               "any_probability_stated": False})
    check("constrained", out["effect"] == LIMITED, out)
    check("it refuses to imply likelihood",
          "do not say which is likely" in out["statement"], out["statement"])


def test_a_clean_consistency_check_is_an_outcome_not_a_blank():
    out = B.consistency_bearing({"n_errors": 0, "n_warnings": 0})
    check("no effect", out["effect"] == NO_EFFECT, out)
    check("and it is stated", bool(out["statement"].strip()))


def test_a_consistency_error_makes_the_answer_unreliable():
    out = B.consistency_bearing({"n_errors": 1, "n_warnings": 0})
    check("changed", out["effect"] == CHANGED, out)
    check("and says the answer is unreliable",
          "unreliable" in out["statement"], out["statement"])


# ── The failing gate is the headline constraint ───────────────────────────

def test_a_failed_statistical_gate_is_named_as_the_biggest_constraint():
    out = B.edge_bearing({"demonstrated_edge": False,
                          "verdict": "INSUFFICIENT_SAMPLE"})
    check("constrained", out["effect"] == LIMITED, out)
    check("it says it is the biggest one",
          "biggest constraint" in out["statement"], out["statement"])
    check("and names the verdict",
          "INSUFFICIENT_SAMPLE" in out["statement"], out["statement"])


# ── The whole object ──────────────────────────────────────────────────────

def _decision():
    return {
        "catalysts": {"status": "OK", "next_event": {"event": "Earnings", "days_away": 56}},
        "risk_budget": {"invalidation_level": 219.36,
                        "invalidation_basis": "PRICE_HISTORY_DERIVED",
                        "inputs_missing": ["avg_cost"]},
        "level_map": {"supports": [{"basis": "TRADED_PRICE"}], "resistances": []},
        "confidence": {"decision_confidence": "NONE", "capped_by": ["statistical (NONE)"]},
        "quality": {"n_good": 4, "n_components": 8, "missing": ["EDGE"], "weak": []},
        "conflict": {"n_conflicts": 1, "n_decision_changing": 0},
        "scenarios": {"scenarios": [1], "any_probability_stated": False},
        "statistical_honesty": {"demonstrated_edge": False, "verdict": "NO_EDGE"},
        "position_context": {"status": "POSITION_CONTEXT_NOT_PROVIDED"},
        "consistency": {"n_errors": 0, "n_warnings": 1, "issues": [{"message": "m"}]},
    }


def test_every_section_gets_a_bearing_with_a_statement():
    out = build_bearing(_decision())
    for k, v in out.items():
        if k.startswith("_"):
            continue
        check(f"{k} has an effect", v["effect"] in
              (CHANGED, NO_EFFECT, LIMITED, UNKNOWN), v)
        check(f"{k} has a statement", bool(v["statement"].strip()), v)


def test_the_summary_does_not_describe_an_empty_remainder():
    """It once said 'the rest are recorded so you can see they were checked'
    on a decision where every section had moved or constrained something."""
    out = build_bearing(_decision())
    s = out["_summary"]
    n = s["n_sections"]
    accounted = (len(s["moved_the_answer"]) + len(s["constrained_the_answer"])
                 + len(s["no_effect"]))
    check("every section is accounted for", accounted <= n, (accounted, n))
    if len(s["moved_the_answer"]) + len(s["constrained_the_answer"]) == n:
        check("no phantom remainder is claimed",
              "remaining" not in s["statement"], s["statement"])


def test_build_bearing_never_raises_on_a_broken_decision():
    for bad in ({}, {"catalysts": None}, {"risk_budget": "nonsense"},
                {"confidence": []}, {"level_map": 5}):
        out = build_bearing(bad)
        check("returns a dict", isinstance(out, dict), out)
        check("with a summary", "_summary" in out, list(out))


def test_bearing_is_derived_not_a_fixed_sentence_per_section():
    """If the statement did not depend on the data, it would be decoration."""
    a = build_bearing(_decision())
    d2 = _decision()
    d2["catalysts"]["next_event"]["days_away"] = 3
    d2["conflict"]["n_decision_changing"] = 1
    b = build_bearing(d2)
    check("catalysts changed with the date",
          a["catalysts"]["statement"] != b["catalysts"]["statement"])
    check("conflict changed with the count",
          a["conflict"]["effect"] != b["conflict"]["effect"])


def test_the_engine_attaches_bearing_before_fingerprinting():
    """Two decisions differing only in what constrained them must not
    fingerprint identically."""
    import inspect
    from decision import engine
    src = inspect.getsource(engine.build_decision_intelligence)
    check("bearing is built", "build_bearing" in src)
    check("before the fingerprint",
          src.index("build_bearing") < src.index("_fingerprint(decision)"))


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
