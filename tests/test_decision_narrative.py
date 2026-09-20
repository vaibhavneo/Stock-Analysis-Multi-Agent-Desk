#!/usr/bin/env python3
"""The LLM guard (Phases 19, 27).

The prompt tells the model not to invent numbers. These tests check that the
instruction is not the safeguard — `validate_narrative()` is. Every one of them
runs offline: no API key, no network, no model call.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision.narrative import (allowed_numbers, build_narrative_prompt,
                                generate_narrative, validate_narrative)
from test_decision_intelligence import build

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


def test_fabricated_price_level_is_caught():
    d = build()
    v = validate_narrative("We see a move to $1234.56 next quarter.", d)
    check("rejected", v["ok"] is False)
    check("the offending number is named", "1234.56" in v["unsupported_numbers"])


def test_numbers_from_the_object_are_accepted():
    d = build()
    support = d["level_map"]["nearest_support"]["price"]
    v = validate_narrative(f"The nearest sourced support is ${support}.", d)
    check("accepted", v["ok"] is True, v["reason"])


def test_uncalibrated_probability_is_caught():
    d = build()
    v = validate_narrative("There is a 72% chance of reaching the target.", d)
    check("rejected", v["ok"] is False)
    check("flagged as an uncalibrated probability",
          any("probability" in b for b in v["banned_phrases"]))


def test_claiming_a_proven_edge_is_caught():
    d = build()
    check("edge is not demonstrated in this fixture",
          d["statistical_edge"]["demonstrated"] is False)
    v = validate_narrative("This setup has a proven edge.", d)
    check("rejected", v["ok"] is False)
    check("the phrase is named", "proven edge" in v["banned_phrases"])


def test_negated_edge_language_is_allowed():
    """The engine's own honesty statements say 'no demonstrated edge'. A guard
    that rejects its own vocabulary would make the narrative impossible."""
    d = build()
    v = validate_narrative("There is no demonstrated edge here.", d)
    check("accepted", v["ok"] is True, v["reason"])


def test_certainty_language_is_caught():
    d = build()
    v = validate_narrative("This will certainly recover.", d)
    check("rejected", v["ok"] is False)
    check("the phrase is named", "will certainly" in v["banned_phrases"])


def test_percentages_of_ratios_are_allowed():
    """A ratio stored as 0.93 is naturally written as 93%."""
    d = build()
    ratio = d["conflict"]["agreement_ratio"]
    if ratio is None:
        return
    v = validate_narrative(f"{round(ratio * 100)}% of directional weight is on one side.", d)
    check("accepted", v["ok"] is True, v["reason"])


def test_prompt_contains_no_secrets():
    """Phase 27: keys must never reach a prompt."""
    d = build()
    system, user = build_narrative_prompt(d)
    blob = (system + user).lower()
    for token in ("sk-", "api_key", "apikey", "deepseek_api_key", "secret", "token="):
        check(f"prompt has no {token!r}", token not in blob)


def test_no_api_key_degrades_cleanly():
    saved = {k: os.environ.pop(k, None) for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")}
    try:
        d = build()
        r = generate_narrative(d)
        check("status is NO_API_KEY", r["status"] == "NO_API_KEY")
        check("no text is fabricated", r["text"] is None)
        check("it says the object stands alone", "stands on its own" in r["message"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_a_lying_model_is_rejected_not_shown():
    """End-to-end with a stub client that invents numbers."""
    class _Msg:
        def __init__(self, c): self.content = c

    class _Choice:
        def __init__(self, c): self.message = _Msg(c)

    class _Resp:
        def __init__(self, c): self.choices = [_Choice(c)]

    class _Completions:
        def create(self, **kw):
            return _Resp("Target $987.65 with an 84% probability. A proven edge.")

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    d = build()
    r = generate_narrative(d, client=_Client(), max_attempts=2)
    check("status is REJECTED", r["status"] == "REJECTED")
    check("the text is withheld, not shown with a caveat", r["text"] is None)
    check("it retried before giving up", r["attempts"] == 2)
    check("the reason is recorded", bool(r["validation"]["reason"]))


def test_an_honest_model_is_accepted():
    support = None

    class _Completions:
        def create(self, **kw):
            return type("R", (), {"choices": [type("C", (), {
                "message": type("M", (), {
                    "content": f"The nearest sourced support is ${support}. "
                               f"There is no demonstrated edge."})()})()]})()

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    d = build()
    support = d["level_map"]["nearest_support"]["price"]
    r = generate_narrative(d, client=_Client())
    check("status is OK", r["status"] == "OK", r.get("validation"))
    check("text is returned", bool(r["text"]))
    check("validation passed", r["validation"]["ok"] is True)


def test_allowlist_covers_the_objects_own_prose():
    """The engine quotes its own numbers inside its sentences; a writer quoting
    those sentences must not be penalised for it."""
    d = build()
    allowed = allowed_numbers(d)
    check("the allowlist is non-trivial", len(allowed) > 50)
    price = d["current_price"]
    check("the current price is allowed", price in allowed or round(price, 2) in allowed)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
