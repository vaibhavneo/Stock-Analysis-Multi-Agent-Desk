#!/usr/bin/env python3
"""The OptionsPilot link.

Everything here runs offline. The client is exercised against stubbed HTTP so
the failure modes that matter — unreachable, 401, a 500 that returns HTML, a
body that is not JSON — are tested deliberately rather than waited for.

Two properties get the most attention, because both are ways the link could do
real harm while looking like it works:

  **The access code must never escape.** It is read from the environment,
  POSTed once, and must appear in no returned payload, no log line and no
  prompt.

  **OptionsPilot's honesty must be carried, not restated.** It orders
  structures by payoff shape and explicitly disclaims that any is best. If this
  app hardcoded a caveat instead of passing theirs through, the caveat would
  survive a change on their side that invalidated it — so a mutation test
  strips the source's honesty and asserts ours disappears with it.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from decision.options_overlay import build_options_overlay, view_for
from optionspilot import client

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


SECRET = "test-access-code-never-leaks-9f3a"


def _raw_payload():
    return {
        "available": True, "symbol": "TEST", "ticker": "TEST", "spot": 100.0,
        "expiry": "2026-10-23", "days": 33, "outcome": "BULLISH",
        "priced_at": "2026-09-20", "live": True,
        "edge": "Candidates are ordered on payoff SHAPE, which is a stated prior.",
        "freshness_note": "Priced from the chain as it is now.",
        "sets": [{
            "available": True, "view": "DIRECTIONAL", "criterion": "payoff shape per unit of loss",
            "ranked_not_best": "These are ORDERED, not recommended.",
            "rules_version": "candidate-rules-v1.0.0",
            "carry_identity": {"deviation_means": "The gap is the volatility smile, not an edge."},
            "liquidity": {"by_structure": [
                {"structure": "bull_call_spread", "all_tradeable": True, "worst_spread_pct": 15.0}]},
            "quoting": {"failed": [], "legs_priced": 2, "legs_wanted": 2},
            "unpriceable": [],
            "candidates": [{
                "rank": 1, "label": "Buy a call spread", "structure": "bull_call_spread",
                "legs": [{"action": "buy", "quantity": 1, "right": "call", "strike": 100.0},
                         {"action": "sell", "quantity": -1, "right": "call", "strike": 110.0}],
                "note": "Cheaper than the call and capped at the short strike.",
                "probability_of_profit": 0.40,
                "profile": {"breakevens": [103.5], "max_loss": -3.5, "max_gain": 6.5},
                "risk_reward": {"ratio": 1.86, "note": "Shape only."},
                "expected_payoff": -0.16, "criterion_value": 1.65,
                "edge": "This ranks SHAPE, not merit. The research baseline reads NO_DEMONSTRATED_EDGE.",
                "research": {"verdict": "NO_DEMONSTRATED_EDGE", "baseline": "V6.0.0",
                             "demonstrated_edge": False},
            }],
        }],
    }


class _Resp:
    def __init__(self, body, status=200):
        self._body = body.encode() if isinstance(body, str) else body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_opener(handler):
    """Replace the module's opener so no socket is ever created."""
    class _Opener:
        def open(self, req, timeout=None):
            return handler(req)
    return _Opener()


# ══════════════════════════════════════════════════════════════════════════
# The client survives every way the dependency can fail
# ══════════════════════════════════════════════════════════════════════════

def test_unreachable_is_reported_not_raised():
    client.reset()
    def boom(req):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))
    with patch.object(client, "_get_opener", lambda: _stub_opener(boom)):
        r = client.instruments("TEST")
    check("available is False", r["available"] is False)
    check("a reason is given", "could not reach OptionsPilot" in r["reason"], r["reason"])


def test_html_error_page_is_not_parsed_as_data():
    """A 500 from OptionsPilot renders HTML. Parsing that as a payload is how a
    broken dependency becomes nonsense on screen instead of a stated outage."""
    client.reset()
    def five_hundred(req):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {},
                                     __import__("io").BytesIO(b"<!doctype html><h1>500</h1>"))
    with patch.object(client, "_get_opener", lambda: _stub_opener(five_hundred)):
        r = client.instruments("TEST")
    check("available is False", r["available"] is False)
    check("the status is named", "500" in r["reason"], r["reason"])
    check("no HTML leaks into the reason", "<" not in r["reason"])


def test_non_json_body_is_an_error_not_a_payload():
    client.reset()
    with patch.object(client, "_get_opener", lambda: _stub_opener(lambda req: _Resp("not json"))):
        r = client.instruments("TEST")
    check("available is False", r["available"] is False)
    check("it says what happened", "not JSON" in r["reason"], r["reason"])


def test_rejected_access_code_is_actionable():
    client.reset()
    def unauthorized(req):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},
                                     __import__("io").BytesIO(b'{"code":"access_required"}'))
    with patch.dict(os.environ, {"OPTIONSPILOT_ACCESS_CODE": SECRET}), \
         patch.object(client, "_get_opener", lambda: _stub_opener(unauthorized)):
        r = client.instruments("TEST")
    check("available is False", r["available"] is False)
    check("it names the rotation problem", "rotated" in r["reason"] or "rejected" in r["reason"],
          r["reason"])
    check("the secret is not echoed", SECRET not in json.dumps(r))


def test_a_ticker_with_no_structures_is_not_an_outage():
    client.reset()
    body = json.dumps({"available": False, "reason": "the options chain could not be read"})
    with patch.object(client, "_get_opener", lambda: _stub_opener(lambda req: _Resp(body))):
        r = client.instruments("TEST")
    check("available is False", r["available"] is False)
    check("OptionsPilot's own reason is carried",
          "chain could not be read" in r["reason"], r["reason"])


# ══════════════════════════════════════════════════════════════════════════
# The access code
# ══════════════════════════════════════════════════════════════════════════

def test_the_access_code_never_appears_in_any_payload():
    client.reset()
    seen = {}

    def handler(req):
        if req.full_url.endswith("/access"):
            seen["login_body"] = req.data
            return _Resp("", status=200)
        return _Resp(json.dumps(_raw_payload()))

    with patch.dict(os.environ, {"OPTIONSPILOT_ACCESS_CODE": SECRET}), \
         patch.object(client, "_get_opener", lambda: _stub_opener(handler)):
        r = client.instruments("TEST")
        st = client.status()

    check("the code was sent to /access", SECRET.encode() in (seen.get("login_body") or b""))
    check("it is absent from the instruments payload", SECRET not in json.dumps(r, default=str))
    check("it is absent from status", SECRET not in json.dumps(st, default=str))
    check("status reports configuredness only", st["access_code_configured"] is True)
    check("status carries no code field",
          not any("code" in k and k != "access_code_configured" for k in st))


def test_no_code_configured_says_so_without_guessing():
    client.reset()
    env = {k: v for k, v in os.environ.items() if k != "OPTIONSPILOT_ACCESS_CODE"}
    def unauthorized(req):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},
                                     __import__("io").BytesIO(b"{}"))
    with patch.dict(os.environ, env, clear=True), \
         patch.object(client, "_get_opener", lambda: _stub_opener(unauthorized)):
        st = client.status()
    check("not available", st["available"] is False)
    check("it names the missing variable",
          "OPTIONSPILOT_ACCESS_CODE" in st["reason"], st["reason"])


# ══════════════════════════════════════════════════════════════════════════
# The overlay carries their honesty rather than asserting its own
# ══════════════════════════════════════════════════════════════════════════

def test_honesty_is_carried_through():
    o = build_options_overlay(_raw_payload(), requested_view="BULLISH")
    joined = " ".join(o["honesty"])
    check("ordered-not-recommended survives", "ORDERED, not recommended" in joined)
    check("the shape prior survives", "payoff SHAPE" in joined)
    check("their research verdict survives", "no demonstrated edge" in joined.lower())
    check("the smile caveat survives", "volatility smile" in joined)
    check("each candidate keeps its own edge note",
          "NO_DEMONSTRATED_EDGE" in (o["candidates"][0]["edge_note"] or ""))


def test_honesty_is_not_hardcoded():
    """MUTATION: strip the source's honesty and ours must vanish with it. A
    caveat this app invents would survive a change on their side that made it
    false."""
    raw = _raw_payload()
    raw.pop("edge", None)
    raw.pop("freshness_note", None)
    raw["sets"][0].pop("ranked_not_best", None)
    raw["sets"][0].pop("carry_identity", None)
    raw["sets"][0]["candidates"][0]["research"] = {"verdict": "DEMONSTRATED_EDGE",
                                                   "demonstrated_edge": True}
    o = build_options_overlay(raw, requested_view="BULLISH")
    check("honesty shrinks when the source drops it", len(o["honesty"]) == 0,
          o["honesty"])


def test_probabilities_are_labelled_as_model_numbers():
    o = build_options_overlay(_raw_payload(), requested_view="BULLISH")
    c = o["candidates"][0]
    check("a probability is shown", c["probability_of_profit"] == 0.40)
    check("its basis is stated", "model number" in c["probability_basis"])
    check("it is not called measured", "measured frequency" in c["probability_basis"])


def test_unavailable_overlay_invents_nothing():
    o = build_options_overlay({"available": False, "reason": "OptionsPilot is down"})
    check("status is UNAVAILABLE", o["status"] == "UNAVAILABLE")
    check("no candidates are fabricated", o["candidates"] == [])
    check("the reason is carried", "down" in o["reason"])
    check("no honesty is asserted about absent data", o["honesty"] == [])


def test_the_direction_source_is_recorded():
    ours = build_options_overlay(_raw_payload(), requested_view="BULLISH")
    theirs = build_options_overlay(_raw_payload(), requested_view=None)
    check("ours is attributed to this engine",
          ours["direction_source"]["source"] == "THIS_ENGINE")
    check("and says so", "this engine's own evidence" in ours["direction_source"]["statement"])
    check("theirs is attributed to OptionsPilot",
          theirs["direction_source"]["source"] == "OPTIONSPILOT")
    check("and says so", "did not supply the direction" in theirs["direction_source"]["statement"])


def test_a_neutral_thesis_hands_over_no_direction():
    """Inventing one would be the link asserting a view neither service holds."""
    check("neutral yields no view", view_for({"direction": "NEUTRAL"}) is None)
    check("no thesis yields no view", view_for(None) is None)
    check("bullish is passed through", view_for({"direction": "BULLISH"}) == "BULLISH")
    check("bearish is passed through", view_for({"direction": "BEARISH"}) == "BEARISH")


def test_legs_are_rendered_in_words():
    o = build_options_overlay(_raw_payload(), requested_view="BULLISH")
    text = o["candidates"][0]["legs_text"]
    check("both legs appear", "call" in text and "$100.00" in text and "$110.00" in text, text)
    check("the actions appear", "uy" in text and "sell" in text, text)


# ══════════════════════════════════════════════════════════════════════════
# The API surface, and what it must not become
# ══════════════════════════════════════════════════════════════════════════

def _client():
    import web.app as app_mod
    app_mod._OPTIONS_CACHE.clear()
    return app_mod, app_mod.app.test_client()


def test_reading_structures_never_triggers_a_run():
    """The read is side-effect free on the other service. A read that quietly
    wrote to another system's journal would be the worst kind of surprise."""
    app_mod, c = _client()
    with patch.object(client, "instruments", return_value=_raw_payload()) as read, \
         patch.object(client, "run_pipeline") as run:
        r = c.get("/api/options/TEST?view=BULLISH")
    check("200", r.status_code == 200)
    check("the read happened", read.called)
    check("no run was triggered", not run.called)


def test_the_view_is_passed_through_and_validated():
    app_mod, c = _client()
    with patch.object(client, "instruments", return_value=_raw_payload()) as read:
        c.get("/api/options/TEST?view=BULLISH")
        check("a valid view is forwarded", read.call_args.kwargs.get("view") == "BULLISH")
        app_mod._OPTIONS_CACHE.clear()
        c.get("/api/options/TEST?view=SIDEWAYS")
        check("an unrecognised view is dropped, not forwarded",
              read.call_args.kwargs.get("view") is None)


def test_the_endpoint_degrades_to_200_with_a_reason():
    """A failing dependency must not turn into a 500 on this service."""
    app_mod, c = _client()
    with patch.object(client, "instruments",
                      side_effect=RuntimeError("boom")):
        r = c.get("/api/options/TEST")
    check("still 200", r.status_code == 200)
    d = r.get_json()
    check("status is UNAVAILABLE", d["status"] == "UNAVAILABLE")
    check("a reason is given", bool(d["reason"]))


def test_the_run_endpoint_is_the_only_write():
    app_mod, _ = _client()
    options_rules = [(r.rule, sorted(r.methods - {"HEAD", "OPTIONS"}))
                     for r in app_mod.app.url_map.iter_rules() if "/api/options" in r.rule]
    writes = [(rule, methods) for rule, methods in options_rules if "POST" in methods]
    check("exactly one write route", len(writes) == 1, writes)
    check("and it is the run", writes[0][0] == "/api/options/run", writes[0])


def test_the_link_adds_no_order_path():
    app_mod, _ = _client()
    rules = [r.rule.lower() for r in app_mod.app.url_map.iter_rules()]
    banned = ("order", "buy", "sell", "trade", "execute", "place", "exercise")
    offenders = [r for r in rules if any(b in r for b in banned)]
    check("no execution route exists anywhere", not offenders, offenders)


def test_the_overlay_states_that_nothing_can_be_executed():
    o = build_options_overlay(_raw_payload(), requested_view="BULLISH")
    check("it says so", "not orders to send" in o["no_execution"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"  {t.__name__}...")
        t()
    total = PASS + FAIL
    print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILED'}: {PASS}/{total} checks")
    sys.exit(0 if FAIL == 0 else 1)
