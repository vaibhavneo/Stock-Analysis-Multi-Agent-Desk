#!/usr/bin/env python3
"""The conversational front door.

Routing is the component in this system whose failures are SILENT: a
mis-routed question still returns a confident, well-formatted answer — just
to a question nobody asked. So the corpus in mas/converse/corpus.py is a
regression gate, not a demo. It was written and frozen before the router
existed, and the thresholds below are set at the measured level so a
vocabulary regression fails the build rather than quietly costing coverage.

The other thing held here is the difference between five outcomes that a bare
capability list would flatten: routed, small talk, a capability question, a
question with no subject, and one that could not be read at all. Each needs a
different thing said back, and collapsing them is how a desk starts answering
questions it was never asked.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas.converse import corpus
from mas.converse.intent import parse, ALL_CAPABILITIES
from mas.converse.symbols import extract, AMBIGUOUS, KNOWN_TICKERS
from mas.converse import session as session_mod
from mas.converse import reply as reply_mod

PASS = 0
FAIL = 0

# Measured on the frozen corpus. Set AT the achieved level: a threshold below
# what the code does is a ratchet that permits silent decay.
MIN_ROUTING = 0.99
MIN_SYMBOLS = 1.00


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def _grade():
    exact = sym_ok = 0
    misroutes = []
    for c in corpus.CASES:
        r = parse(c["q"], context_symbol=c.get("context"))
        if set(r["capabilities"]) == set(c["caps"]):
            exact += 1
        else:
            misroutes.append((c["q"], sorted(c["caps"]), sorted(r["capabilities"])))
        if set(c["symbols"]) <= set(r["symbols"]):
            sym_ok += 1
    n = len(corpus.CASES)
    return exact / n, sym_ok / n, misroutes


# ── The corpus, as a gate ─────────────────────────────────────────────────

def test_corpus_is_frozen_and_covers_every_capability():
    check("corpus is substantial", corpus.size() >= 100, corpus.size())
    covered = {cap for c in corpus.CASES for cap in c["caps"]}
    check("every routable capability has cases",
          covered == set(ALL_CAPABILITIES), sorted(set(ALL_CAPABILITIES) - covered))
    check("it contains cases that must route NOWHERE",
          sum(1 for c in corpus.CASES if not c["caps"]) >= 8)
    check("and cases that depend on the previous turn",
          sum(1 for c in corpus.CASES if c.get("context")) >= 5)


def test_routing_accuracy_on_the_held_out_corpus():
    route, _, misroutes = _grade()
    check(f"routing >= {MIN_ROUTING:.0%}", route >= MIN_ROUTING,
          f"got {route:.1%}; misroutes: {misroutes[:5]}")


def test_symbol_extraction_on_the_held_out_corpus():
    _, syms, _ = _grade()
    check(f"symbols >= {MIN_SYMBOLS:.0%}", syms >= MIN_SYMBOLS, f"got {syms:.1%}")


def test_nothing_is_ever_routed_to_an_unregistered_capability():
    from mas import registry
    known = set(registry.CAPABILITIES)
    for c in corpus.CASES:
        for cap in parse(c["q"], context_symbol=c.get("context"))["capabilities"]:
            check(f"{cap} is registered", cap in known, c["q"])


# ── Symbols: the wrong one is worse than none ─────────────────────────────

def test_english_words_that_are_tickers_do_not_resolve():
    """ALL, IT, ON, A and friends are real US tickers. Analysing Allstate
    because someone said "all of my positions" is a confident answer about a
    company nobody mentioned."""
    for q in ("ALL of my positions", "IT department stocks", "are you sure about that",
              "A good entry point", "is it a buy", "on the other hand"):
        got = extract(q)["symbols"]
        check(f"{q!r} resolves nothing", got == [], got)


def test_the_guard_can_still_be_overridden_explicitly():
    check("$ prefix wins", extract("$ALL")["symbols"] == ["ALL"])
    check("a solo ticker wins", extract("ON")["symbols"] == ["ON"])
    check("a company cue wins", extract("ON semiconductor")["symbols"] == ["ON"])


def test_lower_case_tickers_resolve_but_only_known_ones():
    check("nvda resolves", "NVDA" in extract("nvda vs amd")["symbols"])
    check("amd resolves", "AMD" in extract("nvda vs amd")["symbols"])
    check("a random word does not", extract("zxqv please")["symbols"] == [])
    check("ambiguous words stay excluded from the lexicon",
          not (AMBIGUOUS & KNOWN_TICKERS), sorted(AMBIGUOUS & KNOWN_TICKERS)[:5])


def test_the_lowercase_scan_is_word_anchored():
    """Unanchored, the scan chopped "report" into "repor" + "t" and resolved
    T (AT&T) out of the middle of a word."""
    for q in ("when does it report", "a great quarter", "front and center",
              "is it correlated"):
        check(f"{q!r} pulls no ticker from inside a word",
              extract(q)["symbols"] == [], extract(q)["symbols"])


def test_non_equity_shapes_resolve():
    for q, want in (("BTC-USD", "BTC-USD"), ("bitcoin", "BTC-USD"),
                    ("EURUSD=X", "EURUSD=X"), ("^GSPC", "^GSPC"),
                    ("gold", "GC=F"), ("the s&p 500", "^GSPC")):
        check(f"{q!r} -> {want}", want in extract(q)["symbols"], extract(q)["symbols"])


def test_context_fills_only_when_the_text_resolves_nothing():
    check("elliptical follow-up inherits",
          extract("and the options?", context_symbol="AAPL")["symbols"] == ["AAPL"])
    check("a named follow-up does NOT inherit",
          extract("what about microsoft", context_symbol="AAPL")["symbols"] == ["MSFT"])
    check("inheritance is flagged",
          extract("and the options?", context_symbol="AAPL")["from_context"] is True)


# ── Five outcomes, not one ────────────────────────────────────────────────

def test_small_talk_is_not_a_query():
    for q in ("hi", "hello there", "thanks", "ok"):
        check(f"{q!r} is small talk", parse(q)["kind"] == "SMALL_TALK", parse(q)["kind"])


def test_a_request_with_no_subject_asks_instead_of_guessing():
    """The single most dangerous utterance: it invites a recommendation and
    names nothing. Answering it would mean inventing the subject."""
    for q in ("what's the best stock to buy", "tell me what to do with my money",
              "what should i invest in"):
        r = parse(q)
        check(f"{q!r} needs a subject", r["kind"] == "NEEDS_SUBJECT", r["kind"])
        check("nothing is routed", r["capabilities"] == [], r["capabilities"])
        check("and it says so", bool(r["clarification"]))


def test_unreadable_input_routes_nowhere():
    for q in ("asdfgh", "buy", "i want a long term hold, not a trade"):
        r = parse(q)
        check(f"{q!r} routes nowhere", r["capabilities"] == [], r)
        check("with something to say back",
              bool(r.get("clarification")) or r["kind"] != "QUERY")


def test_a_symbol_requiring_capability_without_a_symbol_asks():
    r = parse("show me some call options")
    check("no symbol -> asks", r["kind"] == "NEEDS_SUBJECT", r["kind"])
    check("the blocked capability is named",
          "option_structures" in (r.get("blocked") or []), r.get("blocked"))


def test_desk_level_questions_need_no_symbol():
    for q, cap in (("how's the market", "market_regime"),
                   ("what's your hit rate", "forward_record"),
                   ("review my portfolio", "portfolio_review")):
        r = parse(q)
        check(f"{q!r} routes", cap in r["capabilities"], r["capabilities"])
        check("without a symbol", r["symbols"] == [], r["symbols"])


def test_a_bare_symbol_is_a_research_request():
    r = parse("TSLA")
    check("routes to research", r["capabilities"] == ["equity_research"])
    check("and explains the default", any("more specific" in w for w in r["why"]), r["why"])


def test_an_options_request_does_not_drag_in_a_full_research_run():
    """'what puts should i buy' asks for puts. The buy verb was pulling an
    equity brief into every options request."""
    r = parse("what puts should i buy on tesla")
    check("options only", r["capabilities"] == ["option_structures"], r["capabilities"])


# ── Session ───────────────────────────────────────────────────────────────

def test_the_subject_carries_but_the_verdict_does_not():
    s = session_mod.get(None)
    session_mod.record(s, "nvda", {"symbols": ["NVDA"], "symbols_from_context": False,
                                   "kind": "QUERY", "capabilities": ["equity_research"]},
                       {"headline": "x"})
    check("subject advanced", s["subject"] == "NVDA")
    check("no verdict is stored",
          not any(k in s for k in ("action", "composite", "verdict", "decision")),
          list(s.keys()))


def test_an_inherited_subject_does_not_re_assert_itself():
    """A turn that only inherited the subject must not re-pin it. Otherwise a
    single mis-parse owns the rest of the conversation."""
    s = session_mod.get(None)
    s["subject"] = "AAPL"
    session_mod.record(s, "and the options?",
                       {"symbols": ["AAPL"], "symbols_from_context": True,
                        "kind": "QUERY", "capabilities": ["option_structures"]},
                       {"headline": "x"})
    check("subject unchanged", s["subject"] == "AAPL")


def test_sessions_are_capped():
    check("there is a cap", session_mod.MAX_SESSIONS > 0)
    check("and a turn cap", session_mod.MAX_TURNS > 0)
    check("and a ttl", session_mod.TTL_SEC > 0)


# ── Replies describe only results that exist ──────────────────────────────

def test_a_declined_specialist_is_reported_not_papered_over():
    parsed = {"kind": "QUERY", "capabilities": ["market_regime"], "symbols": []}
    results = {"market_regime": {"result": None,
                                 "unanswered_reason": "the regime model was unreachable"}}
    out = reply_mod.compose(parsed, results, None)
    text = " ".join(l for b in out["blocks"] for l in b["lines"])
    check("the decline is stated", "unreachable" in text, text)
    check("and is marked as a decline",
          any(b.get("declined") for b in out["blocks"]), out["blocks"])


def test_nothing_is_invented_when_every_specialist_declines():
    parsed = {"kind": "QUERY", "capabilities": ["equity_research"], "symbols": ["AAPL"]}
    out = reply_mod.compose(parsed, {"equity_research": {"result": None,
                                                        "unanswered_reason": "no data"}},
                            "AAPL")
    text = " ".join(l for b in out["blocks"] for l in b["lines"])
    check("no verdict is produced", "BUY" not in text and "SELL" not in text, text)


def test_a_credit_structure_is_not_printed_as_a_loss():
    """Net cost is negative for a credit. Printing "-$1,959.27" beside "sell an
    iron butterfly" reads as a loss rather than premium received."""
    d = {"status": "OK", "is_model_priced": True, "view": "BULLISH",
         "candidates": [
             {"structure": "bull_call_spread", "label": "Buy a call spread",
              "plain": "Pay $100.", "net_cost": 100, "cash_direction": "DEBIT"},
             {"structure": "iron_butterfly", "label": "Sell an iron butterfly",
              "plain": "Collect $1959.27.", "net_cost": -1959.27,
              "cash_direction": "CREDIT"}]}
    lines = reply_mod._say_options(d, "NVDA")
    text = " ".join(lines)
    check("credit reads as collected", "collect $1,959.27" in text.lower(), text)
    check("no negative sign on the credit", "-$1,959.27" not in text, text)


def test_a_ticker_survives_sentence_capitalisation():
    """str.capitalize() lower-cases everything after the first letter, which
    turned BTC-USD into btc-usd inside a verdict."""
    d = {"benchmark": "BTC-USD", "benchmark_label": "bitcoin", "beta": 1.3,
         "r_squared": 0.82, "overlapping_returns": 364,
         "verdict": "bitcoin explains most of ETH-USD's movement"}
    text = " ".join(reply_mod._say_benchmark(d, "ETH-USD"))
    check("the ticker keeps its case", "ETH-USD" in text, text)
    check("and the sentence still starts capitalised", "Bitcoin explains" in text, text)


def test_the_roster_blurb_promises_only_registered_capabilities():
    """A capability list written in prose can drift from the roster. If the
    blurb offers something no agent provides, the desk is advertising a
    capability it will then decline."""
    from mas import registry
    blurb = " ".join(reply_mod.ROSTER_BLURB).lower()
    check("it refuses to place orders", "cannot place orders" in blurb, blurb[:80])
    check("it refuses to pick a name unprompted", "will not pick a stock" in blurb)
    check("every composer maps to a registered capability",
          set(reply_mod.COMPOSERS) <= set(registry.CAPABILITIES),
          sorted(set(reply_mod.COMPOSERS) - set(registry.CAPABILITIES)))


# ── End to end ────────────────────────────────────────────────────────────

def test_a_turn_returns_a_reply_and_a_trace_for_every_kind():
    from mas.converse import turn
    for q in ("hi", "what can you do", "what's the best stock to buy", "asdfgh"):
        out = turn(q)
        check(f"{q!r} replies", bool(out["reply"]["blocks"]), out)
        check("session id issued", bool(out["session"]["session_id"]))
        check("trace present", isinstance(out["trace"], list))


def test_routing_does_no_network_io():
    """The front door must answer instantly and without a key. If parsing
    touched the network, every assertion above would be untrustworthy offline
    and a chat turn would inherit an analysis-agent timeout."""
    import socket
    real = socket.socket

    class Boom(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("parse() opened a socket")

    socket.socket = Boom
    try:
        for c in corpus.CASES:
            parse(c["q"], context_symbol=c.get("context"))
        check("parsing is fully offline", True)
    finally:
        socket.socket = real


def test_an_accumulating_record_is_distinguished_from_an_empty_one():
    """Production held 20 matured predictions and reported "nothing has
    matured yet". Those are different facts, and the second is the useful
    one — it says the record is filling and what it is waiting for."""
    from decision.track_record import LIVE_SAMPLE_READABLE
    empty = reply_mod._say_record({"live_record": {"rows": [], "total_matured": 0}}, "")
    check("truly empty says so", "No frozen prediction has matured" in empty[0], empty)

    partial = reply_mod._say_record(
        {"live_record": {"total_matured": 20,
                         "rows": [{"horizon_days": 1, "n": 18, "readable": False},
                                  {"horizon_days": 5, "n": 12, "readable": False}]}}, "")
    text = " ".join(partial)
    check("it reports what HAS matured", "20 prediction" in text, text)
    check("it names the largest sample", "18" in text, text)
    check("and the bar it is short of", str(LIVE_SAMPLE_READABLE) in text, text)
    check("no win rate is quoted below the bar", "%" not in text, text)


def test_a_readable_record_quotes_the_rate():
    out = reply_mod._say_record(
        {"live_record": {"total_matured": 1498,
                         "rows": [{"horizon_days": 20, "n": 300, "win_rate": 0.515,
                                   "avg_excess_return_pct": -0.34, "readable": True}]}}, "")
    text = " ".join(out)
    check("the rate appears", "51.5%" in text, text)
    check("with its sample size", "300" in text, text)


def test_a_request_for_the_write_up_does_not_also_trigger_a_research_run():
    """'give me the full written analysis' asks for the WRITE-UP. The bare
    'analysis' keyword was dragging a deterministic research run in beside
    it — two different jobs from one phrase."""
    r = parse("give me the full written analysis on META")
    check("narrative only", r["capabilities"] == ["analyst_narrative"],
          r["capabilities"])
    check("and the symbol resolved", r["symbols"] == ["META"], r["symbols"])
    plain = parse("give me the analysis on META")
    check("a plain analysis request still routes to research",
          "equity_research" in plain["capabilities"], plain["capabilities"])


def test_the_written_reasoning_is_never_confused_with_the_interpretive_layer():
    check("reasoning -> narrative",
          parse("explain your reasoning on NVDA")["capabilities"]
          == ["analyst_narrative"])
    check("context -> intelligence",
          parse("historical context on NVDA")["capabilities"]
          == ["market_intelligence"])


def test_a_bounded_analyst_pass_points_somewhere_useful():
    """The five-agent pass takes minutes — longer than an interactive turn
    should hold. In production it ran past the client's patience and returned
    nothing a user could act on. The bound now trips and names the streaming
    route that runs the same agents with progress."""
    import mas.converse.engine as eng
    from mas.agents.desk_capabilities import ANALYST_STREAMING_ROUTE
    from mas import registry

    spec = registry.get("analysts")
    check("bounded well under a minute and a half",
          spec["timeout_sec"] <= 90, spec["timeout_sec"])
    check("the pointer names the streaming route",
          "Analyze Stock" in ANALYST_STREAMING_ROUTE, ANALYST_STREAMING_ROUTE)
    check("and says the numbers do not depend on it",
          "already computed without them" in ANALYST_STREAMING_ROUTE)

    orig = eng._run_capability
    eng._run_capability = lambda cap, sym, params: {
        "capability": cap, "result": None,
        "unanswered_reason": "no answer within 75s"}
    try:
        out = eng.turn("explain your reasoning on NVDA")
        text = " ".join(l for b in out["reply"]["blocks"] for l in b["lines"])
        check("the timeout is replaced by the pointer",
              "Analyze Stock" in text, text[:200])
        check("and the raw timeout is not what the user sees",
              "within 75s" not in text, text[:200])
    finally:
        eng._run_capability = orig


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    r, s, m = _grade()
    print(f"\nrouting {r:.1%} · symbols {s:.1%} · {PASS} passed, {FAIL} failed")
