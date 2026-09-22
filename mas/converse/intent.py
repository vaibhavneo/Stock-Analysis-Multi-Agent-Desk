"""Natural language -> which specialists to ask.

DETERMINISTIC ON PURPOSE
------------------------
An LLM could do this, and it would be worse here for four reasons that all
matter more than flexibility:

  - It can invent a ticker. A hallucinated symbol produces a complete,
    confident analysis of a company nobody asked about.
  - It cannot be graded against a frozen corpus without spending money and
    accepting run-to-run variance in the one component whose failures are
    silent.
  - The desk's analysis path is keyless by design; making the FRONT DOOR
    require a key would be the strictest dependency in the system.
  - `agents/stock_agents.py` runs DeepSeek at a 180-second timeout. That is
    correct for a reasoning agent and unusable for a chat turn.

So routing is rules, measured against `corpus.py`, which was written and
frozen before this file existed. The LLM's place is downstream, polishing
prose over numbers that are already fixed — never choosing the numbers.

NOT ROUTING IS A REAL ANSWER
----------------------------
"tell me what to do with my money" routes nowhere, and must. Guessing a
capability for an utterance that names none produces an authoritative answer
to a question the user did not ask, which is worse than saying "which stock?".
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .symbols import extract as extract_symbols

RESEARCH = "equity_research"
OPTIONS = "option_structures"
BACKTEST = "strategy_backtest"
RECORD = "forward_record"
REGIME = "market_regime"
EVENTS = "event_calendar"
PORTFOLIO = "portfolio_review"
BENCHMARK = "benchmark_relation"

ALL_CAPABILITIES = (RESEARCH, OPTIONS, BACKTEST, RECORD, REGIME, EVENTS,
                    PORTFOLIO, BENCHMARK)

# Capabilities that stand on their own — they answer about the DESK or the
# MARKET, so they need no symbol.
SYMBOL_OPTIONAL = frozenset({BACKTEST, RECORD, REGIME, PORTFOLIO})

SELECT_THRESHOLD = 1.0

# (pattern, capability, weight). Phrases before bare words, because the bare
# words are where the collisions live.
PATTERNS: List[Tuple[str, str, float]] = [
    # ── Options ───────────────────────────────────────────────────────────
    (r"\boptions?\b", OPTIONS, 1.2),
    (r"\bcalls\b|\bputs\b", OPTIONS, 1.2),
    (r"\b(call|put)\s+(spread|option)", OPTIONS, 1.5),
    (r"\b(buy|sell|long|short|write)\s+(a\s+)?(call|put)\b", OPTIONS, 1.5),
    (r"\bcovered\s+call\b|\bprotective\s+put\b", OPTIONS, 1.5),
    (r"\biron\s+(condor|butterfly)\b|\bstraddle\b|\bstrangle\b|\bbutterfly\b",
     OPTIONS, 1.5),
    (r"\bstrikes?\b|\bexpir(y|ation|ies)\b", OPTIONS, 1.0),
    (r"\bpremium\b|\btheta\b|\bimplied\s+vol", OPTIONS, 1.0),
    (r"\bderivatives?\b", OPTIONS, 1.2),
    (r"\bhedge\b|\bhedging\b", OPTIONS, 1.2),
    (r"\bwithout\s+(buying|owning)\s+(the\s+)?shares?\b", OPTIONS, 1.5),
    (r"\bcheapest\s+way\s+to\s+(bet|play|short)\b", OPTIONS, 1.2),
    (r"\bbet\s+against\b", OPTIONS, 1.0),
    (r"\bplay\s+\w+\s+with\b", OPTIONS, 1.0),

    # ── Backtest ──────────────────────────────────────────────────────────
    (r"\bback\s?test(ed|ing)?\b", BACKTEST, 1.5),
    (r"\bhistorical(ly)?\s+(performance|done|worked)\b", BACKTEST, 1.5),
    (r"\bbeat(en)?\s+(buy[\s-]and[\s-]hold|the\s+market)\b", BACKTEST, 1.5),
    (r"\bif\s+i\s+had\s+(followed|bought|used)\b", BACKTEST, 1.5),
    (r"\bwould\s+have\s+happened\b", BACKTEST, 1.5),
    (r"\bdoes\s+(this|it|the)\s+\w*\s*(strategy|approach|thing)?\s*(actually\s+)?(make\s+money|work)\b",
     BACKTEST, 1.5),
    (r"\bprove\s+(it|this)\b", BACKTEST, 1.5),
    (r"\bsharpe\b|\bdrawdown\b|\bcagr\b", BACKTEST, 1.2),
    (r"\bposition\s+rules?\b", BACKTEST, 1.5),
    (r"\bhistorical\s+performance\b|\bperformance\s+of\s+these\s+signals\b",
     BACKTEST, 1.5),
    (r"\bhistorically\b", BACKTEST, 1.2),

    # ── Forward record ────────────────────────────────────────────────────
    (r"\btrack\s+record\b", RECORD, 1.5),
    (r"\bhit\s+rate\b", RECORD, 1.5),
    (r"\bhow\s+accurate\b|\baccuracy\b", RECORD, 1.5),
    (r"\bcalibrat(ed|ion)\b", RECORD, 1.5),
    (r"\b(been|be)\s+right\b", RECORD, 1.2),
    (r"\byour\s+(last\s+)?\w*\s*calls?\b", RECORD, 1.3),
    (r"\bturn(ed)?\s+out\b", RECORD, 1.0),
    (r"\bwhy\s+should\s+i\s+trust\b|\bcan\s+i\s+trust\b", RECORD, 1.5),
    (r"\bpredictions?\b", RECORD, 1.0),
    (r"\b(your|the)\s+record\b", RECORD, 1.5),

    # ── Market regime ─────────────────────────────────────────────────────
    (r"\bmarket\s+(regime|overall|conditions?)\b", REGIME, 1.5),
    (r"\bhow('s|\s+is)\s+the\s+market\b", REGIME, 1.5),
    (r"\brisk[\s-]o(n|ff)\b", REGIME, 1.5),
    (r"\bvix\b", REGIME, 1.2),
    (r"\bcorrection\b|\bbear\s+market\b|\bbull\s+market\b", REGIME, 1.2),
    (r"\bin\s+cash\b", REGIME, 1.2),
    (r"\bmarket\s+(going\s+to\s+)?crash\b|\bcrash\b", REGIME, 1.2),

    # ── Catalysts ─────────────────────────────────────────────────────────
    (r"\bearnings?\b", EVENTS, 1.4),
    (r"\bcatalysts?\b", EVENTS, 1.5),
    (r"\bwhen\s+does\s+\w+\s+report\b|\bwhen\s+(does|do)\s+it\s+report\b",
     EVENTS, 1.5),
    (r"\bon\s+the\s+calendar\b|\bcalendar\b", EVENTS, 1.2),
    (r"\bevents?\s+(that\s+)?could\s+move\b|\bupcoming\s+events?\b", EVENTS, 1.5),
    (r"\breport(s|ing)?\s+(date|on)\b|\breport\s+earnings\b", EVENTS, 1.4),
    (r"\bwhen\s+\w+\s+reports?\b", EVENTS, 1.4),

    # ── Portfolio ─────────────────────────────────────────────────────────
    (r"\bmy\s+(portfolio|holdings|positions|book)\b", PORTFOLIO, 1.6),
    (r"\breview\s+my\b", PORTFOLIO, 1.5),
    (r"\btoo\s+concentrated\b|\bconcentration\b", PORTFOLIO, 1.4),
    (r"\bwhat\s+should\s+i\s+(trim|sell\s+down)\b", PORTFOLIO, 1.4),
    (r"\bdiversif(y|ied|ication)\b", PORTFOLIO, 1.3),
    (r"\ball\s+of\s+my\b", PORTFOLIO, 1.4),

    # ── Benchmark relation ────────────────────────────────────────────────
    (r"\bcorrelat(ed|ion)\b", BENCHMARK, 1.5),
    (r"\bbeta\b", BENCHMARK, 1.4),
    (r"\bjust\s+(following|tracking)\b", BENCHMARK, 1.5),
    (r"\bmoving\s+(on\s+its\s+own|with)\b|\bon\s+its\s+own\s+or\b", BENCHMARK, 1.5),
    (r"\bjust\s+buying\s+the\b", BENCHMARK, 1.5),
    (r"\bdriven\s+by\b", BENCHMARK, 1.2),
    (r"\btracks?\s+the\s+(market|index|s&p)\b", BENCHMARK, 1.4),

    # ── Research, asked explicitly ────────────────────────────────────────
    (r"\banal(yse|yze|ysis)\b", RESEARCH, 1.4),
    (r"\bshould\s+i\s+(buy|sell|add|get\s+out|own)\b", RESEARCH, 1.4),
    (r"\bbuy\s+sell\s+or\s+hold\b|\bbuy,?\s+sell,?\s+or\s+hold\b", RESEARCH, 1.5),
    (r"\bthoughts?\b|\bwhat\s+do\s+you\s+think\b", RESEARCH, 1.2),
    (r"\b(your\s+)?(read|view|verdict|outlook|take)\s+on\b", RESEARCH, 1.3),
    (r"\bfull\s+(picture|analysis|breakdown)\b", RESEARCH, 1.4),
    (r"\bgood\s+buy\b|\bworth\s+(buying|holding)\b", RESEARCH, 1.4),
    (r"\bover\s?bought\b|\bover\s?sold\b", RESEARCH, 1.3),
    (r"\bhow\s+(does|do|is|are)\s+\w+\s+look", RESEARCH, 1.3),
    (r"\bgive\s+me\s+the\s+works\b|\btell\s+me\s+everything\b", RESEARCH, 1.4),
    (r"\bentry\b|\bprice\s+target\b", RESEARCH, 1.1),
    (r"\bworried\s+about\b", RESEARCH, 1.2),
    (r"\btalk\s+me\s+through\b|\bwalk\s+me\s+through\b", RESEARCH, 1.4),
    (r"\bcompare\b|\bvs\.?\b|\bversus\b|\bwhich\s+is\s+better\b", RESEARCH, 1.3),
    (r"\bstock\s+and\s+derivatives?\b", RESEARCH, 1.2),
]

# Utterances that are conversation, not queries. Matched whole-string so a
# greeting embedded in a real question does not suppress it.
SMALL_TALK = re.compile(
    r"^\s*(hi|hey|hello|hello\s+there|yo|thanks|thank\s+you|ty|ok|okay|cool|"
    r"good\s+(morning|afternoon|evening)|how\s+are\s+you|sup)\s*[!.?]*\s*$",
    re.I)

CAPABILITY_QUESTION = re.compile(
    r"\b(what\s+can\s+you\s+do|what\s+do\s+you\s+do|how\s+do\s+you\s+work|"
    r"what\s+are\s+you|help|who\s+are\s+you|capabilities)\b", re.I)

# Asks for a recommendation with no subject. Routing these would be inventing
# the subject, which is the one thing this layer must not do.
NO_SUBJECT = re.compile(
    r"\b(best|top|which)\s+(stock|share|ticker|thing|one)s?\s+(to\s+)?(buy|pick|own)\b"
    r"|\bwhat\s+should\s+i\s+(buy|invest\s+in)\b"
    r"|\bwhat\s+to\s+do\s+with\s+my\s+money\b"
    r"|\btell\s+me\s+what\s+to\s+do\s+with\b", re.I)


# An options noun standing BEFORE a buy/sell verb makes the instrument the
# object of the verb: "what puts should i buy on tesla" asks for puts, not for
# an equity brief. Without this the verb dragged a full research run into
# every options request.
_OPTION_OBJECT_FIRST = re.compile(
    r"\b(calls?|puts?|options?|spreads?|premium|strikes?)\b[^.?!]{0,40}?"
    r"\bshould\s+i\s+(buy|sell|own|add)\b", re.I)
_BUY_VERB_RESEARCH = r"\bshould\s+i\s+(buy|sell|add|get\s+out|own)\b"


def _score(text: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    suppress_buy_verb = bool(_OPTION_OBJECT_FIRST.search(text))
    for pat, cap, w in PATTERNS:
        if suppress_buy_verb and pat == _BUY_VERB_RESEARCH:
            continue
        if re.search(pat, text, re.I):
            scores[cap] = scores.get(cap, 0.0) + w
    return scores


def parse(text: str, context_symbol: Optional[str] = None) -> Dict[str, Any]:
    """Utterance -> {symbols, capabilities, kind, clarification, why}.

    `kind` separates outcomes that a bare capability list would flatten:
    QUERY (routed), SMALL_TALK, CAPABILITY_QUESTION, NEEDS_SUBJECT and
    UNROUTABLE are five different things to say back.
    """
    raw = (text or "").strip()
    out: Dict[str, Any] = {
        "text": raw, "symbols": [], "capabilities": [], "scores": {},
        "kind": "UNROUTABLE", "clarification": None, "why": [],
        "context_symbol": context_symbol, "symbols_from_context": False,
    }
    if not raw:
        out["clarification"] = "Ask me about a stock, the market, or your portfolio."
        return out

    if SMALL_TALK.match(raw):
        out["kind"] = "SMALL_TALK"
        return out

    if CAPABILITY_QUESTION.search(raw) and len(raw.split()) <= 8:
        out["kind"] = "CAPABILITY_QUESTION"
        return out

    sym = extract_symbols(raw, context_symbol=context_symbol)
    out["symbols"] = sym["symbols"]
    out["symbol_basis"] = sym["how"]
    out["symbols_from_context"] = sym["from_context"]

    scores = _score(raw)
    out["scores"] = {k: round(v, 2) for k, v in scores.items()}
    caps = [c for c, v in scores.items() if v >= SELECT_THRESHOLD]

    # A named subject with no stated intent is a research question. This is the
    # single most common utterance a desk gets ("TSLA"), and it fires only when
    # nothing more specific did — "show me calls on TSLA" is not a request for
    # a full equity brief.
    if sym["symbols"] and not caps:
        caps = [RESEARCH]
        out["why"].append(
            f"{sym['symbols'][0]} was named with no more specific question, so "
            "this is read as a request for the full read on it.")

    if NO_SUBJECT.search(raw) and not sym["symbols"]:
        out["kind"] = "NEEDS_SUBJECT"
        out["capabilities"] = []
        out["clarification"] = (
            "I do not pick names for you — I have no ranked universe behind "
            "that question, and answering it would be inventing one. Name a "
            "symbol and I will give you the full read.")
        return out

    # A symbol-requiring capability with no symbol cannot run.
    runnable, blocked = [], []
    for c in caps:
        if c in SYMBOL_OPTIONAL or sym["symbols"]:
            runnable.append(c)
        else:
            blocked.append(c)

    out["capabilities"] = sorted(runnable)
    out["blocked"] = blocked

    if blocked and not runnable:
        out["kind"] = "NEEDS_SUBJECT"
        out["clarification"] = (
            "Which symbol? " + ", ".join(b.replace("_", " ") for b in blocked)
            + " needs one, and I would rather ask than guess.")
        return out

    if not out["capabilities"]:
        out["kind"] = "UNROUTABLE"
        out["clarification"] = (
            "I could not tell what you want from that. Try naming a symbol, or "
            "ask about the market, your portfolio, or how this desk has done.")
        return out

    out["kind"] = "QUERY"
    if sym["from_context"]:
        out["why"].append(
            f"No symbol in that message, so it is read as still being about "
            f"{sym['symbols'][0]}.")
    return out
