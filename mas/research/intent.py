"""What KIND of decision is being asked for.

This is a different and prior question to "which specialist answers it".
The capability router maps an utterance to agents; this maps it to the shape
of the decision. Conflating them is what produced the baseline failure:
"Should I add to my IONQ position?" and "Tell me about IONQ" both routed to
`equity_research` and received the same fixed computation, when only the
first is a position-management question and only the first should reach the
add engine, its averaging-down guard, and the position context.

Exactly ONE intent per utterance. Capabilities are plural because a question
can need several specialists; the decision being asked for is singular, and
allowing two would mean the plan had not actually decided anything.

Graded against `intent_corpus.py`, which was written and frozen first.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .position import extract as _extract_position
from .intent_corpus import (
    GENERAL_RESEARCH, NEW_ENTRY, EXISTING_POSITION, ADD_TO_POSITION,
    REDUCE_POSITION, EXIT_POSITION, SHORT_TERM_SETUP, MEDIUM_TERM_SETUP,
    LONG_TERM_THESIS, EARNINGS_PREVIEW, EVENT_ANALYSIS, RISK_ANALYSIS,
    COMPARISON, PORTFOLIO_CONTEXT, OPTIONS_ANALYSIS, INVALIDATION, NONE,
)

ALL_INTENTS = (
    GENERAL_RESEARCH, NEW_ENTRY, EXISTING_POSITION, ADD_TO_POSITION,
    REDUCE_POSITION, EXIT_POSITION, SHORT_TERM_SETUP, MEDIUM_TERM_SETUP,
    LONG_TERM_THESIS, EARNINGS_PREVIEW, EVENT_ANALYSIS, RISK_ANALYSIS,
    COMPARISON, PORTFOLIO_CONTEXT, OPTIONS_ANALYSIS, INVALIDATION, NONE,
)

# Intents that only make sense about something already held.
POSITION_INTENTS = frozenset({ADD_TO_POSITION, REDUCE_POSITION, EXIT_POSITION,
                              EXISTING_POSITION, PORTFOLIO_CONTEXT})

SELECT_THRESHOLD = 1.0


def _stated_position(text: str) -> bool:
    """Did the user assert they hold the thing, in words or numbers?"""
    try:
        return bool(_extract_position(text).get("owns"))
    except Exception:
        return False

# Ordered: the first pattern that matches at the top weight decides. Position
# actions sit above general research because "should I add to my NVDA" is a
# position question that also mentions a stock, not the reverse.
PATTERNS: List[Tuple[str, str, float]] = [
    # ── Not a research question at all ────────────────────────────────────
    (r"^\s*(hi|hey|hello|thanks|thank\s+you|ok|okay|cool)\s*[!.?]*\s*$", NONE, 5.0),
    (r"\bwhat\s+can\s+you\s+do\b|\bhow\s+do\s+you\s+work\b", NONE, 5.0),
    (r"\b(best|top|which)\s+stocks?\s+to\s+buy\b"
     r"|\bwhat\s+should\s+i\s+(buy|invest\s+in)\b(?!\s+more\b)", NONE, 5.0),
    # "add a watchlist", "reduce the noise" — the verb without a security.
    (r"\badd\s+(a|an|the)\s+(watchlist|alert|column|note|filter)\b", NONE, 5.0),
    (r"\breduce\s+the\s+(noise|clutter|verbosity)\b", NONE, 5.0),
    (r"\bbuy\s+more\s+time\b", NONE, 5.0),

    # ── Portfolio, before the single-name position intents ────────────────
    (r"\bmy\s+(portfolio|holdings|book)\b", PORTFOLIO_CONTEXT, 3.0),
    (r"\breview\s+my\b|\btoo\s+concentrated\b|\bconcentration\b", PORTFOLIO_CONTEXT, 2.5),
    # Scope outranks the verb. "What should I trim across the book" names a
    # reduce action but asks a portfolio question, and answering it about one
    # name would answer a question that was not asked.
    (r"\bacross\s+the\s+book\b|\ball\s+of\s+my\s+positions\b"
     r"|\bacross\s+my\s+(portfolio|holdings)\b", PORTFOLIO_CONTEXT, 3.5),

    # ── Invalidation ──────────────────────────────────────────────────────
    (r"\binvalidat(e|es|ed|ion)\b", INVALIDATION, 3.0),
    (r"\bchange\s+(your|my)\s+mind\b", INVALIDATION, 3.0),
    (r"\bmake\s+(you|me)\s+wrong\b|\bprove\s+(you|me)\s+wrong\b", INVALIDATION, 3.0),
    (r"\bbreaks?\s+the\s+(bull|bear|base)?\s*(case|thesis)\b", INVALIDATION, 3.0),
    (r"\bwhat\s+am\s+i\s+missing\b|\bwhat\s+could\s+go\s+wrong\b", INVALIDATION, 2.6),

    # ── Options ───────────────────────────────────────────────────────────
    (r"\boptions?\b|\bcalls?\b|\bputs?\b|\bspread\b|\biron\s+condor\b"
     r"|\bstraddle\b|\bstrangle\b|\bbutterfly\b", OPTIONS_ANALYSIS, 2.6),
    (r"\bhedge\b|\bhedging\b", OPTIONS_ANALYSIS, 2.6),

    # ── Position actions ──────────────────────────────────────────────────
    (r"\baverage\s+down\b|\baveraging\s+down\b|\bdoubl(e|ing)\s+down\b",
     ADD_TO_POSITION, 3.2),
    (r"\b(should|shall|can)\s+i\s+add\b|\bshould\s+i\s+buy\s+more\b", ADD_TO_POSITION, 3.0),
    (r"\badd\s+(to|more)\b.{0,24}\b(position|holding|stake)\b", ADD_TO_POSITION, 3.0),
    (r"\badd\s+to\s+my\b|\btop(ping)?\s+up\b", ADD_TO_POSITION, 3.0),
    (r"\bbuy\s+more\s+(of\s+)?(it|them|this)\b", ADD_TO_POSITION, 2.8),
    (r"\badd\s+to\s+the\s+winner\b", ADD_TO_POSITION, 2.8),
    (r"\badd\s+more\s+[A-Z]{2,5}\b", ADD_TO_POSITION, 2.8),

    (r"\btrim\b|\bscale\s+(out|back)\b|\btake\s+(some\s+)?profit", REDUCE_POSITION, 3.0),
    (r"\breduce\s+(my|the)\s+(position|holding|stake|exposure)\b", REDUCE_POSITION, 3.0),
    (r"\bshould\s+i\s+reduce\b", REDUCE_POSITION, 3.0),
    (r"\bi'?m\s+overweight\b|\bover\s?weight\s+[A-Z]{2,5}\b", REDUCE_POSITION, 2.8),
    (r"\bsell\s+(some|half|part)\b", REDUCE_POSITION, 2.8),

    (r"\bshould\s+i\s+sell\s+(my|all|out)\b|\bsell\s+my\b", EXIT_POSITION, 3.0),
    (r"\bget\s+out\s+of\b|\btime\s+to\s+(get\s+out|exit|sell)\b", EXIT_POSITION, 3.0),
    (r"\bclose\s+(the|my)\s+position\b|\bexit\s+(the|my)\s+position\b", EXIT_POSITION, 3.0),
    (r"\bcut\s+(my\s+)?loss(es)?\b|\bdump\s+it\b", EXIT_POSITION, 3.0),

    # ── Holding, with no action named ─────────────────────────────────────
    (r"\bi\s+(own|hold)\b|\bi'?m\s+long\b|\bi\s+bought\b", EXISTING_POSITION, 2.0),
    (r"\bhow'?s\s+my\b|\bmy\s+[A-Z]{2,5}\b", EXISTING_POSITION, 2.0),

    # ── Earnings and events ───────────────────────────────────────────────
    (r"\bearnings?\s+(preview|report|call)\b|\bbefore\s+earnings\b"
     r"|\binto\s+earnings\b|\bthrough\s+earnings\b", EARNINGS_PREVIEW, 3.0),
    (r"\bearnings?\b", EARNINGS_PREVIEW, 2.2),
    (r"\bcatalysts?\b|\bevents?\s+(that\s+)?could\s+move\b"
     r"|\bupcoming\s+events?\b", EVENT_ANALYSIS, 2.6),

    # ── Risk ──────────────────────────────────────────────────────────────
    (r"\bbiggest\s+risks?\b|\bhow\s+risky\b|\brisk(s|y)?\s+(over|in|for)\b",
     RISK_ANALYSIS, 2.8),
    (r"\bmy\s+downside\b|\bhow\s+much\s+could\s+i\s+lose\b"
     r"|\bworst\s+case\b", RISK_ANALYSIS, 2.8),

    # ── Comparison ────────────────────────────────────────────────────────
    (r"\bcompare\b|\bversus\b|\bvs\.?\b", COMPARISON, 2.8),
    (r"\bwhich\s+is\s+better\b|\bwhich\s+(one|should\s+i)\b.{0,20}\bor\b",
     COMPARISON, 2.6),
    (r"\b[A-Z]{2,5}\s+or\s+[A-Z]{2,5}\b", COMPARISON, 2.6),

    # ── Horizon ───────────────────────────────────────────────────────────
    (r"\bnext\s+(few\s+)?(days?|week)\b|\bthis\s+week\b|\bswing\s+trade\b"
     r"|\bshort[\s-]term\b", SHORT_TERM_SETUP, 2.6),
    (r"\bnext\s+(couple|few)\s+of?\s*months?\b|\bnext\s+(3|three|6|six)\s+months?\b"
     r"|\bmedium[\s-]term\b", MEDIUM_TERM_SETUP, 2.6),
    (r"\blong[\s-]term\b|\bnext\s+(few\s+)?years?\b|\bdecade\b"
     r"|\bfor\s+(the\s+)?years?\b", LONG_TERM_THESIS, 2.6),

    # ── New entry ─────────────────────────────────────────────────────────
    (r"\bshould\s+i\s+buy\b(?!\s+more\b)", NEW_ENTRY, 2.4),
    (r"\bgood\s+entry\b|\bstart\s+a\s+position\b|\bopen(ing)?\s+a\s+position\b",
     NEW_ENTRY, 2.6),
    (r"\bat\s+what\s+price\s+would\b|\bwhat\s+would\s+make\s+.{0,24}\battractive\b",
     NEW_ENTRY, 2.6),
    (r"\bworth\s+buying\b|\bis\s+it\s+a\s+buy\b", NEW_ENTRY, 2.4),

    # ── General ───────────────────────────────────────────────────────────
    (r"\btell\s+me\s+about\b|\bwhat\s+do\s+you\s+think\b|\bthoughts?\b"
     r"|\bhow\s+does\s+\w+\s+look\b|\boverview\b", GENERAL_RESEARCH, 1.4),
]


def classify(text: str, context_symbol: Optional[str] = None,
             has_symbol: bool = True, n_symbols: int = 0) -> Dict[str, Any]:
    """The single research intent this utterance expresses.

    `n_symbols` settles the one ambiguity patterns cannot: a horizon phrase
    beside two named securities is a comparison over that horizon, not a
    thesis about one of them.
    """
    raw = (text or "").strip()
    out: Dict[str, Any] = {
        "text": raw, "intent": NONE, "score": 0.0, "scores": {},
        "is_position_question": False, "matched": [], "why": "",
    }
    if not raw:
        out["why"] = "nothing was asked"
        return out
    n_symbols = max(int(n_symbols or 0), 1 if has_symbol else 0)

    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}
    for pat, intent, w in PATTERNS:
        if re.search(pat, raw, re.I):
            scores[intent] = max(scores.get(intent, 0.0), w)
            matched.setdefault(intent, []).append(pat)

    if not scores:
        # A bare symbol, or a sentence naming one with no stated intent, is a
        # request for the general read. Without a symbol it is nothing.
        if has_symbol or context_symbol:
            out.update({"intent": GENERAL_RESEARCH, "score": 1.0,
                        "why": "a name with no more specific question"})
        else:
            out["why"] = "no intent and no symbol"
        return out

    # A horizon phrase is a MODIFIER of a comparison, not a rival to it.
    # "AAPL or MSFT for the next year" is a comparison with a horizon, and
    # letting the horizon win answers about one name over a year instead of
    # comparing two. Two named symbols is what settles it.
    if COMPARISON in scores and n_symbols >= 2:
        for horizon in (LONG_TERM_THESIS, MEDIUM_TERM_SETUP, SHORT_TERM_SETUP):
            if horizon in scores:
                scores[COMPARISON] = max(scores[COMPARISON], scores[horizon] + 0.4)

    best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
    intent, score = best
    if score < SELECT_THRESHOLD:
        intent = GENERAL_RESEARCH if (has_symbol or context_symbol) else NONE

    out.update({
        "intent": intent, "score": round(score, 2),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "matched": matched.get(intent, []),
        # Ownership is NOT derivable from intent alone. "How do I hedge my
        # NVDA position" is an options question about something held, and
        # "should I hold through earnings" is an earnings question about
        # something held. Both need the position context; neither has a
        # position-management intent. So the utterance is consulted too.
        "is_position_question": (intent in POSITION_INTENTS
                                 or bool(_stated_position(raw))),
        "why": f"matched {len(matched.get(intent, []))} pattern(s) for {intent}",
    })
    return out
