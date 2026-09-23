"""Held-out corpus for RESEARCH INTENT — written before the classifier exists.

The capability router (`mas/converse/intent.py`) answers "which specialist".
This answers a different and prior question: "what KIND of decision is being
asked for". They are not the same, and conflating them is what produced the
baseline failure — "Should I add to my IONQ position?" and "Tell me about
IONQ" both routed to `equity_research` and got the same fixed computation,
when only the first one is a position-management question.

Same discipline as the routing corpus: written and frozen BEFORE
`mas/research/intent.py` exists, phrased the way people speak, deliberately
not derived from keyword lists that do not yet exist. Whatever the first
measured number is, it is reported as the baseline.

FIELDS
    q          the utterance
    intent     the single research intent it expresses
    position   True when the utterance implies the user HOLDS the thing
    ctx        prior-turn subject, for elliptical cases
"""
from __future__ import annotations

from typing import Any, Dict, List

GENERAL_RESEARCH = "GENERAL_RESEARCH"
NEW_ENTRY = "NEW_ENTRY"
EXISTING_POSITION = "EXISTING_POSITION"
ADD_TO_POSITION = "ADD_TO_POSITION"
REDUCE_POSITION = "REDUCE_POSITION"
EXIT_POSITION = "EXIT_POSITION"
SHORT_TERM_SETUP = "SHORT_TERM_SETUP"
MEDIUM_TERM_SETUP = "MEDIUM_TERM_SETUP"
LONG_TERM_THESIS = "LONG_TERM_THESIS"
EARNINGS_PREVIEW = "EARNINGS_PREVIEW"
EVENT_ANALYSIS = "EVENT_ANALYSIS"
RISK_ANALYSIS = "RISK_ANALYSIS"
COMPARISON = "COMPARISON"
PORTFOLIO_CONTEXT = "PORTFOLIO_CONTEXT"
OPTIONS_ANALYSIS = "OPTIONS_ANALYSIS"
INVALIDATION = "INVALIDATION"
NONE = "NONE"

CASES: List[Dict[str, Any]] = [
    # ── Adding to a position ──────────────────────────────────────────────
    {"q": "Should I add to my IONQ position?", "intent": ADD_TO_POSITION, "position": True},
    {"q": "should i add more NVDA", "intent": ADD_TO_POSITION, "position": True},
    {"q": "is it worth topping up my AAPL", "intent": ADD_TO_POSITION, "position": True},
    {"q": "I own 44 shares at $197.80. Should I add?", "intent": ADD_TO_POSITION,
     "position": True},
    {"q": "IONQ dropped 10%. Should I average down?", "intent": ADD_TO_POSITION,
     "position": True},
    {"q": "thinking of doubling down on TSLA", "intent": ADD_TO_POSITION, "position": True},
    {"q": "add to the winner or leave it", "intent": ADD_TO_POSITION, "position": True,
     "ctx": "NVDA"},
    {"q": "should i buy more of it", "intent": ADD_TO_POSITION, "position": True,
     "ctx": "AMD"},

    # ── Reducing ──────────────────────────────────────────────────────────
    {"q": "Should I reduce my position?", "intent": REDUCE_POSITION, "position": True},
    {"q": "should i trim NVDA", "intent": REDUCE_POSITION, "position": True},
    {"q": "take some profit off the table on AAPL?", "intent": REDUCE_POSITION,
     "position": True},
    {"q": "I'm overweight TSLA, what do I do", "intent": REDUCE_POSITION, "position": True},
    {"q": "scale out of half my position", "intent": REDUCE_POSITION, "position": True,
     "ctx": "META"},

    # ── Exiting ───────────────────────────────────────────────────────────
    {"q": "should i sell my NVDA", "intent": EXIT_POSITION, "position": True},
    {"q": "time to get out of AMD?", "intent": EXIT_POSITION, "position": True},
    {"q": "should i close the position", "intent": EXIT_POSITION, "position": True,
     "ctx": "INTC"},
    {"q": "cut my losses on PLTR?", "intent": EXIT_POSITION, "position": True},
    {"q": "dump it or hold", "intent": EXIT_POSITION, "position": True, "ctx": "COIN"},

    # ── Holding an existing position, no action named ─────────────────────
    {"q": "I own NVDA, what now", "intent": EXISTING_POSITION, "position": True},
    {"q": "I'm long AAPL from 180", "intent": EXISTING_POSITION, "position": True},
    {"q": "how's my TSLA doing", "intent": EXISTING_POSITION, "position": True},
    {"q": "I hold 200 shares of AMD", "intent": EXISTING_POSITION, "position": True},

    # ── New entry ─────────────────────────────────────────────────────────
    {"q": "Should I buy NVDA now?", "intent": NEW_ENTRY, "position": False},
    {"q": "is this a good entry on AAPL", "intent": NEW_ENTRY, "position": False},
    {"q": "where should i start a position in AMD", "intent": NEW_ENTRY, "position": False},
    {"q": "worth opening a position in META", "intent": NEW_ENTRY, "position": False},
    {"q": "What would make this stock attractive?", "intent": NEW_ENTRY, "position": False,
     "ctx": "NVDA"},
    {"q": "at what price would TSLA be a buy", "intent": NEW_ENTRY, "position": False},

    # ── General research ──────────────────────────────────────────────────
    {"q": "NVDA", "intent": GENERAL_RESEARCH, "position": False},
    {"q": "tell me about palantir", "intent": GENERAL_RESEARCH, "position": False},
    {"q": "what do you think of AMD", "intent": GENERAL_RESEARCH, "position": False},
    {"q": "how does apple look", "intent": GENERAL_RESEARCH, "position": False},

    # ── Horizon-specific ──────────────────────────────────────────────────
    {"q": "is NVDA a good trade for the next few days", "intent": SHORT_TERM_SETUP,
     "position": False},
    {"q": "swing trade setup on AMD this week", "intent": SHORT_TERM_SETUP, "position": False},
    {"q": "what's the setup over the next couple of months for AAPL",
     "intent": MEDIUM_TERM_SETUP, "position": False},
    {"q": "is TSLA a good hold for the next few years", "intent": LONG_TERM_THESIS,
     "position": False},
    {"q": "long term thesis on META", "intent": LONG_TERM_THESIS, "position": False},
    {"q": "is NVDA a decade holding", "intent": LONG_TERM_THESIS, "position": False},

    # ── Earnings and events ───────────────────────────────────────────────
    {"q": "What should I watch before earnings?", "intent": EARNINGS_PREVIEW,
     "position": False, "ctx": "NVDA"},
    {"q": "NVDA earnings preview", "intent": EARNINGS_PREVIEW, "position": False},
    {"q": "how does AAPL usually trade into earnings", "intent": EARNINGS_PREVIEW,
     "position": False},
    {"q": "should i hold through earnings", "intent": EARNINGS_PREVIEW, "position": True,
     "ctx": "AMD"},
    {"q": "what catalysts are coming for TSLA", "intent": EVENT_ANALYSIS, "position": False},
    {"q": "is there an event that could move META", "intent": EVENT_ANALYSIS,
     "position": False},

    # ── Risk ──────────────────────────────────────────────────────────────
    {"q": "What are the biggest risks over the next month?", "intent": RISK_ANALYSIS,
     "position": False, "ctx": "NVDA"},
    {"q": "how risky is AMD here", "intent": RISK_ANALYSIS, "position": False},
    {"q": "what's my downside on TSLA", "intent": RISK_ANALYSIS, "position": False},
    {"q": "how much could i lose", "intent": RISK_ANALYSIS, "position": False, "ctx": "AAPL"},

    # ── Invalidation — the one that was UNROUTABLE ────────────────────────
    {"q": "What could invalidate the thesis?", "intent": INVALIDATION, "position": False,
     "ctx": "NVDA"},
    {"q": "what would change your mind on AMD", "intent": INVALIDATION, "position": False},
    {"q": "what would make you wrong about TSLA", "intent": INVALIDATION, "position": False},
    {"q": "what breaks the bull case", "intent": INVALIDATION, "position": False,
     "ctx": "META"},
    {"q": "what am i missing here", "intent": INVALIDATION, "position": False, "ctx": "NVDA"},

    # ── Comparison ────────────────────────────────────────────────────────
    {"q": "Compare these two stocks.", "intent": COMPARISON, "position": False,
     "ctx": "NVDA"},
    {"q": "NVDA vs AMD", "intent": COMPARISON, "position": False},
    {"q": "which is better, google or meta", "intent": COMPARISON, "position": False},
    {"q": "AAPL or MSFT for the next year", "intent": COMPARISON, "position": False},

    # ── Portfolio ─────────────────────────────────────────────────────────
    {"q": "review my portfolio", "intent": PORTFOLIO_CONTEXT, "position": True},
    {"q": "am i too concentrated", "intent": PORTFOLIO_CONTEXT, "position": True},
    {"q": "how are my holdings doing", "intent": PORTFOLIO_CONTEXT, "position": True},
    {"q": "what should i trim across the book", "intent": PORTFOLIO_CONTEXT,
     "position": True},

    # ── Options ───────────────────────────────────────────────────────────
    {"q": "show me call options on AAPL", "intent": OPTIONS_ANALYSIS, "position": False},
    {"q": "how do i hedge my NVDA position", "intent": OPTIONS_ANALYSIS, "position": True},
    {"q": "iron condor on QQQ", "intent": OPTIONS_ANALYSIS, "position": False},
    {"q": "what puts should i buy on tesla", "intent": OPTIONS_ANALYSIS, "position": False},

    # ── Not a research question ───────────────────────────────────────────
    {"q": "hi", "intent": NONE, "position": False},
    {"q": "what can you do", "intent": NONE, "position": False},
    {"q": "thanks", "intent": NONE, "position": False},
    {"q": "what's the best stock to buy", "intent": NONE, "position": False},

    # ── Traps ─────────────────────────────────────────────────────────────
    {"q": "I want to add a watchlist", "intent": NONE, "position": False,
     "note": "'add' is not always a position action"},
    {"q": "should i buy more time before deciding", "intent": NONE, "position": False,
     "note": "'buy more' without a security"},
    {"q": "my position on this is that it's overvalued", "intent": GENERAL_RESEARCH,
     "position": False, "ctx": "NVDA",
     "note": "'my position' here is an opinion, not a holding"},
    {"q": "reduce the noise in this analysis", "intent": NONE, "position": False,
     "note": "'reduce' with no holding"},
]


def size() -> int:
    return len(CASES)


def by_intent() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in CASES:
        out[c["intent"]] = out.get(c["intent"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
