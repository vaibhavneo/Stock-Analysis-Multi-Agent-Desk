"""A held-out corpus of things people actually type, written BEFORE the router.

WHY THIS FILE COMES FIRST
-------------------------
A router graded on questions written from its own keyword lists cannot fail.
It will score 95% and tell you nothing, because every phrasing it was built to
catch is the only phrasing it was tested on — and the coverage gap, which is
the whole risk, is invisible by construction. "We improved routing" is then an
unfalsifiable claim.

So this corpus is written first and frozen. It is deliberately NOT derived
from `mas/converse/intent.py` — that file does not exist yet as this is
written. The phrasings come from how someone would speak to a desk: casual,
elliptical, sometimes without a ticker, sometimes without punctuation,
sometimes asking two things at once.

Whatever the first measured score is, it gets reported as the baseline. A low
number here is information; a high number from a self-written corpus is not.

FIELDS
------
    q         the utterance
    caps      capabilities that MUST be routed to (order irrelevant)
    symbols   symbols that must be extracted ([] = none expected)
    note      why this case is here, when it is not obvious
    context   prior turn's resolved symbol, for follow-up cases
"""
from __future__ import annotations

from typing import Any, Dict, List

# Capability names this corpus expects to exist. Kept here rather than
# imported so the corpus stays readable on its own.
RESEARCH = "equity_research"
OPTIONS = "option_structures"
BACKTEST = "strategy_backtest"
RECORD = "forward_record"
REGIME = "market_regime"
EVENTS = "event_calendar"
PORTFOLIO = "portfolio_review"
BENCHMARK = "benchmark_relation"

CASES: List[Dict[str, Any]] = [
    # ── Plain research, ticker given ──────────────────────────────────────
    {"q": "should i buy AAPL", "caps": [RESEARCH], "symbols": ["AAPL"]},
    {"q": "what do you think about NVDA right now", "caps": [RESEARCH], "symbols": ["NVDA"]},
    {"q": "TSLA", "caps": [RESEARCH], "symbols": ["TSLA"],
     "note": "a bare ticker is a complete question to a desk"},
    {"q": "$MSFT thoughts?", "caps": [RESEARCH], "symbols": ["MSFT"]},
    {"q": "is amd a good buy at these levels", "caps": [RESEARCH], "symbols": ["AMD"]},
    {"q": "give me the full picture on META", "caps": [RESEARCH], "symbols": ["META"]},
    {"q": "worth holding GOOGL or should i get out", "caps": [RESEARCH], "symbols": ["GOOGL"]},
    {"q": "talk me through amazon", "caps": [RESEARCH], "symbols": ["AMZN"],
     "note": "company name, not ticker"},
    {"q": "how does apple look", "caps": [RESEARCH], "symbols": ["AAPL"]},
    {"q": "i'm thinking of adding to my nvidia position", "caps": [RESEARCH], "symbols": ["NVDA"]},
    {"q": "what's your read on microsoft", "caps": [RESEARCH], "symbols": ["MSFT"]},
    {"q": "netflix — buy sell or hold", "caps": [RESEARCH], "symbols": ["NFLX"]},
    {"q": "analyse coinbase for me", "caps": [RESEARCH], "symbols": ["COIN"]},
    {"q": "should i be worried about intel", "caps": [RESEARCH], "symbols": ["INTC"]},
    {"q": "whats the verdict on palantir", "caps": [RESEARCH], "symbols": ["PLTR"],
     "note": "no apostrophe, common in typed chat"},

    # ── Non-equity asset classes ──────────────────────────────────────────
    {"q": "how's bitcoin looking", "caps": [RESEARCH], "symbols": ["BTC-USD"]},
    {"q": "should i buy ethereum", "caps": [RESEARCH], "symbols": ["ETH-USD"]},
    {"q": "BTC-USD", "caps": [RESEARCH], "symbols": ["BTC-USD"]},
    {"q": "what about gold", "caps": [RESEARCH], "symbols": ["GC=F"]},
    {"q": "euro dollar view", "caps": [RESEARCH], "symbols": ["EURUSD=X"]},
    {"q": "is the s&p overbought", "caps": [RESEARCH], "symbols": ["^GSPC"]},
    {"q": "thoughts on solana", "caps": [RESEARCH], "symbols": ["SOL-USD"]},
    {"q": "spy or qqq right now", "caps": [RESEARCH], "symbols": ["SPY", "QQQ"],
     "note": "two symbols, comparison"},

    # ── Options ───────────────────────────────────────────────────────────
    {"q": "show me call options on AAPL", "caps": [OPTIONS], "symbols": ["AAPL"]},
    {"q": "what puts should i buy on tesla", "caps": [OPTIONS], "symbols": ["TSLA"]},
    {"q": "NVDA call spread ideas", "caps": [OPTIONS], "symbols": ["NVDA"]},
    {"q": "how do i play meta with options", "caps": [OPTIONS], "symbols": ["META"]},
    {"q": "is there a way to get exposure to amd without buying shares",
     "caps": [OPTIONS], "symbols": ["AMD"],
     "note": "describes options without naming them"},
    {"q": "premium selling ideas on SPY", "caps": [OPTIONS], "symbols": ["SPY"]},
    {"q": "i want to hedge my apple position", "caps": [OPTIONS], "symbols": ["AAPL"],
     "note": "hedging is an options question"},
    {"q": "iron condor on QQQ", "caps": [OPTIONS], "symbols": ["QQQ"]},
    {"q": "what's the cheapest way to bet against intel", "caps": [OPTIONS], "symbols": ["INTC"]},
    {"q": "derivatives on bitcoin", "caps": [OPTIONS], "symbols": ["BTC-USD"]},
    {"q": "covered call on my MSFT", "caps": [OPTIONS], "symbols": ["MSFT"]},
    {"q": "strike and expiry for a bullish nvda trade", "caps": [OPTIONS], "symbols": ["NVDA"]},

    # ── Research AND options together ─────────────────────────────────────
    {"q": "full analysis on AAPL including options", "caps": [RESEARCH, OPTIONS],
     "symbols": ["AAPL"]},
    {"q": "should i buy nvda and if so how — shares or calls",
     "caps": [RESEARCH, OPTIONS], "symbols": ["NVDA"]},
    {"q": "tell me everything about tesla, stock and derivatives",
     "caps": [RESEARCH, OPTIONS], "symbols": ["TSLA"]},

    # ── Backtesting ───────────────────────────────────────────────────────
    {"q": "does this strategy actually make money", "caps": [BACKTEST], "symbols": []},
    {"q": "backtest AAPL", "caps": [BACKTEST], "symbols": ["AAPL"]},
    {"q": "has your approach beaten buy and hold on NVDA", "caps": [BACKTEST],
     "symbols": ["NVDA"]},
    {"q": "show me the historical performance of these signals", "caps": [BACKTEST],
     "symbols": []},
    {"q": "if i had followed this on msft for the last 5 years what would have happened",
     "caps": [BACKTEST], "symbols": ["MSFT"]},
    {"q": "prove it works", "caps": [BACKTEST], "symbols": [],
     "note": "the shortest way anyone asks for a backtest"},
    {"q": "what's the sharpe on this", "caps": [BACKTEST], "symbols": []},
    {"q": "run the position rules on AMD", "caps": [BACKTEST], "symbols": ["AMD"]},

    # ── Forward record / calibration ──────────────────────────────────────
    {"q": "how accurate have your calls been", "caps": [RECORD], "symbols": []},
    {"q": "what's your hit rate", "caps": [RECORD], "symbols": []},
    {"q": "have you been right lately", "caps": [RECORD], "symbols": []},
    {"q": "show me your track record", "caps": [RECORD], "symbols": []},
    {"q": "are your predictions calibrated", "caps": [RECORD], "symbols": []},
    {"q": "how did your last AAPL call turn out", "caps": [RECORD], "symbols": ["AAPL"]},
    {"q": "why should i trust this", "caps": [RECORD], "symbols": [],
     "note": "trust is the calibration question in plain words"},

    # ── Market regime ─────────────────────────────────────────────────────
    {"q": "how's the market overall", "caps": [REGIME], "symbols": []},
    {"q": "what's the current market regime", "caps": [REGIME], "symbols": []},
    {"q": "is this risk on or risk off", "caps": [REGIME], "symbols": []},
    {"q": "should i be in cash right now", "caps": [REGIME], "symbols": []},
    {"q": "what's the vix telling us", "caps": [REGIME], "symbols": []},
    {"q": "are we in a correction", "caps": [REGIME], "symbols": []},

    # ── Catalysts / events ────────────────────────────────────────────────
    {"q": "when does NVDA report earnings", "caps": [EVENTS], "symbols": ["NVDA"]},
    {"q": "any catalysts coming up for apple", "caps": [EVENTS], "symbols": ["AAPL"]},
    {"q": "is there anything on the calendar for TSLA", "caps": [EVENTS], "symbols": ["TSLA"]},
    {"q": "earnings date for microsoft", "caps": [EVENTS], "symbols": ["MSFT"]},
    {"q": "what events could move amd in the next month", "caps": [EVENTS], "symbols": ["AMD"]},

    # ── Portfolio ─────────────────────────────────────────────────────────
    {"q": "review my portfolio", "caps": [PORTFOLIO], "symbols": []},
    {"q": "how are my holdings doing", "caps": [PORTFOLIO], "symbols": []},
    {"q": "am i too concentrated", "caps": [PORTFOLIO], "symbols": []},
    {"q": "what should i trim", "caps": [PORTFOLIO], "symbols": []},
    {"q": "look at my positions and tell me what to do", "caps": [PORTFOLIO], "symbols": []},

    # ── Benchmark relation ────────────────────────────────────────────────
    {"q": "is ETH just following bitcoin", "caps": [BENCHMARK], "symbols": ["ETH-USD"]},
    {"q": "how correlated is NVDA to the market", "caps": [BENCHMARK], "symbols": ["NVDA"]},
    {"q": "what's AAPL's beta", "caps": [BENCHMARK], "symbols": ["AAPL"]},
    {"q": "is tesla moving on its own or with the index", "caps": [BENCHMARK],
     "symbols": ["TSLA"]},
    {"q": "am i just buying the s&p again if i buy QQQ", "caps": [BENCHMARK],
     "symbols": ["QQQ"]},

    # ── Follow-ups: the symbol lives in the conversation, not the sentence ─
    {"q": "what about microsoft", "caps": [RESEARCH], "symbols": ["MSFT"],
     "context": "AAPL"},
    {"q": "and the options?", "caps": [OPTIONS], "symbols": ["AAPL"], "context": "AAPL",
     "note": "no symbol in the text at all; it must come from context"},
    {"q": "when does it report", "caps": [EVENTS], "symbols": ["NVDA"], "context": "NVDA"},
    {"q": "backtest that", "caps": [BACKTEST], "symbols": ["TSLA"], "context": "TSLA"},
    {"q": "how about calls instead", "caps": [OPTIONS], "symbols": ["AMD"], "context": "AMD"},
    {"q": "is it correlated to the market", "caps": [BENCHMARK], "symbols": ["META"],
     "context": "META"},
    {"q": "same for google", "caps": [RESEARCH], "symbols": ["GOOGL"], "context": "AAPL"},

    # ── Ambiguous / unanswerable: must NOT fabricate a route ──────────────
    {"q": "hi", "caps": [], "symbols": [], "note": "a greeting is not a query"},
    {"q": "hello there", "caps": [], "symbols": []},
    {"q": "thanks", "caps": [], "symbols": []},
    {"q": "what can you do", "caps": [], "symbols": [],
     "note": "a capability question, answered by the roster, not by routing"},
    {"q": "will the market crash", "caps": [REGIME], "symbols": [],
     "note": "answerable as regime, but the reply must refuse the prediction"},
    {"q": "what's the best stock to buy", "caps": [], "symbols": [],
     "note": "no symbol and no universe — must ask, not guess"},
    {"q": "tell me what to do with my money", "caps": [], "symbols": [],
     "note": "must not be routed to anything"},
    {"q": "asdfgh", "caps": [], "symbols": []},
    {"q": "buy", "caps": [], "symbols": [], "note": "a verb with no object"},

    # ── Collisions the matcher must not fall for ──────────────────────────
    {"q": "is the call on AAPL earnings already priced in", "caps": [EVENTS, OPTIONS],
     "symbols": ["AAPL"],
     "note": "'call' here is genuinely ambiguous; both routes are defensible"},
    {"q": "i want a long term hold, not a trade", "caps": [], "symbols": [],
     "note": "'hold' is a verdict word but this is not a query about one"},
    {"q": "ALL of my positions", "caps": [PORTFOLIO], "symbols": [],
     "note": "ALL must not resolve as the ticker ALL (Allstate)"},
    {"q": "IT department stocks", "caps": [], "symbols": [],
     "note": "IT is a real ticker (Gartner) but here it is a word"},
    {"q": "ON semiconductor", "caps": [RESEARCH], "symbols": ["ON"],
     "note": "ON genuinely IS the ticker, disambiguated by the company name"},
    {"q": "are you sure about that", "caps": [], "symbols": [],
     "note": "'sure' must not match anything"},
    {"q": "A good entry for NVDA", "caps": [RESEARCH], "symbols": ["NVDA"],
     "note": "'A' is a ticker (Agilent) but here it is an article"},

    # ── Multi-symbol and comparison ───────────────────────────────────────
    {"q": "compare AAPL and MSFT", "caps": [RESEARCH], "symbols": ["AAPL", "MSFT"]},
    {"q": "nvda vs amd", "caps": [RESEARCH], "symbols": ["NVDA", "AMD"]},
    {"q": "which is better, google or meta", "caps": [RESEARCH], "symbols": ["GOOGL", "META"]},

    # ── Everything at once ────────────────────────────────────────────────
    {"q": "give me the works on NVDA — analysis, options, backtest and your record",
     "caps": [RESEARCH, OPTIONS, BACKTEST, RECORD], "symbols": ["NVDA"]},
    {"q": "AAPL analysis plus when it reports", "caps": [RESEARCH, EVENTS],
     "symbols": ["AAPL"]},
    {"q": "should i buy tesla and how has this strategy done historically",
     "caps": [RESEARCH, BACKTEST], "symbols": ["TSLA"]},
]


def size() -> int:
    return len(CASES)


def by_capability() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in CASES:
        for cap in (c["caps"] or ["<none>"]):
            out[cap] = out.get(cap, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
