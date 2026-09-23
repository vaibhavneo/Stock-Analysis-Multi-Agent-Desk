"""Getting the symbol out of a sentence, without inventing one.

THE FAILURE THAT MATTERS
------------------------
Resolving the WRONG symbol is far worse than resolving none. "Show me ALL of
my positions" must not analyse Allstate; "the IT department" must not analyse
Gartner; "A good entry" must not analyse Agilent. Every one of those is a real
US ticker that is also an ordinary English word, and a confident analysis of
the wrong company is indistinguishable from a correct one until the user acts
on it.

So ambiguous tickers — the ones that collide with English — only resolve when
something in the sentence disambiguates them: a `$` prefix, a company-name
cue, or the utterance being nothing but the symbol. Everything else is left
unresolved, and an unresolved symbol is a question to ask, not a gap to fill.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..asset_class import CRYPTO_QUOTE_CURRENCIES

# Spoken names -> symbol. Names, not tickers: people say "apple", not "AAPL".
NAMES: Dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "tesla": "TSLA",
    "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
    "facebook": "META", "netflix": "NFLX", "intel": "INTC", "amd": "AMD",
    "coinbase": "COIN", "palantir": "PLTR", "broadcom": "AVGO",
    "salesforce": "CRM", "oracle": "ORCL", "adobe": "ADBE", "cisco": "CSCO",
    "qualcomm": "QCOM", "micron": "MU", "uber": "UBER", "airbnb": "ABNB",
    "shopify": "SHOP", "paypal": "PYPL", "disney": "DIS", "walmart": "WMT",
    "costco": "COST", "berkshire": "BRK-B", "jpmorgan": "JPM",
    "goldman": "GS", "boeing": "BA", "ford": "F", "starbucks": "SBUX",
    "pfizer": "PFE", "moderna": "MRNA", "exxon": "XOM", "chevron": "CVX",
    "arm": "ARM", "snowflake": "SNOW", "datadog": "DDOG", "crowdstrike": "CRWD",
    "supermicro": "SMCI", "lilly": "LLY", "visa": "V", "mastercard": "MA",
    # Crypto
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "ether": "ETH-USD", "eth": "ETH-USD",
    "solana": "SOL-USD", "sol": "SOL-USD",
    "dogecoin": "DOGE-USD", "doge": "DOGE-USD",
    "cardano": "ADA-USD", "ripple": "XRP-USD", "xrp": "XRP-USD",
    # ETFs, which people type in lower case as often as not. "spy" is also an
    # English verb, but on a stock desk the fund is overwhelmingly the meaning;
    # the rest are not words at all.
    "spy": "SPY", "qqq": "QQQ", "iwm": "IWM", "dia": "DIA", "voo": "VOO",
    "vti": "VTI", "arkk": "ARKK", "smh": "SMH", "xlk": "XLK", "xle": "XLE",
    "xlf": "XLF", "gld": "GLD", "slv": "SLV", "tlt": "TLT",
    # Indices, FX, commodities
    "gold": "GC=F", "silver": "SI=F", "oil": "CL=F", "crude": "CL=F",
    "nasdaq": "^IXIC", "dow": "^DJI", "vix": "^VIX", "russell": "^RUT",
}

# Multi-word names, matched before single tokens.
PHRASE_NAMES: List[Tuple[str, str]] = [
    (r"\bs\s*&\s*p\s*500\b", "^GSPC"), (r"\bs\s*&\s*p\b", "^GSPC"),
    (r"\bsp\s*500\b", "^GSPC"), (r"\bs\s*and\s*p\b", "^GSPC"),
    (r"\beuro\s*[/-]?\s*dollar\b", "EURUSD=X"), (r"\beur\s*[/-]\s*usd\b", "EURUSD=X"),
    (r"\beurusd\b", "EURUSD=X"), (r"\bdollar\s*yen\b", "USDJPY=X"),
    (r"\bon\s+semi(conductor)?\b", "ON"),
    (r"\bberkshire\s+hathaway\b", "BRK-B"),
]

# Tickers that are also ordinary English. These NEVER resolve from a bare
# lowercase word; they need a $ prefix, a name cue, or a solo utterance.
AMBIGUOUS = frozenset({
    "A", "ALL", "AN", "AND", "ARE", "AS", "AT", "BE", "BIG", "BUY", "BY",
    "CAN", "CAR", "CASH", "DD", "DO", "EAT", "FAST", "FOR", "GO", "GOOD",
    "HAS", "HE", "HOLD", "HUGE", "IF", "IN", "IS", "IT", "ITS", "KEY", "LOW",
    "ME", "MY", "NEW", "NICE", "NO", "NOW", "OF", "ON", "ONE", "OPEN", "OR",
    "OUT", "PLAY", "REAL", "RUN", "SAFE", "SEE", "SO", "SURE", "TELL", "THE",
    "TO", "UP", "US", "VERY", "WELL", "WHY", "YOU",
})

def _known_tickers() -> frozenset:
    """Symbols that may be written in lower case.

    People type "nvda vs amd", not "NVDA vs AMD", and requiring capitals threw
    away the symbol — which then blocked the capability that had correctly
    fired, so a routing success looked like a routing failure. Sourced from
    `watchlist.txt`, the universe this desk already curates, so the lexicon
    cannot drift away from what the engine actually covers. Ambiguous words are
    excluded even when they are in the file.
    """
    import os
    out = set(NAMES.values())
    path = os.path.join(os.path.dirname(__file__), "..", "..", "watchlist.txt")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip().upper()
                if line and not line.startswith("#"):
                    out.add(line.split()[0])
    except OSError:
        pass
    return frozenset(t for t in out if t not in AMBIGUOUS)


KNOWN_TICKERS = _known_tickers()

# Single letters are excluded from the BARE-CAPITALS scan for the same reason
# they are excluded from the lower-case one. "Should I add to my IONQ
# position?" resolved ['I', 'IONQ'], and "I own 44 shares at $197.80. Should I
# add?" resolved ONLY ['I'] — the pronoun, read as a ticker, analysing a
# company nobody named. AMBIGUOUS listed IF/IN/IS/IT/ITS but not I, and the
# guard only ran on the lower-case path. A one-letter ticker still resolves
# when it is written with a $ or is the entire message.
_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{2,5})(?![A-Za-z0-9])")
_SOLO_OR_DOLLAR_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Z])(?![A-Za-z0-9])")
_SHAPED_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    r"\^[A-Z]{1,6}"                       # ^GSPC
    r"|[A-Z]{2,6}=[XF]"                   # EURUSD=X, GC=F
    r"|[A-Z0-9]{2,6}-(?:" + "|".join(sorted(CRYPTO_QUOTE_CURRENCIES)) + r")"
    r"|[A-Z]{1,5}-[A-Z]"                  # BRK-B
    r")(?![A-Za-z0-9])")
_DOLLAR_RE = re.compile(r"\$([A-Za-z]{1,5})(?![A-Za-z0-9])")


def extract(text: str, context_symbol: Optional[str] = None) -> Dict[str, Any]:
    """Symbols in the order they appear, plus how each was resolved.

    `context_symbol` is the previous turn's subject. It is used ONLY when the
    utterance resolves nothing of its own — "and the options?" is about
    whatever was just discussed, while "what about microsoft" is not.
    """
    raw = text or ""
    found: List[str] = []
    how: Dict[str, str] = {}

    def add(sym: str, basis: str):
        sym = sym.upper()
        if sym not in found:
            found.append(sym)
            how[sym] = basis

    # 1. Shapes are unambiguous — BTC-USD, ^GSPC, GC=F, BRK-B.
    for m in _SHAPED_RE.finditer(raw.upper()):
        add(m.group(1), "symbol shape")

    # 2. $-prefixed is an explicit assertion by the user; it beats every guard.
    for m in _DOLLAR_RE.finditer(raw):
        add(m.group(1), "written with a $ prefix")

    # 3. Multi-word names before single tokens, so "on semiconductor" wins
    #    over the word "on".
    low = raw.lower()
    for pat, sym in PHRASE_NAMES:
        if re.search(pat, low):
            add(sym, "company or market name")

    # 4. Spoken names.
    for word in re.findall(r"[a-z][a-z&.\-]*", low):
        if word in NAMES:
            add(NAMES[word], "company or market name")

    # 5. Known tickers written in lower case. Restricted to the curated
    #    universe: accepting any 1-5 letter word as a ticker would make every
    #    sentence a minefield.
    # \b-anchored, and never single-letter. Unanchored, this regex chopped
    # "report" into "repor" + "t" and resolved T (AT&T) out of the middle of a
    # word — a wrong company, silently, from a sentence that named none.
    # Single letters are excluded outright: the lower-case "a", "t", "f" and
    # "v" in ordinary prose are never the tickers that share their spelling.
    for word in re.findall(r"\b[a-z]{2,5}\b", low):
        up = word.upper()
        if up in KNOWN_TICKERS and up not in AMBIGUOUS:
            add(up, "a known symbol, written in lower case")

    # 6. Bare upper-case tokens. Ambiguous ones are refused here.
    upper_tokens = _TICKER_RE.findall(raw)
    solo = raw.strip().upper().lstrip("$")
    # A single-letter ticker is only ever meant when it stands alone or is
    # written with a $. Anything else is a pronoun or an article.
    if len(solo) == 1 and solo.isalpha():
        upper_tokens = [solo] + upper_tokens
    upper_tokens += _SOLO_OR_DOLLAR_RE.findall(raw)
    for tok in upper_tokens:
        if tok in AMBIGUOUS and solo != tok:
            continue
        if not re.search(rf"(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])", raw):
            continue
        add(tok, "written in capitals")

    if found:
        return {"symbols": found, "how": how, "from_context": False,
                "context_symbol": context_symbol}

    if context_symbol:
        return {"symbols": [context_symbol.upper()],
                "how": {context_symbol.upper(): "carried from the previous turn"},
                "from_context": True, "context_symbol": context_symbol}

    return {"symbols": [], "how": {}, "from_context": False,
            "context_symbol": context_symbol}
