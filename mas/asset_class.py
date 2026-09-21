"""
Asset class as a first-class concept.

WHY THIS EXISTS
---------------
Every analytic in this repo was written for a US equity and quietly assumes
one. Three assumptions in particular are wrong for anything else:

1. **The calendar.** `sqrt(252)` annualization appears in
   `tools/market_data.py`, `xsection/features.py`, `decision/catalysts.py` and
   nine other places. Bitcoin trades 365 days a year. Annualizing BTC's daily
   vol on a 252-day calendar understates it by a factor of
   `sqrt(365/252)` = 1.204 — every volatility number comes out ~20% too low,
   and that error propagates into option prices, position sizes and scenario
   bands, all of which look perfectly reasonable.

2. **The pillars.** `backtest/pillars.py::_pillar` substitutes `50.0` for a
   missing score, and the composite is a *weighted sum* over
   `CORE_WEIGHTS = {technical: .40, algo: .40, fundamentals: .20}`. A coin has
   no income statement, so the fundamentals pillar contributes a hard
   `50.0 * 0.20 = 10` points of fabricated middle to every crypto composite.
   A flag on the pillar does not undo its weight.

3. **The instruments.** Listed US options exist for equities, ETFs and index
   products. They do not exist, in any venue this system can see, for spot
   crypto or FX. Offering a "bull call spread on BTC" priced off a model as if
   it were a quotable structure would be a fabrication.

So classification is not a routing convenience — it is the correctness
boundary. A symbol whose class is unknown is answered as UNKNOWN, never as a
default equity, because the failure mode of guessing is silent and confident.

NO I/O
------
`classify()` is a pure function of the symbol's shape plus whatever metadata
the caller already has. It never fetches. Callers that hold a provider payload
can pass `metadata=` to sharpen EQUITY vs ETF; callers that do not still get a
correct calendar and capability mask, which is what the math depends on.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

# ── The taxonomy ──────────────────────────────────────────────────────────
EQUITY = "EQUITY"
ETF = "ETF"
INDEX = "INDEX"
CRYPTO = "CRYPTO"
FX = "FX"
COMMODITY = "COMMODITY"
UNKNOWN = "UNKNOWN"

ALL_CLASSES = (EQUITY, ETF, INDEX, CRYPTO, FX, COMMODITY, UNKNOWN)

# Options availability, as an honest three-way rather than a boolean.
LISTED = "LISTED"                # exchange-listed, quotable, chain fetchable
NO_ACCESSIBLE_VENUE = "NO_ACCESSIBLE_VENUE"   # may trade somewhere; not here
NONE_EXIST = "NONE_EXIST"


@dataclass(frozen=True)
class ClassSpec:
    """What is true about an asset class, stated once so no analytic has to
    re-derive it (or forget to)."""
    asset_class: str
    label: str
    days_per_year: int
    sessions: str
    # Capability mask — which inputs genuinely exist for this class.
    has_fundamentals: bool
    has_earnings: bool
    has_sec_filings: bool
    has_short_interest: bool
    has_analyst_ratings: bool
    has_equity_beta: bool
    has_retail_sentiment: bool
    options: str
    # Annualized-volatility bands, in PERCENT, that define this class's
    # regimes. Equity's 40/20 thresholds are not universal: BTC sits near 50%
    # in an ordinary month, so on the equity bands every coin is permanently
    # HIGH — which fires `vol_expanding` risk vetoes and clamps every crypto
    # composite into the HOLD band by construction. FX sits near 8%, so on the
    # same bands no currency is ever anything but LOW.
    vol_high_pct: float = 40.0
    vol_medium_pct: float = 20.0
    # Pillars whose inputs cannot exist for this class. The composite must
    # renormalize over the rest rather than scoring these a neutral 50.
    inapplicable_pillars: tuple = field(default_factory=tuple)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["inapplicable_pillars"] = list(self.inapplicable_pillars)
        return d


SPECS: Dict[str, ClassSpec] = {
    EQUITY: ClassSpec(
        asset_class=EQUITY, label="listed equity",
        days_per_year=252, sessions="weekday",
        has_fundamentals=True, has_earnings=True, has_sec_filings=True,
        has_short_interest=True, has_analyst_ratings=True,
        has_equity_beta=True, has_retail_sentiment=True,
        options=LISTED,
        vol_high_pct=40.0, vol_medium_pct=20.0,
        note="The class every analytic in this repo was written for.",
    ),
    ETF: ClassSpec(
        asset_class=ETF, label="exchange-traded fund",
        days_per_year=252, sessions="weekday",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=True, has_analyst_ratings=False,
        has_equity_beta=True, has_retail_sentiment=True,
        options=LISTED,
        vol_high_pct=32.0, vol_medium_pct=16.0,
        inapplicable_pillars=("fundamentals", "research"),
        note=("A fund has holdings, not earnings. EDGAR companyfacts carries "
              "no income statement for it, so the fundamentals pillar has no "
              "input rather than a weak one."),
    ),
    INDEX: ClassSpec(
        asset_class=INDEX, label="index",
        days_per_year=252, sessions="weekday",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=False, has_analyst_ratings=False,
        has_equity_beta=False, has_retail_sentiment=False,
        options=LISTED,
        vol_high_pct=30.0, vol_medium_pct=15.0,
        inapplicable_pillars=("fundamentals", "research", "social"),
        note=("Not directly holdable. Levels and volatility are real; "
              "position advice on the index itself is not."),
    ),
    CRYPTO: ClassSpec(
        asset_class=CRYPTO, label="crypto asset",
        days_per_year=365, sessions="24/7",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=False, has_analyst_ratings=False,
        has_equity_beta=False, has_retail_sentiment=True,
        options=NO_ACCESSIBLE_VENUE,
        vol_high_pct=85.0, vol_medium_pct=45.0,
        inapplicable_pillars=("fundamentals", "research"),
        note=("Trades continuously, so the 365-day calendar is the correct "
              "annualization. Crypto options trade on venues this system "
              "does not read, so any structure priced here is MODEL-priced."),
    ),
    FX: ClassSpec(
        asset_class=FX, label="currency pair",
        days_per_year=260, sessions="24/5",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=False, has_analyst_ratings=False,
        has_equity_beta=False, has_retail_sentiment=False,
        options=NO_ACCESSIBLE_VENUE,
        vol_high_pct=15.0, vol_medium_pct=8.0,
        inapplicable_pillars=("fundamentals", "research", "social"),
        note=("A pair is a relative price, not an asset with a balance "
              "sheet. Its drift is dominated by the rate differential, "
              "which this system does not model — so directional calls on "
              "FX carry a standing caveat."),
    ),
    COMMODITY: ClassSpec(
        asset_class=COMMODITY, label="commodity future",
        days_per_year=252, sessions="weekday",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=False, has_analyst_ratings=False,
        has_equity_beta=False, has_retail_sentiment=False,
        options=LISTED,
        vol_high_pct=45.0, vol_medium_pct=22.0,
        inapplicable_pillars=("fundamentals", "research", "social"),
        note=("A continuous front-month series splices contracts; roll "
              "effects are in the returns and are not separated here."),
    ),
    UNKNOWN: ClassSpec(
        asset_class=UNKNOWN, label="unrecognized symbol",
        days_per_year=252, sessions="unknown",
        has_fundamentals=False, has_earnings=False, has_sec_filings=False,
        has_short_interest=False, has_analyst_ratings=False,
        has_equity_beta=False, has_retail_sentiment=False,
        options=NONE_EXIST,
        vol_high_pct=40.0, vol_medium_pct=20.0,
        inapplicable_pillars=("fundamentals", "research", "social"),
        note=("Deliberately barren. An unclassified symbol must not inherit "
              "equity capabilities by default — that is the silent failure "
              "this module exists to prevent."),
    ),
}


# ── Shape rules ───────────────────────────────────────────────────────────
# Quote currencies that appear on the right of a crypto pair. A US-listed
# equity never carries one of these as a share-class suffix (those are single
# letters: BRK-B, BF-B), so the rule does not collide with them.
CRYPTO_QUOTE_CURRENCIES = frozenset({
    "USD", "USDT", "USDC", "BUSD", "DAI", "EUR", "GBP", "JPY", "CAD",
    "AUD", "CHF", "KRW", "INR", "BRL", "BTC", "ETH",
})

_FIAT = frozenset({
    "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "CNY", "CNH",
    "HKD", "SGD", "INR", "KRW", "MXN", "BRL", "ZAR", "SEK", "NOK", "DKK",
    "PLN", "TRY", "RUB", "THB", "TWD", "ILS",
})


def normalize(symbol: str) -> str:
    """Upper-case and strip. Deliberately does not rewrite separators — a
    symbol's punctuation carries its namespace and changing it would erase
    exactly the signal this module reads."""
    return (symbol or "").strip().upper()


def _shape_class(sym: str) -> tuple[Optional[str], str]:
    """Classify from the symbol's shape alone. Returns (class|None, reason)."""
    if not sym:
        return UNKNOWN, "the symbol was empty"

    if sym.startswith("^"):
        return INDEX, f"'{sym}' carries the '^' index prefix"

    if sym.endswith("=F"):
        return COMMODITY, f"'{sym}' carries the '=F' futures suffix"

    if sym.endswith("=X"):
        base = sym[:-2]
        # EURUSD=X (pair) and EUR=X (implied USD cross) are both FX.
        if len(base) in (3, 6) or (base[:3] in _FIAT):
            return FX, f"'{sym}' carries the '=X' currency suffix"
        return FX, f"'{sym}' carries the '=X' currency suffix"

    if "-" in sym:
        base, _, quote = sym.rpartition("-")
        if base and quote in CRYPTO_QUOTE_CURRENCIES:
            return CRYPTO, (f"'{sym}' is a PAIR quoted in {quote}, which is a "
                            f"crypto quote currency, not an equity share class")
        # BRK-B, BF-B: a single-letter suffix is a share class.
        if base and len(quote) == 1 and quote.isalpha():
            return EQUITY, f"'{sym}' has a single-letter share-class suffix"

    return None, ""


# yfinance / provider `quoteType` values, mapped to this taxonomy.
_QUOTE_TYPE = {
    "EQUITY": EQUITY, "ETF": ETF, "MUTUALFUND": ETF, "INDEX": INDEX,
    "CRYPTOCURRENCY": CRYPTO, "CURRENCY": FX, "FUTURE": COMMODITY,
}


def classify(symbol: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify a symbol. Pure — never fetches.

    `metadata` is an already-fetched provider payload (e.g. the dict from
    `tools.market_data.fetch_fundamentals`). It is consulted ONLY to separate
    EQUITY from ETF, which the symbol's shape cannot distinguish. Shape rules
    win over metadata everywhere else: a provider that labels `BTC-USD` an
    equity is wrong, and the shape is not.

    Returns a dict rather than a ClassSpec so it can ride in JSON responses
    and be journalled alongside the decision it shaped.
    """
    sym = normalize(symbol)
    shaped, reason = _shape_class(sym)

    if shaped is not None and shaped is not UNKNOWN:
        cls, basis = shaped, "SYMBOL_SHAPE"
    elif shaped is UNKNOWN:
        cls, basis = UNKNOWN, "SYMBOL_SHAPE"
    else:
        # Shape is silent: a bare alphabetic ticker. Could be equity or ETF.
        cls, basis = None, "METADATA"
        if metadata:
            qt = str(metadata.get("quoteType") or "").upper()
            if qt in _QUOTE_TYPE:
                cls = _QUOTE_TYPE[qt]
                reason = f"the data provider reports quoteType={qt}"
            elif metadata.get("trailingEps") is not None or metadata.get("sector"):
                cls = EQUITY
                reason = ("the provider returned issuer fundamentals "
                          "(sector/EPS), which only an operating company has")
            elif metadata.get("fundFamily") or metadata.get("navPrice") is not None:
                cls = ETF
                reason = "the provider returned fund fields (fundFamily/NAV)"
        if cls is None:
            # No metadata to separate them. A bare ticker on a US venue is
            # overwhelmingly an equity or an ETF, and both share the 252-day
            # calendar — which is the part the math depends on. Say so, and
            # mark the confidence, rather than pretending to certainty.
            cls, basis = EQUITY, "ASSUMED"
            reason = ("a bare alphabetic ticker with no metadata to separate "
                      "equity from ETF; both share the 252-day calendar, so "
                      "the volatility math is correct either way")

    spec = SPECS[cls]
    return {
        "symbol": sym,
        "asset_class": cls,
        "label": spec.label,
        "basis": basis,
        "reason": reason,
        "confident": basis in ("SYMBOL_SHAPE", "METADATA"),
        "spec": spec.to_dict(),
    }


def spec_for(symbol_or_class: str) -> ClassSpec:
    """The ClassSpec for a class name, or for a symbol classified by shape."""
    key = normalize(symbol_or_class)
    if key in SPECS:
        return SPECS[key]
    return SPECS[classify(key)["asset_class"]]


def vol_regime(hv_annualized_pct: Optional[float], symbol_or_class: str) -> Optional[str]:
    """LOW / MEDIUM / HIGH on this class's own bands.

    Returns None when there is no volatility number, so a caller cannot
    mistake "not measured" for "calm"."""
    if hv_annualized_pct is None:
        return None
    spec = spec_for(symbol_or_class)
    v = float(hv_annualized_pct)
    if v > spec.vol_high_pct:
        return "HIGH"
    if v > spec.vol_medium_pct:
        return "MEDIUM"
    return "LOW"


def days_per_year(symbol_or_class: str) -> int:
    """The annualization calendar. The single most consequential number in
    this module: using 252 for a 24/7 asset understates its volatility by
    ~20%, and every downstream risk number inherits the error."""
    return spec_for(symbol_or_class).days_per_year


def applicable_pillars(symbol_or_class: str, all_pillars) -> tuple:
    """The pillars whose inputs genuinely exist for this class."""
    spec = spec_for(symbol_or_class)
    return tuple(p for p in all_pillars if p not in spec.inapplicable_pillars)
