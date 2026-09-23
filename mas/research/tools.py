"""What the orchestrator can actually reach, and what it cannot.

THE DEFECT THIS CLOSES
----------------------
`mas/registry.json` registered AGENTS. `financial_data/registry.json`
registered PROVIDERS. Nothing connected them, and every agent reached for
data directly — `from financial_data import get_bars_df` — so the planner had
no visibility into data acquisition and no way to choose. Six providers,
including SEC filings, FRED macro, CBOE volatility and the earnings calendar,
were invisible to planning. That is the "tool exists but the orchestrator does
not know it exists" class, and it is the reason the planner could not do
anything more interesting than pick an agent.

DISCOVERY, NOT ASSUMPTION
-------------------------
Availability is PROBED, not declared. A tool whose key is absent is
`UNAVAILABLE` with the variable named; a tool whose module will not import is
`ERROR` with the exception. The planner is then working from what is true of
this deployment rather than from what the repository contains — which is the
difference between planning and hoping.

The outcome vocabulary is shared with tool execution, because "I could not
get this" and "I did not ask for this" must never collapse into one state.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

# Execution outcomes. Every one of these is a different thing to tell a user.
SUCCESS = "SUCCESS"
EMPTY = "EMPTY"              # ran, returned nothing, and nothing is the answer
STALE = "STALE"              # returned data older than the decision needs
PARTIAL = "PARTIAL"          # some of what was asked for
RATE_LIMITED = "RATE_LIMITED"
TIMEOUT = "TIMEOUT"
UNAVAILABLE = "UNAVAILABLE"  # cannot be reached at all in this deployment
ERROR = "ERROR"              # raised; a bug, not a decline
NOT_REQUESTED = "NOT_REQUESTED"

OUTCOMES = (SUCCESS, EMPTY, STALE, PARTIAL, RATE_LIMITED, TIMEOUT,
            UNAVAILABLE, ERROR, NOT_REQUESTED)

# What a tool yields. The planner reasons over these, never over module names.
PRICE_CURRENT = "price_current"
PRICE_INTRADAY = "price_intraday"
PRICE_HISTORY = "price_history"
FUNDAMENTALS = "fundamentals"
FILINGS = "filings"
EARNINGS = "earnings"
ANALYST = "analyst_coverage"
NEWS = "news"
SOCIAL = "social_sentiment"
MACRO = "macro"
VOLATILITY_INDEX = "volatility_index"
OPTIONS_CHAIN = "options_chain"
EVENT_CALENDAR = "event_calendar"

# How current a tool's output can possibly be. A planner that does not know
# this cannot tell a settled close from a last trade, which is the bug that
# put a day-old price under the label PRICE NOW.
REALTIME = "REALTIME"          # last trade / live quote
INTRADAY = "INTRADAY"          # minute bars, within the session
SETTLED_DAILY = "SETTLED_DAILY"
FILED = "FILED"                # published on a filing date
SCHEDULED = "SCHEDULED"        # a calendar of future dates
VINTAGE = "VINTAGE"            # current vintage, revised in place


class Tool:
    """One reachable source of evidence."""

    def __init__(self, tool_id: str, name: str, produces: List[str],
                 freshness: str, requires_key: Optional[str] = None,
                 est_ms: int = 500, reliability: float = 0.7,
                 probe: Optional[Callable[[], Any]] = None,
                 asset_classes: Optional[List[str]] = None,
                 notes: str = "", deterministic: bool = True,
                 decision_use: bool = True):
        self.id = tool_id
        self.name = name
        self.produces = produces
        self.freshness = freshness
        self.requires_key = requires_key
        self.est_ms = est_ms
        self.reliability = reliability
        self._probe = probe
        self.asset_classes = asset_classes or ["EQUITY", "ETF", "INDEX",
                                               "CRYPTO", "FX", "COMMODITY"]
        self.notes = notes
        self.deterministic = deterministic
        self.decision_use = decision_use

    def availability(self) -> Dict[str, Any]:
        """Probed, not declared."""
        if self.requires_key and not os.environ.get(self.requires_key):
            return {"state": UNAVAILABLE, "reason":
                    f"{self.requires_key} is not set in this environment"}
        if self._probe is not None:
            try:
                res = self._probe()
                if res is False:
                    return {"state": UNAVAILABLE,
                            "reason": "the probe reported the source unreachable"}
            except Exception as e:
                return {"state": ERROR,
                        "reason": f"{type(e).__name__}: {str(e)[:140]}"}
        return {"state": SUCCESS, "reason": ""}

    def to_dict(self) -> Dict[str, Any]:
        av = self.availability()
        return {"id": self.id, "name": self.name, "produces": list(self.produces),
                "freshness": self.freshness, "requires_key": self.requires_key,
                "est_ms": self.est_ms, "reliability": self.reliability,
                "asset_classes": list(self.asset_classes), "notes": self.notes,
                "deterministic": self.deterministic,
                "decision_use": self.decision_use,
                "available": av["state"] == SUCCESS, "state": av["state"],
                "reason": av["reason"]}


def _probe_import(module: str) -> Callable[[], Any]:
    def _p():
        __import__(module)
        return True
    return _p


def _probe_provider(provider_id: str) -> Callable[[], Any]:
    """A provider is reachable if the registry lists it and its module loads."""
    def _p():
        from financial_data import gateway
        reg = gateway.load_registry()
        p = (reg.get("providers") or {}).get(provider_id)
        if not p:
            return False
        key = p.get("requires_key")
        if key and not os.environ.get(key):
            return False
        gateway._load_module(p)
        return True
    return _p


def _build() -> Dict[str, Tool]:
    t: List[Tool] = [
        Tool("quote_last_trade", "Last trade (yfinance fast_info)",
             [PRICE_CURRENT], REALTIME, est_ms=600, reliability=0.6,
             probe=_probe_import("yfinance"),
             notes="The only REALTIME source here. Never feeds a computation "
                   "calibrated on settled bars; it is what makes a stale "
                   "headline price detectable."),
        Tool("bars_intraday", "Intraday minute bars (yfinance)",
             [PRICE_INTRADAY], INTRADAY, est_ms=1200, reliability=0.55,
             probe=_probe_import("yfinance"),
             notes="Shows whether a session has traded that the settled "
                   "series has not yet published."),
        Tool("bars_daily", "Daily OHLCV via the FinancialDataGateway",
             [PRICE_HISTORY], SETTLED_DAILY, est_ms=700, reliability=0.85,
             probe=_probe_provider("yfinance"),
             notes="The series every indicator, pillar and backtest is "
                   "computed on. Provider choice is the gateway's, not the "
                   "planner's."),
        Tool("edgar_fundamentals", "SEC EDGAR XBRL companyfacts",
             [FUNDAMENTALS, FILINGS], FILED, est_ms=1800, reliability=1.0,
             probe=_probe_provider("sec-edgar"),
             asset_classes=["EQUITY"],
             notes="As-reported and keyed by FILING date, which is what makes "
                   "point-in-time fundamentals possible at all."),
        Tool("quote_fundamentals", "Issuer snapshot (yfinance info)",
             [FUNDAMENTALS, ANALYST], VINTAGE, est_ms=900, reliability=0.5,
             probe=_probe_import("yfinance"), asset_classes=["EQUITY", "ETF"],
             notes="RESTATED, not point-in-time. Carries analyst coverage and "
                   "the identity fields that separate an ETF from an issuer."),
        Tool("earnings_calendar", "Earnings dates and actuals",
             [EARNINGS, EVENT_CALENDAR], SCHEDULED, est_ms=900,
             reliability=0.7, probe=_probe_import("yfinance"),
             asset_classes=["EQUITY"],
             notes="Future dates are estimates that move; no probability is "
                   "attached to any of them."),
        Tool("finnhub_events", "Finnhub earnings calendar",
             [EARNINGS, EVENT_CALENDAR], SCHEDULED, requires_key="FINNHUB_API_KEY",
             est_ms=800, reliability=0.8, asset_classes=["EQUITY"],
             notes="Key-gated. Absent, the yfinance calendar is the fallback."),
        Tool("news", "Company news headlines", [NEWS], VINTAGE, est_ms=1100,
             reliability=0.4, probe=_probe_import("yfinance"),
             deterministic=False, decision_use=False,
             notes="Explanation only. Headline counts are not evidence of "
                   "direction and no decision field reads them."),
        Tool("reddit", "Reddit mention sentiment", [SOCIAL], VINTAGE,
             est_ms=2500, reliability=0.3, deterministic=False,
             asset_classes=["EQUITY", "ETF", "CRYPTO"],
             notes="Low reliability by construction: mention counts are a "
                   "popularity measure, and popularity is not direction."),
        Tool("stocktwits", "StockTwits bull/bear ratio", [SOCIAL], VINTAGE,
             est_ms=2000, reliability=0.3, deterministic=False,
             asset_classes=["EQUITY", "ETF", "CRYPTO"],
             notes="Self-reported sentiment from a self-selected population."),
        Tool("fred_macro", "FRED macro series", [MACRO], VINTAGE, est_ms=1400,
             reliability=0.9, probe=_probe_provider("fred"),
             notes="CURRENT VINTAGE ONLY — revised in place, so it is live "
                   "context and never a backtest input."),
        Tool("cboe_vix", "CBOE VIX history", [VOLATILITY_INDEX, MACRO],
             SETTLED_DAILY, est_ms=900, reliability=0.95,
             probe=_probe_provider("cboe"),
             notes="Published at close and not restated, so unlike the FRED "
                   "series this one is point-in-time honest."),
        Tool("optionspilot_chain", "OptionsPilot live option chain",
             [OPTIONS_CHAIN], REALTIME, requires_key="OPTIONSPILOT_ACCESS_CODE",
             est_ms=2500, reliability=0.9, asset_classes=["EQUITY", "ETF"],
             notes="A separate service over HTTP. Real quotes and a real IV "
                   "surface, over a bounded ~31-name universe."),
    ]
    return {x.id: x for x in t}


_TOOLS: Optional[Dict[str, Tool]] = None


def all_tools() -> Dict[str, Tool]:
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = _build()
    return _TOOLS


def discover(asset_class: Optional[str] = None) -> Dict[str, Any]:
    """What this deployment can actually reach, right now.

    Probing costs an import and an env lookup per tool — cheap, and the
    alternative is a planner that believes the repository instead of the
    machine it is running on.
    """
    rows = []
    for t in all_tools().values():
        if asset_class and asset_class not in t.asset_classes:
            d = t.to_dict()
            d.update({"available": False, "state": UNAVAILABLE,
                      "reason": f"this tool does not cover {asset_class}"})
            rows.append(d)
            continue
        rows.append(t.to_dict())
    available = [r for r in rows if r["available"]]
    produced = sorted({p for r in available for p in r["produces"]})
    missing = sorted({p for r in rows for p in r["produces"]} - set(produced))
    return {
        "asset_class": asset_class,
        "tools": sorted(rows, key=lambda r: (not r["available"], r["id"])),
        "n_available": len(available), "n_total": len(rows),
        "evidence_reachable": produced,
        "evidence_unreachable": missing,
        "statement": (
            f"{len(available)} of {len(rows)} tools are reachable"
            + (f" for {asset_class}" if asset_class else "")
            + (f"; no tool here can supply {', '.join(missing)}." if missing
               else "; every evidence kind has at least one source.")),
    }


def tools_for(evidence_kind: str, asset_class: Optional[str] = None,
              available_only: bool = True) -> List[Tool]:
    """Tools that produce this evidence, most reliable first."""
    out = []
    for t in all_tools().values():
        if evidence_kind not in t.produces:
            continue
        if asset_class and asset_class not in t.asset_classes:
            continue
        if available_only and t.availability()["state"] != SUCCESS:
            continue
        out.append(t)
    return sorted(out, key=lambda t: (-t.reliability, t.est_ms))
