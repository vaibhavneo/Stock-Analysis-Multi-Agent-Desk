"""The desk's own analytical functions, registered as sub-agents.

WHY THESE MOVED INTO THE ROSTER
-------------------------------
Backtesting, calibration, market regime, the catalyst calendar and the
portfolio review all existed and all worked — each reachable from exactly one
fixed endpoint, which meant they were reachable only by a user who already
knew they existed and which button produced them.

That is the same "computed but never consulted" failure this codebase keeps
finding, one level up: the capability is not missing, the ROUTE to it is. A
conversational front door cannot offer "how accurate have you been" unless
something can be asked that question, so each of these is now an adapter
behind a named capability, and the planner reaches them the same way it
reaches OptionsPilot.

Every adapter follows `mas/contract.py`: it returns an AgentResult for every
outcome and never raises. A specialist that can raise past the executor can
take down a conversation, which is exactly the coupling the options work
removed last time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..asset_class import spec_for
from ..contract import AgentRequest, AgentResult, ok, unavailable, error


# ══════════════════════════════════════════════════════════════════════════
# Market regime — about the market, so it needs no symbol
# ══════════════════════════════════════════════════════════════════════════
class Regime:
    AGENT_ID = "regime"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        try:
            from intelligence.regime import compute_market_regime
            r = compute_market_regime()
            if not r:
                return unavailable(Regime.AGENT_ID, request.capability,
                                   "the regime model returned nothing")
            return ok(Regime.AGENT_ID, request.capability, r,
                      price_basis="PRICE_HISTORY_DERIVED")
        except Exception as e:
            return error(Regime.AGENT_ID, request.capability, e)


# ══════════════════════════════════════════════════════════════════════════
# Forward record — what this desk has actually got right, measured live
# ══════════════════════════════════════════════════════════════════════════
class TrackRecord:
    AGENT_ID = "track_record"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        try:
            from data import prediction_ledger as pl
            by_h = pl.calibration_report_all_horizons()
            if not by_h:
                return unavailable(
                    TrackRecord.AGENT_ID, request.capability,
                    "no frozen predictions have been graded yet, so there is "
                    "no live record to report")
            from decision.track_record import build_live_record
            live = build_live_record(by_h)
            return ok(TrackRecord.AGENT_ID, request.capability,
                      {"live_record": live, "by_horizon": by_h},
                      price_basis="MEASURED_LIVE")
        except Exception as e:
            return error(TrackRecord.AGENT_ID, request.capability, e)


# ══════════════════════════════════════════════════════════════════════════
# Catalysts
# ══════════════════════════════════════════════════════════════════════════
class Catalysts:
    AGENT_ID = "catalysts"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        spec = spec_for(asset_class)
        if not spec.has_earnings:
            return False, (f"a {spec.label} has no earnings calendar — there "
                           f"are no scheduled company events to list")
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        try:
            from decision.catalysts import build_catalyst_timeline
            t = build_catalyst_timeline(request.symbol,
                                        request.params.get("price"),
                                        request.params.get("atr"))
            if not t or t.get("status") == "UNAVAILABLE":
                return unavailable(
                    Catalysts.AGENT_ID, request.capability,
                    (t or {}).get("waiting_for")
                    or "no catalyst calendar was available for this symbol")
            return ok(Catalysts.AGENT_ID, request.capability, t)
        except Exception as e:
            return error(Catalysts.AGENT_ID, request.capability, e)


# ══════════════════════════════════════════════════════════════════════════
# Backtest — does following this actually beat holding the thing?
# ══════════════════════════════════════════════════════════════════════════
class Backtester:
    AGENT_ID = "backtester"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        symbol = request.symbol
        if not symbol:
            return unavailable(
                Backtester.AGENT_ID, request.capability,
                "a backtest needs a symbol to run on")
        try:
            from financial_data import get_bars_df
            df = get_bars_df(symbol, period=request.params.get("period", "5y"))
        except Exception as e:
            return unavailable(Backtester.AGENT_ID, request.capability,
                               f"no price history for {symbol} "
                               f"({type(e).__name__})")
        try:
            # Lazy, and from the route module on purpose: this is the SAME
            # racer the decision brief uses, so the conversational answer and
            # the brief cannot disagree about whether the strategy works.
            from web.app import _build_backtest_all
            res = _build_backtest_all(symbol, df)
            if not res:
                return unavailable(Backtester.AGENT_ID, request.capability,
                                   "the strategy race produced no result")
            return ok(Backtester.AGENT_ID, request.capability, res,
                      price_basis="PRICE_HISTORY_DERIVED")
        except Exception as e:
            return error(Backtester.AGENT_ID, request.capability, e)


# ══════════════════════════════════════════════════════════════════════════
# Portfolio — only answerable when the user has actually supplied holdings
# ══════════════════════════════════════════════════════════════════════════
class Portfolio:
    AGENT_ID = "portfolio"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        holdings = request.params.get("holdings") or []
        if not holdings:
            return unavailable(
                Portfolio.AGENT_ID, request.capability,
                "no holdings were supplied. Positions are never inferred — "
                "add them in My Portfolio and ask again, and nothing will be "
                "assumed about what you own.")
        try:
            from agents.portfolio_brief import build_portfolio_brief
            brief = build_portfolio_brief(holdings)
            return ok(Portfolio.AGENT_ID, request.capability, brief)
        except Exception as e:
            return error(Portfolio.AGENT_ID, request.capability, e)


# ══════════════════════════════════════════════════════════════════════════
# Research — the desk's own full read
# ══════════════════════════════════════════════════════════════════════════
class Research:
    AGENT_ID = "research"

    @staticmethod
    def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
        if spec_for(asset_class).asset_class == "UNKNOWN":
            return False, ("the symbol could not be classified, so none of the "
                           "analytics know which calendar or pillars apply")
        return True, ""

    @staticmethod
    def run(request: AgentRequest) -> AgentResult:
        if not request.symbol:
            return unavailable(Research.AGENT_ID, request.capability,
                               "research needs a symbol")
        try:
            if request.params.get("full"):
                # The complete Decision Intelligence object. Lazy import: the
                # builder lives in the route module, and mas/ must not import
                # web/ at module load or the cycle closes.
                from web.app import _build_decision_intelligence
                d = _build_decision_intelligence(
                    request.symbol,
                    period=request.params.get("period", "5y"))
                return ok(Research.AGENT_ID, request.capability,
                          {"decision": d, "depth": "FULL"})
            from ..core import core_read
            r = core_read(request.symbol,
                          period=request.params.get("period", "1y"))
            if r.get("status") != "OK":
                return unavailable(Research.AGENT_ID, request.capability,
                                   r.get("reason") or "no read was produced")
            return ok(Research.AGENT_ID, request.capability,
                      {"core": r, "depth": "FAST"},
                      price_basis="PRICE_HISTORY_DERIVED")
        except Exception as e:
            return error(Research.AGENT_ID, request.capability, e)
