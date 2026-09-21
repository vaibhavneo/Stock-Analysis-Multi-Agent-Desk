"""The OptionsPilot sub-agent — the one specialist that is a separate service.

This is a thin adapter over `optionspilot/client.py`, which already does the
hard part (session auth, timeouts, never raising). Its job is to translate
that client's `available` + `reason` shape into the uniform AgentResult, and
to decline EARLY and cheaply for the cases the planner can know about without
a network round trip.

Declining early matters more than it looks. OptionsPilot covers a bounded
universe of ~31 names. For any other symbol — and for every crypto, FX, index
and commodity underlying — the correct answer is "out of universe", and
spending a 45-second HTTP timeout to discover that on every request would make
the orchestrator feel broken when it is working exactly as designed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from ..contract import AgentRequest, AgentResult, ok, unavailable, error

AGENT_ID = "options_pilot"

SERVES = {"EQUITY", "ETF"}


def available(symbol: str, asset_class: str) -> Tuple[bool, str]:
    """Cheap, local pre-checks only. The expensive truth (is the service up,
    is this name in its universe) is discovered by running."""
    if asset_class not in SERVES:
        return False, (f"OptionsPilot covers listed equities and ETFs; "
                       f"{symbol} is classified {asset_class}, which no venue "
                       "it reads lists options on")
    if not os.environ.get("OPTIONSPILOT_ACCESS_CODE"):
        return False, ("OPTIONSPILOT_ACCESS_CODE is not set in this service's "
                       "environment, so the link cannot authenticate. "
                       "OptionsPilot refuses every route without it by design.")
    return True, ""


def status() -> Dict[str, Any]:
    try:
        from optionspilot import client
        return client.status()
    except Exception as e:
        return {"available": False,
                "reason": f"the options link could not be used ({type(e).__name__})"}


def run(request: AgentRequest) -> AgentResult:
    cap = request.capability
    ready, why = available(request.symbol, request.asset_class)
    if not ready:
        return unavailable(AGENT_ID, cap, why)

    if cap not in ("option_structures", "implied_volatility"):
        return unavailable(AGENT_ID, cap, f"this agent does not offer {cap!r}")

    try:
        from optionspilot import client
        from decision.options_overlay import build_options_overlay

        view = (request.params.get("view") or "").upper() or None
        if view not in (None, "BULLISH", "BEARISH"):
            view = None

        raw = client.instruments(request.symbol, view=view)
        overlay = build_options_overlay(raw, requested_view=view)

        if overlay.get("status") != "OK":
            return unavailable(AGENT_ID, cap,
                               overlay.get("reason")
                               or "OptionsPilot returned no usable structures")

        if cap == "implied_volatility":
            surface = (overlay.get("surface") or raw.get("surface")
                       if isinstance(raw, dict) else None)
            if not surface:
                return unavailable(AGENT_ID, cap,
                                   "OptionsPilot answered but carried no "
                                   "volatility surface for this name")
            return ok(AGENT_ID, cap, {"symbol": request.symbol,
                                      "surface": surface},
                      price_basis="TRADED_PRICE", service="optionspilot")

        return ok(AGENT_ID, cap, overlay, price_basis="TRADED_PRICE",
                  service="optionspilot", view_sent=view)
    except Exception as e:
        return error(AGENT_ID, cap, e)
