"""
broker.providers.robinhood — the ONLY file in this codebase allowed to
reference real Robinhood MCP tool names. Written after, not before, live
discovery (see broker/discover_tools.py's output) confirmed exactly what
https://agent.robinhood.com/mcp/trading (serverInfo: robinhood-trading
v1.1.4, 54 tools total) actually exposes — nothing here is guessed.

Read-only by design, matching the user's explicit choice (recommendations +
a manual trade queue, not automatic execution): this module calls exactly
three tools — get_accounts, get_equity_positions, get_portfolio — and NO
other tool from that server's 54. In particular it never references
place_equity_order, place_option_order, cancel_equity_order,
cancel_option_order, or exercise_option, which the server does expose.
tests/test_broker_no_trading.py enforces this boundary by grepping the
whole broker/ package for those names.

Discovery also surfaced something the plan flagged as a real risk before
confirming it either way: two accounts existed on this OAuth grant — the
user's real default brokerage account (agentic_allowed: false — reads
still work, writes don't) and a separate, empty "Agentic" sub-account
(agentic_allowed: true) matching Robinhood's own "your agent trades in a
dedicated account separate from the rest of your portfolio" framing.
get_default_holdings() deliberately targets is_default, never
agentic_allowed, for exactly that reason — the dedicated sandbox account is
not what a user asking "what do I actually hold" wants to see.

No dedicated crypto-positions tool exists in this tool set (only an
aggregate crypto_value dollar figure via get_portfolio) — Bitcoin shows up
as a total, not an itemized holding, and that limitation is surfaced
explicitly in this module's return shape rather than silently dropped.
"""
from __future__ import annotations

from typing import Optional

from .. import oauth
from ..mcp_client import MCPClient

MCP_SERVER_URL = "https://agent.robinhood.com/mcp/trading"


class RobinhoodError(RuntimeError):
    """The MCP server reachable and authenticated, but the account/position
    data it returned doesn't make sense to act on (e.g. no accounts at
    all) — distinct from an oauth.NotConnectedError (never authenticated)
    or an MCPError (transport/protocol failure)."""


def _open_client() -> MCPClient:
    access_token = oauth.ensure_fresh_access_token()
    client = MCPClient(MCP_SERVER_URL, access_token)
    client.initialize()
    return client


def _call(client: MCPClient, tool_name: str, arguments: Optional[dict] = None) -> dict:
    """Unwrap one tools/call result down to its actual payload. Confirmed
    live (get_accounts/get_equity_positions/get_portfolio's outputSchema,
    and the actual response bodies) that every one of these tools wraps its
    payload as {"data": {...}, "guide": "..."} inside structuredContent —
    "guide" is server-authored prose about how to *present* the data to an
    end user, not data itself, so it's dropped here rather than carried
    through this module's own return shape. Falls back to parsing
    content[0].text as JSON for a server that only populates the older
    plain-text content field, since the raw wire shape wasn't something to
    assume blind without a live server actually confirming it."""
    result = client.call_tool(tool_name, arguments or {})
    if result.get("isError"):
        content = result.get("content") or []
        text = content[0].get("text") if content else str(result)
        raise RobinhoodError(f"{tool_name} returned an error: {text}")
    if "structuredContent" in result:
        payload = result["structuredContent"]
    else:
        content = result.get("content") or []
        if content and content[0].get("type") == "text":
            import json
            payload = json.loads(content[0]["text"])
        else:
            raise RobinhoodError(f"{tool_name}: no structuredContent or parseable text content")
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def _normalize_equity_positions(raw_positions: list[dict]) -> list[dict]:
    """Robinhood's raw position rows -> the exact {ticker, shares, avg_cost}
    shape /api/portfolio-brief already accepts (web/app.py:848-912) — the
    one and only shape this module's holdings output has to match, so the
    existing recommendation/brief logic needs zero changes. Positions with
    zero (or unparseable) quantity are dropped — a fully-closed position
    (type == 'empty' in Robinhood's own data, confirmed live) is not a
    holding to size a trade queue around."""
    out = []
    for p in raw_positions or []:
        try:
            shares = float(p.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        if shares <= 0:
            continue
        avg_cost_raw = p.get("average_buy_price")
        try:
            avg_cost = float(avg_cost_raw) if avg_cost_raw not in (None, "") else None
        except (TypeError, ValueError):
            avg_cost = None
        out.append({"ticker": p["symbol"], "shares": shares, "avg_cost": avg_cost})
    return out


def get_default_holdings() -> dict:
    """The one function the Flask route (GET /api/broker/positions) calls.
    One MCP session, three read-only tool calls, real account picked
    deliberately by is_default rather than agentic_allowed (see module
    docstring). Returns:

      {"account_number": str,
       "holdings": [{"ticker","shares","avg_cost"}, ...],   # equities only
       "crypto_value_usd": float | None,   # aggregate only, not itemized
       "cash_usd": float | None,
       "total_value_usd": float | None}
    """
    with _open_client() as client:
        accounts = _call(client, "get_accounts").get("accounts") or []
        if not accounts:
            raise RobinhoodError("no brokerage accounts visible to this OAuth grant")
        account = next((a for a in accounts if a.get("is_default")), accounts[0])
        account_number = account["account_number"]

        positions = _call(client, "get_equity_positions",
                           {"account_number": account_number}).get("positions") or []
        holdings = _normalize_equity_positions(positions)

        portfolio = _call(client, "get_portfolio", {"account_number": account_number})

    def _f(key: str) -> Optional[float]:
        v = portfolio.get(key)
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    return {
        "account_number": account_number,
        "holdings": holdings,
        "crypto_value_usd": _f("crypto_value"),
        "cash_usd": _f("cash"),
        "total_value_usd": _f("total_value"),
    }
