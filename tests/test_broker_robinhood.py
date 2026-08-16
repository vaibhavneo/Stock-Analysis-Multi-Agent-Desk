"""
Verification for broker/providers/robinhood.py — response unwrapping and
position normalization.

Run: python3 tests/test_broker_robinhood.py

The fixtures below are the real shapes confirmed live during M2's discovery
against https://agent.robinhood.com/mcp/trading (serverInfo: robinhood-
trading v1.1.4) and the real /api/broker/positions response this app served
against the user's own account — not guessed. Uses fake MCPClient/oauth
objects rather than real network or a real token, so it runs offline and
never touches a live account.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.providers.robinhood import (
    RobinhoodError, _call, _normalize_equity_positions, get_default_holdings,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


class FakeMCPClient:
    """Stands in for broker.mcp_client.MCPClient. call_tool_results maps
    tool name -> the raw tools/call 'result' object (before this module's
    own _call() unwraps it) — the exact shape confirmed live in M2."""
    def __init__(self, call_tool_results):
        self._results = call_tool_results
        self.calls = []

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self._results[name]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def test_call_unwraps_structured_content_data_envelope():
    """Confirmed live (and the bug this test guards against — an earlier
    version of get_default_holdings() forgot this exact unwrap and raised
    'no brokerage accounts visible' against a real, populated account):
    every one of these tools wraps its payload as
    {"data": {...}, "guide": "..."} inside structuredContent."""
    client = FakeMCPClient({
        "get_accounts": {
            "isError": False,
            "structuredContent": {
                "data": {"accounts": [{"account_number": "123", "is_default": True}]},
                "guide": "some presentation instructions",
            },
        },
    })
    result = _call(client, "get_accounts")
    check("_call unwraps to the 'data' payload, dropping 'guide'",
          result == {"accounts": [{"account_number": "123", "is_default": True}]})


def test_call_falls_back_to_text_content():
    client = FakeMCPClient({
        "get_portfolio": {
            "isError": False,
            "content": [{"type": "text", "text": '{"data": {"cash": "100.00"}, "guide": "x"}'}],
        },
    })
    result = _call(client, "get_portfolio")
    check("_call parses JSON out of content[0].text when structuredContent is absent",
          result == {"cash": "100.00"})


def test_call_raises_on_isError():
    client = FakeMCPClient({
        "get_accounts": {"isError": True, "content": [{"type": "text", "text": "no access"}]},
    })
    raised = False
    try:
        _call(client, "get_accounts")
    except RobinhoodError as e:
        raised = True
        check("RobinhoodError message includes the server's error text", "no access" in str(e))
    check("isError:true raises RobinhoodError", raised)


def test_normalize_drops_closed_and_zero_positions():
    """Real shape from the user's own account (M2): a fully-closed position
    (TSLA) came back with type:'empty' and quantity:'0.000000' — that must
    not show up as a $0 holding in the trade queue."""
    raw = [
        {"symbol": "TSLA", "quantity": "0.000000", "type": "empty"},
        {"symbol": "IONQ", "quantity": "175.000000", "average_buy_price": "44.180000", "type": "long"},
        {"symbol": "GARBAGE", "quantity": "not-a-number", "average_buy_price": "1.00"},
    ]
    out = _normalize_equity_positions(raw)
    check("zero-quantity position dropped", all(h["ticker"] != "TSLA" for h in out))
    check("unparseable-quantity position dropped", all(h["ticker"] != "GARBAGE" for h in out))
    check("real position kept with correct ticker/shares/avg_cost",
          out == [{"ticker": "IONQ", "shares": 175.0, "avg_cost": 44.18}], f"got {out}")


def test_normalize_handles_missing_avg_cost():
    out = _normalize_equity_positions([
        {"symbol": "XYZ", "quantity": "1.0", "average_buy_price": None},
    ])
    check("missing average_buy_price becomes None, not a crash",
          out == [{"ticker": "XYZ", "shares": 1.0, "avg_cost": None}], f"got {out}")


def test_get_default_holdings_picks_is_default_not_agentic_allowed():
    """The exact risk the plan flagged before it was confirmed either way:
    Robinhood's own agentic sub-account is separate from the real portfolio
    (confirmed live in M2 — the real account was is_default:true,
    agentic_allowed:false; the sandbox was the opposite). This must never
    silently prefer the agentic account."""
    import broker.providers.robinhood as rh
    fake = FakeMCPClient({
        "get_accounts": {"isError": False, "structuredContent": {"data": {"accounts": [
            {"account_number": "AGENTIC-1", "is_default": False, "agentic_allowed": True},
            {"account_number": "REAL-1", "is_default": True, "agentic_allowed": False},
        ]}, "guide": ""}},
        "get_equity_positions": {"isError": False, "structuredContent": {"data": {
            "positions": [{"symbol": "AAPL", "quantity": "1.0", "average_buy_price": "100.0"}],
        }, "guide": ""}},
        "get_portfolio": {"isError": False, "structuredContent": {"data": {
            "crypto_value": "5.00", "cash": "10.00", "total_value": "115.00",
        }, "guide": ""}},
    })
    orig_open = rh._open_client
    rh._open_client = lambda: fake
    try:
        result = get_default_holdings()
    finally:
        rh._open_client = orig_open

    check("picked the is_default account, not the agentic_allowed one",
          result["account_number"] == "REAL-1")
    check("holdings normalized correctly", result["holdings"] == [
        {"ticker": "AAPL", "shares": 1.0, "avg_cost": 100.0}])
    check("crypto/cash/total surfaced as floats",
          result["crypto_value_usd"] == 5.0 and result["cash_usd"] == 10.0
          and result["total_value_usd"] == 115.0)

    equity_call = [c for c in fake.calls if c[0] == "get_equity_positions"][0]
    check("get_equity_positions called with the REAL account, not the agentic one",
          equity_call[1] == {"account_number": "REAL-1"})


if __name__ == "__main__":
    test_call_unwraps_structured_content_data_envelope()
    test_call_falls_back_to_text_content()
    test_call_raises_on_isError()
    test_normalize_drops_closed_and_zero_positions()
    test_normalize_handles_missing_avg_cost()
    test_get_default_holdings_picks_is_default_not_agentic_allowed()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — broker.providers.robinhood: unwrapping + normalization + account selection")
