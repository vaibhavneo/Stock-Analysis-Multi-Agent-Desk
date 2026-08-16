"""
Guard-rail: broker/ must never reference an order-placement, order-
cancellation, or option-exercise tool name, anywhere, in any file.

Run: python3 tests/test_broker_no_trading.py

The user explicitly chose "recommendations + a manual trade queue" over
automatic execution, even though Robinhood's own MCP server exposes real
trading tools (confirmed live in M2's discovery: place_equity_order,
place_option_order, cancel_equity_order, cancel_option_order,
exercise_option, review_equity_order, review_option_order, cancel_option_exercise
— 54 tools total, of which broker/providers/robinhood.py deliberately calls
exactly 3: get_accounts, get_equity_positions, get_portfolio).

This test doesn't re-verify that boundary by reading the code and trusting
it stays that way — it greps the actual committed source on every run for
real CALL SITES specifically (a tool name passed as the first quoted
argument to _call(client, ...) or .call_tool(...), the only two ways this
codebase ever invokes an MCP tool — see broker/providers/robinhood.py), not
bare textual mentions. That distinction matters: this module's own
docstring names every forbidden tool by hand to explain what it does NOT
call, which a naive "does this string appear anywhere" grep would itself
flag as a violation — checked, this is not a hypothetical, it's what the
first draft of this test actually did. Only an executable invocation counts
as a violation; documentation explaining the boundary does not.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BROKER_DIR = ROOT / "broker"

# Tool name passed as the first quoted argument to _call(client, "...") or
# .call_tool("...") — the only two call shapes broker/ ever uses to invoke
# an MCP tool (see broker/providers/robinhood.py's _call() and
# broker/mcp_client.py's MCPClient.call_tool()).
CALL_SITE_RE = re.compile(
    r"""(?:_call\(\s*client\s*,\s*|\.call_tool\(\s*)['"]([A-Za-z_]+)['"]""")

# Confirmed live via tools/list against the real Robinhood MCP server (M2) —
# every tool that places, cancels, or exercises a real order. Not the full
# 54-tool list: only the read-only ones (get_accounts, get_equity_positions,
# get_portfolio, ...) are legitimate call sites and are absent from this set
# on purpose.
FORBIDDEN_TOOLS = {
    "place_equity_order", "place_option_order",
    "cancel_equity_order", "cancel_option_order", "cancel_option_exercise",
    "exercise_option",
    "review_equity_order", "review_option_order",   # pre-trade review still implies a pending order flow
}

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def test_no_trading_tool_call_sites_in_broker_package():
    offenders = []
    all_call_sites = []
    for path in sorted(BROKER_DIR.rglob("*.py")):
        text = path.read_text()
        for m in CALL_SITE_RE.finditer(text):
            tool_name = m.group(1)
            line_no = text[:m.start()].count("\n") + 1
            all_call_sites.append(f"{path.relative_to(ROOT)}:{line_no}: {tool_name}")
            if tool_name in FORBIDDEN_TOOLS:
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {tool_name}")
    check("no order-placement/cancellation/exercise tool CALL SITE in broker/",
          not offenders, "; ".join(offenders) if offenders else "")
    check("at least one real (read-only) call site exists, proving the regex isn't just matching nothing",
          len(all_call_sites) > 0, "; ".join(all_call_sites))


if __name__ == "__main__":
    test_no_trading_tool_call_sites_in_broker_package()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — broker/ contains no trading/order-placement tool calls")
