"""
broker — OAuth + MCP client machinery for connecting a live brokerage account
to the stock_agent app.

Protocol-generic pieces (oauth.py, mcp_client.py, token_store.py, keys.py) know
nothing about any specific broker. Everything that references a real vendor
(tool names, response shapes) lives under broker/providers/ — currently just
providers/robinhood.py, written only after live tool discovery confirmed what
Robinhood's MCP server actually exposes (see broker/discover_tools.py).

This package is read-only by design: it can fetch positions/account data, and
nothing here places an order. That is a deliberate scope boundary, not an
oversight — see the guard-rail test in tests/test_broker_no_trading.py.
"""
