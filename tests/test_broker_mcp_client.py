"""
Verification for broker/mcp_client.py — the hand-rolled MCP JSON-RPC-over-
HTTP client.

Run: python3 tests/test_broker_mcp_client.py

Offline/deterministic via httpx.MockTransport — no real Robinhood server
needed (the real server was exercised live during M2's discovery; see
broker/discover_tools.py). What's checked here is the transport-level
contract this client promises: correct JSON-RPC framing, the full MCP
lifecycle handshake, session-id capture/replay, and handling both
content-type flavors (application/json and text/event-stream) the
Streamable HTTP transport spec allows a server to answer with.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from broker.mcp_client import MCPClient, MCPError

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _server(calls, fail_tool_name=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        method = body.get("method")

        if method == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"protocolVersion": "2025-06-18", "capabilities": {},
                           "serverInfo": {"name": "test-server", "version": "1.0"}},
            }, headers={"Mcp-Session-Id": "sess-abc"})

        if method == "notifications/initialized":
            return httpx.Response(202)

        if method == "tools/list":
            sse = "event: message\ndata: " + json.dumps({
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"tools": [{"name": "get_thing"}]},
            }) + "\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

        if method == "tools/call":
            if body["params"]["name"] == fail_tool_name:
                return httpx.Response(200, json={
                    "jsonrpc": "2.0", "id": body["id"],
                    "error": {"code": -32000, "message": "boom"},
                })
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"content": [{"type": "text", "text": "ok"}], "isError": False},
            })

        return httpx.Response(500, text=f"unexpected method {method}")

    return handler


def test_full_handshake_and_session_replay():
    calls = []
    transport = httpx.MockTransport(_server(calls))
    client = MCPClient("https://example.test/mcp", "tok-123", transport=transport)

    init = client.initialize()
    check("initialize() returns server info",
          init.get("serverInfo", {}).get("name") == "test-server")
    check("session id captured from initialize response header",
          client._session_id == "sess-abc")

    tools = client.list_tools()
    check("list_tools() parses an SSE-framed response",
          tools == [{"name": "get_thing"}])

    result = client.call_tool("get_thing", {"x": 1})
    check("call_tool() returns the tool result",
          result.get("content", [{}])[0].get("text") == "ok")

    check("exactly 4 HTTP calls made (initialize, notify, list, call)", len(calls) == 4)
    ids = [c["id"] for c in calls if "id" in c]
    check("request ids increment only for actual requests, not the notification",
          ids == [1, 2, 3], f"got {ids}")
    notif = [c for c in calls if c.get("method") == "notifications/initialized"][0]
    check("the notification itself carries no id", "id" not in notif)

    client.close()


def test_bearer_header_and_session_header_sent():
    seen_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": body["id"],
                "result": {"serverInfo": {"name": "x"}},
            }, headers={"Mcp-Session-Id": "sess-1"})
        return httpx.Response(202)

    transport = httpx.MockTransport(handler)
    client = MCPClient("https://example.test/mcp", "secret-token", transport=transport)
    client.initialize()

    check("Authorization: Bearer header sent on the first call",
          seen_headers[0].get("authorization") == "Bearer secret-token")
    check("Mcp-Session-Id replayed on the notification after initialize",
          seen_headers[1].get("mcp-session-id") == "sess-1")
    client.close()


def test_tool_error_raises_mcp_error():
    calls = []
    transport = httpx.MockTransport(_server(calls, fail_tool_name="get_thing"))
    client = MCPClient("https://example.test/mcp", "tok", transport=transport)
    client.initialize()
    raised = False
    try:
        client.call_tool("get_thing", {})
    except MCPError as e:
        raised = True
        check("MCPError message includes the server's error text", "boom" in str(e))
    check("tools/call error response raises MCPError", raised)
    client.close()


def test_context_manager_closes():
    transport = httpx.MockTransport(lambda r: httpx.Response(202))
    with MCPClient("https://example.test/mcp", "tok", transport=transport) as client:
        check("client usable inside a with-block", isinstance(client, MCPClient))


if __name__ == "__main__":
    test_full_handshake_and_session_replay()
    test_bearer_header_and_session_header_sent()
    test_tool_error_raises_mcp_error()
    test_context_manager_closes()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — broker.mcp_client: JSON-RPC framing + lifecycle + sessions")
