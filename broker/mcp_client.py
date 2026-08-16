"""
broker.mcp_client — a minimal MCP (Model Context Protocol) client over the
Streamable HTTP transport, hand-rolled with httpx rather than the official
`mcp` SDK.

Why hand-rolled: the `mcp` SDK requires Python >=3.10, but this app's
Railway runtime is pinned to 3.9.18 (runtime.txt) — adding the SDK would
force an unrelated runtime bump. It also pulls in ~11 transitive
dependencies (anyio, starlette, uvicorn, pydantic>=2.12, ...) that are
server-side machinery this app never needs, since it is only ever an MCP
*client*. The rest of this codebase already hand-rolls direct HTTP calls to
every data provider (financial_data/providers/*.py — no vendor SDKs there
either), so this matches house style rather than deviating from it.

Protocol-generic: this file knows nothing about Robinhood specifically. It
implements exactly the JSON-RPC 2.0 framing + lifecycle handshake the MCP
spec defines — initialize -> notifications/initialized -> tools/list /
tools/call — nothing more. Any MCP-over-HTTP server could use this client.

The Streamable HTTP transport lets a server answer either
`Content-Type: application/json` (a single JSON-RPC response object) or
`Content-Type: text/event-stream` (the same response wrapped as one or more
SSE `data:` frames) — this client handles both, since which one a given
server picks isn't something a client is supposed to assume in advance.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx


class MCPError(RuntimeError):
    """A JSON-RPC error object came back, or the transport/handshake broke in
    a way that isn't a plain network failure — distinct from httpx's own
    exceptions so callers can tell 'the server said no' from 'the network is
    down' without inspecting exception internals."""


class MCPClient:
    """One MCP session against one server. Not thread-safe across
    concurrent tool calls (the request id counter and session id are
    instance state) — callers needing concurrency should use one instance
    per call, which is how broker/providers/robinhood.py uses this."""

    def __init__(self, base_url: str, access_token: str,
                 transport: Optional[httpx.BaseTransport] = None,
                 timeout: float = 20.0):
        self.base_url = base_url
        self.access_token = access_token
        self._session_id: Optional[str] = None
        self._next_id = 1
        self._client = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── JSON-RPC transport ──────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.access_token}",
        }
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _post(self, envelope: dict) -> Optional[dict]:
        """POST one JSON-RPC message. Returns the parsed response object for
        a request (has 'id'), or None for a notification (no response body
        expected, MCP servers return 202 Accepted with an empty body)."""
        resp = self._client.post(self.base_url, headers=self._headers(),
                                  json=envelope)
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        if "id" not in envelope:
            resp.raise_for_status()
            return None
        resp.raise_for_status()
        return self._parse_body(resp)

    def _parse_body(self, resp: httpx.Response) -> dict:
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(resp.text)
        return resp.json()

    @staticmethod
    def _parse_sse(body: str) -> dict:
        """Extract the JSON-RPC response object from one or more SSE
        `data: {...}` frames. A single JSON-RPC call yields exactly one
        response frame in practice; this returns the last data frame found,
        since that is the one carrying the actual result (earlier frames,
        if any, would be intermediate progress notifications the spec
        allows but this minimal client does not otherwise act on)."""
        last: Optional[dict] = None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                last = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if last is None:
            raise MCPError(f"no parseable SSE data frame in response: {body[:200]!r}")
        return last

    def _call(self, method: str, params: Optional[dict] = None) -> Any:
        req_id = self._next_id
        self._next_id += 1
        envelope = {"jsonrpc": "2.0", "id": req_id, "method": method,
                    "params": params or {}}
        result = self._post(envelope)
        if result is None:
            raise MCPError(f"expected a response to {method!r}, got none")
        if "error" in result:
            err = result["error"]
            raise MCPError(f"{method} failed: {err.get('message', err)} "
                            f"(code {err.get('code')})")
        return result.get("result")

    def _notify(self, method: str, params: Optional[dict] = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ── MCP lifecycle + tool calls ──────────────────────────────────────

    def initialize(self, client_name: str = "stock_agent",
                    client_version: str = "1.0") -> dict:
        """Full MCP handshake: initialize request, then the
        notifications/initialized notification the spec requires before any
        other method is allowed. Returns the server's initialize result
        (capabilities, serverInfo, protocolVersion)."""
        result = self._call("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": client_version},
        })
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        result = self._call("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        return self._call("tools/call", {"name": name, "arguments": arguments or {}})
