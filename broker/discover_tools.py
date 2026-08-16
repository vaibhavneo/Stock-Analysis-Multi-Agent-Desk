"""
One-time, interactive: complete a real Robinhood OAuth login (in the user's
own browser — this script never sees a password), capture the callback on a
local loopback listener, exchange the code for a token, then call the MCP
server's tools/list for real so broker/providers/robinhood.py can be written
against actual tool names instead of guessed ones.

Run:  python3 -m broker.discover_tools

Prints the authorize URL to open, waits (up to 10 minutes) for the
localhost:5051 callback, then prints the raw initialize + tools/list output
and, if it finds a stock/crypto positions balance in the response, echoes
that too as a first sanity check against what the user actually holds.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from broker.keys import get_key
from broker.mcp_client import MCPClient
from broker.oauth import (
    generate_pkce_pair, generate_state, build_authorize_url,
    complete_authorization, OAuthError,
)

LOCAL_REDIRECT_URI = "http://localhost:5051/api/broker/callback"
MCP_SERVER_URL = "https://agent.robinhood.com/mcp/trading"

_result: dict = {}
_done = threading.Event()


def _make_handler(expected_state: str, code_verifier: str):
    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass   # keep stdout clean for the discovery output below

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/broker/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = parse_qs(parsed.query)
            error = qs.get("error", [None])[0]
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]

            if error:
                _result["error"] = error
            elif state != expected_state:
                _result["error"] = f"state mismatch (possible CSRF): got {state!r}"
            elif not code:
                _result["error"] = "no authorization code in callback"
            else:
                _result["code"] = code

            body = (b"<html><body><h2>Robinhood connected.</h2>"
                    b"You can close this tab and return to the terminal.</body></html>"
                    if "code" in _result else
                    f"<html><body><h2>Something went wrong</h2><p>{_result.get('error')}</p></body></html>".encode())
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
            _done.set()

    return CallbackHandler


def main() -> None:
    client_id = get_key("ROBINHOOD_CLIENT_ID", "broker.discover_tools")
    code_verifier, code_challenge = generate_pkce_pair()
    state = generate_state()
    authorize_url = build_authorize_url(client_id, LOCAL_REDIRECT_URI, state, code_challenge)

    server = HTTPServer(("localhost", 5051), _make_handler(state, code_verifier))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("=" * 70)
    print("Open this URL in your browser and log in / approve on Robinhood's")
    print("own site. Waiting up to 10 minutes for the callback...")
    print("=" * 70)
    print(authorize_url)
    print("=" * 70)
    sys.stdout.flush()

    got_callback = _done.wait(timeout=600)
    server.shutdown()

    if not got_callback:
        print("Timed out waiting for the OAuth callback.", file=sys.stderr)
        sys.exit(1)
    if "error" in _result:
        print(f"OAuth callback reported an error: {_result['error']}", file=sys.stderr)
        sys.exit(1)

    print("\nCallback received, exchanging code for a token...")
    try:
        tokens = complete_authorization(_result["code"], LOCAL_REDIRECT_URI, code_verifier)
    except OAuthError as e:
        print(f"Token exchange failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Token stored. Has refresh_token: {bool(tokens.get('refresh_token'))}, "
          f"expires_at: {tokens.get('expires_at')}")

    print("\nCalling MCP initialize + tools/list against the real server...")
    with MCPClient(MCP_SERVER_URL, tokens["access_token"]) as mcp:
        init_result = mcp.initialize()
        print("\n=== initialize() result ===")
        print(json.dumps(init_result, indent=2))

        tools = mcp.list_tools()
        print(f"\n=== tools/list: {len(tools)} tool(s) ===")
        print(json.dumps(tools, indent=2))


if __name__ == "__main__":
    main()
