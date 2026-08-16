"""
One-time Dynamic Client Registration against Robinhood's Agentic Trading
MCP server. Run by hand:

    python3 -m broker.register_client

Prints the full raw registration response (so client_secret presence, if
any, is visible rather than assumed) and the client_id to add to .env /
Railway as ROBINHOOD_CLIENT_ID. Safe to re-run — registering again just
issues a new client_id, it doesn't affect any previously-issued token.
"""
from __future__ import annotations

import json
import sys

from broker.oauth import register_client

PROD_REDIRECT_URI = "https://agentic-ai-production-aea7.up.railway.app/api/broker/callback"
LOCAL_REDIRECT_URI = "http://localhost:5051/api/broker/callback"


def main() -> None:
    try:
        result = register_client(redirect_uris=[PROD_REDIRECT_URI, LOCAL_REDIRECT_URI],
                                  client_name="stock_agent")
    except Exception as e:
        print(f"Registration with both redirect_uris failed: {e}", file=sys.stderr)
        print("Retrying with production redirect_uri only...", file=sys.stderr)
        result = register_client(redirect_uris=[PROD_REDIRECT_URI], client_name="stock_agent")

    print("\n=== Raw registration response ===")
    print(json.dumps(result, indent=2))

    client_id = result.get("client_id")
    has_secret = "client_secret" in result
    print("\n=== Summary ===")
    print(f"client_id: {client_id}")
    print(f"client_secret present: {has_secret}"
          + (f" (value: {result['client_secret'][:8]}...)" if has_secret else ""))
    print(f"redirect_uris registered: {result.get('redirect_uris')}")
    print("\nAdd to stock_agent/.env and Railway variables:")
    print(f"ROBINHOOD_CLIENT_ID={client_id}")


if __name__ == "__main__":
    main()
