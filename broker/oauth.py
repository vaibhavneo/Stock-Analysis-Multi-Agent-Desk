"""
broker.oauth — OAuth 2.1 authorization-code + PKCE flow, Dynamic Client
Registration, and token refresh, against Robinhood's Agentic Trading MCP
server.

The PKCE/DCR/token-exchange functions here are protocol-generic (they take
the authorization/registration/token endpoint URLs as arguments, defaulting
to Robinhood's); the module-level ROBINHOOD_* constants are the only
Robinhood-specific thing in this file, kept here rather than duplicated at
every call site since Robinhood is currently the only provider this app
talks to. Verified live against
https://agent.robinhood.com/.well-known/oauth-authorization-server:

  authorization_endpoint: https://robinhood.com/oauth
  registration_endpoint:  https://agent.robinhood.com/oauth/trading/register
  token_endpoint:         https://api.robinhood.com/oauth2/token/
  token_endpoint_auth_methods_supported: ["none"]   <- public client, no secret
  code_challenge_methods_supported: ["S256"]         <- PKCE required
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Optional

import httpx

from . import token_store
from .keys import get_key

ROBINHOOD_AUTHORIZATION_ENDPOINT = "https://robinhood.com/oauth"
ROBINHOOD_REGISTRATION_ENDPOINT = "https://agent.robinhood.com/oauth/trading/register"
ROBINHOOD_TOKEN_ENDPOINT = "https://api.robinhood.com/oauth2/token/"
ROBINHOOD_SCOPE = "internal"

# How much earlier than the real expiry to treat an access token as stale,
# so an in-flight request doesn't race the token expiring mid-call.
REFRESH_SKEW_SECONDS = 60


class OAuthError(RuntimeError):
    """Registration or token exchange failed against the server — distinct
    from NotConnectedError (no token at all) since 'the flow broke' and
    'never started' call for different messages to the user."""


class NotConnectedError(RuntimeError):
    """No stored token — the user has not completed the Robinhood connect
    flow yet (or disconnected). Callers should route to /api/broker/connect,
    not retry."""


def generate_pkce_pair() -> tuple[str, str]:
    """(code_verifier, code_challenge) per RFC 7636. verifier: a 43-128 char
    URL-safe string; challenge: base64url(sha256(verifier)), no padding."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def register_client(redirect_uris: list[str], client_name: str = "stock_agent",
                     registration_endpoint: str = ROBINHOOD_REGISTRATION_ENDPOINT) -> dict:
    """Dynamic Client Registration (RFC 7591) — run once, by hand (see
    broker/register_client.py), to obtain a client_id. Returns the full raw
    response so the caller can inspect it directly — e.g. whether a
    client_secret came back despite the advertised "none" auth method,
    which is unconfirmed until this actually runs against the live
    endpoint."""
    resp = httpx.post(registration_endpoint, json={
        "client_name": client_name,
        "redirect_uris": redirect_uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }, timeout=20.0)
    if resp.status_code >= 400:
        raise OAuthError(f"client registration failed: {resp.status_code} {resp.text[:500]}")
    return resp.json()


def build_authorize_url(client_id: str, redirect_uri: str, state: str,
                         code_challenge: str,
                         authorization_endpoint: str = ROBINHOOD_AUTHORIZATION_ENDPOINT,
                         scope: str = ROBINHOOD_SCOPE) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "scope": scope,
    }
    return str(httpx.URL(authorization_endpoint, params=params))


def exchange_code_for_token(code: str, redirect_uri: str, code_verifier: str,
                             client_id: str,
                             token_endpoint: str = ROBINHOOD_TOKEN_ENDPOINT) -> dict:
    resp = httpx.post(token_endpoint, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }, timeout=20.0)
    if resp.status_code >= 400:
        raise OAuthError(f"token exchange failed: {resp.status_code} {resp.text[:500]}")
    return resp.json()


def refresh_access_token(refresh_token: str, client_id: str,
                          token_endpoint: str = ROBINHOOD_TOKEN_ENDPOINT) -> dict:
    resp = httpx.post(token_endpoint, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }, timeout=20.0)
    if resp.status_code >= 400:
        raise OAuthError(f"token refresh failed: {resp.status_code} {resp.text[:500]}")
    return resp.json()


def _persist_token_response(token_response: dict, previous: Optional[dict]) -> dict:
    """Merge a token endpoint response into the stored shape, defensively:
    some OAuth 2.1 servers rotate the refresh_token on every use, some
    don't (unconfirmed for Robinhood until this runs live) — keep the
    previous refresh_token only when the new response didn't include one."""
    now = time.time()
    expires_in = token_response.get("expires_in")
    record = {
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token")
                          or (previous or {}).get("refresh_token"),
        "token_type": token_response.get("token_type", "Bearer"),
        "expires_at": (now + float(expires_in)) if expires_in is not None else None,
        "obtained_at": now,
    }
    token_store.save(record)
    return record


def complete_authorization(code: str, redirect_uri: str, code_verifier: str) -> dict:
    client_id = get_key("ROBINHOOD_CLIENT_ID", "broker.oauth")
    token_response = exchange_code_for_token(code, redirect_uri, code_verifier, client_id)
    return _persist_token_response(token_response, previous=None)


def ensure_fresh_access_token() -> str:
    """The one function most callers want: a currently-valid access token,
    refreshing on disk (never in a process-global — see token_store's
    module docstring on the --workers 2 hazard) if the stored one is stale
    or close to expiring."""
    record = token_store.load()
    if record is None:
        raise NotConnectedError(
            "no stored Robinhood token — connect first via /api/broker/connect")
    expires_at = record.get("expires_at")
    if expires_at is not None and time.time() < (expires_at - REFRESH_SKEW_SECONDS):
        return record["access_token"]
    refresh_token = record.get("refresh_token")
    if not refresh_token:
        raise NotConnectedError(
            "stored Robinhood token has no refresh_token and is expired — "
            "reconnect via /api/broker/connect")
    client_id = get_key("ROBINHOOD_CLIENT_ID", "broker.oauth")
    token_response = refresh_access_token(refresh_token, client_id)
    updated = _persist_token_response(token_response, previous=record)
    return updated["access_token"]


def is_connected() -> bool:
    return token_store.load() is not None
