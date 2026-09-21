"""
Calling OptionsPilot over HTTP.

WHY A CLIENT AND NOT AN IMPORT

    OptionsPilot computes options structures against a live chain, freezes
    ideas into its own journal, and holds its own research freeze. It is a
    deployed service with its own release cycle. Importing it would fork the
    options logic into two copies that drift; calling it keeps one.

    This is the mirror of `desk/client.py` inside OptionsPilot, which reads
    THIS service for direction. Both halves of the link therefore behave the
    same way, which matters when one of them breaks.

WHAT HAPPENS WHEN IT IS DOWN

    The Stock Agent keeps working on equity evidence alone, and SAYS SO. Every
    return value carries `available` and, when false, a `reason` a person can
    act on. No caller may assume OptionsPilot answered — a panel that looks
    identical whether or not it did is the worst failure available here,
    because nothing on screen reveals which happened.

AUTHENTICATION

    OptionsPilot gates everything behind a shared access code, and the gate is
    SESSION-COOKIE based: the code is POSTed once to `/access`, which sets a
    cookie that later calls carry. So this client logs in once per process and
    reuses the cookie.

    The code comes from OPTIONSPILOT_ACCESS_CODE and is never logged, never
    placed in a returned payload, and never put in an LLM prompt. `status()`
    reports whether a code is configured, never any part of its value.

WHAT THIS CLIENT WILL NOT DO

    There is no write path here beyond `run_pipeline()`, which is explicitly
    user-triggered. Neither service has a broker order path and this link does
    not add one.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from typing import Any, Dict, Optional, Tuple

DEFAULT_BASE_URL = "https://optionspilot-production-ba41.up.railway.app"

# A dependency that can take this app down by being slow is a dependency that
# will take this app down. The instruments call prices a live options chain and
# was measured at ~13s, so its timeout is generous while the cheap calls are
# held short.
DEFAULT_TIMEOUT = 20.0
INSTRUMENTS_TIMEOUT = 45.0
RUN_TIMEOUT = 120.0

_lock = threading.Lock()
_opener: Optional[Any] = None
_authenticated = False


def base_url() -> str:
    return (os.getenv("OPTIONSPILOT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _access_code() -> Optional[str]:
    code = os.getenv("OPTIONSPILOT_ACCESS_CODE")
    return code.strip() if code and code.strip() else None


def reset() -> None:
    """Drop the session. Used by tests, and after an auth failure so the next
    call re-logs-in rather than reusing a cookie the server has forgotten."""
    global _opener, _authenticated
    with _lock:
        _opener = None
        _authenticated = False


def _get_opener():
    global _opener
    if _opener is None:
        _opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))
    return _opener


def _request(path: str, timeout: float, method: str = "GET",
             body: Optional[Dict[str, Any]] = None,
             form: Optional[Dict[str, str]] = None
             ) -> Tuple[Optional[Any], Optional[str], Optional[int]]:
    """One HTTP call. Returns (payload, error, status) and NEVER raises.

    `payload` is the decoded JSON when the response was JSON. A non-JSON body
    is an error, not a payload: OptionsPilot returns an HTML error page on a
    500, and parsing that as data is how a broken dependency turns into
    nonsense on screen instead of a stated outage.
    """
    url = base_url() + path
    data = None
    headers = {"Accept": "application/json", "User-Agent": "StockAgent/1.0 (options-link)"}
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _get_opener().open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            raw = e.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
        if status == 401:
            return None, "OptionsPilot rejected the access code", status
        try:
            payload = json.loads(raw)
        except ValueError:
            # A 500 from OptionsPilot renders an HTML error page. Say that
            # rather than surfacing markup as data.
            return None, f"OptionsPilot returned HTTP {status}", status
        return payload, f"OptionsPilot returned HTTP {status}", status
    except urllib.error.URLError as e:
        return None, f"could not reach OptionsPilot ({type(e.reason).__name__})", None
    except Exception as e:
        return None, f"could not reach OptionsPilot ({type(e).__name__})", None

    try:
        return json.loads(raw), None, status
    except ValueError:
        return None, "OptionsPilot returned a response that was not JSON", status


def _ensure_authenticated() -> Optional[str]:
    """Log in once per process. Returns an error string, or None on success.

    When no code is configured this is a no-op: a local OptionsPilot with no
    ACCESS_CODE set serves without a gate, and failing here would make the
    integration untestable without a production credential.
    """
    global _authenticated
    code = _access_code()
    if not code:
        return None
    with _lock:
        if _authenticated:
            return None
        _, error, status = _request("/access", DEFAULT_TIMEOUT, method="POST",
                                    form={"code": code})
        # /access redirects on success; a 401 means the code was not accepted.
        if status == 401 or (error and "rejected" in error):
            return ("OptionsPilot rejected the configured access code. It was most likely "
                    "rotated on OptionsPilot without OPTIONSPILOT_ACCESS_CODE being updated "
                    "here.")
        if error and status is None:
            return error
        _authenticated = True
        return None


def _call(path: str, timeout: float, method: str = "GET",
          body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Authenticate if needed, then call. Always returns an `available` dict."""
    auth_error = _ensure_authenticated()
    if auth_error:
        return {"available": False, "reason": auth_error, "authenticated": False}

    payload, error, status = _request(path, timeout, method=method, body=body)
    if status == 401:
        # The cookie went stale mid-process; drop it so the next call retries.
        reset()
        return {"available": False, "authenticated": False,
                "reason": "OptionsPilot requires an access code and the one configured here "
                          "was not accepted."}
    if error:
        return {"available": False, "reason": error, "http_status": status}
    if payload is None:
        return {"available": False, "reason": "OptionsPilot returned an empty response"}
    return {"available": True, "payload": payload, "http_status": status}


# ── Public surface ────────────────────────────────────────────────────────

def status() -> Dict[str, Any]:
    """Reachability and configuration. Never reveals any part of the code."""
    configured = _access_code() is not None
    result = _call("/api/status", DEFAULT_TIMEOUT)
    out: Dict[str, Any] = {
        "available": result["available"],
        "base_url": base_url(),
        "access_code_configured": configured,
    }
    if not result["available"]:
        out["reason"] = result["reason"]
        if not configured:
            out["reason"] += (" No OPTIONSPILOT_ACCESS_CODE is set in this service's "
                              "environment, which is required for the deployed OptionsPilot.")
        return out
    payload = result["payload"] or {}
    out.update({
        "paper_only": payload.get("paper_only"),
        "latest_run": payload.get("latest_run"),
        # OptionsPilot reads THIS desk for direction. Its own coverage report
        # says whether that read is producing anything usable, which is the
        # most useful diagnostic the link exposes in either direction.
        "desk_coverage": payload.get("desk_coverage"),
    })
    return out


def instruments(ticker: str, view: Optional[str] = None) -> Dict[str, Any]:
    """Candidate option structures for one symbol, priced against a live chain.

    `view` is THIS engine's directional read, handed over explicitly. That is
    the point of the link: OptionsPilot otherwise derives direction from a
    stored desk view, and its own status reports that most names carry no fresh
    view at all. Passing ours means the two agents agree by construction rather
    than by coincidence.
    """
    ticker = (ticker or "").upper().strip()
    if not ticker:
        return {"available": False, "reason": "no ticker given"}
    path = f"/api/instruments/{urllib.parse.quote(ticker)}"
    if view:
        path += "?" + urllib.parse.urlencode({"view": view})
    result = _call(path, INSTRUMENTS_TIMEOUT)
    if not result["available"]:
        return {"available": False, "ticker": ticker, "reason": result["reason"],
                "authenticated": result.get("authenticated")}
    payload = result["payload"] or {}
    if not payload.get("available"):
        return {"available": False, "ticker": ticker,
                "reason": payload.get("reason") or "OptionsPilot has no structures for this symbol"}
    payload["ticker"] = ticker
    payload["available"] = True
    return payload


def equity_read(ticker: str) -> Dict[str, Any]:
    """OptionsPilot's own read of the stock from daily bars — its realised
    expected move, which is what sizes every structure it builds."""
    ticker = (ticker or "").upper().strip()
    result = _call(f"/api/equity/{urllib.parse.quote(ticker)}", DEFAULT_TIMEOUT)
    if not result["available"]:
        return {"available": False, "ticker": ticker, "reason": result["reason"]}
    payload = result["payload"] or {}
    payload.setdefault("available", True)
    payload["ticker"] = ticker
    return payload


def run_pipeline(universe: Optional[list] = None) -> Dict[str, Any]:
    """Trigger one OptionsPilot pipeline pass.

    THE ONLY WRITE THIS LINK PERFORMS. It runs OptionsPilot's ranking over a
    universe and freezes whatever ideas clear its bar into ITS journal — a real,
    durable side effect on another service. It is therefore never called
    automatically: the UI exposes it as an explicit, separate action, and the
    per-ticker read below needs no trigger at all.
    """
    result = _call("/api/run", RUN_TIMEOUT, method="POST",
                   body={"universe": universe} if universe else {})
    if not result["available"]:
        return {"available": False, "reason": result["reason"]}
    payload = result["payload"] or {}
    payload.setdefault("available", True)
    return payload
