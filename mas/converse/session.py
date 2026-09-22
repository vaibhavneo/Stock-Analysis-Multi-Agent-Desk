"""Conversation state — small on purpose.

WHAT IS CARRIED, AND WHAT IS NOT
--------------------------------
Only the SUBJECT carries between turns. "and the options?" is about whatever
was just discussed; that is genuinely elliptical speech and refusing to
resolve it would make the desk feel stupid.

What does NOT carry is the verdict. Re-answering "should I buy it?" from a
cached read would quietly serve yesterday's price as today's advice, and the
user has no way to see that it happened. Every turn re-runs the specialists.

Sessions live in memory with a hard cap. This is a research desk, not a
messaging product: losing a conversation costs a re-ask, while persisting one
would put a user's holdings and questions into a store nobody asked for.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

MAX_SESSIONS = 200
MAX_TURNS = 40
TTL_SEC = 60 * 60 * 6

_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _prune(now: float) -> None:
    dead = [k for k, v in _sessions.items() if now - v["touched"] > TTL_SEC]
    for k in dead:
        _sessions.pop(k, None)
    if len(_sessions) > MAX_SESSIONS:
        for k, _ in sorted(_sessions.items(),
                           key=lambda kv: kv[1]["touched"])[:len(_sessions) - MAX_SESSIONS]:
            _sessions.pop(k, None)


def get(session_id: Optional[str]) -> Dict[str, Any]:
    now = time.time()
    with _lock:
        _prune(now)
        if session_id and session_id in _sessions:
            s = _sessions[session_id]
            s["touched"] = now
            return s
        sid = session_id or uuid.uuid4().hex[:16]
        s = {"id": sid, "created": now, "touched": now,
             "subject": None, "turns": []}
        _sessions[sid] = s
        return s


def record(session: Dict[str, Any], user_text: str, parsed: Dict[str, Any],
           reply: Dict[str, Any]) -> None:
    with _lock:
        session["touched"] = time.time()
        # The subject advances only when a turn actually resolved one from its
        # OWN text. A turn that merely inherited the subject must not re-assert
        # it, or a mis-parse would pin the conversation to the wrong name for
        # the rest of the session.
        if parsed.get("symbols") and not parsed.get("symbols_from_context"):
            session["subject"] = parsed["symbols"][0]
        session["turns"].append({
            "at": time.time(),
            "user": user_text,
            "kind": parsed.get("kind"),
            "capabilities": parsed.get("capabilities"),
            "symbols": parsed.get("symbols"),
            "headline": reply.get("headline"),
        })
        if len(session["turns"]) > MAX_TURNS:
            session["turns"] = session["turns"][-MAX_TURNS:]


def public(session: Dict[str, Any]) -> Dict[str, Any]:
    return {"session_id": session["id"], "subject": session.get("subject"),
            "turns": len(session.get("turns") or [])}
