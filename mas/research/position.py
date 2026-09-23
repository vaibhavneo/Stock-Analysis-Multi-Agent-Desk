"""Position context, taken from what the user actually said.

THE GAP THIS CLOSES
-------------------
"I own 44 shares at $197.80. Should I add?" reached the engine as a bare
question. The numbers were parsed by nothing and propagated nowhere, so the
answer came back with POSITION_CONTEXT_NOT_PROVIDED and both ownership
branches — while the user had just supplied the context in the sentence.

NOTHING IS EVER INFERRED
------------------------
A field is extracted only when the user stated it. Shares without a cost
basis yields shares and an explicitly absent cost, never a guessed one —
`decision/position.py` already refuses to assume ownership, and guessing here
would defeat that from the outside. What IS produced is an explicit list of
what is still missing, so the orchestrator can ask once rather than either
inventing it or asking repeatedly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# "44 shares", "200 sh", "44 units"
_SHARES = re.compile(
    r"(?<![\w.])(\d{1,9}(?:[.,]\d+)?)\s*(?:shares?|sh\b|units?|contracts?)", re.I)
# "at $197.80", "from 180", "cost basis 42.5", "@ 197.8"
_COST = re.compile(
    r"(?:\bat\b|\bfrom\b|@|\bcost\s*basis\b|\bbasis\s*(?:of|is)?|\baverage\b"
    r"|\bavg\b|\bpaid\b)\s*\$?\s*(\d{1,9}(?:\.\d+)?)", re.I)
# "$12,500 position", "position worth 12500"
_VALUE = re.compile(
    r"\$\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,9}(?:\.\d+)?)\s*"
    r"(?:position|invested|worth|in\s+it)", re.I)
# "portfolio is 250k", "my portfolio of $250,000"
_PORTFOLIO = re.compile(
    r"portfolio\s*(?:is|of|=|worth)?\s*\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|m)?",
    re.I)

# Phrases that assert ownership without any number.
_OWNS = re.compile(
    r"\bi\s+(own|hold|have|bought|am\s+long|'m\s+long)\b"
    r"|\bmy\s+(position|shares|holding|stake|book)\b"
    r"|\bi'?m\s+(long|in)\b"
    r"|\bmy\s+[A-Z]{2,5}\b", re.I)

# "my position on this is that…" is an opinion, not a holding. Without this
# the ownership phrase matcher reads a figure of speech as a portfolio.
_OPINION = re.compile(r"\bmy\s+position\s+(on|is)\b.{0,30}\b(is|that)\b", re.I)


def _num(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return None


def extract(text: str) -> Dict[str, Any]:
    """Position context stated in the utterance. Never infers."""
    raw = text or ""
    out: Dict[str, Any] = {
        "stated": False, "owns": None, "shares": None, "avg_cost": None,
        "position_value": None, "portfolio_value": None,
        "missing": [], "source": "the message you just sent", "notes": [],
    }

    opinion = bool(_OPINION.search(raw))
    m = _SHARES.search(raw)
    if m:
        out["shares"] = _num(m.group(1))
    m = _COST.search(raw)
    if m:
        out["avg_cost"] = _num(m.group(1))
    m = _VALUE.search(raw)
    if m:
        out["position_value"] = _num(m.group(1))
    m = _PORTFOLIO.search(raw)
    if m:
        v = _num(m.group(1))
        if v is not None:
            mult = {"k": 1_000, "m": 1_000_000}.get((m.group(2) or "").lower(), 1)
            out["portfolio_value"] = v * mult

    has_numbers = any(out[k] is not None for k in
                      ("shares", "avg_cost", "position_value"))
    owns_phrase = bool(_OWNS.search(raw)) and not opinion
    if opinion:
        out["notes"].append(
            "\"my position\" here reads as an opinion, not a holding, so no "
            "ownership was inferred from it.")

    out["owns"] = True if (has_numbers or owns_phrase) else None
    out["stated"] = has_numbers or owns_phrase

    if out["owns"]:
        for field, label in (("shares", "share count"),
                             ("avg_cost", "average cost"),
                             ("portfolio_value", "portfolio value")):
            if out[field] is None:
                out["missing"].append(label)
    return out


def merge(stated: Dict[str, Any],
          supplied: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Combine what was said with what the application already holds.

    The message wins on conflict — a user restating their basis is correcting
    the record, not contradicting it. Asking again for something already in
    either source is the behaviour this exists to prevent.
    """
    out = dict(stated or {})
    src = dict(supplied or {})
    for field in ("shares", "avg_cost", "portfolio_value", "position_value"):
        if out.get(field) is None and src.get(field) is not None:
            out[field] = src[field]
            out.setdefault("from_application", []).append(field)
    if any(out.get(f) is not None for f in
           ("shares", "avg_cost", "position_value")):
        out["owns"] = True
        out["stated"] = True
    out["missing"] = [lbl for f, lbl in (("shares", "share count"),
                                         ("avg_cost", "average cost"),
                                         ("portfolio_value", "portfolio value"))
                      if out.get(f) is None] if out.get("owns") else []
    return out


def to_engine(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The shape `decision/` expects, or None when ownership is unknown.

    `avg_cost` is the engine's gate for treating a position as real, so a
    context with shares but no basis returns None rather than a partial dict
    the engine would read as an assumed holding.
    """
    if not ctx or not ctx.get("owns") or ctx.get("avg_cost") is None:
        return None
    out: Dict[str, Any] = {"avg_cost": float(ctx["avg_cost"])}
    for k in ("shares", "portfolio_value"):
        if ctx.get(k) is not None:
            out[k] = float(ctx[k])
    return out


def describe(ctx: Dict[str, Any]) -> str:
    if not ctx or not ctx.get("stated"):
        return "No position was stated, so nothing about ownership is assumed."
    bits: List[str] = []
    if ctx.get("shares") is not None:
        bits.append(f"{ctx['shares']:g} shares")
    if ctx.get("avg_cost") is not None:
        bits.append(f"at ${ctx['avg_cost']:,.2f}")
    if ctx.get("portfolio_value") is not None:
        bits.append(f"in a ${ctx['portfolio_value']:,.0f} portfolio")
    head = ("You hold " + " ".join(bits) + ".") if bits else "You hold this."
    if ctx.get("missing"):
        head += (" Still unknown: " + ", ".join(ctx["missing"])
                 + " — those are asked for once, never assumed.")
    return head
