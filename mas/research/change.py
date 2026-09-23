"""What is different since the last time this question was asked.

An analysis that cannot say what changed forces the reader to diff two briefs
in their head, and the thing they most need — "the technical evidence went
from neutral to constructive, and here is why" — is exactly what a fresh
snapshot cannot express on its own.

WHAT COUNTS AS A CHANGE
-----------------------
Not any difference in a number. A composite moving 61.4 → 61.6 is noise, and
reporting it as change would bury the one line that matters. A change is
recorded when it crosses a boundary the reader acts on: a direction flips, an
item appears or disappears, freshness decays, or a value moves more than its
own threshold.

INVALIDATED is kept separate from DETERIORATED. Something that got worse is
still the same claim; something invalidated is no longer the claim at all,
and collapsing them loses the distinction between "the case weakened" and
"the case is gone".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

NEW = "NEW"
IMPROVED = "IMPROVED"
DETERIORATED = "DETERIORATED"
UNCHANGED = "UNCHANGED"
INVALIDATED = "INVALIDATED"
STALE = "STALE"
DISAPPEARED = "DISAPPEARED"

# How much a numeric value must move before it is worth mentioning, as a
# fraction of the previous value. Below this it is noise wearing a headline.
MATERIAL_MOVE = 0.05

DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL", "NOT_DIRECTIONAL")


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _classify(prev: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    pd_, cd = prev.get("direction"), cur.get("direction")
    pv, cv = _num(prev.get("value")), _num(cur.get("value"))

    # A direction flip is the change readers act on, so it outranks magnitude.
    if pd_ != cd and pd_ in DIRECTIONS and cd in DIRECTIONS:
        if pd_ == "NOT_DIRECTIONAL" or cd == "NOT_DIRECTIONAL":
            kind = DETERIORATED if cd == "NOT_DIRECTIONAL" else IMPROVED
        elif pd_ in ("BEARISH", "NEUTRAL") and cd == "BULLISH":
            kind = IMPROVED
        elif pd_ in ("BULLISH", "NEUTRAL") and cd == "BEARISH":
            kind = DETERIORATED
        else:
            kind = DETERIORATED if cd == "NEUTRAL" else IMPROVED
        return {"kind": kind,
                "why": (f"direction moved from {pd_.lower()} to {cd.lower()}"),
                "from": pd_, "to": cd}

    if cur.get("freshness") == "STALE" and prev.get("freshness") != "STALE":
        return {"kind": STALE, "why": "the source has gone stale since last time",
                "from": prev.get("freshness"), "to": "STALE"}

    if not cur.get("usable_for_decision") and prev.get("usable_for_decision"):
        return {"kind": INVALIDATED,
                "why": ("it no longer qualifies as decision evidence — this is "
                        "not the same claim weakened, it is the claim gone"),
                "from": True, "to": False}

    if pv is not None and cv is not None and pv != 0:
        move = (cv - pv) / abs(pv)
        if abs(move) >= MATERIAL_MOVE:
            return {"kind": IMPROVED if move > 0 else DETERIORATED,
                    "why": f"the value moved {move:+.1%}, past the "
                           f"{MATERIAL_MOVE:.0%} threshold for materiality",
                    "from": pv, "to": cv}
        return {"kind": UNCHANGED,
                "why": f"moved {move:+.1%}, inside the noise threshold",
                "from": pv, "to": cv}

    return {"kind": UNCHANGED, "why": "no material difference",
            "from": prev.get("value"), "to": cur.get("value")}


def compare(previous: Optional[Dict[str, Any]],
            current: Dict[str, Any]) -> Dict[str, Any]:
    """Diff two research results. Never raises."""
    cur_items = {i["key"]: i for i in
                 ((current.get("evidence") or {}).get("items") or [])}

    if not previous:
        return {"has_previous": False, "changes": [], "n_material": 0,
                "statement": ("No earlier analysis of this name to compare "
                              "against, so nothing here is a change — it is a "
                              "first reading."),
                "previous_at": None}

    prev_items = {i["key"]: i for i in
                  ((previous.get("evidence") or {}).get("items") or [])}

    changes: List[Dict[str, Any]] = []
    for key, cur in cur_items.items():
        if key not in prev_items:
            changes.append({"key": key, "source": cur.get("source"),
                            "kind": NEW, "why": "it was not measured last time",
                            "statement": cur.get("statement")})
            continue
        cls = _classify(prev_items[key], cur)
        changes.append({"key": key, "source": cur.get("source"), **cls,
                        "statement": cur.get("statement")})

    for key, prev in prev_items.items():
        if key not in cur_items:
            changes.append({
                "key": key, "source": prev.get("source"), "kind": DISAPPEARED,
                "why": ("it was measured last time and is absent now, which is "
                        "a gap rather than a neutral reading"),
                "statement": prev.get("statement")})

    material = [c for c in changes if c["kind"] not in (UNCHANGED,)]

    prev_syn = (previous.get("synthesis") or {}).get("consensus")
    cur_syn = (current.get("synthesis") or {}).get("consensus")
    consensus_moved = prev_syn != cur_syn

    lines: List[str] = []
    if consensus_moved:
        lines.append(f"The overall read moved from "
                     f"{str(prev_syn or 'none').replace('_', ' ').lower()} to "
                     f"{str(cur_syn or 'none').replace('_', ' ').lower()}.")
    flips = [c for c in material if c["kind"] in (IMPROVED, DETERIORATED)
             and c.get("from") in DIRECTIONS]
    for c in flips[:3]:
        lines.append(f"{c['source']} evidence changed from "
                     f"{str(c['from']).lower()} to {str(c['to']).lower()}.")
    gone = [c for c in material if c["kind"] in (DISAPPEARED, INVALIDATED)]
    if gone:
        lines.append(f"No longer available: "
                     f"{', '.join(c['source'] for c in gone[:3])}.")
    fresh_new = [c for c in material if c["kind"] == NEW]
    if fresh_new:
        lines.append(f"New since last time: "
                     f"{', '.join(c['source'] for c in fresh_new[:3])}.")
    if not lines:
        lines.append("Nothing material changed since the last analysis — the "
                     "numbers moved inside their noise thresholds.")

    return {"has_previous": True, "changes": changes,
            "n_material": len(material),
            "consensus_moved": consensus_moved,
            "previous_consensus": prev_syn, "current_consensus": cur_syn,
            "statement": " ".join(lines),
            "previous_at": previous.get("as_of")}
