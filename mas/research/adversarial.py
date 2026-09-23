"""Argue against the answer, from the evidence that produced it.

WHAT THIS IS NOT
----------------
It is not a bearish-prose generator. Asking a model to "give the bear case"
produces fluent opposition unconnected to anything measured, which is worse
than no challenge because it reads like analysis.

WHAT IT DOES
------------
It works on the evidence ledger and the synthesis, and every challenge it
raises names the item it came from. Five questions, all answerable from
structure:

  1. What is the single strongest piece of evidence against the conclusion?
  2. What evidence is MISSING that would most change it?
  3. Where could this be a false positive — thin samples, one source, a
     derived level standing in for a traded one?
  4. What historical precedent argues the other way?
  5. What exact condition would falsify it?

A challenge with no evidence behind it is not produced at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evidence import Ledger, Item, TIER_RANK, STALE

BULLISH, BEARISH = "BULLISH", "BEARISH"


def challenge(result: Dict[str, Any]) -> Dict[str, Any]:
    """The strongest case against the current reading. Never invents one."""
    syn = result.get("synthesis") or {}
    items = [i for i in ((result.get("evidence") or {}).get("items") or [])]
    plan = result.get("plan") or {}

    leaning = (BULLISH if (syn.get("bullish_weight") or 0)
               > (syn.get("bearish_weight") or 0) else BEARISH)
    opposite = BEARISH if leaning == BULLISH else BULLISH

    out: Dict[str, Any] = {
        "leaning": leaning, "challenges": [], "missing_that_would_matter": [],
        "false_positive_risks": [], "falsifiers": [],
        "statement": "", "had_material": False,
    }

    # 1. Strongest opposing evidence that actually exists.
    against = sorted([i for i in items if i.get("direction") == opposite],
                     key=lambda i: -(i.get("weight") or 0))
    if against:
        top = against[0]
        out["challenges"].append({
            "kind": "OPPOSING_EVIDENCE",
            "statement": (f"{top['source']} reads the other way: "
                          f"{top['statement']}"),
            "from": top["key"], "weight": top.get("weight")})
    else:
        out["challenges"].append({
            "kind": "NO_OPPOSING_EVIDENCE",
            "statement": ("Nothing measured here argues the other way — which "
                          "is itself worth noting, because a reading with no "
                          "opposition usually means the opposing evidence was "
                          "not looked for rather than that it does not exist."),
            "from": None, "weight": None})

    # 2. Missing evidence, ranked by whether the plan called it required.
    required = set(plan.get("required_evidence") or [])
    unavailable = [i for i in items if "unavailable" in (i.get("flags") or [])]
    for i in unavailable:
        out["missing_that_would_matter"].append({
            "what": i["key"], "why": i["statement"],
            "required": any(r in i["key"] for r in required)})
    for m in (plan.get("missing") or []):
        out["missing_that_would_matter"].append(
            {"what": m["what"], "why": m["why"], "required": True})

    # 3. False-positive risks, each tied to a measured weakness.
    for i in items:
        flags = i.get("flags") or []
        if "insufficient_sample" in flags:
            out["false_positive_risks"].append({
                "what": i["key"],
                "why": ("rests on too little history to distinguish skill "
                        "from luck")})
        if "no_demonstrated_edge" in flags:
            out["false_positive_risks"].append({
                "what": i["key"],
                "why": ("the strategy did not beat simply holding, so a "
                        "directional read here is a thesis and not a "
                        "demonstrated edge")})
        if "derived_only" in flags:
            out["false_positive_risks"].append({
                "what": i["key"],
                "why": ("every level is arithmetic on the current price "
                        "rather than a level the market has defended")})
        if i.get("freshness") == STALE:
            out["false_positive_risks"].append({
                "what": i["key"], "why": "it is working from stale data"})
    if (syn.get("n_directional_sources") or 0) < 2:
        out["false_positive_risks"].append({
            "what": "synthesis",
            "why": ("only one source takes a direction, so the reading rests "
                    "on a single measurement with nothing to corroborate it")})

    # 4. Falsifiers, from levels and catalysts that exist.
    by_key = {i["key"]: i for i in items}
    if "research:levels" in by_key:
        out["falsifiers"].append({
            "condition": "a close through the nearest sourced level against "
                         "the reading",
            "from": "research:levels"})
    cat = by_key.get("catalyst:next")
    if cat and isinstance(cat.get("value"), (int, float)):
        hz = (plan.get("horizon") or {}).get("days")
        inside = hz is not None and cat["value"] <= hz
        out["falsifiers"].append({
            "condition": (f"{cat['statement']} resolving against the reading"
                          + ("" if inside else
                             " — though it falls outside this horizon, so it "
                             "cannot falsify a position closed before it")),
            "from": "catalyst:next"})

    out["had_material"] = bool(against or unavailable
                               or out["false_positive_risks"])

    lines: List[str] = []
    lines.append(out["challenges"][0]["statement"])
    if out["false_positive_risks"]:
        top = out["false_positive_risks"][0]
        lines.append(f"Where this could be a false positive: {top['why']}.")
    if out["missing_that_would_matter"]:
        m = out["missing_that_would_matter"][0]
        lines.append(f"The absence that would most change it: {m['what']}.")
    if out["falsifiers"]:
        lines.append(f"It would be falsified by {out['falsifiers'][0]['condition']}.")
    out["statement"] = " ".join(lines)
    return out
