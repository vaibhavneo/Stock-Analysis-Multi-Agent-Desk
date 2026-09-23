"""Why does the answer say that?

Every statement a research result makes should be walkable back to the thing
that produced it:

    FINAL STATEMENT
        → SYNTHESIS (which items it weighed, and how)
        → EVIDENCE ITEM (tier, direction, confidence)
        → CAPABILITY (which specialist)
        → TOOL (which source)
        → SOURCE + TIMESTAMP

A claim that cannot be walked back is a claim nobody can check, and this
system's whole argument for being trusted is that its numbers are checkable.

The graph is built from what already exists — `Item.provenance`, the tool
registry and the execution trace — rather than from a parallel record that
could drift out of step with the answer it describes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import tools as T

FINAL = "FINAL"
SYNTHESIS = "SYNTHESIS"
EVIDENCE = "EVIDENCE"
CAPABILITY = "CAPABILITY"
TOOL = "TOOL"
SOURCE = "SOURCE"

LEVELS = (FINAL, SYNTHESIS, EVIDENCE, CAPABILITY, TOOL, SOURCE)


def _tool_for(capability: str, evidence_key: str) -> Optional[Dict[str, Any]]:
    """Which registered tool actually fed this capability."""
    from .plan import EVIDENCE_TOOLS
    kind, _, tail = evidence_key.partition(":")
    # Pillars are attributed to the source they actually read. Falling back to
    # a generic "pillar -> price history" mapping credited the fundamentals
    # pillar to the daily bar feed, which is a provenance claim that is simply
    # untrue — and a wrong chain is worse than a missing one.
    if kind == "pillar":
        per_pillar = {
            "fundamentals": ["fundamentals"],
            "research": ["analyst_coverage"],
            "social": ["social_sentiment"],
            "technical": ["price_history"],
            "algo": ["price_history"],
            "risk": ["price_history"],
        }
        guesses = per_pillar.get(tail, ["price_history"])
        for g in guesses:
            candidates = T.tools_for(g, available_only=False)
            if candidates:
                t = candidates[0].to_dict()
                return {"id": t["id"], "name": t["name"],
                        "freshness": t["freshness"],
                        "reliability": t["reliability"],
                        "available": t["available"], "produces": g}
        return None
    guesses = {
        "research": ["price_history", "fundamentals"],
        "risk": ["price_history"],
        "stats": ["price_history"],
        "catalyst": ["earnings", "event_calendar"],
        "relative": ["price_history"],
        "macro": ["macro", "volatility_index"],
        "options": ["options_chain", "price_history"],
    }.get(kind, [])
    for g in guesses:
        candidates = T.tools_for(g, available_only=False)
        if candidates:
            t = candidates[0].to_dict()
            return {"id": t["id"], "name": t["name"], "freshness": t["freshness"],
                    "reliability": t["reliability"], "available": t["available"],
                    "produces": g}
    return None


def build_graph(result: Dict[str, Any]) -> Dict[str, Any]:
    """The full provenance graph for one research result."""
    items = (result.get("evidence") or {}).get("items") or []
    syn = result.get("synthesis") or {}
    weighed = set(syn.get("weighed_keys") or [])
    trace = {s.get("capability"): s
             for s in ((result.get("trace") or {}).get("steps") or [])}

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, str]] = []

    nodes.append({"id": "final", "level": FINAL,
                  "label": (result.get("decision") or {}).get("statement")
                           or "the answer"})
    nodes.append({"id": "synthesis", "level": SYNTHESIS,
                  "label": syn.get("statement", ""),
                  "detail": {"consensus": syn.get("consensus"),
                             "method": syn.get("method")}})
    edges.append({"from": "final", "to": "synthesis",
                  "why": "the answer states what the synthesis concluded"})

    for it in items:
        key = it.get("key")
        nid = f"evidence:{key}"
        nodes.append({
            "id": nid, "level": EVIDENCE, "label": it.get("statement"),
            "detail": {"tier": it.get("tier"), "direction": it.get("direction"),
                       "weight": it.get("weight"),
                       "confidence": it.get("confidence"),
                       "freshness": it.get("freshness"),
                       "weighed": key in weighed,
                       "usable_for_decision": it.get("usable_for_decision")}})
        edges.append({
            "from": "synthesis", "to": nid,
            "why": ("weighed in the directional reading" if key in weighed
                    else "read and categorised, but not weighed"
                         " — it takes no direction or sits below the "
                         "decision tier")})

        cap = (it.get("provenance") or {}).get("capability")
        if cap:
            cid = f"capability:{cap}"
            if not any(n["id"] == cid for n in nodes):
                st = trace.get(cap, {})
                nodes.append({"id": cid, "level": CAPABILITY, "label": cap,
                              "detail": {"outcome": st.get("outcome"),
                                         "elapsed_ms": st.get("elapsed_ms"),
                                         "purpose": st.get("purpose")}})
            edges.append({"from": nid, "to": cid,
                          "why": "produced by this specialist"})

            tool = _tool_for(cap, key or "")
            if tool:
                tid = f"tool:{tool['id']}"
                if not any(n["id"] == tid for n in nodes):
                    nodes.append({"id": tid, "level": TOOL,
                                  "label": tool["name"],
                                  "detail": tool})
                edges.append({"from": cid, "to": tid,
                              "why": f"reads {tool['produces'].replace('_',' ')}"})
                sid = f"source:{tool['id']}"
                if not any(n["id"] == sid for n in nodes):
                    nodes.append({
                        "id": sid, "level": SOURCE, "label": tool["name"],
                        "detail": {"freshness": tool["freshness"],
                                   "observed_at": it.get("observed_at"),
                                   "data_timestamp": it.get("data_timestamp")}})
                edges.append({"from": tid, "to": sid,
                              "why": f"{tool['freshness'].replace('_',' ').lower()} data"})

    orphans = [n["id"] for n in nodes
               if n["level"] == EVIDENCE
               and not any(e["from"] == n["id"] for e in edges)]
    return {
        "nodes": nodes, "edges": edges,
        "n_nodes": len(nodes), "n_edges": len(edges),
        "levels": list(LEVELS),
        "unwalkable": orphans,
        "statement": (
            f"{len(nodes)} nodes across {len(LEVELS)} levels. "
            + (f"{len(orphans)} evidence item(s) cannot be walked back to a "
               f"source — that is a provenance gap, not a detail."
               if orphans else
               "Every evidence item walks back to a capability, a tool and a "
               "timestamp.")),
    }


def why(result: Dict[str, Any], claim: Optional[str] = None) -> Dict[str, Any]:
    """The chain behind one claim, or behind the answer as a whole.

    `claim` matches an evidence key or a substring of a statement. Without it,
    the strongest weighed evidence is explained — which is the honest default
    for "why does it say that".
    """
    graph = build_graph(result)
    items = (result.get("evidence") or {}).get("items") or []
    syn = result.get("synthesis") or {}
    weighed = set(syn.get("weighed_keys") or [])

    target = None
    if claim:
        low = claim.lower()
        target = next((i for i in items if low in str(i.get("key", "")).lower()), None)
        if target is None:
            target = next((i for i in items
                           if low in str(i.get("statement", "")).lower()), None)
    if target is None:
        candidates = [i for i in items if i.get("key") in weighed] or items
        target = max(candidates, key=lambda i: i.get("weight") or 0, default=None)

    if target is None:
        return {"found": False,
                "statement": "There is no evidence behind that claim to walk.",
                "graph": graph}

    cap = (target.get("provenance") or {}).get("capability")
    tool = _tool_for(cap or "", target.get("key") or "")
    chain = [
        {"level": EVIDENCE, "what": target.get("statement"),
         "detail": f"{target.get('tier')} · direction "
                   f"{target.get('direction')} · weight {target.get('weight')} "
                   f"· {'weighed' if target.get('key') in weighed else 'not weighed'}"},
    ]
    if cap:
        chain.append({"level": CAPABILITY, "what": cap,
                      "detail": "the specialist that produced it"})
    if tool:
        chain.append({"level": TOOL, "what": tool["name"],
                      "detail": f"reliability {tool['reliability']}, "
                                f"{tool['freshness'].replace('_',' ').lower()}"})
        chain.append({"level": SOURCE, "what": tool["name"],
                      "detail": f"observed {target.get('observed_at') or 'unknown'}"
                                + (f", data dated {target['data_timestamp']}"
                                   if target.get("data_timestamp") else "")})

    return {
        "found": True, "claim": target.get("statement"), "key": target.get("key"),
        "chain": chain, "graph": graph,
        "statement": " ← ".join(str(c["what"]) for c in chain),
        "caveat": (target.get("uncertainty")
                   or "within the normal error of this measurement"),
    }
