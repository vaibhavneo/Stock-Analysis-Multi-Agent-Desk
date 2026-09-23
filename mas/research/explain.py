"""Pass two: turn a finished research result into prose.

THE SEPARATION
--------------
Pass one — planning, tool selection, execution, evidence, synthesis, decision
— uses NO language model at all. Every number is fixed before this module is
reached. So the boundary Phase 33 asks for is not merely enforced here; it is
structural, because there is nothing for a model to influence by the time it
is called.

What the model may do: read the structured result and say it in better
English. What it may not do is add a number, a level, a probability, a
catalyst or a source. `decision/narrative.py` already implements exactly that
guard — an allowlist built from the object's own values — so this reuses it
rather than inventing a second, weaker one.

Without a key, the deterministic composition is the answer. It is not a
degraded mode; it is the same content in plainer sentences.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from . import reply as _reply


def deterministic(result: Dict[str, Any]) -> Dict[str, Any]:
    """The answer, composed without a model. Always available."""
    out = _reply.compose(result)
    out["source"] = "deterministic"
    out["llm_used"] = False
    return out


def explain(result: Dict[str, Any], use_llm: bool = False,
            timeout_sec: float = 30.0) -> Dict[str, Any]:
    """Pass two. Falls back to the deterministic composition on any problem.

    `use_llm` is opt-in, not a default: the deterministic version is already
    the complete answer, and a model call adds latency and a failure mode to
    something that had neither.
    """
    base = deterministic(result)
    if not use_llm:
        base["note"] = ("Composed deterministically. No model was asked, and "
                        "none was needed — every number was fixed before this "
                        "step.")
        return base

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        base["note"] = ("No model key is set, so this is the deterministic "
                        "composition. It is the same content, not a reduced "
                        "version of it.")
        return base

    try:
        from decision.narrative import generate_narrative
        # The SAME guard the brief uses: a numeric allowlist built from the
        # object's own values, so a number the structure does not contain
        # cannot appear in the prose.
        payload = {
            "ticker": (result.get("plan") or {}).get("symbols", [None])[0],
            "synthesis": result.get("synthesis"),
            "decision": result.get("decision"),
            "evidence": result.get("evidence"),
            "change": result.get("change"),
        }
        nar = generate_narrative(payload)
        if nar and nar.get("text"):
            base["blocks"] = [{"capability": None,
                               "lines": [nar["text"]]}] + base["blocks"]
            base["source"] = "llm_over_structure"
            base["llm_used"] = True
            base["guard"] = nar.get("guard") or "numeric allowlist applied"
            base["note"] = ("Written by a model over the structure above, then "
                            "checked number by number against it.")
        else:
            base["note"] = ("The model produced nothing usable, so this is the "
                            "deterministic composition.")
    except Exception as e:
        base["note"] = (f"The explanation pass failed ({type(e).__name__}), so "
                        f"this is the deterministic composition. The research "
                        f"itself is unaffected — it completed before this step.")
    return base
