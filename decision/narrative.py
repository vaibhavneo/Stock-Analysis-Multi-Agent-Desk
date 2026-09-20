"""
LLM synthesis layer (Phase 19) — the model explains; it never decides.

The structured Decision Intelligence object is COMPLETE before this module
runs. Nothing here can write back into it. The model's only job is to turn a
finished object into readable prose that answers the fourteen questions the
specification lists, citing the structured evidence it drew each statement from.

The interesting engineering is not the prompt — it is `validate_narrative()`.

Telling a model "do not invent price levels" is a request. Checking is a
guarantee. So every number in the generated text is extracted and matched
against an allowlist built from the structured object itself. A number the
object does not contain is `unsupported`, and unsupported numbers make the
narrative fail validation. A failing narrative is not shown as analysis; the
structured object is shown on its own, which is always sufficient.

The allowlist is built from the object's own values (and their common
renderings — 2 decimal places, 1 decimal place, integer, percentage), plus a
small set of universally-safe integers (small counts, years already present in
the object). It is deliberately conservative: a false positive costs a
regenerated paragraph, a false negative ships a fabricated price level.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

SYSTEM_PROMPT = """You are writing the explanation section of a research decision brief.

Absolute rules:
- Every number you write MUST already appear in the structured data you are given.
  Do not compute new numbers, do not round to a different value, do not estimate,
  do not interpolate. If a number is not in the data, describe it in words instead.
- Never state a probability. The data tells you whether any probability is
  calibrated; where it is not, say "probability not calibrated".
- Never claim a statistical edge, significance, or that something is "proven"
  unless the data's statistical_edge.demonstrated is true.
- Never invent a catalyst, an event date, a financial metric, a technical signal,
  a historical result, or a price level.
- Never recommend a transaction. Describe the decision state and its conditions.
- Cite the structured field you are drawing on, in square brackets, after each
  claim — e.g. [thesis.statement], [level_map.nearest_support], [conflict].

You are explaining a decision that has already been made deterministically. You
are not making it, revising it, or softening it."""

QUESTIONS = [
    "What is happening?",
    "Why is it happening, according to the evidence?",
    "What is the dominant thesis?",
    "What disagrees with it?",
    "What would the market have to do to confirm it?",
    "What would invalidate it?",
    "What are the attractive entry conditions?",
    "Under what conditions would adding be justified?",
    "Under what conditions would reducing be justified?",
    "What are the exit conditions?",
    "Which catalysts matter?",
    "What is unknown?",
    "What should be monitored?",
    "What would change the system's mind?",
]

# Numbers that are safe regardless of the object: small counts a writer will
# naturally use ("three reasons", "the first of two"), and 0/1/100 as rhetorical
# anchors ("0%", "100% of the weight").
UNIVERSALLY_SAFE = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 50.0}

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# Ordered-list and heading markers at the start of a line. The prompt asks for
# fourteen numbered answers, so the model writes "8. What are the conditions
# for adding?" — and "8" is not a claim about the security. Stripping the
# marker is exact; adding 6..14 to the safe set would have quietly licensed a
# fabricated "8%" as well.
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[#*\-]{0,6}[ \t]*)?\d{1,2}[.)][ \t]+", re.M)


def _strip_list_markers(text: str) -> str:
    return _LIST_MARKER_RE.sub("", text)


def _walk_numbers(obj: Any, out: Set[float], depth: int = 0) -> None:
    """Collect every numeric value anywhere in the structured object, including
    numbers embedded in its prose fields — the engine's own sentences already
    quote its own numbers, and a writer quoting those sentences must not be
    penalised for it."""
    if depth > 12:
        return
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        return
    if isinstance(obj, str):
        for m in _NUMBER_RE.findall(obj):
            try:
                out.add(float(m.replace(",", "")))
            except ValueError:
                pass
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out, depth + 1)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _walk_numbers(v, out, depth + 1)


def allowed_numbers(decision: Dict[str, Any]) -> Set[float]:
    """Every number the narrative is permitted to state, plus the renderings a
    writer would reasonably produce from them (2dp, 1dp, integer, and the
    percentage form of a 0-1 ratio)."""
    raw: Set[float] = set()
    _walk_numbers(decision, raw)

    allowed: Set[float] = set(UNIVERSALLY_SAFE)
    for v in raw:
        if v != v:                       # NaN
            continue
        allowed.add(v)
        allowed.add(round(v, 2))
        allowed.add(round(v, 1))
        allowed.add(float(round(v)))
        allowed.add(abs(v))
        allowed.add(round(abs(v), 2))
        allowed.add(round(abs(v), 1))
        allowed.add(float(round(abs(v))))
        # A ratio in 0..1 is routinely written as a percentage.
        if 0.0 <= abs(v) <= 1.0:
            allowed.add(round(abs(v) * 100, 1))
            allowed.add(float(round(abs(v) * 100)))
    return allowed


def validate_narrative(text: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    """Check the generated prose against the structured object.

    Returns `ok=False` with the offending values when the text states a number
    the object does not contain, or uses banned certainty language, or asserts a
    proven edge the gate did not grant.
    """
    if not text:
        return {"ok": False, "reason": "empty narrative", "unsupported_numbers": [],
                "banned_phrases": [], "n_numbers_checked": 0}

    allowed = allowed_numbers(decision)
    unsupported: List[str] = []
    checked = 0
    for m in _NUMBER_RE.findall(_strip_list_markers(text)):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        checked += 1
        if v in allowed or round(v, 2) in allowed or round(abs(v), 2) in allowed:
            continue
        unsupported.append(m)

    lowered = text.lower()
    banned: List[str] = []
    # Certainty language. The system never has grounds for any of these.
    for phrase in ("guaranteed", "will certainly", "is certain to", "risk-free",
                   "cannot lose", "sure thing", "definitely will"):
        if phrase in lowered:
            banned.append(phrase)

    edge_demonstrated = bool((decision.get("statistical_edge") or {}).get("demonstrated"))
    if not edge_demonstrated:
        for phrase in ("proven edge", "statistically significant", "demonstrated edge",
                       "statistically proven"):
            # Allowed only in an explicitly negating context, which the engine's
            # own honesty statements use ("no demonstrated edge").
            for match in re.finditer(re.escape(phrase), lowered):
                window = lowered[max(0, match.start() - 40):match.start()]
                if not any(neg in window for neg in ("no ", "not ", "without ", "lacks ",
                                                     "absence of ", "isn't ", "is not ")):
                    banned.append(phrase)

    # Probability claims where none is calibrated.
    if not (decision.get("scenarios") or {}).get("any_probability_stated"):
        if re.search(r"\b\d{1,3}\s?%\s+(chance|probability|likelihood|likely)", lowered) or \
           re.search(r"(probability|chance|odds)\s+of\s+\d", lowered):
            banned.append("numeric probability without calibration")

    return {
        "ok": not unsupported and not banned,
        "unsupported_numbers": sorted(set(unsupported)),
        "banned_phrases": sorted(set(banned)),
        "n_numbers_checked": checked,
        "reason": ("ok" if not unsupported and not banned else
                   "; ".join(filter(None, [
                       f"{len(set(unsupported))} number(s) not present in the structured data"
                       if unsupported else "",
                       f"banned language: {', '.join(sorted(set(banned)))}" if banned else ""]))),
    }


def _compact(decision: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of the object the model actually needs.

    Trimmed deliberately: a prompt carrying every evidence item's provenance
    dict is mostly tokens the writer cannot use, and a larger prompt means more
    surface for the model to paraphrase a number into a new one.
    """
    ds = decision.get("decision_state") or {}
    return {
        "ticker": decision.get("ticker"),
        "current_price": decision.get("current_price"),
        "horizon_days": decision.get("horizon_days"),
        "decision_state": {
            "headline": ds.get("headline_state"),
            "reason": ds.get("headline_reason"),
            "ownership": ds.get("ownership"),
            "if_owned": (ds.get("if_owned") or {}).get("state"),
            "if_owned_reason": (ds.get("if_owned") or {}).get("reason"),
            "if_not_owned": (ds.get("if_not_owned") or {}).get("state"),
            "if_not_owned_reason": (ds.get("if_not_owned") or {}).get("reason"),
        },
        "thesis": {k: (decision.get("thesis") or {}).get(k) for k in
                   ("direction", "strength", "statement", "supporting_evidence",
                    "opposing_evidence", "horizon_note")},
        "statistical_edge": {k: (decision.get("statistical_edge") or {}).get(k) for k in
                             ("level", "demonstrated", "verdict", "basis",
                              "sizing_consequence", "honesty")},
        "thesis_edge_relation": (decision.get("thesis_edge_relation") or {}).get("statement"),
        "horizon_read": (decision.get("horizon_read") or {}).get("summaries"),
        "conflict": {"summary": (decision.get("conflict") or {}).get("summary"),
                     "conflicts": [{"name": c.get("name"), "why": c.get("why"),
                                    "resolution": c.get("resolution")}
                                   for c in (decision.get("conflict") or {}).get("conflicts", [])[:4]]},
        "level_map": {"nearest_support": (decision.get("level_map") or {}).get("nearest_support"),
                      "nearest_resistance": (decision.get("level_map") or {}).get("nearest_resistance"),
                      "major_support": (decision.get("level_map") or {}).get("major_support"),
                      "major_resistance": (decision.get("level_map") or {}).get("major_resistance")},
        "entry": {"status": (decision.get("entry") or {}).get("status"),
                  "reason": (decision.get("entry") or {}).get("reason"),
                  "conditions": (decision.get("entry") or {}).get("conditions")},
        "sizing": decision.get("sizing"),
        "position_context": {"status": (decision.get("position_context") or {}).get("status"),
                             "message": (decision.get("position_context") or {}).get("message"),
                             "unrealized_pl_pct": (decision.get("position_context") or {}).get("unrealized_pl_pct")},
        "add_analysis": {"verdict": (decision.get("add_analysis") or {}).get("verdict"),
                         "headline": (decision.get("add_analysis") or {}).get("headline"),
                         "blockers": (decision.get("add_analysis") or {}).get("blockers"),
                         "rule": (decision.get("add_analysis") or {}).get("rule")},
        "playbook": {k: (decision.get("playbook") or {}).get(k) for k in
                     ("hold_conditions", "add_conditions", "reduce_conditions",
                      "exit_conditions", "do_not_add_conditions", "review_date")},
        "scenarios": [{k: s.get(k) for k in ("name", "thesis", "expected_direction",
                                             "conditions", "invalidation", "probability",
                                             "probability_basis")}
                      for s in (decision.get("scenarios") or {}).get("scenarios", [])],
        "mind_changers": decision.get("mind_changers"),
        "catalysts": {"status": (decision.get("catalysts") or {}).get("status"),
                      "waiting_for": (decision.get("catalysts") or {}).get("waiting_for"),
                      "upcoming": (decision.get("catalysts") or {}).get("upcoming", [])[:2]},
        "monitoring": decision.get("monitoring"),
        "confidence": {"decision_confidence": (decision.get("confidence") or {}).get("decision_confidence"),
                       "summary": (decision.get("confidence") or {}).get("summary"),
                       "capped_by": (decision.get("confidence") or {}).get("capped_by")},
        "quality": {"summary": (decision.get("quality") or {}).get("summary"),
                    "missing": (decision.get("quality") or {}).get("missing")},
        "statistical_honesty": decision.get("statistical_honesty"),
        "consistency": {"summary": (decision.get("consistency") or {}).get("summary")},
    }


def build_narrative_prompt(decision: Dict[str, Any]) -> Tuple[str, str]:
    """(system, user). The user message is the structured object plus the
    fourteen questions, and nothing else — no free-text context the model could
    treat as a licence to improvise."""
    import json
    body = json.dumps(_compact(decision), indent=1, default=str)
    questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(QUESTIONS))
    user = (
        f"Structured decision data for {decision.get('ticker')}:\n\n{body}\n\n"
        f"Write the explanation. Answer each of these, in order, with a short heading "
        f"for each:\n{questions}\n\n"
        f"Keep it tight — a few sentences per question. Cite the structured field in "
        f"square brackets after each substantive claim. Every number you write must "
        f"already appear above."
    )
    return SYSTEM_PROMPT, user


def generate_narrative(decision: Dict[str, Any], client: Any = None,
                       model: str = "deepseek-chat",
                       max_attempts: int = 2) -> Dict[str, Any]:
    """Generate and VALIDATE the narrative.

    Returns `{status, text, validation, attempts}`. `status`:
      OK              generated and validated
      REJECTED        generated but failed validation; `text` is withheld
      NO_API_KEY      no key configured — the structured object stands alone
      ERROR           the provider call failed

    The structured object is always sufficient on its own; a missing or rejected
    narrative degrades the page's readability, never its correctness.
    """
    import os
    system, user = build_narrative_prompt(decision)

    if client is None:
        key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return {"status": "NO_API_KEY", "text": None, "validation": None, "attempts": 0,
                    "message": ("No LLM key configured. The structured decision stands on its "
                                "own — the narrative is an explanation of it, never a source "
                                "for it.")}
        try:
            from agents.stock_agents import _get_client
            client = _get_client(key)
        except Exception as e:
            return {"status": "ERROR", "text": None, "validation": None, "attempts": 0,
                    "message": f"client unavailable: {e}"}

    last_validation = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.2,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return {"status": "ERROR", "text": None, "validation": None,
                    "attempts": attempt, "message": str(e)[:200]}

        validation = validate_narrative(text, decision)
        last_validation = validation
        if validation["ok"]:
            return {"status": "OK", "text": text, "validation": validation,
                    "attempts": attempt}
        # Tell the model exactly what failed and let it try once more.
        user = (user + "\n\nYour previous attempt was REJECTED: "
                + validation["reason"]
                + ". Rewrite it using only numbers present in the structured data above.")

    return {"status": "REJECTED", "text": None, "validation": last_validation,
            "attempts": max_attempts,
            "message": ("The generated narrative stated numbers that are not in the structured "
                        "data and was withheld. The structured decision above is unaffected.")}
