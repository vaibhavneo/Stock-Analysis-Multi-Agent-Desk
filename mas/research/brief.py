"""The Decision Brief as the final synthesis layer.

Fifteen named sections, in the order a reader needs them: the state and the
one-sentence answer first, then why, then what moved, then the cases, the
map, the two ownership branches, catalysts, risk, agreement, statistical
honesty, what would change it, monitoring, and data quality last.

EVERY SECTION IS BUILT FROM STRUCTURE THAT EXISTS
-------------------------------------------------
A section with nothing behind it renders as NOT_PRODUCED with the reason,
never as an empty heading and never as prose improvised to fill it. The
ordering is the argument: a reader who stops after the second section has the
answer and its single strongest constraint, which is the honest short version.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

NOT_PRODUCED = "NOT_PRODUCED"

SECTION_ORDER = (
    "current_state", "one_sentence", "why", "what_changed", "thesis",
    "entry_map", "if_you_own_it", "if_you_do_not", "catalysts", "risk",
    "agreement", "statistical_honesty", "what_would_change_my_mind",
    "monitoring", "data_quality",
)


def _s(key: str, title: str, status: str, lines: List[str],
       source: str = "") -> Dict[str, Any]:
    return {"key": key, "title": title, "status": status,
            "lines": [str(l).strip() for l in lines if l and str(l).strip()],
            "source": source}


def _missing(key: str, title: str, why: str) -> Dict[str, Any]:
    return _s(key, title, NOT_PRODUCED, [why])


def build(result: Dict[str, Any]) -> Dict[str, Any]:
    """The fifteen-section brief for one research result. Never raises."""
    plan = result.get("plan") or {}
    dec = result.get("decision") or {}
    syn = result.get("synthesis") or {}
    ev = (result.get("evidence") or {}).get("items") or []
    chg = result.get("change") or {}
    adv = result.get("adversarial") or {}
    val = result.get("validation") or {}
    disc = result.get("tool_discovery") or {}
    # The brief is a DOCUMENT, not an answer: it carries every section that
    # can be built, while `result["decision"]` stays scoped to what the
    # question asked. Reading the scoped set left bull/base/bear and the
    # not-owned branch empty on an add question purely because that question
    # had not requested them.
    full = dec
    try:
        from .decision import build as _bd
        raw = result.get("_decision_object")
        if raw is not None:
            full = _bd(raw, plan.get("intent", ""), all_sections=True)
    except Exception:
        full = dec
    by_key = {s.get("key"): s for s in (full.get("sections") or [])}
    symbols = ", ".join(plan.get("symbols") or []) or "this"

    def sec(name):
        s = by_key.get(name)
        return s if s and s.get("status") != "NOT_PRODUCED" else None

    out: List[Dict[str, Any]] = []

    # 1. CURRENT STATE
    st = sec("state")
    out.append(_s("current_state", "Current state",
                  st["status"] if st else NOT_PRODUCED,
                  [st["headline"]] if st else
                  ["No decision state was produced for this question."],
                  "decision_state") if st else
               _missing("current_state", "Current state",
                        "No decision state was produced for this question."))

    # 2. ONE-SENTENCE SYNTHESIS
    one = []
    if st:
        one.append(f"{symbols}: {st['status'].replace('_', ' ').lower()} — "
                   f"{st['headline']}")
    elif syn.get("statement"):
        one.append(syn["statement"].split(".")[0] + ".")
    out.append(_s("one_sentence", "In one sentence",
                  "STATED" if one else NOT_PRODUCED,
                  one or ["Not enough was produced to state one."], "synthesis"))

    # 3. WHY — strongest evidence, strongest constraint, strongest contradiction
    usable = sorted([i for i in ev if i.get("usable_for_decision")],
                    key=lambda i: -(i.get("weight") or 0))
    why: List[str] = []
    if usable:
        why.append(f"Strongest evidence: {usable[0]['statement']} "
                   f"({usable[0]['source']}, "
                   f"{usable[0]['tier'].replace('_', ' ').lower()}).")
    constraints = [i for i in ev
                   if {"no_demonstrated_edge", "insufficient_sample",
                       "derived_only"} & set(i.get("flags") or [])]
    if constraints:
        why.append(f"Strongest constraint: {constraints[0]['statement']} "
                   f"— {constraints[0].get('uncertainty') or ''}")
    if (adv.get("challenges") or [{}])[0].get("statement"):
        why.append(f"Strongest contradiction: {adv['challenges'][0]['statement']}")
    elif syn.get("n_decision_changing"):
        why.append(f"{syn['n_decision_changing']} decision-changing "
                   f"disagreement(s) in the evidence.")
    out.append(_s("why", "Why the answer is what it is",
                  "STATED" if why else NOT_PRODUCED,
                  why or ["No evidence reached the decision."], "evidence"))

    # 4. WHAT CHANGED
    out.append(_s("what_changed", "What changed",
                  "NO_PRIOR" if not chg.get("has_previous") else
                  ("CHANGED" if chg.get("n_material") else "UNCHANGED"),
                  [chg.get("statement", "")], "change"))

    # 5. THESIS — bull / base / bear
    sc = sec("scenarios")
    out.append(_s("thesis", "Bull, base, bear",
                  sc["status"] if sc else NOT_PRODUCED,
                  (sc["detail"] if sc else
                   ["No scenarios were produced for this question."]),
                  "scenarios"))

    # 6. ENTRY MAP
    en = sec("entry")
    rk = sec("risk")
    entry_lines = list(en["detail"]) if en else []
    if en and en.get("headline"):
        entry_lines.insert(0, en["headline"])
    if rk:
        inval = next((d for d in rk["detail"] + [rk["headline"]]
                      if "Invalidation" in str(d)), None)
        if inval:
            entry_lines.append(inval)
    # Status follows whether lines were actually assembled, not whether one
    # particular source existed — the risk section can supply the
    # invalidation even when no entry plan was produced.
    out.append(_s("entry_map", "Entry map",
                  (en["status"] if en else
                   ("PARTIAL" if entry_lines else NOT_PRODUCED)),
                  entry_lines or ["No entry plan was produced."], "entry_plan"))

    # 7/8. THE TWO OWNERSHIP BRANCHES
    owns = (plan.get("position") or {}).get("owns")
    own_lines: List[str] = []
    for name, label in (("playbook", "Hold while"), ("add", "Add if"),
                        ("reduce", "Reduce if"), ("exit", "Exit if")):
        s2 = sec(name)
        if s2:
            own_lines.append(f"**{label}** — {s2['status'].replace('_', ' ')}: "
                             f"{s2['headline']}")
            own_lines.extend(s2["detail"][:2])
    out.append(_s("if_you_own_it", "If you own it",
                  "STATED" if own_lines else NOT_PRODUCED,
                  own_lines or ["No owned-branch conditions were produced."],
                  "playbook"))

    not_own = []
    if en:
        not_own.append(f"**Enter if** — {en['headline'] or en['status']}")
        not_own.extend(en["detail"][:2])
    if st and "AVOID" in str(st.get("status", "")):
        not_own.append(f"**Avoid** — {st['headline']}")
    out.append(_s("if_you_do_not", "If you do not own it",
                  "STATED" if not_own else NOT_PRODUCED,
                  not_own or ["No not-owned branch was produced."], "entry"))

    # 9. CATALYST MAP
    cats = [i for i in ev if str(i.get("key", "")).startswith("catalyst")]
    hz_days = (plan.get("horizon") or {}).get("days")
    cat_lines = []
    for c in cats:
        days = c.get("value")
        inside = (isinstance(days, (int, float)) and isinstance(hz_days, (int, float))
                  and days <= hz_days)
        cat_lines.append(
            f"{c['statement']} — "
            + ("inside this horizon, so a position held that long carries it."
               if inside else
               f"OUTSIDE this {hz_days}-day horizon; it cannot move a position "
               f"closed before it."))
    out.append(_s("catalysts", "Catalyst map",
                  "STATED" if cat_lines else NOT_PRODUCED,
                  cat_lines or ["No dated event was found for this name."],
                  "catalysts"))

    # 10. RISK
    out.append(_s("risk", "Risk", rk["status"] if rk else NOT_PRODUCED,
                  ([rk["headline"]] + rk["detail"]) if rk else
                  ["No risk budget was produced."], "risk_budget"))

    # 11. EVIDENCE AGREEMENT / CONFLICT
    out.append(_s("agreement", "What agrees, what does not",
                  syn.get("consensus") or NOT_PRODUCED,
                  [syn.get("statement", "")]
                  + [f"{c['sources'][0]} vs {c['sources'][1]}: {c['resolution']}"
                     for c in (syn.get("conflicts") or [])[:2]],
                  "synthesis"))

    # 12. STATISTICAL HONESTY
    stat = [i for i in ev if str(i.get("key", "")).startswith("stats")]
    stat_lines = [f"{i['statement']} — {i.get('uncertainty') or ''}" for i in stat]
    out.append(_s("statistical_honesty", "Statistical honesty",
                  "STATED" if stat_lines else NOT_PRODUCED,
                  stat_lines or ["No statistical evidence was produced."],
                  "statistical"))

    # 13. WHAT WOULD CHANGE MY MIND
    inv = sec("invalidation")
    out.append(_s("what_would_change_my_mind", "What would change my mind",
                  inv["status"] if inv else NOT_PRODUCED,
                  (inv["detail"] if inv else
                   ["No invalidation conditions were produced."]),
                  "mind_changers"))

    # 14. MONITORING PLAN
    mon = sec("monitoring")
    out.append(_s("monitoring", "Monitoring plan",
                  mon["status"] if mon else NOT_PRODUCED,
                  ([mon["headline"]] + mon["detail"]) if mon else
                  ["No monitoring plan was produced."], "monitoring"))

    # 15. DATA QUALITY
    dq = [disc.get("statement", "")]
    gaps = [i["statement"] for i in ev if "unavailable" in (i.get("flags") or [])]
    dq.extend(gaps)
    if syn.get("stale_evidence"):
        dq.append(f"Stale and down-weighted: {', '.join(syn['stale_evidence'])}.")
    acct = result.get("accounting") or {}
    if acct.get("statement"):
        dq.append(acct["statement"])
    if val.get("summary"):
        dq.append(val["summary"])
    out.append(_s("data_quality", "Data quality", "STATED", dq, "tool_discovery"))

    produced = [s for s in out if s["status"] != NOT_PRODUCED]
    return {
        "symbol": (plan.get("symbols") or [None])[0],
        "intent": plan.get("intent"),
        "sections": out,
        "order": list(SECTION_ORDER),
        "n_sections": len(out),
        "n_produced": len(produced),
        "statement": (
            f"All {len(out)} brief sections were produced."
            if len(produced) == len(out) else
            f"{len(produced)} of {len(out)} brief sections were produced; the "
            f"other {len(out) - len(produced)} are named with the reason "
            f"rather than filled in."),
    }
