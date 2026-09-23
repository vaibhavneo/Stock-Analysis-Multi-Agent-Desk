"""What each section of the working actually CHANGED about the answer.

THE PROBLEM
-----------
"Show the working" grew into a set of panels that each display a lot of data
and none of which say what it did. The catalyst calendar reports an earnings
date 56 days out without saying that this is outside the next-few-weeks plan
and inside the other two. Risk detail lists ATR, drawdown and a concentration
figure marked "not computable" without saying which number set the
invalidation level or what the missing ones would have unlocked.

A panel that shows data without saying what it changed is asking the reader to
do the synthesis the engine was supposed to do. It also hides the more useful
fact: most of these inputs did NOT change the verdict, and saying so is
information, not an absence of it.

This is the same discipline the compute side already follows — a value that is
computed but never consulted is a bug — applied to presentation. `conflict`
already counts `n_decision_changing`; this generalizes that idea to every
section.

WHAT `changed` MEANS
--------------------
True when the section's inputs moved the verdict, the levels, the sizing or
the confidence. False is a real answer and the common one. `UNKNOWN` is used
only where the engine genuinely cannot tell, and never as a hedge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .horizon_plan import HORIZON_SPECS, HORIZON_ORDER

CHANGED = "CHANGED"
NO_EFFECT = "NO_EFFECT"
LIMITED = "LIMITED"        # it capped or constrained rather than moved
UNKNOWN = "UNKNOWN"


def _b(effect: str, statement: str, detail: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"effect": effect, "changed": effect in (CHANGED, LIMITED),
            "statement": statement, "detail": detail or []}


# ── Catalysts ─────────────────────────────────────────────────────────────

def catalysts_bearing(catalysts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Which horizons the next scheduled event actually falls inside.

    An earnings date is only a catalyst for a plan whose holding period
    reaches it. Reporting "56 days away" with three horizon cards above and
    no mapping between them leaves the reader to do that arithmetic — and the
    answer changes which card it belongs to.
    """
    cat = catalysts or {}
    nxt = cat.get("next_event") or cat.get("next_scheduled_event")
    if cat.get("status") == "UNAVAILABLE" or not nxt:
        return _b(NO_EFFECT,
                  "No dated event was found, so nothing here narrowed any "
                  "horizon. That is an absence of data, not a quiet calendar.",
                  [cat.get("waiting_for")] if cat.get("waiting_for") else [])

    days = nxt.get("days_away")
    if days is None:
        return _b(UNKNOWN, "An event is listed without a date, so it cannot be "
                           "placed inside or outside any horizon.")

    inside = [n for n in HORIZON_ORDER if days <= HORIZON_SPECS[n]["days"]]
    outside = [n for n in HORIZON_ORDER if n not in inside]
    label = lambda n: HORIZON_SPECS[n]["label"].lower()
    name = nxt.get("event") or "the next event"

    if not inside:
        return _b(NO_EFFECT,
                  f"{name} is {days} days out, past every horizon shown — it "
                  f"does not bear on any of these plans.",
                  [f"Longest horizon here is {HORIZON_SPECS[HORIZON_ORDER[-1]]['label'].lower()} "
                   f"({HORIZON_SPECS[HORIZON_ORDER[-1]]['days']} days)."])

    detail = [f"Inside: {', '.join(label(n) for n in inside)}."]
    if outside:
        detail.append(f"Outside: {', '.join(label(n) for n in outside)} — the "
                      f"event lands after that plan's horizon ends.")
    return _b(CHANGED,
              f"{name} in {days} days falls inside "
              f"{'the ' + label(inside[0]) if len(inside) == 1 else str(len(inside)) + ' of the 3 horizons'}"
              f", so a position held that long carries it.",
              detail)


# ── Risk ──────────────────────────────────────────────────────────────────

# What each absent input would have unlocked. Saying "not computable" without
# saying what would make it computable reads as a defect rather than a choice.
# Whole sentences, not fragments. The first version stored noun phrases and
# prefixed them with "no ", which produced "Without avg cost: no your
# position's gain or loss" — a template seam showing through in the one place
# the text is trying to explain a gap clearly.
UNLOCKED_BY = {
    "avg_cost": ("Without your average cost there is no gain or loss figure, "
                 "and no way to say whether the invalidation is already "
                 "behind you."),
    "shares": ("Without a share count the risk to invalidation stays a "
               "percentage rather than a dollar amount."),
    "portfolio_value": ("Without a portfolio value, position weight and "
                        "concentration cannot be computed — which is why that "
                        "chip reads 'not computable'."),
}


def risk_bearing(risk_budget: Optional[Dict[str, Any]],
                 position_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rb = risk_budget or {}
    level = rb.get("invalidation_level")
    basis = rb.get("invalidation_basis")
    missing = list(rb.get("inputs_missing") or [])

    if level is None:
        return _b(UNKNOWN, "No invalidation level was produced, so risk did "
                           "not constrain anything here.")

    basis_text = {
        "TRADED_PRICE": "a price this stock has actually traded at",
        "PRICE_HISTORY_DERIVED": "a level derived from its own price history",
        "CURRENT_PRICE_DERIVED": "arithmetic on today's close, not a level the "
                                 "market has defended",
        "MODEL": "a model, not an observed level",
    }.get(basis, basis or "an unstated basis")

    detail = [f"The invalidation at {level} comes from {basis_text}."]
    for m in missing:
        if m in UNLOCKED_BY:
            detail.append(UNLOCKED_BY[m])

    if missing:
        return _b(LIMITED,
                  f"Risk set the exit at {level} and capped what could be sized: "
                  f"{len(missing)} position input{'s' if len(missing) != 1 else ''} "
                  f"were not supplied, so the figures here are per-share and "
                  f"proportional rather than in dollars.",
                  detail)
    return _b(CHANGED,
              f"Risk set the exit at {level} and sized the position against it.",
              detail)


# ── Levels ────────────────────────────────────────────────────────────────

def levels_bearing(level_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    lm = level_map or {}
    rows: List[Dict[str, Any]] = []
    for key in ("supports", "resistances"):
        rows.extend(lm.get(key) or [])
    if not rows:
        return _b(NO_EFFECT, "No levels were produced, so none of the prices "
                             "above came from observed trading.")
    traded = sum(1 for r in rows if r.get("basis") == "TRADED_PRICE")
    n = len(rows)
    if traded == 0:
        return _b(LIMITED,
                  f"None of the {n} levels is a price this stock has traded at "
                  f"— every one is arithmetic on the current price, which is a "
                  f"weaker claim than a level the market has defended.")
    return _b(CHANGED,
              f"{traded} of {n} levels are prices actually traded at; the rest "
              f"are derived. The entry and exit above are drawn from these.",
              [f"A derived level is arithmetic, not a place the market has "
               f"shown it cares about."])


# ── Confidence and data quality ───────────────────────────────────────────

def confidence_bearing(confidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    c = confidence or {}
    capped = list(c.get("capped_by") or [])
    level = c.get("decision_confidence") or c.get("overall")
    if not capped:
        return _b(NO_EFFECT,
                  f"Confidence came out {level or 'unstated'} with nothing "
                  f"holding it down.")
    # The cap labels already carry their own level, so naming the level twice
    # produced "Confidence is NONE because of statistical (NONE)" — circular,
    # and it hides the useful half, which is WHICH dimension did the capping.
    dims = ", ".join(str(x).split(" (")[0] for x in capped)
    if str(level).upper() == "NONE":
        return _b(LIMITED,
                  f"Confidence could not be established at all: {dims} is the "
                  f"dimension that held it there. That ceiling — not the "
                  f"direction of the evidence — is what keeps the size above "
                  f"at its minimum.",
                  [c.get("rule")] if c.get("rule") else [])
    return _b(LIMITED,
              f"Confidence is {level} because {dims} capped it — that ceiling "
              f"is what stops the size above from being larger, not the "
              f"direction of the evidence.",
              [c.get("rule")] if c.get("rule") else [])


def quality_bearing(quality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    q = quality or {}
    missing = list(q.get("missing") or [])
    weak = list(q.get("weak") or [])
    n_good, n_total = q.get("n_good"), q.get("n_components")
    if not missing and not weak:
        return _b(NO_EFFECT,
                  f"All {n_total} data components are present and good, so "
                  f"nothing was down-weighted for data quality.")
    return _b(LIMITED,
              f"{n_good} of {n_total} data components are good. "
              + (f"Missing: {', '.join(missing)}. " if missing else "")
              + (f"Weak: {', '.join(weak)}. " if weak else "")
              + "Each absence lowers confidence rather than the score.")


# ── Conflict ──────────────────────────────────────────────────────────────

def conflict_bearing(conflict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cf = conflict or {}
    n = cf.get("n_conflicts") or 0
    changing = cf.get("n_decision_changing") or 0
    if n == 0:
        return _b(NO_EFFECT, "Nothing in the evidence contradicted anything "
                             "else, so no disagreement had to be resolved.")
    if changing == 0:
        return _b(NO_EFFECT,
                  f"{n} disagreement{'s' if n != 1 else ''} named, none of "
                  f"which would change the decision — they are recorded so "
                  f"the agreement above is not mistaken for unanimity.")
    return _b(CHANGED,
              f"{changing} of {n} disagreements are decision-changing: had "
              f"they resolved the other way the verdict would differ.")


# ── Scenarios ─────────────────────────────────────────────────────────────

def scenarios_bearing(scenarios: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sc = scenarios or {}
    rows = sc.get("scenarios") or []
    if not rows:
        return _b(NO_EFFECT, "No scenarios were produced.")
    if not sc.get("any_probability_stated"):
        return _b(LIMITED,
                  f"{len(rows)} scenarios with NO probability attached to any "
                  f"of them. They describe what each outcome would look like "
                  f"and what it would need; they do not say which is likely, "
                  f"because this engine has not measured itself enough to.",
                  [sc.get("probability_policy")] if sc.get("probability_policy") else [])
    return _b(CHANGED, f"{len(rows)} scenarios, with stated probabilities.")


# ── Statistical honesty / track record ────────────────────────────────────

def edge_bearing(honesty: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    h = honesty or {}
    if h.get("demonstrated_edge"):
        return _b(CHANGED,
                  "The statistical gate passed, which is what permits a "
                  "position size above the minimum.")
    return _b(LIMITED,
              f"The statistical gate did NOT pass ({h.get('verdict') or 'no edge shown'}). "
              f"This is the single biggest constraint on the answer above: it "
              f"is why the size is what it is, whatever the evidence says.")


# ── Position ──────────────────────────────────────────────────────────────

def position_bearing(position_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pc = position_context or {}
    if (pc.get("status") or "").upper().startswith("POSITION_CONTEXT_NOT"):
        return _b(LIMITED,
                  "No holding was supplied, so both the own-it and don't-own-it "
                  "answers are shown above and neither is assumed.",
                  ["Entering an average cost makes the advice specific to you."])
    return _b(CHANGED, "Your position was used to make the advice specific.")


# ── Consistency ───────────────────────────────────────────────────────────

def consistency_bearing(consistency: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    c = consistency or {}
    errs, warns = c.get("n_errors") or 0, c.get("n_warnings") or 0
    if errs == 0 and warns == 0:
        return _b(NO_EFFECT, "Every internal cross-check agreed; nothing here "
                             "qualifies the answer above.")
    if errs:
        return _b(CHANGED,
                  f"{errs} internal contradiction{'s' if errs != 1 else ''} — "
                  f"the answer above should be treated as unreliable until "
                  f"resolved.")
    first = (c.get("issues") or [{}])[0]
    return _b(LIMITED,
              f"{warns} warning{'s' if warns != 1 else ''}, shown rather than "
              f"silently reconciled"
              + (f": {first.get('message')}" if first.get("message") else "."))


SECTION_ORDER = ("edge", "risk", "confidence", "catalysts", "levels",
                 "conflict", "scenarios", "quality", "position", "consistency")


def build_bearing(decision: Dict[str, Any]) -> Dict[str, Any]:
    """One bearing statement per section of the working.

    Never raises: a bearing that fails to compute is omitted, because a
    missing explanation is better than a wrong one attached to real numbers.
    """
    d = decision or {}
    out: Dict[str, Any] = {}
    builders = {
        "catalysts": lambda: catalysts_bearing(d.get("catalysts")),
        "risk": lambda: risk_bearing(d.get("risk_budget"), d.get("position_context")),
        "levels": lambda: levels_bearing(d.get("level_map")),
        "confidence": lambda: confidence_bearing(d.get("confidence")),
        "quality": lambda: quality_bearing(d.get("quality")),
        "conflict": lambda: conflict_bearing(d.get("conflict")),
        "scenarios": lambda: scenarios_bearing(d.get("scenarios")),
        "edge": lambda: edge_bearing(d.get("statistical_honesty")),
        "position": lambda: position_bearing(d.get("position_context")),
        "consistency": lambda: consistency_bearing(d.get("consistency")),
    }
    for name, fn in builders.items():
        try:
            out[name] = fn()
        except Exception:
            continue

    changed = [k for k, v in out.items() if v.get("effect") == CHANGED]
    limited = [k for k, v in out.items() if v.get("effect") == LIMITED]
    n = len([k for k in out if not k.startswith("_")])
    rest = n - len(changed) - len(limited)
    # Only claim a remainder when there is one. The first version said "the
    # rest are recorded so you can see they were checked" on a decision where
    # every section had moved or constrained something — describing a group
    # with nothing in it.
    tail = (f"; the remaining {rest} were checked and changed nothing."
            if rest > 0 else ", and every section here did one or the other.")
    out["_summary"] = {
        "n_sections": n,
        "moved_the_answer": sorted(changed),
        "constrained_the_answer": sorted(limited),
        "no_effect": sorted(k for k, v in out.items()
                            if not k.startswith("_") and v.get("effect") == NO_EFFECT),
        "statement": (
            f"{len(changed)} section{'s' if len(changed) != 1 else ''} moved "
            f"the answer and {len(limited)} constrained it{tail}"),
    }
    return out
