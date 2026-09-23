"""A question becomes a research plan, before anything runs.

WHAT WAS THERE BEFORE
---------------------
Intent mapped 1:1 onto a capability and the capability ran a fixed
computation. "Should I add to my IONQ position?" and "Tell me about IONQ"
produced the same work. There was no representation of what the question
NEEDED — only of which module would be called — so nothing could reason about
sufficiency, freshness, cost, or what was missing.

WHAT A PLAN IS FOR
------------------
It is the object that lets the orchestrator answer three questions it could
not previously ask:

  1. What evidence does THIS decision actually require?
  2. Which of that can this deployment reach, and which cannot?
  3. What is still missing, and is the question answerable without it?

The plan is built with NO I/O. It can therefore be inspected before a single
tool runs, asserted against in tests without mocking a network, and shown to
the user as "here is what I am about to do" — which is the only way a
research step that never happened becomes visible.

EVIDENCE, NOT MODULES
---------------------
Requirements are expressed as evidence kinds (`price_history`, `earnings`,
`position_context`), never as module names. That indirection is what makes a
missing tool a stated gap rather than a silent absence: the plan can say
"this decision needs an option chain and none is reachable" instead of simply
not calling something.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from . import tools as T
from .intent import classify as classify_intent
from .intent_corpus import (
    GENERAL_RESEARCH, NEW_ENTRY, EXISTING_POSITION, ADD_TO_POSITION,
    REDUCE_POSITION, EXIT_POSITION, SHORT_TERM_SETUP, MEDIUM_TERM_SETUP,
    LONG_TERM_THESIS, EARNINGS_PREVIEW, EVENT_ANALYSIS, RISK_ANALYSIS,
    COMPARISON, PORTFOLIO_CONTEXT, OPTIONS_ANALYSIS, INVALIDATION, NONE,
)
from .position import extract as extract_position, merge as merge_position

# Evidence kinds the DESK produces (as opposed to the tool-level kinds in
# tools.py, which are raw data). These are what specialists return.
THESIS = "thesis"
TECHNICAL_STRUCTURE = "technical_structure"
STATISTICAL_EDGE = "statistical_edge"
LEVELS = "levels"
RISK = "risk"
POSITION_CONTEXT = "position_context"
CATALYSTS = "catalysts"
SCENARIOS = "scenarios"
BENCHMARK = "benchmark_relation"
FORWARD_RECORD = "forward_record"
REGIME = "regime"
OPTIONS = "option_structures"
NARRATIVE = "analyst_narrative"

SHORT, MEDIUM, LONG, UNSPECIFIED = "SHORT", "MEDIUM", "LONG", "UNSPECIFIED"

HORIZON_DAYS = {SHORT: 21, MEDIUM: 126, LONG: 252, UNSPECIFIED: 91}

_HORIZON_PATTERNS = [
    (r"\bnext\s+(few\s+)?(days?|week)\b|\bthis\s+week\b|\bswing\b"
     r"|\bshort[\s-]term\b|\bnext\s+month\b", SHORT),
    (r"\bnext\s+(couple|few)\s+of?\s*months?\b|\b(3|three|6|six)\s+months?\b"
     r"|\bmedium[\s-]term\b", MEDIUM),
    (r"\blong[\s-]term\b|\bnext\s+(few\s+)?years?\b|\bdecade\b"
     r"|\bfor\s+(the\s+)?years?\b|\bnext\s+year\b", LONG),
]

# Which evidence each decision actually needs. This table IS the planner's
# knowledge: it is the difference between "run everything" and "run what this
# question depends on".
@dataclass(frozen=True)
class IntentSpec:
    objective: str
    required: tuple
    optional: tuple = ()
    needs_position: bool = False
    default_horizon: str = UNSPECIFIED
    note: str = ""


SPECS: Dict[str, IntentSpec] = {
    ADD_TO_POSITION: IntentSpec(
        objective="Decide whether adding to an existing holding is justified",
        required=(POSITION_CONTEXT, THESIS, LEVELS, RISK, STATISTICAL_EDGE,
                  TECHNICAL_STRUCTURE),
        optional=(CATALYSTS, BENCHMARK, SCENARIOS),
        needs_position=True, default_horizon=MEDIUM,
        note="A lower price is not a better thesis. This decision needs the "
             "thesis and the invalidation distance, not just the drawdown."),
    REDUCE_POSITION: IntentSpec(
        objective="Decide whether to reduce an existing holding",
        required=(POSITION_CONTEXT, THESIS, RISK, LEVELS),
        optional=(CATALYSTS, STATISTICAL_EDGE, BENCHMARK),
        needs_position=True, default_horizon=MEDIUM),
    EXIT_POSITION: IntentSpec(
        objective="Decide whether the thesis has broken enough to exit",
        required=(POSITION_CONTEXT, THESIS, LEVELS, RISK),
        optional=(CATALYSTS, STATISTICAL_EDGE),
        needs_position=True, default_horizon=MEDIUM,
        note="Exit is a thesis question, not a price question. Normal "
             "volatility is not invalidation."),
    EXISTING_POSITION: IntentSpec(
        objective="Report the state of an existing holding and what to watch",
        required=(POSITION_CONTEXT, THESIS, LEVELS, RISK),
        optional=(CATALYSTS, SCENARIOS, STATISTICAL_EDGE),
        needs_position=True, default_horizon=MEDIUM),
    NEW_ENTRY: IntentSpec(
        objective="Decide whether and where to start a position",
        required=(THESIS, LEVELS, RISK, STATISTICAL_EDGE, TECHNICAL_STRUCTURE),
        optional=(CATALYSTS, SCENARIOS, BENCHMARK, REGIME),
        default_horizon=MEDIUM),
    GENERAL_RESEARCH: IntentSpec(
        objective="Give the full read on a security",
        required=(THESIS, LEVELS, RISK),
        optional=(STATISTICAL_EDGE, CATALYSTS, BENCHMARK, SCENARIOS),
        default_horizon=UNSPECIFIED),
    SHORT_TERM_SETUP: IntentSpec(
        objective="Assess the near-term setup",
        required=(TECHNICAL_STRUCTURE, LEVELS, RISK),
        optional=(CATALYSTS, STATISTICAL_EDGE), default_horizon=SHORT,
        note="Fundamentals move slower than this horizon, so they are "
             "optional rather than required here."),
    MEDIUM_TERM_SETUP: IntentSpec(
        objective="Assess the setup over one to six months",
        required=(THESIS, TECHNICAL_STRUCTURE, LEVELS, RISK),
        optional=(CATALYSTS, STATISTICAL_EDGE, SCENARIOS),
        default_horizon=MEDIUM),
    LONG_TERM_THESIS: IntentSpec(
        objective="Assess the multi-year case",
        required=(THESIS, RISK),
        optional=(LEVELS, BENCHMARK, SCENARIOS), default_horizon=LONG,
        note="Chart structure decays as a long-horizon input, so it is not "
             "required here even though it is for a swing question."),
    EARNINGS_PREVIEW: IntentSpec(
        objective="Assess what the coming earnings event means for this name",
        required=(CATALYSTS, THESIS, RISK, LEVELS),
        optional=(STATISTICAL_EDGE, TECHNICAL_STRUCTURE, OPTIONS),
        default_horizon=SHORT),
    EVENT_ANALYSIS: IntentSpec(
        objective="Identify scheduled events that could move this name",
        required=(CATALYSTS,), optional=(RISK, THESIS),
        default_horizon=MEDIUM),
    RISK_ANALYSIS: IntentSpec(
        objective="Quantify what can go wrong and how far",
        required=(RISK, LEVELS, SCENARIOS),
        optional=(CATALYSTS, BENCHMARK, REGIME), default_horizon=SHORT),
    INVALIDATION: IntentSpec(
        objective="State what would falsify the current thesis",
        required=(THESIS, LEVELS, SCENARIOS, RISK),
        optional=(CATALYSTS, STATISTICAL_EDGE), default_horizon=MEDIUM,
        note="This is the question the old router could not answer at all."),
    COMPARISON: IntentSpec(
        objective="Compare two or more securities on the same basis",
        required=(THESIS, RISK, BENCHMARK, LEVELS),
        optional=(STATISTICAL_EDGE, SCENARIOS), default_horizon=UNSPECIFIED,
        note=("The decision sections describe the FIRST named symbol. This "
              "engine builds a decision for one security at a time, so a "
              "comparison puts two reads side by side rather than producing "
              "a single verdict over both — and says so rather than letting "
              "the reader assume otherwise.")),
    PORTFOLIO_CONTEXT: IntentSpec(
        objective="Assess the book rather than a single name",
        required=(POSITION_CONTEXT, RISK), optional=(BENCHMARK, REGIME),
        needs_position=True, default_horizon=UNSPECIFIED),
    OPTIONS_ANALYSIS: IntentSpec(
        objective="Express a view in options, or price a structure",
        required=(OPTIONS, THESIS), optional=(RISK, LEVELS, CATALYSTS),
        default_horizon=SHORT),
    NONE: IntentSpec(objective="Not a research question", required=()),
}

# Evidence kind -> the registered capability that produces it.
EVIDENCE_CAPABILITY = {
    THESIS: "equity_research",
    TECHNICAL_STRUCTURE: "equity_research",
    LEVELS: "equity_research",
    RISK: "equity_research",
    STATISTICAL_EDGE: "strategy_backtest",
    CATALYSTS: "event_calendar",
    BENCHMARK: "benchmark_relation",
    FORWARD_RECORD: "forward_record",
    REGIME: "market_regime",
    OPTIONS: "option_structures",
    SCENARIOS: "equity_research",
    POSITION_CONTEXT: None,          # supplied by the user, never fetched
    NARRATIVE: "analyst_narrative",
}

# Evidence kind -> the raw tool evidence it depends on.
EVIDENCE_TOOLS = {
    THESIS: [T.PRICE_HISTORY, T.FUNDAMENTALS],
    TECHNICAL_STRUCTURE: [T.PRICE_HISTORY],
    LEVELS: [T.PRICE_HISTORY],
    RISK: [T.PRICE_HISTORY],
    STATISTICAL_EDGE: [T.PRICE_HISTORY],
    CATALYSTS: [T.EARNINGS, T.EVENT_CALENDAR],
    BENCHMARK: [T.PRICE_HISTORY],
    REGIME: [T.MACRO, T.VOLATILITY_INDEX],
    OPTIONS: [T.OPTIONS_CHAIN, T.PRICE_HISTORY],
    SCENARIOS: [T.PRICE_HISTORY],
    FORWARD_RECORD: [],
    POSITION_CONTEXT: [],
    NARRATIVE: [T.PRICE_HISTORY, T.FUNDAMENTALS, T.SOCIAL],
}

FAST, DEEP, ADVERSARIAL = "FAST", "DEEP", "ADVERSARIAL"
DEPTH_BUDGET_MS = {FAST: 6_000, DEEP: 20_000, ADVERSARIAL: 45_000}


def detect_horizon(text: str, default: str = UNSPECIFIED) -> Dict[str, Any]:
    for pat, h in _HORIZON_PATTERNS:
        if re.search(pat, text or "", re.I):
            return {"horizon": h, "days": HORIZON_DAYS[h], "stated": True,
                    "why": "the question named a timeframe"}
    return {"horizon": default, "days": HORIZON_DAYS[default], "stated": False,
            "why": f"no timeframe was stated; {default.lower()} is this "
                   f"question's natural horizon"}


@dataclass
class ResearchPlan:
    question: str
    objective: str
    intent: str
    symbols: List[str] = field(default_factory=list)
    asset_class: str = "UNKNOWN"
    horizon: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, Any] = field(default_factory=dict)
    required_evidence: List[str] = field(default_factory=list)
    optional_evidence: List[str] = field(default_factory=list)
    capabilities: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[Dict[str, Any]] = field(default_factory=list)
    stages: List[List[str]] = field(default_factory=list)
    depth: str = FAST
    budget_ms: int = DEPTH_BUDGET_MS[FAST]
    answerable: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> List[str]:
        out = [f"Objective: {self.objective}."]
        if self.symbols:
            out.append(f"Subject: {', '.join(self.symbols)} ({self.asset_class}), "
                       f"over {self.horizon.get('horizon', '').lower()} "
                       f"(~{self.horizon.get('days')} days) — "
                       f"{self.horizon.get('why')}.")
        if self.required_evidence:
            out.append("Needs: " + ", ".join(
                e.replace("_", " ") for e in self.required_evidence) + ".")
        for c in self.capabilities:
            out.append(f"Ask {c['capability'].replace('_', ' ')} — {c['why']}")
        for m in self.missing:
            out.append(f"Missing: {m['what']} — {m['why']}")
        return out


def build_plan(question: str,
               symbols: Optional[Sequence[str]] = None,
               asset_class: str = "EQUITY",
               context_symbol: Optional[str] = None,
               supplied_position: Optional[Dict[str, Any]] = None,
               depth: str = FAST) -> ResearchPlan:
    """Question -> plan. No I/O beyond probing which tools exist."""
    q = (question or "").strip()
    syms = [s.upper() for s in (symbols or [])]
    intent_res = classify_intent(q, context_symbol=context_symbol,
                                 has_symbol=bool(syms), n_symbols=len(syms))
    intent = intent_res["intent"]
    spec = SPECS.get(intent, SPECS[GENERAL_RESEARCH])

    plan = ResearchPlan(question=q, objective=spec.objective, intent=intent,
                        symbols=syms, asset_class=asset_class, depth=depth,
                        budget_ms=DEPTH_BUDGET_MS.get(depth,
                                                      DEPTH_BUDGET_MS[FAST]))
    if spec.note:
        plan.notes.append(spec.note)

    if intent == NONE:
        plan.answerable = False
        plan.objective = spec.objective
        plan.missing.append({"what": "a research question",
                             "why": "nothing here asks for analysis"})
        return plan

    plan.horizon = detect_horizon(q, spec.default_horizon)

    # Position context: stated, then merged with whatever the application
    # already holds. Never inferred.
    stated = extract_position(q)
    plan.position = merge_position(stated, supplied_position)

    required = list(spec.required)
    optional = list(spec.optional)

    # A position-management question without a position is not answerable as
    # asked. Saying so is the answer; guessing a holding would not be.
    if spec.needs_position and not plan.position.get("owns"):
        plan.answerable = False
        plan.missing.append({
            "what": "your position",
            "why": ("this decision is about something you hold, and nothing "
                    "here says you hold it. Tell me shares and average cost "
                    "and the answer becomes specific to you; without them I "
                    "would be inventing the thing being decided about.")})

    if not syms and intent not in (PORTFOLIO_CONTEXT,):
        plan.answerable = False
        plan.missing.append({"what": "a symbol",
                             "why": "this question is about a security and "
                                    "none was named"})

    plan.required_evidence = required
    plan.optional_evidence = optional

    # Capabilities, derived from evidence rather than named directly.
    seen: Dict[str, Dict[str, Any]] = {}
    for kind, necessity in ([(k, "required") for k in required]
                            + [(k, "optional") for k in optional]):
        cap = EVIDENCE_CAPABILITY.get(kind)
        if cap is None:
            continue
        row = seen.setdefault(cap, {"capability": cap, "necessity": necessity,
                                    "for_evidence": [], "why": ""})
        row["for_evidence"].append(kind)
        if necessity == "required":
            row["necessity"] = "required"
    for cap, row in seen.items():
        kinds = ", ".join(k.replace("_", " ") for k in row["for_evidence"])
        row["why"] = f"it is the source of {kinds}"
    plan.capabilities = sorted(seen.values(),
                               key=lambda r: (r["necessity"] != "required",
                                              r["capability"]))

    # Tools, discovered rather than assumed.
    need_tools: Dict[str, str] = {}
    for kind in required:
        for tk in EVIDENCE_TOOLS.get(kind, []):
            need_tools.setdefault(tk, "required")
    for kind in optional:
        for tk in EVIDENCE_TOOLS.get(kind, []):
            need_tools.setdefault(tk, "optional")
    # Every decision shows a price, so the freshness check is never optional.
    need_tools.setdefault(T.PRICE_CURRENT, "required")

    for tk, necessity in sorted(need_tools.items()):
        candidates = T.tools_for(tk, asset_class=asset_class,
                                 available_only=True)
        if candidates:
            # Value per second of latency, not reliability alone. The old
            # selector took the most reliable source regardless of what it
            # cost to wait for, so a 2500ms chain outranked a 120ms model even
            # for evidence the plan had marked optional.
            from .budget import tool_value
            scored = sorted(
                ((c, tool_value(c.reliability, c.est_ms, necessity))
                 for c in candidates), key=lambda cv: -cv[1])
            best, value = scored[0]
            runner_up = scored[1] if len(scored) > 1 else None
            why = (f"best value for {tk.replace('_', ' ')}: reliability "
                   f"{best.reliability:.2f} at {best.est_ms}ms "
                   f"({value:.2f} per second of latency)")
            if runner_up and runner_up[0].reliability > best.reliability:
                why += (f"; {runner_up[0].id} is more reliable but "
                        f"{runner_up[0].est_ms - best.est_ms}ms slower, which "
                        f"does not pay for {necessity} evidence")
            plan.tools.append({
                "evidence": tk, "tool": best.id, "necessity": necessity,
                "freshness": best.freshness, "reliability": best.reliability,
                "est_ms": best.est_ms, "value_per_sec": value,
                "alternatives": [c.id for c, _ in scored[1:]],
                "why": why})
            continue
        offline = T.tools_for(tk, asset_class=asset_class, available_only=False)
        reason = (offline[0].to_dict()["reason"] if offline
                  else f"no tool in this system produces {tk}")
        plan.tools.append({"evidence": tk, "tool": None,
                           "necessity": necessity, "why": reason})
        entry = {"what": tk.replace("_", " "), "why": reason}
        plan.missing.append(entry)
        if necessity == "required":
            plan.notes.append(
                f"{tk.replace('_', ' ')} is required for this question and "
                f"no reachable tool provides it, so that part of the answer "
                f"will say UNAVAILABLE rather than be estimated.")

    # Each specialist gets the specific question it must settle, what it is
    # handed, and the fields it owes back. "Analyze this stock" produces five
    # overlapping essays; a specific question produces an answer the
    # orchestrator can compare without reading any of them.
    try:
        from .delegation import attach as _attach_briefs
        _attach_briefs(plan)
    except Exception:
        pass

    # Independent capabilities run together; nothing here depends on another
    # capability's output, so one parallel stage is honest. Sequencing appears
    # when a dependency does (see execute.py).
    caps = [c["capability"] for c in plan.capabilities]
    plan.stages = [caps] if caps else []
    return plan
