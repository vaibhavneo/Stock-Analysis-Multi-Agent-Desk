"""
Catalyst engine (Phase 14) — "what am I waiting for?"

The app already fetches earnings dates, consensus estimates, surprise history
and analyst actions (tools/market_data.py::fetch_earnings / fetch_analyst_ratings)
and hands them to an LLM prompt. They have never reached any decision surface:
`analyze_stock()` does not return them, so the Decision Brief cannot see them.
This module turns that same already-fetched data into a structured timeline.

Two hard rules, both of which cost information rather than invent it:

  1. **No speculation about unknown events.** Only events with a source-supplied
     date appear. There is no "product launch expected in Q3", no "possible
     regulatory decision" — if a provider did not give a date, the event does
     not exist here.
  2. **No fabricated probabilities.** Every catalyst carries
     `probability: None` and `probability_basis: "NOT_CALIBRATED"`. What it does
     carry is a *measured* magnitude: the ATR-implied move over the days to the
     event, which is arithmetic on realized volatility, not a forecast of
     direction.

Bull and bear implications are stated as the two branches of the event, not as
predictions of which branch occurs. "A miss reprices the stock downward" is a
statement about what the event is; "the stock will miss" would be a claim this
module has no basis for and never makes.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from decision.evidence import DecisionEvidence

# Beyond this, a scheduled event is real but cannot plausibly drive today's
# decision, so it is listed as context rather than as something to wait for.
ACTIONABLE_WINDOW_DAYS = 90

# An event inside this window dominates near-term planning: entering just
# before a binary repricing is a materially different decision from entering
# after it.
IMMINENT_WINDOW_DAYS = 14


def _as_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.fromisoformat(str(v)[:10]).date()
    except (ValueError, TypeError):
        return None


def fetch_catalyst_inputs(ticker: str) -> Dict[str, Any]:
    """Pull the raw calendar/consensus/surprise/analyst data.

    Best-effort by construction: every provider call is individually guarded,
    because a missing catalyst feed must degrade the catalyst section only —
    never the decision. Returns a dict with a `flags` list naming exactly what
    could not be fetched.
    """
    out: Dict[str, Any] = {"ticker": ticker.upper(), "flags": []}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
    except Exception as e:
        out["flags"].append(f"provider_unavailable:{type(e).__name__}")
        return out

    try:
        cal = t.calendar or {}
        out["calendar"] = {k: v for k, v in cal.items()} if isinstance(cal, dict) else {}
    except Exception:
        out["calendar"] = {}
        out["flags"].append("calendar_unavailable")

    try:
        eh = t.earnings_history
        if eh is not None and not eh.empty:
            out["earnings_history"] = [
                {"period": str(idx)[:10],
                 "eps_actual": (None if r.get("epsActual") != r.get("epsActual") else r.get("epsActual")),
                 "eps_estimate": (None if r.get("epsEstimate") != r.get("epsEstimate") else r.get("epsEstimate")),
                 "surprise_pct": (None if r.get("surprisePercent") != r.get("surprisePercent") else r.get("surprisePercent"))}
                for idx, r in eh.tail(8).iterrows()
            ]
    except Exception:
        out["flags"].append("earnings_history_unavailable")

    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            out["analyst_actions"] = [
                {"date": str(idx)[:10], "firm": r.get("Firm"),
                 "action": r.get("Action"), "to_grade": r.get("ToGrade"),
                 "from_grade": r.get("FromGrade"),
                 "price_target": r.get("currentPriceTarget"),
                 "prior_price_target": r.get("priorPriceTarget")}
                for idx, r in ud.head(8).iterrows()
            ]
    except Exception:
        out["flags"].append("analyst_actions_unavailable")

    return out


# Calendar days to trading days. Event dates are calendar dates; √t volatility
# scaling counts trading sessions. Scaling by calendar days overstates the move
# by ~25% (√(365/252)), which on a 90-day horizon is the difference between a
# plausible number and an alarming one.
TRADING_DAYS_PER_CALENDAR_DAY = 252.0 / 365.0


def _implied_move_pct(atr_14: Optional[float], current_price: Optional[float],
                      calendar_days: int) -> Optional[float]:
    """ATR-implied magnitude of ordinary drift over the window BEFORE the event
    — √t scaling of the realized daily range across the trading sessions in
    that window.

    This is not the size of the event's own move (that would need options data
    this app does not have). It answers "how far does this stock normally
    travel between now and then?", which is what determines whether a planned
    entry level is likely to be reached before the event at all.
    """
    if not atr_14 or not current_price or current_price <= 0 or calendar_days <= 0:
        return None
    sessions = max(1.0, calendar_days * TRADING_DAYS_PER_CALENDAR_DAY)
    daily = float(atr_14) / float(current_price)
    return round(min(daily * math.sqrt(sessions), 0.95) * 100, 1)


def _surprise_record(history: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Measured beat/miss record — a fact about the company's reporting history,
    with no claim that it repeats."""
    if not history:
        return None
    rows = [h for h in history if h.get("surprise_pct") is not None]
    if not rows:
        return None
    beats = sum(1 for h in rows if h["surprise_pct"] > 0)
    avg = sum(h["surprise_pct"] for h in rows) / len(rows)
    return {"n_quarters": len(rows), "beats": beats,
            "avg_surprise_pct": round(avg, 2),
            "statement": (f"Beat consensus in {beats} of the last {len(rows)} reported "
                          f"quarters (average surprise {avg:+.1f}%). Past reporting "
                          f"behaviour, not a prediction.")}


def build_catalyst_timeline(
    ticker: str,
    current_price: Optional[float] = None,
    atr_14: Optional[float] = None,
    raw: Optional[Dict[str, Any]] = None,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """The structured catalyst timeline.

    `status` is:
      OK            — at least one dated forward event
      NO_EVENTS     — the feeds worked and there is genuinely nothing scheduled
      UNAVAILABLE   — the feeds could not be read
    Distinguishing the last two matters: "nothing is coming" and "we could not
    look" lead to different decisions about how long a plan stays valid.
    """
    raw = raw if raw is not None else fetch_catalyst_inputs(ticker)
    today = as_of or date.today()
    flags = list(raw.get("flags") or [])
    cal = raw.get("calendar") or {}
    events: List[Dict[str, Any]] = []

    # ── Earnings ─────────────────────────────────────────────────────────
    ed = cal.get("Earnings Date")
    ed_list = ed if isinstance(ed, (list, tuple)) else ([ed] if ed else [])
    earnings_dates = [d for d in (_as_date(x) for x in ed_list) if d]
    surprise = _surprise_record(raw.get("earnings_history"))

    for d in sorted(earnings_dates):
        days_away = (d - today).days
        if days_away < 0:
            continue
        implied = _implied_move_pct(atr_14, current_price, max(days_away, 1))
        est_avg, est_hi, est_lo = (cal.get("Earnings Average"), cal.get("Earnings High"),
                                   cal.get("Earnings Low"))
        dispersion = None
        if est_avg and est_hi is not None and est_lo is not None and est_avg != 0:
            dispersion = round(abs(est_hi - est_lo) / abs(est_avg) * 100, 1)

        uncertainty_bits = []
        if dispersion is not None:
            uncertainty_bits.append(
                f"analyst EPS estimates span ${est_lo:.2f}–${est_hi:.2f} around a "
                f"${est_avg:.2f} average ({dispersion:.0f}% dispersion)")
        if implied is not None:
            uncertainty_bits.append(
                f"ordinary volatility implies the price travels ±{implied:.1f}% over the "
                f"{days_away} days BEFORE the event (this does not size the event's own move — "
                f"that would need options data this app does not collect)")
        if not uncertainty_bits:
            uncertainty_bits.append("no consensus or volatility data available to size the event")

        events.append({
            "date": d.isoformat(),
            "days_away": days_away,
            "event": "Quarterly earnings report",
            "category": "EARNINGS",
            "scheduled": True,
            "relevance": ("HIGH" if days_away <= IMMINENT_WINDOW_DAYS else
                          "MEDIUM" if days_away <= ACTIONABLE_WINDOW_DAYS else "LOW"),
            "consensus": ({"eps_average": est_avg, "eps_high": est_hi, "eps_low": est_lo,
                           "estimate_dispersion_pct": dispersion}
                          if est_avg is not None else None),
            "implied_move_to_event_pct": implied,
            "bull_implication": ("An in-line or better print with intact guidance removes the "
                                 "largest scheduled uncertainty in the window and lets the "
                                 "existing thesis run without this overhang."),
            "bear_implication": ("A miss or a guidance cut is the most likely single-session "
                                 "repricing event in the window, and would invalidate a thesis "
                                 "that rests on fundamental improvement."),
            "uncertainty": "; ".join(uncertainty_bits) + ".",
            "probability": None,
            "probability_basis": "NOT_CALIBRATED",
            "history": surprise,
            "source": "yfinance calendar + earnings history",
            "provenance": {"module": "decision/catalysts.py",
                           "feed": "yfinance Ticker.calendar / Ticker.earnings_history"},
        })

    # ── Dividend / ex-dividend ───────────────────────────────────────────
    for key, label in (("Ex-Dividend Date", "Ex-dividend date"),
                       ("Dividend Date", "Dividend payment date")):
        d = _as_date(cal.get(key))
        if not d:
            continue
        days_away = (d - today).days
        if days_away < 0:
            continue
        events.append({
            "date": d.isoformat(),
            "days_away": days_away,
            "event": label,
            "category": "DIVIDEND",
            "scheduled": True,
            "relevance": "LOW",
            "consensus": None,
            "implied_move_to_event_pct": None,
            "bull_implication": "Confirms the distribution is being maintained.",
            "bear_implication": ("Price adjusts down by the distribution on the ex-date — a "
                                 "mechanical move, not a signal."),
            "uncertainty": "Date is scheduled; amount may change.",
            "probability": None,
            "probability_basis": "NOT_CALIBRATED",
            "history": None,
            "source": "yfinance calendar",
            "provenance": {"module": "decision/catalysts.py", "feed": "yfinance Ticker.calendar"},
        })

    events.sort(key=lambda e: e["days_away"])

    # ── Recent analyst actions: past events, kept as context only ────────
    recent_actions = []
    for a in (raw.get("analyst_actions") or [])[:5]:
        d = _as_date(a.get("date"))
        if not d:
            continue
        age = (today - d).days
        if age < 0 or age > 120:
            continue
        pt, prior = a.get("price_target"), a.get("prior_price_target")
        move = None
        if pt and prior and prior > 0:
            move = round((pt / prior - 1) * 100, 1)
        recent_actions.append({
            "date": d.isoformat(), "days_ago": age, "firm": a.get("firm"),
            "action": a.get("action"), "from_grade": a.get("from_grade"),
            "to_grade": a.get("to_grade"), "price_target": pt,
            "prior_price_target": prior, "price_target_change_pct": move,
        })

    upcoming = [e for e in events if e["days_away"] <= ACTIONABLE_WINDOW_DAYS]

    # "What am I waiting for?" means the next event that could change the
    # decision, not the next date on the calendar. An ex-dividend date is a
    # mechanical price adjustment; treating it as the thing being waited on
    # would bury a real earnings catalyst behind a non-event.
    next_scheduled = events[0] if events else None
    material = [e for e in events if e["relevance"] != "LOW"]
    next_event = material[0] if material else next_scheduled

    if events:
        status = "OK"
    elif flags:
        status = "UNAVAILABLE"
    else:
        status = "NO_EVENTS"

    if next_event:
        waiting_for = (f"{next_event['event']} on {next_event['date']} "
                       f"({next_event['days_away']} days away).")
    elif status == "UNAVAILABLE":
        waiting_for = ("No catalyst calendar could be read — treat the plan as having no "
                       "known scheduled checkpoint, not as having none.")
    else:
        waiting_for = "Nothing scheduled in the feeds consulted."

    return {
        "status": status,
        "ticker": ticker.upper(),
        "as_of": today.isoformat(),
        "events": events,
        "upcoming": upcoming,
        "next_event": next_event,
        "next_scheduled_event": next_scheduled,
        "n_upcoming": len(upcoming),
        "recent_analyst_actions": recent_actions,
        "waiting_for": waiting_for,
        "flags": flags,
        "disclaimer": ("Only events with a provider-supplied date appear here. No catalyst "
                       "probability is stated because none is calibrated."),
    }


def evidence_from_catalysts(catalysts: Optional[Dict[str, Any]]) -> List[DecisionEvidence]:
    """A scheduled catalyst is NOT_DIRECTIONAL by construction.

    An upcoming earnings report is neither bullish nor bearish — it is a
    scheduled increase in uncertainty. Giving it a direction would be inventing
    the outcome, which is the exact failure mode Phase 5 forbids.
    """
    if not catalysts or catalysts.get("status") != "OK":
        return []
    out: List[DecisionEvidence] = []
    for e in catalysts.get("upcoming", [])[:3]:
        implied = e.get("implied_move_to_event_pct")
        out.append(DecisionEvidence(
            source=f"catalyst:{e['category'].lower()}:{e['date']}",
            category="CATALYST",
            metric="scheduled_event",
            observation=(f"{e['event']} in {e['days_away']} days"
                         + (f"; ordinary volatility implies ±{implied:.1f}% of drift before then"
                            if implied is not None else "")),
            direction="NOT_DIRECTIONAL",
            magnitude=min(1.0, (implied or 0.0) / 20.0),
            horizon="SHORT" if e["days_away"] <= 30 else "MEDIUM",
            reliability=0.9 if e.get("scheduled") else 0.4,
            validation_status="UNVALIDATED",
            as_of=catalysts.get("as_of"),
            data_quality="OK",
            provenance=e.get("provenance") or {},
            decision_relevance="DECISIVE" if e["relevance"] == "HIGH" else "CONTEXT",
            raw_value=e["date"],
            flags=[],
        ))
    return out
