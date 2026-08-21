"""
Structural validation for the prediction agent's parsed JSON output.

agents/stock_agents.py::run_prediction_agent already extracts and
json.loads()'s the model's response defensively (regex extraction, one
retry, a hardcoded fallback on total failure) - but a response that IS
valid JSON can still be structurally wrong in ways that check never
catches: a missing key, "action": "Maybe" instead of one of the three
allowed values, upside_pct as the string "15.5%" instead of a number, or
a stray 1550 that was meant to be 15.5. validate_prediction() catches
these specifically, and never raises - mirrors tools/eval/judges.py's
Verdict, which also never raises on a malformed model response.

PREDICTION_SCHEMA (the prompt-facing example) lives in stock_agents.py;
the field spec below is a separate, stricter description of the same
shape. Keeping them in agents/stock_agents.py and here respectively -
rather than one deriving from the other - is a real, acknowledged risk
of drift; the two are re-synced by hand whenever either changes, and
this module's REQUIRED_KEYS should be checked against
stock_agents.PREDICTION_SCHEMA's keys any time one is edited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number

ACTIONS = {"BUY", "SELL", "HOLD"}
CONVICTIONS = {"HIGH", "MEDIUM", "LOW"}

# key -> expected type(s). A tuple means "any of these types is OK."
FIELD_TYPES = {
    "action":           str,
    "conviction":       str,
    "time_horizon":     str,
    "time_horizon_days": (int, float),
    "entry_price":      str,
    "target_price":     str,
    "stop_loss":        str,
    "upside_pct":       (int, float),
    "downside_pct":     (int, float),
    "risk_reward":      str,
    "summary":          str,
    "bull_case":        str,
    "bear_case":        str,
    "key_catalysts":    list,
    "watch_levels":     dict,
    "scores":           dict,
}

REQUIRED_KEYS = tuple(FIELD_TYPES)

# Generous bounds - wide enough to never reject a genuine outlier call,
# tight enough to catch an obvious unit error (e.g. 1550 meant as 15.5).
_PCT_BOUND = 300.0
_SCORE_BOUND = (0.0, 10.0)


@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)


def _check_type(key: str, value, expected) -> str | None:
    if isinstance(expected, tuple):
        # bool is a subclass of int - reject it explicitly for numeric fields
        # so a stray `true`/`false` doesn't silently pass as 1/0.
        if isinstance(value, bool) or not isinstance(value, expected):
            return f"{key}: expected {expected}, got {type(value).__name__}"
        return None
    if not isinstance(value, expected):
        return f"{key}: expected {expected.__name__}, got {type(value).__name__}"
    return None


def validate_prediction(data: dict) -> ValidationResult:
    """Never raises - always returns a ValidationResult, even for garbage input."""
    errors: list = []

    if not isinstance(data, dict):
        return ValidationResult(valid=False, errors=[f"top-level value is {type(data).__name__}, not a dict"])

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key}")

    for key, expected in FIELD_TYPES.items():
        if key not in data:
            continue
        err = _check_type(key, data[key], expected)
        if err:
            errors.append(err)

    if "action" in data and isinstance(data["action"], str) and data["action"] not in ACTIONS:
        errors.append(f"action: {data['action']!r} not in {sorted(ACTIONS)}")

    if "conviction" in data and isinstance(data["conviction"], str) and data["conviction"] not in CONVICTIONS:
        errors.append(f"conviction: {data['conviction']!r} not in {sorted(CONVICTIONS)}")

    for pct_key in ("upside_pct", "downside_pct"):
        val = data.get(pct_key)
        if isinstance(val, Number) and not isinstance(val, bool) and abs(val) > _PCT_BOUND:
            errors.append(f"{pct_key}: {val} is outside the plausible ±{_PCT_BOUND}% range")

    scores = data.get("scores")
    if isinstance(scores, dict):
        lo, hi = _SCORE_BOUND
        for sub_key, sub_val in scores.items():
            if not isinstance(sub_val, Number) or isinstance(sub_val, bool):
                errors.append(f"scores.{sub_key}: expected a number, got {type(sub_val).__name__}")
            elif not (lo <= sub_val <= hi):
                errors.append(f"scores.{sub_key}: {sub_val} outside plausible {lo}-{hi} range")

    return ValidationResult(valid=not errors, errors=errors)
