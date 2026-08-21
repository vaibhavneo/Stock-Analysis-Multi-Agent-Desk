"""
Verification for Phase 4: native structured output (response_format) in
agents/stock_agents.py::_call, and its wiring into run_prediction_agent.

Run: python3 tests/test_call_response_format.py

Offline/deterministic - client.chat.completions.create is stubbed via
unittest.mock; no network, no API key. Separately confirmed LIVE
(2026-08-21) that deepseek-v4-pro actually honors
response_format={"type": "json_object"} (2.1s round trip, clean JSON) -
this file only guards the Python-level wiring around that fact, not the
live model behavior itself.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stock_agents import _call, run_prediction_agent

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:66s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def _fake_client(content_or_exc):
    client = MagicMock()

    def _create(**kwargs):
        if isinstance(content_or_exc, Exception):
            raise content_or_exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content_or_exc))])

    client.chat.completions.create.side_effect = _create
    return client


def test_call_without_response_format_omits_the_kwarg():
    client = _fake_client("hello")
    _call(client, "sys", "user")
    kwargs = client.chat.completions.create.call_args.kwargs
    check("response_format is not sent when not requested",
          "response_format" not in kwargs, str(kwargs))


def test_call_with_response_format_includes_the_kwarg():
    client = _fake_client('{"ok": true}')
    _call(client, "sys", "user", response_format={"type": "json_object"})
    kwargs = client.chat.completions.create.call_args.kwargs
    check("response_format is sent when requested",
          kwargs.get("response_format") == {"type": "json_object"}, str(kwargs))


def test_call_retries_without_response_format_on_rejection():
    client = MagicMock()
    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise Exception("Unsupported parameter: 'response_format'")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="plain text ok"))])

    client.chat.completions.create.side_effect = _create
    result = _call(client, "sys", "user", response_format={"type": "json_object"})
    check("falls back to a plain-text call when response_format is rejected",
          result == "plain text ok", repr(result))
    check("exactly 2 calls made (1 with response_format, 1 without)", len(calls) == 2)
    check("first call had response_format, second didn't",
          "response_format" in calls[0] and "response_format" not in calls[1])


def test_call_returns_agent_error_string_when_both_attempts_fail():
    client = _fake_client(RuntimeError("timeout"))
    result = _call(client, "sys", "user", response_format={"type": "json_object"})
    check("returns an '[Agent error:' string, never raises",
          result.startswith("[Agent error:"), repr(result))


_VALID_TEMPLATE = {
    "action": "BUY", "conviction": "HIGH", "time_horizon": "medium-term (1-3 months)",
    "time_horizon_days": 90, "entry_price": "$100", "target_price": "$120",
    "stop_loss": "$90", "upside_pct": 20.0, "downside_pct": -10.0, "risk_reward": "2:1",
    "summary": "ok", "bull_case": "ok", "bear_case": "ok", "key_catalysts": [],
    "watch_levels": {"support": "$95", "resistance": "$115"},
    "scores": {"fundamentals": 7, "technical": 7, "social": 6, "algo": 7, "overall": 7},
}


def _run_with_stub_call(raw_response):
    with patch("agents.stock_agents._call") as mock_call:
        mock_call.return_value = raw_response
        result = run_prediction_agent(
            client=object(), ticker="TEST", company_name="Test Co", current_price=100.0,
            fundamentals_analysis="...", technical_analysis="...",
            signal_summary={"score": 50, "direction": "NEUTRAL"},
        )
        return result, mock_call


def test_run_prediction_agent_requests_json_object_mode():
    result, mock_call = _run_with_stub_call(json.dumps(_VALID_TEMPLATE))
    _args, kwargs = mock_call.call_args
    check("run_prediction_agent requests JSON object mode",
          kwargs.get("response_format") == {"type": "json_object"}, str(kwargs))
    check("only 1 call made for an already-valid response", mock_call.call_count == 1)


# ── Defense-in-depth: extraction still recovers from wrapped/prefixed JSON ──
# response_format reduces but doesn't guarantee zero malformed responses -
# the regex extraction from before Phase 4 stays in place unchanged.

def test_extraction_survives_markdown_fences():
    wrapped = "```json\n" + json.dumps(dict(_VALID_TEMPLATE, action="HOLD")) + "\n```"
    result, _ = _run_with_stub_call(wrapped)
    check("markdown-fenced JSON is still extracted and validated",
          result["action"] == "HOLD", str(result))


def test_extraction_survives_leading_prose():
    prefixed = "Here is my decision:\n" + json.dumps(dict(_VALID_TEMPLATE, action="SELL"))
    result, _ = _run_with_stub_call(prefixed)
    check("JSON prefixed by leading prose is still extracted and validated",
          result["action"] == "SELL", str(result))


if __name__ == "__main__":
    test_call_without_response_format_omits_the_kwarg()
    test_call_with_response_format_includes_the_kwarg()
    test_call_retries_without_response_format_on_rejection()
    test_call_returns_agent_error_string_when_both_attempts_fail()
    test_run_prediction_agent_requests_json_object_mode()
    test_extraction_survives_markdown_fences()
    test_extraction_survives_leading_prose()
    print("\n" + "=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — _call response_format wiring + extraction defense-in-depth")
