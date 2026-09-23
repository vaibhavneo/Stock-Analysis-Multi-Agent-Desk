#!/usr/bin/env python3
"""Canary: no credential may leave the process in anything a user sees.

The research pipeline touches keyed providers, an HTTP service and an LLM.
Each is a place a secret could ride out in a trace, an evidence item, a plan,
or a prompt. The failure is silent — a leaked key looks like any other string
— so it is asserted rather than reviewed.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0
FAIL = 0

# Planted values. If any appears in output, something is serialising the
# environment rather than the evidence.
CANARIES = {
    "DEEPSEEK_API_KEY": "sk-CANARY-deepseek-3f9a2b",
    "OPTIONSPILOT_ACCESS_CODE": "CANARY-optionspilot-7c1d",
    "TIINGO_API_KEY": "CANARY-tiingo-91be",
    "FINNHUB_API_KEY": "CANARY-finnhub-44af",
}


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label} {detail}")
    assert condition, f"{label} {detail}"


def _leaked(blob: str):
    return [name for name, value in CANARIES.items() if value in blob]


def _with_canaries():
    saved = {k: os.environ.get(k) for k in CANARIES}
    os.environ.update(CANARIES)
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_tool_discovery_never_echoes_a_key_value():
    """Discovery reports WHICH variable is missing, which is useful. It must
    never report what the variable contains."""
    saved = _with_canaries()
    try:
        from mas.research.tools import discover
        for ac in ("EQUITY", "CRYPTO", None):
            blob = json.dumps(discover(ac))
            check(f"discover({ac}) leaks nothing", not _leaked(blob),
                  _leaked(blob))
    finally:
        _restore(saved)


def test_a_research_plan_never_carries_a_credential():
    saved = _with_canaries()
    try:
        from mas.research.plan import build_plan
        for q in ("Should I add to my NVDA position?",
                  "show me options on NVDA", "NVDA"):
            blob = json.dumps(build_plan(q, symbols=["NVDA"]).to_dict())
            check(f"plan for {q!r} leaks nothing", not _leaked(blob),
                  _leaked(blob))
    finally:
        _restore(saved)


def test_the_execution_trace_never_carries_a_credential():
    saved = _with_canaries()
    try:
        from mas.research.execute import execute
        from mas.research.plan import build_plan

        plan = build_plan("I own 10 at 100. Should I add to NVDA?",
                          symbols=["NVDA"])

        def runner(cap):
            # An adapter that leaks the environment into its error, which is
            # exactly how a key reaches a user-visible trace.
            raise RuntimeError(f"auth failed with {os.environ['DEEPSEEK_API_KEY']}")

        out = execute(plan, runner, timeout_sec=3)
        blob = json.dumps({"steps": {k: v.to_dict() for k, v in out["steps"].items()},
                           "summary": out["summary"]})
        leaks = _leaked(blob)
        check("a leaking adapter is contained or redacted",
              not leaks, f"LEAKED: {leaks}")
    finally:
        _restore(saved)


def test_api_status_does_not_publish_a_key():
    saved = _with_canaries()
    try:
        from web.app import app
        app.config["TESTING"] = True
        c = app.test_client()
        for path in ("/api/status", "/api/mas/agents", "/api/research/tools",
                     "/api/maintenance"):
            r = c.get(path)
            blob = r.get_data(as_text=True)
            check(f"{path} leaks nothing", not _leaked(blob), _leaked(blob))
    finally:
        _restore(saved)


def test_the_registry_declares_key_names_never_key_values():
    from mas import registry
    blob = json.dumps(registry.load())
    for name in CANARIES:
        if name in blob:
            check(f"{name} appears only as a variable NAME",
                  CANARIES[name] not in blob)
    check("registry carries no canary value", not _leaked(blob), _leaked(blob))


def test_evidence_items_carry_provenance_not_credentials():
    saved = _with_canaries()
    try:
        from mas.research.evidence import Item, Ledger, OBSERVATION, FRESH
        L = Ledger()
        L.add(Item("k", OBSERVATION, "x", "src", reliability=0.5,
                   confidence=0.5, freshness=FRESH,
                   provenance={"capability": "equity_research",
                               "tool": "bars_daily"}))
        blob = json.dumps(L.to_dict())
        check("ledger leaks nothing", not _leaked(blob), _leaked(blob))
    finally:
        _restore(saved)


def test_no_source_file_hard_codes_a_secret_looking_literal():
    """A key committed to the repository is the one leak no runtime check
    catches."""
    import re
    root = os.path.join(os.path.dirname(__file__), "..")
    pattern = re.compile(r"""(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})""")
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", ".cache", "node_modules",
                                    "logs", ".venv")]
        for fn in filenames:
            if not fn.endswith((".py", ".json", ".md", ".html", ".sh", ".txt")):
                continue
            path = os.path.join(dirpath, fn)
            if os.path.basename(path) == "test_secrets.py":
                continue
            try:
                with open(path, "r", errors="ignore") as f:
                    body = f.read()
            except OSError:
                continue
            if pattern.search(body):
                offenders.append(os.path.relpath(path, root))
    check("no committed secret-shaped literal", not offenders, offenders)


if __name__ == "__main__":
    import traceback
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception:
                traceback.print_exc()
    print(f"\n{PASS} passed, {FAIL} failed")
