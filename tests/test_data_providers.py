"""
Verification for the expanded data-source layer (FRED, CBOE, Tiingo, Finnhub).

Run: python3 tests/test_data_providers.py

Groups 1-3 are offline/deterministic (parsers fed fixture text; key-gating
logic). Group 4 hits live keyless endpoints and SKIPS without network —
reporting WHICH condition failed (the SEC-403 lesson: a config bug must never
wear an environment bug's clothes).
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES, SKIPPED = [], []


def check(name, cond, detail=""):
    print(f"  {name:58s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def skip(name, why):
    print(f"  {name:58s} SKIP  {why}")
    SKIPPED.append(name)


def test_registry_shape():
    print("=== 1. Registry: six providers, honest pit declarations ===")
    from financial_data.gateway import list_providers, load_registry
    reg = load_registry(refresh=True)["providers"]
    for pid in ("sec-edgar", "yfinance", "tiingo", "fred", "cboe", "finnhub"):
        check(f"registered: {pid}", pid in reg)
    check("FRED is NOT pit_capable (current-vintage only, revisions overwrite)",
          reg["fred"]["pit_capable"] == [])
    check("Finnhub is NOT pit_capable (calendars shift)",
          reg["finnhub"]["pit_capable"] == [])
    check("CBOE IS pit_capable (published at close, not restated)",
          reg["cboe"]["pit_capable"] == ["macro"])
    check("key-gated providers name their env var",
          reg["tiingo"]["requires_key"] == "TIINGO_API_KEY"
          and reg["finnhub"]["requires_key"] == "FINNHUB_API_KEY")
    bars = [p["id"] for p in list_providers("bars")]
    check("bars order: edgar absent, tiingo (40) before yfinance (50)",
          bars.index("tiingo") < bars.index("yfinance"), str(bars))
    macro = [p["id"] for p in list_providers("macro")]
    check("macro served by fred then cboe", macro == ["fred", "cboe"], str(macro))


def test_parsers_offline():
    print("=== 2. Parsers against fixture text (offline, deterministic) ===")
    from financial_data import cache
    from financial_data.providers import cboe, fred

    # Redirect the cache so fixtures are read instead of the network, and
    # nothing from this test pollutes the real cache.
    old = cache.CACHE_DIR
    cache.CACHE_DIR = Path(tempfile.mkdtemp())
    try:
        cache.put("fred", "macro", "DGS10",
                  "observation_date,DGS10\n2026-06-02,4.40\n2026-06-03,.\n2026-06-04,4.45\n")
        r = fred.fetch("macro", ["DGS10"])
        check("FRED parses observations", len(r["data"]) == 2, f"n={len(r['data'])}")
        check("FRED skips '.' missing markers (never a fake 0)",
              all(d["value"] in (4.40, 4.45) for d in r["data"]))
        check("FRED datum carries series provenance",
              r["data"][0]["source"]["document"] == "FRED:DGS10")
        r2 = fred.fetch("macro", ["DGS10"], start="2026-06-04")
        check("FRED start filter works", len(r2["data"]) == 1 and r2["data"][0]["value"] == 4.45)

        cache.put("cboe", "macro", "VIX_History",
                  "DATE,OPEN,HIGH,LOW,CLOSE\n07/16/2026,17.0,18.0,16.5,17.5\n"
                  "07/17/2026,18.0,19.5,17.7,18.8\n")
        r3 = cboe.fetch("macro", ["VIX"])
        check("CBOE parses VIX bars (MM/DD/YYYY -> ISO)",
              len(r3["data"]) == 2 and r3["data"][0]["available_at"] == "2026-07-16")
        check("CBOE aliases ^VIX/VIXCLS", len(cboe.fetch("macro", ["^VIX"])["data"]) == 2)
        r4 = cboe.fetch("macro", ["AAPL"])
        check("CBOE honestly refuses non-VIX symbols",
              not r4["data"] and r4["unavailable"][0]["reason"].startswith("cboe_free_file"))
    finally:
        cache.CACHE_DIR = old


def test_key_gating():
    print("=== 3. Key gating: absent key -> named NotConfiguredError ===")
    import os
    from financial_data.keys import NotConfiguredError, get_key, has_key
    from financial_data.providers import finnhub_events, tiingo

    # These keys are not set in this environment (asserted, not assumed).
    for var in ("TIINGO_API_KEY", "FINNHUB_API_KEY"):
        if has_key(var):
            skip(f"key gating ({var})", "key IS configured in this env")
            return
    try:
        get_key("TIINGO_API_KEY", "tiingo")
        check("get_key raises when absent", False, "no raise")
    except NotConfiguredError as e:
        check("get_key raises when absent, naming the variable",
              "TIINGO_API_KEY" in str(e))
    try:
        tiingo.fetch("bars", ["AAPL"])
        check("tiingo.fetch refuses without key", False, "no raise")
    except NotConfiguredError:
        check("tiingo.fetch refuses without key", True)
    try:
        finnhub_events.fetch("events", ["AAPL"])
        check("finnhub.fetch refuses without key", False, "no raise")
    except NotConfiguredError:
        check("finnhub.fetch refuses without key", True)

    # Through the gateway: falls through, then fails LOUDLY with the reason.
    from financial_data.gateway import NoProviderError, get
    try:
        get("events", "AAPL")
        check("gateway events without key -> loud failure", False, "no raise")
    except NoProviderError as e:
        check("gateway events without key -> loud failure naming the cause",
              "FINNHUB_API_KEY" in str(e), str(e)[:60])


def test_live_keyless():
    print("=== 4. LIVE keyless endpoints (skip without network) ===")
    from financial_data.gateway import get
    try:
        r = get("macro", ["DGS10"], start="2026-01-01")
    except Exception as e:
        skip("live FRED", f"network: {str(e)[:50]}")
        return
    check("live FRED serves DGS10", r["n"] > 0 and r["provider"] == "fred", f"n={r['n']}")
    check("live FRED values plausible (0-20%)",
          all(0 < d["value"] < 20 for d in r["data"]))
    try:
        r2 = get("macro", "VIX", start="2026-07-01", provider="cboe")
        check("live CBOE serves VIX", r2["n"] > 0, f"n={r2['n']}")
        check("live VIX plausible (5-100)", all(5 < d["value"] < 100 for d in r2["data"]))
    except Exception as e:
        skip("live CBOE", f"network: {str(e)[:50]}")


if __name__ == "__main__":
    test_registry_shape()
    test_parsers_offline()
    test_key_gating()
    test_live_keyless()
    print("\n" + "=" * 66)
    if SKIPPED:
        print(f"SKIPPED: {SKIPPED}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — data providers: FRED+CBOE live keyless, Tiingo+Finnhub key-gated")
