"""
Verification for evidence-backed PIT fundamentals (FIL — EDGAR live wiring).

Run: python3 tests/test_fundamentals_pit.py

Group 1 is offline/deterministic (stub gateway) and proves the load-bearing
behaviors: ratios computed correctly, every number recorded as a traceable
claim, non-filers labelled unavailable rather than faked. Group 2 hits live
EDGAR and SKIPS (never fails) without network/config.
"""
import sys
import tempfile
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES, SKIPPED = [], []
TEST_DB = Path(tempfile.mkdtemp()) / "fpit.db"


def check(name, cond, detail=""):
    print(f"  {name:56s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def skip(name, why):
    print(f"  {name:56s} SKIP  {why}")
    SKIPPED.append(name)


def test_offline_with_stub():
    print("=== 1. Offline: ratios + evidence + honest unavailability ===")
    from data import ledger
    ledger.set_db_path(TEST_DB)

    import financial_data.gateway as gw
    from financial_data.schemas import make_datum, make_source

    def datum(concept, value):
        return make_datum("fundamentals_pit", value, "2024-02-01",
                          make_source("stub", document=f"ACCN-{concept}", ref=f"us-gaap:{concept}"),
                          symbol="TESTCO", concept=concept, unit="USD", period_end="2023-12-31")

    stub = types.ModuleType("stub_fund")
    def _fetch(kind, symbols, as_of=None, concepts=None, reliability=1.0, **kw):
        if "TESTCO" in [s.upper() for s in symbols]:
            vals = {"revenue": 1000, "net_income": 200, "gross_profit": 400,
                    "operating_income": 300, "assets": 5000, "equity": 800,
                    "long_term_debt": 400, "operating_cash_flow": 250}
            return {"data": [datum(c, v) for c, v in vals.items()],
                    "unavailable": [], "warnings": []}
        return {"data": [], "unavailable": [{"symbol": symbols[0], "reason": "no_sec_cik (not a filer)"}],
                "warnings": []}
    stub.fetch = _fetch
    sys.modules["stub_fund"] = stub
    reg = gw.load_registry()
    reg["providers"]["_stub_fund"] = {
        "id": "_stub_fund", "module": "stub_fund", "kinds": ["fundamentals_pit"],
        "pit_capable": ["fundamentals_pit"], "reliability": 1.0, "priority": 1, "caveats": {}}
    try:
        from agents.fundamentals_pit import analyze_fundamentals_pit
        r = analyze_fundamentals_pit("TESTCO", as_of="2024-06-01", run_id="t")
        check("available for a filer", r["available"])
        check("as_of_honored surfaced", r["as_of_honored"] is True)
        check("net_margin computed correctly (200/1000)",
              abs(r["ratios"]["net_margin"]["value"] - 0.2) < 1e-9,
              str(r["ratios"]["net_margin"]["value"]))
        check("debt_to_equity computed (400/800)",
              abs(r["ratios"]["debt_to_equity"]["value"] - 0.5) < 1e-9)
        check("every ratio carries a claim_id",
              all(v["claim_id"] for v in r["ratios"].values()))
        check("raw concepts trace to an accession",
              r["concepts"]["revenue"]["accession"] == "ACCN-revenue")

        # The claim is real and traceable.
        cid = r["ratios"]["net_margin"]["claim_id"]
        e = ledger.explain(cid)
        check("claim traces to its two datums", len(e["evidence"]) == 2)
        check("claim carries the PIT flag", e["as_of_honored"] is True)

        # Non-filer: labelled unavailable, never fabricated.
        r2 = analyze_fundamentals_pit("NOTAFILER", as_of="2024-06-01")
        check("non-filer -> available False", r2["available"] is False)
        check("non-filer -> reason names the cause, no fake numbers",
              r2["reason"] and not r2["ratios"] and not r2["concepts"],
              str(r2["reason"])[:40])

        # Divide-by-zero safety: a zero denominator drops the ratio, never inf/NaN.
        def _fz(kind, symbols, as_of=None, concepts=None, reliability=1.0, **kw):
            return {"data": [datum("net_income", 200), datum("revenue", 0)],
                    "unavailable": [], "warnings": []}
        stub.fetch = _fz
        r3 = analyze_fundamentals_pit("TESTCO", as_of="2024-06-01")
        check("zero denominator drops the ratio (no inf/NaN claim)",
              "net_margin" not in r3["ratios"])
    finally:
        reg["providers"].pop("_stub_fund", None)
        sys.modules.pop("stub_fund", None)
        gw._module_cache.pop("stub_fund", None)


def test_live():
    print("=== 2. LIVE EDGAR (skips without network/config) ===")
    from data import ledger
    ledger.set_db_path(TEST_DB)
    from agents.fundamentals_pit import analyze_fundamentals_pit
    from financial_data.providers.edgar import NotConfiguredError, resolve_cik

    try:
        if resolve_cik("AAPL") is None:
            skip("live AAPL", "no CIK"); return
    except NotConfiguredError as e:
        skip("live AAPL", f"NOT CONFIGURED: {str(e).splitlines()[0][:40]}"); return
    except Exception as e:
        skip("live AAPL", f"network: {str(e)[:40]}"); return

    r = analyze_fundamentals_pit("AAPL", as_of="2024-01-01", run_id="live")
    check("live AAPL available + PIT-honored", r["available"] and r["as_of_honored"])
    check("live net_margin plausible (0<m<0.6)",
          0 < r["ratios"].get("net_margin", {}).get("value", -1) < 0.6,
          str(r["ratios"].get("net_margin", {}).get("value")))
    check("live: no input filed after as_of",
          all(all(v <= "2024-01-01" for v in rr["inputs_filed"].values())
              for rr in r["ratios"].values()))


if __name__ == "__main__":
    test_offline_with_stub()
    test_live()
    print("\n" + "=" * 64)
    if SKIPPED:
        print(f"SKIPPED: {SKIPPED}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — evidence-backed PIT fundamentals (EDGAR -> EvidenceLedger)")
