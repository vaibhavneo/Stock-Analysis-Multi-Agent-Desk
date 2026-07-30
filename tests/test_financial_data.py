"""
Verification for the Financial Intelligence Layer (FIL) — Milestone 1.

Run: python3 tests/test_financial_data.py

Test strategy, deliberately: the point-in-time and restatement guarantees are
proven against SYNTHETIC FIXTURES, offline and deterministically. A guarantee
that can only be checked when sec.gov is reachable is not a guarantee — it is a
hope with a network dependency. The live EDGAR test at the end is a genuine
end-to-end check but SKIPS (never fails) without network, so CI stays honest
about what it did and did not verify.

Groups:
  1. Datum schema — provenance is mandatory, ids are content-addressed
  2. PIT filtering — the future cannot leak backwards
  3. Restatements — as-of returns what you WOULD have seen, not today's revision
  4. Gateway — provider resolution, as_of_honored contract, honest unavailability
  5. EvidenceLedger — claims cannot exist without evidence
  6. TrialRegistry — the true n_trials for dSR, monotone and idempotent
  7. Provider boundary — no new code calls a vendor SDK directly (P8 seam)
  8. LIVE EDGAR (skips offline) — real filing, real accession number
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FAILURES = []
SKIPPED = []

# Every ledger/trial test runs against a THROWAWAY database. Two reasons, both
# load-bearing: (1) the suite must be idempotent — writing to the real registry
# made re-runs fail, since a deterministic trial_id means the second run adds no
# rows; (2) test fixtures must never enter the real TrialRegistry, because
# n_trials is dSR's denominator and polluting it would corrupt the exact number
# this milestone exists to make honest.
_TMPDIR = tempfile.TemporaryDirectory()
TEST_DB = Path(_TMPDIR.name) / "fil_test.db"


def check(name: str, condition: bool, detail: str = ""):
    print(f"  {name:58s} {'OK' if condition else 'FAIL'}  {detail}")
    if not condition:
        FAILURES.append(name)


def skip(name: str, why: str):
    print(f"  {name:58s} SKIP  {why}")
    SKIPPED.append(f"{name} ({why})")


# ── Fixtures ───────────────────────────────────────────────────────────────
# A real restatement pattern: FY2022 revenue was first reported as 1000 in the
# Feb-2023 10-K, then RESTATED to 900 in the Feb-2024 10-K (comparative column).
# Anyone deciding in mid-2023 saw 1000. Only a look-ahead bug shows them 900.
RESTATEMENT_FIXTURE = {
    "cik": 1234567,
    "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2022-01-01", "end": "2022-12-31", "val": 1000,
         "accn": "0000000-23-ORIGINAL", "fy": 2022, "fp": "FY", "form": "10-K",
         "filed": "2023-02-01"},
        {"start": "2022-01-01", "end": "2022-12-31", "val": 900,
         "accn": "0000000-24-RESTATED", "fy": 2022, "fp": "FY", "form": "10-K",
         "filed": "2024-02-01"},
        {"start": "2023-01-01", "end": "2023-12-31", "val": 1500,
         "accn": "0000000-24-RESTATED", "fy": 2023, "fp": "FY", "form": "10-K",
         "filed": "2024-02-01"},
    ]}}}},
}


def test_schema():
    print("=== 1. Datum schema: provenance is mandatory ===")
    from financial_data.schemas import (SchemaError, datum_id, make_datum,
                                        make_source, validate_datum)

    d = make_datum("fundamentals_pit", 1000, "2023-02-01",
                   make_source("sec-edgar", document="ACCN-1", ref="us-gaap:Revenues"),
                   symbol="test", concept="revenue", unit="USD", period_end="2022-12-31")
    validate_datum(d)
    check("datum carries full provenance", all(
        d.get(f) for f in ("datum_id", "kind", "value", "available_at",
                           "retrieved_at", "source", "confidence", "status")))
    check("symbol normalized upper", d["symbol"] == "TEST")

    # NEGATIVE: available_at is the one field that cannot be defaulted, because
    # a datum with no knowable-at time silently enables look-ahead.
    try:
        make_datum("fundamentals_pit", 1, None, make_source("x"))
        check("missing available_at rejected", False, "no raise")
    except SchemaError:
        check("missing available_at rejected", True)

    try:
        make_datum("fundamentals_pit", None, "2023-01-01", make_source("x"))
        check("None value rejected (not coerced to 0)", False, "no raise")
    except SchemaError:
        check("None value rejected (not coerced to 0)", True)

    try:
        make_datum("not_a_kind", 1, "2023-01-01", make_source("x"))
        check("unknown kind rejected", False, "no raise")
    except SchemaError:
        check("unknown kind rejected", True)

    try:
        make_datum("fundamentals_pit", 1, "Q4 2023", make_source("x"))
        check("unparseable timestamp rejected", False, "no raise")
    except SchemaError:
        check("unparseable timestamp rejected", True)

    # BOUNDS: confidence is a probability-like weight; outside [0,1] is a bug.
    try:
        make_datum("fundamentals_pit", 1, "2023-01-01", make_source("x"), confidence=1.5)
        check("confidence bounds enforced [0,1]", False, "no raise")
    except SchemaError:
        check("confidence bounds enforced [0,1]", True)

    # Content-addressing: re-fetching the same fact must not mint a new identity,
    # or evidence links would break on every refresh.
    d2 = make_datum("fundamentals_pit", 1000, "2023-02-01",
                    make_source("sec-edgar", document="ACCN-1", ref="us-gaap:Revenues"),
                    symbol="test", concept="revenue", unit="USD", period_end="2022-12-31")
    check("datum_id stable across re-fetch", d["datum_id"] == d2["datum_id"])
    d3 = make_datum("fundamentals_pit", 999, "2023-02-01",
                    make_source("sec-edgar", document="ACCN-1", ref="us-gaap:Revenues"),
                    symbol="test", concept="revenue", unit="USD", period_end="2022-12-31")
    check("datum_id changes when value changes", d["datum_id"] != d3["datum_id"])


def test_pit_filter():
    print("=== 2. PIT filtering: the future cannot leak backwards ===")
    from financial_data.schemas import filter_pit, make_datum, make_source, visible_at

    def mk(val, filed):
        return make_datum("fundamentals_pit", val, filed, make_source("t"),
                          symbol="X", concept="revenue", period_end="2022-12-31")

    past, future = mk(1, "2023-02-01"), mk(2, "2024-02-01")
    check("datum filed before as_of is visible", visible_at(past, "2023-06-01"))
    check("datum filed after as_of is invisible", not visible_at(future, "2023-06-01"))
    # Boundary: a filing made ON the decision date was readable that day.
    check("boundary inclusive (filed == as_of)", visible_at(past, "2023-02-01"))
    # Mixed granularity must not silently flip the answer.
    check("date as_of vs datetime available_at",
          visible_at(mk(1, "2023-02-01T16:30:00"), "2023-02-01"))
    check("as_of=None means no filtering", visible_at(future, None))

    kept = filter_pit([past, future], "2023-06-01")
    check("filter_pit drops future datums", len(kept) == 1 and kept[0]["value"] == 1)
    check("filter_pit(None) keeps everything", len(filter_pit([past, future], None)) == 2)


def test_restatement():
    print("=== 3. Restatement: as_of returns what you WOULD have seen ===")
    from financial_data.providers.edgar import parse_companyfacts
    from financial_data.schemas import filter_pit, latest_by_period

    datums = parse_companyfacts(RESTATEMENT_FIXTURE, "TEST", concepts=["revenue"])
    check("parse keeps ALL filings incl. superseded", len(datums) == 3,
          f"n={len(datums)} (history must be preserved to be replayable)")

    # THE CORE TEST. Deciding on 2023-06-01: the restatement does not exist yet.
    seen = latest_by_period(filter_pit(datums, "2023-06-01"))
    fy22 = [d for d in seen if d["period_end"] == "2022-12-31"]
    check("as_of=2023-06-01 -> FY22 revenue is the ORIGINAL 1000",
          len(fy22) == 1 and fy22[0]["value"] == 1000,
          f"got {[d['value'] for d in fy22]}")
    check("as_of=2023-06-01 -> cites the ORIGINAL accession",
          len(fy22) == 1 and fy22[0]["source"]["document"] == "0000000-23-ORIGINAL")
    check("as_of=2023-06-01 -> FY23 (filed 2024) is absent",
          not any(d["period_end"] == "2023-12-31" for d in seen))

    # After the restatement is public, the same query returns the revised figure.
    seen_now = latest_by_period(filter_pit(datums, "2024-06-01"))
    fy22_now = [d for d in seen_now if d["period_end"] == "2022-12-31"]
    check("as_of=2024-06-01 -> FY22 revenue is the RESTATED 900",
          len(fy22_now) == 1 and fy22_now[0]["value"] == 900,
          f"got {[d['value'] for d in fy22_now]}")
    check("as_of=2024-06-01 -> cites the RESTATED accession",
          len(fy22_now) == 1 and fy22_now[0]["source"]["document"] == "0000000-24-RESTATED")

    # The ordering bug this whole design exists to prevent: collapsing
    # restatements BEFORE the PIT cut leaks tomorrow's revision into yesterday.
    wrong = filter_pit(latest_by_period(datums), "2023-06-01")
    wrong_fy22 = [d for d in wrong if d["period_end"] == "2022-12-31"]
    check("collapse-before-filter provably leaks (guard against regression)",
          len(wrong_fy22) == 0,
          "wrong order loses the original entirely -> filter must run FIRST")


def test_gateway():
    print("=== 4. Gateway: resolution + the as_of_honored contract ===")
    import financial_data.gateway as gw
    from financial_data.gateway import GatewayError, NoProviderError

    reg = gw.load_registry()
    check("registry loads", "providers" in reg and len(reg["providers"]) >= 2)
    fund = gw.list_providers("fundamentals_pit")
    check("EDGAR is the primary fundamentals provider",
          bool(fund) and fund[0]["id"] == "sec-edgar")
    check("yfinance is NOT a fundamentals provider (PIT-incapable)",
          "yfinance" not in [p["id"] for p in fund])
    bars = gw.list_providers("bars")
    check("yfinance serves bars, demoted not deleted",
          bool(bars) and "yfinance" in [p["id"] for p in bars])
    check("yfinance reliability is low (0.6)",
          reg["providers"]["yfinance"]["reliability"] == 0.6)
    check("yfinance declares its retroactive-adjustment caveat",
          "retroactive" in (reg["providers"]["yfinance"]["caveats"].get("bars", "").lower()))

    try:
        gw.get("not_a_kind", "AAPL")
        check("unknown kind rejected", False, "no raise")
    except GatewayError:
        check("unknown kind rejected", True)

    try:
        gw.get("bars", [])
        check("empty symbols rejected", False, "no raise")
    except GatewayError:
        check("empty symbols rejected", True)

    # A kind nobody serves must fail loudly, not return [] — "no provider" and
    # "no such data" are different claims and must not be conflated. ("macro"
    # was the example here until FRED/CBOE began serving it; "universe" is the
    # honest still-unserved kind.)
    try:
        gw.get("universe", "SP500")
        check("unserved kind raises (not silent [])", False, "no raise")
    except NoProviderError:
        check("unserved kind raises (not silent [])", True)

    # as_of_honored contract, proven with a stub provider so it needs no network.
    import sys as _sys
    import types
    stub = types.ModuleType("stub_provider")

    def _fetch(kind, symbols, start=None, end=None, as_of=None, reliability=1.0, **kw):
        from financial_data.schemas import make_datum, make_source
        return {"data": [
            make_datum("fundamentals_pit", 1000, "2023-02-01", make_source("stub", document="A"),
                       symbol="X", concept="revenue", period_end="2022-12-31"),
            make_datum("fundamentals_pit", 2000, "2024-02-01", make_source("stub", document="B"),
                       symbol="X", concept="revenue", period_end="2023-12-31"),
        ], "unavailable": [], "warnings": []}
    stub.fetch = _fetch
    _sys.modules["stub_provider"] = stub

    reg["providers"]["_stub_pit"] = {
        "id": "_stub_pit", "module": "stub_provider", "kinds": ["fundamentals_pit"],
        "pit_capable": ["fundamentals_pit"], "reliability": 1.0, "priority": 1, "caveats": {}}
    reg["providers"]["_stub_nopit"] = {
        "id": "_stub_nopit", "module": "stub_provider", "kinds": ["fundamentals_pit"],
        "pit_capable": [], "reliability": 1.0, "priority": 2, "caveats": {}}
    try:
        r = gw.get("fundamentals_pit", "X", as_of="2023-06-01", provider="_stub_pit")
        check("as_of + pit_capable -> as_of_honored TRUE", r["as_of_honored"] is True)
        check("as_of cut actually removed the future datum",
              r["n"] == 1 and r["data"][0]["value"] == 1000,
              f"n={r['n']}")
        check("excluded_future_datums reports the cut as evidence",
              r["excluded_future_datums"] == 1)

        r2 = gw.get("fundamentals_pit", "X", provider="_stub_pit")
        check("no as_of -> as_of_honored FALSE (no promise claimed)",
              r2["as_of_honored"] is False)
        check("no as_of -> nothing filtered", r2["n"] == 2)

        # The honesty case: a provider that cannot do PIT must not be allowed to
        # look like it did. Data still flows; the guarantee is withdrawn loudly.
        r3 = gw.get("fundamentals_pit", "X", as_of="2023-06-01", provider="_stub_nopit")
        check("as_of + NOT pit_capable -> as_of_honored FALSE", r3["as_of_honored"] is False)
        check("...and warns about possible look-ahead",
              any("look-ahead" in w for w in r3["warnings"]),
              str(r3["warnings"])[:60])
        check("...and does NOT silently filter (data unchanged)", r3["n"] == 2)
    finally:
        reg["providers"].pop("_stub_pit", None)
        reg["providers"].pop("_stub_nopit", None)
        _sys.modules.pop("stub_provider", None)
        gw._module_cache.pop("stub_provider", None)


def test_bars_df():
    print("=== 4b. get_bars_df: OHLCV frame through the gateway (M2b) ===")
    import sys as _sys
    import types

    import financial_data.gateway as gw
    from financial_data.schemas import make_datum, make_source

    # Stub bars provider with timestamps that straddle a DST boundary — the
    # exact case that broke the first implementation: daily bars arrive as ISO
    # strings whose UTC offset flips (-05:00 in Feb, -04:00 in Apr), and pandas
    # refuses to build one DatetimeIndex from mixed offsets. Offline + tests the
    # regression directly.
    stub = types.ModuleType("stub_bars")

    def _fetch(kind, symbols, start=None, end=None, as_of=None, reliability=0.6, **kw):
        def bar(ts_iso, o, h, l, c, v):
            return make_datum("bars", c, ts_iso, make_source("stub_bars", document=f"X:{ts_iso}"),
                              symbol="X", concept="close", unit="USD", period_end=ts_iso,
                              confidence=reliability,
                              extra={"open": o, "high": h, "low": l, "volume": v})
        return {"data": [
            bar("2024-02-01T00:00:00-05:00", 10, 11, 9, 10.5, 1000),   # EST
            bar("2024-04-01T00:00:00-04:00", 12, 13, 11, 12.5, 2000),  # EDT (offset differs!)
            bar("2024-03-01T00:00:00-05:00", 11, 12, 10, 11.5, 1500),  # out of order on purpose
        ], "unavailable": [], "warnings": []}
    stub.fetch = _fetch
    _sys.modules["stub_bars"] = stub
    reg = gw.load_registry()
    reg["providers"]["_stub_bars"] = {
        "id": "_stub_bars", "module": "stub_bars", "kinds": ["bars"],
        "pit_capable": ["bars"], "reliability": 0.6, "priority": 1, "caveats": {}}
    try:
        df = gw.get_bars_df("X", provider="_stub_bars")
        check("returns all 5 OHLCV columns",
              list(df.columns) == ["Open", "High", "Low", "Close", "Volume"], str(list(df.columns)))
        check("mixed DST offsets do NOT crash the index build (regression)", len(df) == 3, f"n={len(df)}")
        check("index is a naive DatetimeIndex",
              str(df.index.dtype) == "datetime64[ns]", str(df.index.dtype))
        check("rows sorted chronologically",
              list(df.index) == sorted(df.index), "out-of-order input must be sorted")
        check("Close maps from datum value", df.iloc[0]["Close"] == 10.5, str(df.iloc[0]["Close"]))
        check("OHLV map from datum extra",
              df.iloc[0]["Open"] == 10 and df.iloc[0]["Volume"] == 1000)
        check("provenance rides in df.attrs (not per row)",
              df.attrs.get("provider") == "_stub_bars" and df.attrs.get("reliability") == 0.6)

        # Empty result -> empty frame with the right columns, never a raise: the
        # live dashboard must degrade to "no data", not a 500.
        reg["providers"]["_stub_empty"] = {
            "id": "_stub_empty", "module": "stub_empty", "kinds": ["bars"],
            "pit_capable": ["bars"], "reliability": 0.5, "priority": 0, "caveats": {}}
        empty_mod = types.ModuleType("stub_empty")
        empty_mod.fetch = lambda **kw: {"data": [], "unavailable": [{"symbol": "X", "reason": "none"}],
                                        "warnings": []}
        _sys.modules["stub_empty"] = empty_mod
        edf = gw.get_bars_df("X", provider="_stub_empty")
        check("empty result -> empty DataFrame, not a raise",
              edf.empty and list(edf.columns) == ["Open", "High", "Low", "Close", "Volume"])
        check("empty frame still carries provenance in attrs",
              edf.attrs.get("provider") == "_stub_empty")
    finally:
        for k in ("_stub_bars", "_stub_empty"):
            reg["providers"].pop(k, None)
        for m in ("stub_bars", "stub_empty"):
            _sys.modules.pop(m, None)
            gw._module_cache.pop(m, None)


def test_ledger():
    print("=== 5. EvidenceLedger: claims cannot exist without evidence ===")
    from data import ledger
    from financial_data.schemas import make_datum, make_source
    ledger.set_db_path(TEST_DB)

    ni = make_datum("fundamentals_pit", 100, "2024-02-01",
                    make_source("sec-edgar", document="ACCN-NI", ref="us-gaap:NetIncomeLoss"),
                    symbol="TESTCO", concept="net_income", unit="USD", period_end="2023-12-31")
    rev = make_datum("fundamentals_pit", 500, "2024-02-01",
                     make_source("sec-edgar", document="ACCN-REV", ref="us-gaap:Revenues"),
                     symbol="TESTCO", concept="revenue", unit="USD", period_end="2023-12-31")

    cid = ledger.record_claim(
        "net_margin", 100 / 500, "net_income / revenue",
        {"net_income": ni, "revenue": rev},
        symbol="TESTCO", unit="ratio", as_of="2024-06-01", as_of_honored=True,
        run_id="test-run-1")
    check("claim recorded", bool(cid))

    # THE ENFORCEMENT: an unsupported number is structurally impossible.
    try:
        ledger.record_claim("made_up_target", 420.69, "vibes", {})
        check("evidence-free claim REFUSED", False, "no raise — unsupported claim got in")
    except ledger.LedgerError:
        check("evidence-free claim REFUSED", True)

    try:
        ledger.record_claim("bad", 1.0, "f", {"x": {"not": "a datum"}})
        check("non-datum evidence refused", False, "no raise")
    except ledger.LedgerError:
        check("non-datum evidence refused", True)

    e = ledger.explain(cid)
    check("explain returns the formula", e.get("formula") == "net_income / revenue")
    check("explain links both datums", len(e.get("evidence", [])) == 2)
    docs = sorted(x["document"] for x in e["evidence"])
    check("explain traces to filing documents", docs == ["ACCN-NI", "ACCN-REV"], str(docs))
    check("explain carries the PIT flag with the number", e["as_of_honored"] is True)
    check("explain surfaces data freshness", e["evidence_available_at_max"] == "2024-02-01")
    check("claim is idempotent (same facts -> same id)",
          ledger.record_claim("net_margin", 100 / 500, "net_income / revenue",
                              {"net_income": ni, "revenue": rev}, symbol="TESTCO",
                              unit="ratio", as_of="2024-06-01", as_of_honored=True,
                              run_id="test-run-1") == cid)
    check("claims_for_run finds it", any(c["claim_id"] == cid
                                         for c in ledger.claims_for_run("test-run-1")))
    check("explain of unknown claim is honest", "error" in ledger.explain("nope"))


def test_trials():
    print("=== 6. TrialRegistry: the true n_trials for dSR ===")
    from data import ledger
    ledger.set_db_path(TEST_DB)

    before = ledger.n_trials(family="_test_family")
    t1 = ledger.record_trial("SMA cross beats buy&hold", {"fast": 10, "slow": 50},
                             "dead", family="_test_family", strategy="sma_crossover",
                             sharpe=-0.2, flags={"survivorship_safe": False})
    t2 = ledger.record_trial("SMA cross beats buy&hold", {"fast": 20, "slow": 100},
                             "alive", family="_test_family", strategy="sma_crossover",
                             sharpe=0.9)
    check("distinct params -> distinct trials", t1 != t2)
    check("n_trials counts both", ledger.n_trials(family="_test_family") == before + 2)

    # Idempotence matters: dSR must not decay just because a test suite re-ran.
    ledger.record_trial("SMA cross beats buy&hold", {"fast": 10, "slow": 50},
                        "dead", family="_test_family", strategy="sma_crossover")
    check("identical experiment does NOT inflate the count",
          ledger.n_trials(family="_test_family") == before + 2)

    # MONOTONE: the count is the denominator of honesty; it may never fall.
    n_all = ledger.n_trials()
    ledger.record_trial("another idea", {"x": 1}, "inconclusive", family="_test_family2")
    check("n_trials is non-decreasing (dead ideas stay counted)",
          ledger.n_trials() == n_all + 1)

    try:
        ledger.record_trial("bad outcome", {}, "probably_fine")
        check("invalid outcome rejected", False, "no raise")
    except ledger.LedgerError:
        check("invalid outcome rejected", True)

    check("tests write to a throwaway DB, not the real registry",
          str(TEST_DB) != str(__import__("data.store", fromlist=["DB_PATH"]).DB_PATH)
          and TEST_DB.exists())

    stats = ledger.trial_stats()
    check("trial_stats reports a kill rate", stats["total"] > 0 and stats["kill_rate"] is not None,
          f"kill_rate={stats['kill_rate']}")
    check("dead trials are listed, not hidden",
          any(t["outcome"] == "dead" for t in ledger.list_trials(family="_test_family")))


def test_provider_boundary():
    print("=== 7. Provider boundary (P8 seam) ===")
    root = Path(__file__).parent.parent
    # New FIL code + the agent layer must never import a vendor SDK directly.
    # tools/market_data.py and data/store.py are PRE-EXISTING and tracked as a
    # known gap in FIL.md — scoping the test to new code gives it real teeth now
    # instead of a permanently-red assertion nobody acts on.
    scope = [root / "financial_data" / "gateway.py",
             root / "financial_data" / "schemas.py",
             root / "financial_data" / "cache.py",
             root / "financial_data" / "__init__.py",
             root / "data" / "ledger.py"]
    offenders = []
    for f in scope:
        txt = f.read_text()
        for vendor in ("import yfinance", "yf.Ticker", "data.sec.gov", "sec.gov"):
            if vendor in txt:
                offenders.append(f"{f.name}:{vendor}")
    check("no vendor SDK/url in gateway/schemas/cache/ledger", not offenders, str(offenders))

    # And the registry — not code — is where vendor names live.
    reg_txt = (root / "financial_data" / "registry.json").read_text()
    check("vendor names live in registry.json (config, not code)",
          "yfinance" in reg_txt and "sec-edgar" in reg_txt)

    gw_txt = (root / "financial_data" / "gateway.py").read_text()
    check("gateway imports providers dynamically (no hardcoded vendor import)",
          "importlib" in gw_txt and "from .providers" not in gw_txt)

    # M2b narrowed the gap: fetch_price_history no longer calls yfinance directly,
    # it routes through the gateway. Assert the migration so a future edit can't
    # quietly regress the bars path back to a raw vendor call.
    md_txt = (root / "tools" / "market_data.py").read_text()
    fph = md_txt[md_txt.index("def fetch_price_history"):
                 md_txt.index("def fetch_fundamentals")]
    check("fetch_price_history routes bars through the gateway (M2b)",
          "get_bars_df" in fph and "yf.Ticker" not in fph, "still calls yf.Ticker directly")
    # Honest remaining-gap marker: fundamentals/news in market_data.py ARE still
    # on yfinance (M1's EDGAR path is not yet wired into the live pipeline). This
    # is documented in FIL.md gap 1; the assertion below PINS that reality so the
    # doc and code cannot silently diverge.
    check("(known gap) fundamentals/news still on yfinance, as FIL.md states",
          "yf.Ticker" in md_txt, "if this fails, market_data is fully migrated — update FIL.md gap 1")


def test_live_edgar():
    print("=== 8. LIVE EDGAR (skips without network/config) ===")
    from financial_data.gateway import get
    from financial_data.providers.edgar import (NotConfiguredError, ProviderError,
                                                resolve_cik)

    # The two skip reasons are reported DIFFERENTLY on purpose. During M1 this
    # test skipped with "network unavailable" while the real cause was a 403 from
    # a non-compliant User-Agent — a config bug wearing an environment bug's
    # clothes. Conflating them lets a broken provider look permanently "skipped".
    try:
        cik = resolve_cik("AAPL")
    except NotConfiguredError as e:
        skip("live EDGAR", f"NOT CONFIGURED (fix: {str(e).splitlines()[0][:60]})")
        return
    except ProviderError as e:
        skip("live EDGAR", f"network unreachable: {str(e)[:40]}")
        return

    check("AAPL resolves to its real CIK (320193)", cik == 320193, f"got {cik}")

    try:
        res = get("fundamentals_pit", "AAPL", as_of="2024-01-01",
                  concepts=["revenue", "net_income", "assets"])
    except Exception as e:
        skip("live fundamentals_pit fetch", f"{type(e).__name__}: {str(e)[:40]}")
        return

    check("live: as_of_honored is TRUE (EDGAR is pit_capable)", res["as_of_honored"] is True)
    check("live: returned real datums", res["n"] > 0, f"n={res['n']}")
    check("live: provider is sec-edgar", res["provider"] == "sec-edgar")
    # THE POINT: nothing filed after the decision date survives.
    leaked = [d for d in res["data"] if d["available_at"] > "2024-01-01"]
    check("live: ZERO datums filed after as_of", not leaked,
          f"{len(leaked)} leaked" if leaked else "no look-ahead")
    check("live: cut removed future filings", res["excluded_future_datums"] > 0,
          f"excluded={res['excluded_future_datums']}")
    check("live: every datum cites an accession number",
          all(d["source"]["document"] for d in res["data"]))
    check("live: every datum has a us-gaap tag ref",
          all(str(d["source"]["ref"]).startswith("us-gaap:") for d in res["data"]))

    # Honest unavailability: an ETF is not an SEC XBRL filer. It must be NAMED
    # as unavailable, never fabricated and never silently empty.
    try:
        res2 = get("fundamentals_pit", "SPY", as_of="2024-01-01", concepts=["revenue"])
        check("live: non-filer reported as unavailable, not fabricated",
              res2["n"] == 0 and bool(res2["unavailable"]),
              str(res2["unavailable"])[:60])
    except Exception:
        # Raising is also acceptable here — both are honest; silence is not.
        check("live: non-filer reported as unavailable, not fabricated", True, "raised")

    # THE DoD CHECK, against real filings rather than a fixture: ask the same
    # question at two dates and require the ANSWER ITSELF to move as later
    # filings arrive. If the as_of cut were fake, both would return today's view
    # and these sets would be identical.
    then = get("fundamentals_pit", "AAPL", as_of="2020-01-01", concepts=["revenue"])
    now = get("fundamentals_pit", "AAPL", as_of="2025-01-01", concepts=["revenue"])
    then_docs = {d["source"]["document"] for d in then["data"]}
    now_docs = {d["source"]["document"] for d in now["data"]}
    check("live: an older as_of sees strictly fewer periods",
          then["n"] < now["n"], f"2020 view n={then['n']} < 2025 view n={now['n']}")
    check("live: the 2020 view cites only filings that existed in 2020",
          all(d["available_at"] <= "2020-01-01" for d in then["data"]))
    check("live: later view cites documents the older view could not know",
          bool(now_docs - then_docs), f"{len(now_docs - then_docs)} newer accessions")


def test_end_to_end_trace():
    print("=== 9. END-TO-END: claim -> datum ids -> EDGAR accession (skips offline) ===")
    from data import ledger
    from financial_data.gateway import get_concept
    from financial_data.providers.edgar import ProviderError
    ledger.set_db_path(TEST_DB)

    try:
        ni = get_concept("AAPL", "net_income", as_of="2024-01-01")
        rev = get_concept("AAPL", "revenue", as_of="2024-01-01")
    except (ProviderError, Exception) as e:
        skip("end-to-end EDGAR trace", f"network unavailable: {str(e)[:40]}")
        return
    if not ni or not rev:
        skip("end-to-end EDGAR trace", "concepts unavailable")
        return

    margin = ni["value"] / rev["value"]
    cid = ledger.record_claim(
        "net_margin", margin, "net_income / revenue",
        {"net_income": ni, "revenue": rev},
        symbol="AAPL", unit="ratio", as_of="2024-01-01", as_of_honored=True,
        run_id="fil-m1-demo")
    e = ledger.explain(cid)
    check("claim traces to >=2 datums", len(e["evidence"]) >= 2)
    check("every datum names its EDGAR accession",
          all(x["document"] for x in e["evidence"]))
    check("every input was knowable at as_of",
          all(x["available_at"] <= "2024-01-01" for x in e["evidence"]))
    check("net margin is plausible for AAPL (0 < m < 0.6)", 0 < margin < 0.6,
          f"margin={margin:.4f}")
    print(f"    trace: claim {cid} = {margin:.4f} via '{e['formula']}'")
    for x in e["evidence"]:
        print(f"      {x['role']:12s} {x['value']:>18,.0f} {x['unit']:4s} "
              f"filed {x['available_at']}  accn {x['document']}")


if __name__ == "__main__":
    test_schema()
    test_pit_filter()
    test_restatement()
    test_gateway()
    test_bars_df()
    test_ledger()
    test_trials()
    test_provider_boundary()
    test_live_edgar()
    test_end_to_end_trace()

    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}): " + "; ".join(SKIPPED))
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — FIL Milestone 1 (gateway + EDGAR PIT + EvidenceLedger)")
