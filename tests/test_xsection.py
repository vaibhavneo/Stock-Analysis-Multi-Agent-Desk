"""
Verification for the survivorship-safe cross-sectional ranking engine (M-F3).

Run: python3 tests/test_xsection.py

Offline/deterministic on the reference fixture (temp DB). Proves every survivor-
ship / look-ahead / determinism property the mission requires.
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:62s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def setup():
    from data import store
    from xsection import ranking
    store._DB_PATH = Path(tempfile.mkdtemp()) / "s.db"; store.DB_PATH = store._DB_PATH
    ranking.set_db_path(store._DB_PATH)
    return ranking


def test_survivorship_membership():
    print("=== 1. Survivorship-safe point-in-time membership ===")
    from xsection.universe import FixtureUniverseProvider, UniverseIncomplete
    p = FixtureUniverseProvider()
    m2020 = {x["ticker_as_of"] for x in p.members("2020-06-01")}
    m2023 = {x["ticker_as_of"] for x in p.members("2023-06-01")}
    check("removed/delisted-LATER constituent present in historical universe (LMBD@2020)",
          "LMBD" in m2020)
    check("newly-added constituent absent before its entry date (NUMT not in 2020)",
          "NUMT" not in m2020)
    check("added constituent appears after entry date (NUMT in 2023)", "NUMT" in m2023)
    check("delisted name gone from a later universe (LMBD not in 2023)", "LMBD" not in m2023)
    check("provider is labelled survivorship_safe", p.survivorship_safe is True)
    try:
        p.members("2010-01-01")
        check("out-of-coverage raises UNIVERSE_INCOMPLETE", False)
    except UniverseIncomplete as e:
        check("out-of-coverage raises UNIVERSE_INCOMPLETE", "UNIVERSE_INCOMPLETE" in str(e))
    # the paid provider is BLOCKED, never faking membership
    from xsection.universe import PaidUniverseProvider
    try:
        PaidUniverseProvider().members("2020-06-01")
        check("paid provider BLOCKED without dataset (never fakes)", False)
    except UniverseIncomplete as e:
        check("paid provider BLOCKED without dataset (never fakes)", "BLOCKED" in str(e))


def test_security_identity():
    print("=== 2. Security identity: ticker change + reused ticker ===")
    from xsection.universe import FixtureUniverseProvider
    sm = FixtureUniverseProvider().security_master()
    check("ticker change preserves identity (KAPB@2020 == KPPA@2023 -> same security_id)",
          sm.resolve("KAPB", "2020-01-01") == sm.resolve("KPPA", "2023-01-01") == "SEC0009")
    a, b = sm.resolve("XILO", "2020-06-01"), sm.resolve("XILO", "2024-06-01")
    check("reused ticker does NOT merge companies (XILO -> two different security_ids)",
          a == "SEC0013" and b == "SEC0014" and a != b)
    check("ticker_as_of reflects the effective ticker at the date",
          sm.ticker_as_of("SEC0009", "2020-01-01") == "KAPB"
          and sm.ticker_as_of("SEC0009", "2023-01-01") == "KPPA")


def test_pit_features_no_lookahead():
    print("=== 3. PIT features: future filings/prices cannot enter earlier ranks ===")
    from xsection.features import compute_features, synthetic_fundamentals
    from xsection.universe import FixtureUniverseProvider
    p = FixtureUniverseProvider()
    sm = p.security_master()
    as_of = "2021-06-15"
    sec = sm.get("SEC0001")
    # synthetic fundamentals: every returned quarter was FILED on/before as_of
    funds = synthetic_fundamentals(sec, as_of)
    check("every fundamental filing is filed on/before as_of (no future filings)",
          all(f["filed"] <= as_of for f in funds) and len(funds) > 0)
    # a later as_of reveals strictly more (or equal) filings — monotone, never fewer
    later = synthetic_fundamentals(sec, "2023-06-15")
    check("a later as_of never has FEWER historical filings", len(later) >= len(funds))
    # feature record provenance is complete
    member = next(m for m in p.members(as_of) if m["security_id"] == "SEC0001")
    px = p.prices("SEC0001", "2019-01-02", as_of)
    row = compute_features(member, as_of, px, p.prices(p.benchmark_id(), "2019-01-02", as_of), sec)
    f0 = row["features"][0]
    for key in ("feature_name", "raw_value", "available_at", "source", "data_quality",
                "feature_version", "formula"):
        check(f"feature record carries provenance field: {key}", key in f0)
    # every fundamental feature's available_at (filed) <= as_of
    fund_feats = [f for f in row["features"] if f.get("period_end")]
    check("fundamental features are available_at <= as_of (PIT)",
          all(f["available_at"] <= as_of for f in fund_feats) and len(fund_feats) > 0)
    # LOOK-AHEAD: appending future prices must not change PIT features
    px_future = px.copy()
    fut_idx = pd.date_range("2021-06-16", "2022-01-01", freq="B")
    px_future = pd.concat([px_future, pd.Series(9999.0, index=fut_idx)])
    row2 = compute_features(member, as_of, px_future, p.prices(p.benchmark_id(), "2019-01-02", as_of), sec)
    mom6_a = next(f["raw_value"] for f in row["features"] if f["feature_name"] == "mom_6m")
    mom6_b = next(f["raw_value"] for f in row2["features"] if f["feature_name"] == "mom_6m")
    check("appending future prices does NOT change a PIT feature", mom6_a == mom6_b)


def test_estimates_rejected():
    print("=== 4. Current-vintage estimates rejected for historical dates ===")
    from xsection.features import compute_features
    from xsection.universe import FixtureUniverseProvider
    p = FixtureUniverseProvider()
    m = next(x for x in p.members("2022-01-03") if x["security_id"] == "SEC0002")
    row = compute_features(m, "2022-01-03", p.prices("SEC0002", "2019-01-02", "2022-01-03"),
                           p.prices(p.benchmark_id(), "2019-01-02", "2022-01-03"),
                           p.security_master().get("SEC0002"))
    rev = next(f for f in row["features"] if f["feature_name"] == "earnings_revision")
    check("estimate/revision feature is UNAVAILABLE (never a current-vintage value)",
          rev["raw_value"] is None and rev["source"] == "unavailable")
    check("missingness reason names the absence of PIT estimate history",
          rev["missingness_reason"] == "no_pit_estimate_history")
    check("security flagged that estimates are unavailable",
          "estimates_unavailable_no_pit_history" in row["flags"])


def test_normalization():
    print("=== 5. Robust normalization + sector-relative correctness ===")
    from xsection import normalize as nz
    # Outlier robustness: a huge outlier must not blow up percentile ranks.
    vals = [1.0, 2.0, 3.0, 4.0, 1e9]
    pr = nz.percentile_rank(vals)
    check("percentile rank is outlier-immune (huge value -> rank 1.0, others unmoved)",
          pr[-1] == 1.0 and pr[0] == 0.0 and abs(pr[1] - 0.25) < 1e-9)
    w = nz.winsorize(vals, 0.2)
    check("winsorize clips extreme values", max(w) < 1e9)
    # Sector-relative: a name is ranked WITHIN its sector, not the whole tape.
    v = [10.0, 20.0, 100.0, 200.0]
    sectors = ["A", "A", "B", "B"]
    sr = nz.sector_relative_rank(v, sectors)
    check("sector-relative rank ranks within sector (top of A == top of B == 1.0)",
          sr[1] == 1.0 and sr[3] == 1.0 and sr[0] == 0.0 and sr[2] == 0.0,
          str(sr))
    # robust_z uses median/MAD (not mean/std)
    z = nz.robust_z([1, 2, 3, 4, 1000])
    check("robust_z centres on the median (outlier doesn't shift the centre)",
          z[2] == 0.0, str(z[2]))


def test_data_quality_gates():
    print("=== 6. Data-quality gates: missing data lowers confidence / excludes ===")
    from xsection import ranking as rk
    rk_mod = setup()
    # SEC0014 (XILO 2023 IPO) has little history at 2023-06-01 -> low confidence.
    r = rk_mod.run_ranking("2023-06-01", persist=False)
    xilo = next((x for x in r["ranked"] if x["ticker_as_of"] == "XILO"), None)
    check("young/low-history name has LOW data confidence but is still ranked",
          xilo is not None and xilo["data_confidence"] < 0.6, str(xilo["data_confidence"]) if xilo else "missing")
    # A ranking very early (insufficient history for everyone but recent IPOs) excludes names.
    early = rk_mod.run_ranking("2019-04-01", persist=False)
    check("insufficient-history securities are excluded with a reason",
          early.get("status") in ("OK", "NO_ELIGIBLE_SECURITIES")
          and (early.get("n_excluded", 0) > 0 or early.get("status") == "NO_ELIGIBLE_SECURITIES"),
          f"status={early.get('status')} excluded={early.get('n_excluded')}")


def test_weights_preregistered():
    print("=== 7. Ranking weights match the pre-registered ExperimentRegistry config ===")
    from backtest import experiments
    from xsection import ranking as rk
    rk_mod = setup()
    r = rk_mod.run_ranking("2023-06-01", persist=False)
    check("ranking config validates (weights sum to 1, risk not positive alpha)",
          experiments.validate_ranking_config() == [])
    check("ranking uses the pre-registered config hash",
          r["config_hash"] == experiments.ranking_config_hash("xsec-v1"))
    check("weights in the result == pre-registered weights",
          r["weights"] == experiments.ranking_config("xsec-v1")["weights"])
    check("risk is a penalty, not an alpha weight",
          "risk" not in r["weights"] and r["risk_penalty"] > 0)


def test_determinism_and_immutability():
    print("=== 8. Determinism, idempotence, immutability, action-order independence ===")
    from xsection import ranking as rk
    rk_mod = setup()
    r1 = rk_mod.run_ranking("2022-06-01", persist=True)
    r2 = rk_mod.run_ranking("2022-06-01", persist=True)          # identical re-run
    check("identical runs produce identical decision fingerprints",
          r1["decision_fingerprint"] == r2["decision_fingerprint"])
    check("re-running is idempotent (same ranking_run_id, no duplicate)",
          r1["ranking_run_id"] == r2["ranking_run_id"]
          and len(rk_mod.list_rankings()) == 1)
    # ACTION ORDER: running a DIFFERENT date's ranking in between must not change
    # the 2022 ranking's fingerprint.
    rk_mod.run_ranking("2023-01-01", persist=True)
    r3 = rk_mod.run_ranking("2022-06-01", persist=True)
    check("UI/API action order cannot change a ranking's result",
          r1["decision_fingerprint"] == r3["decision_fingerprint"])
    # IMMUTABILITY: the ledger row cannot be edited/deleted.
    import sqlite3
    conn = sqlite3.connect(str(rk_mod._db()))
    try:
        conn.execute("UPDATE ranking_runs SET as_of='1999-01-01' WHERE ranking_run_id=?",
                     (r1["ranking_run_id"],))
        check("ranking runs are immutable (UPDATE blocked)", False)
    except sqlite3.IntegrityError:
        check("ranking runs are immutable (UPDATE blocked)", True)
    except Exception as e:
        check("ranking runs are immutable (UPDATE blocked)", "immutable" in str(e).lower())
    finally:
        conn.close()


def test_no_llm_in_numeric_path():
    print("=== 9. No LLM prose changes deterministic ranks ===")
    from xsection import ranking as rk
    rk_mod = setup()
    r = rk_mod.run_ranking("2022-06-01", persist=False)
    fp = r["decision_fingerprint"]
    # Attaching a narrative to the result must not change the fingerprint (the
    # fingerprint is computed from ranks only; prose is display-only).
    r["llm_narrative"] = "This name looks like a screaming buy!!! moon!!!"
    from xsection.ranking import _hash
    fp2 = _hash({"as_of": r["as_of"], "universe": r["universe_id"], "cfg": r["config_hash"],
                 "ranks": [(x["security_id"], x["rank"], x["composite_raw"]) for x in r["ranked"]],
                 "excluded": sorted(e["security_id"] for e in r["excluded"])})
    check("narrative text does not change the deterministic fingerprint", fp == fp2)


def test_evaluation_delisting_and_costs():
    print("=== 10. Evaluation: delisted names included; costs reduce performance ===")
    from xsection import evaluate, ranking as rk
    from xsection.universe import get_provider
    setup()
    p = get_provider()
    # 2020-06-01: LMBD delists 2020-09-15 -> realized within 252d, must be INCLUDED.
    r = rk.run_ranking("2020-06-01", universe_id="reference-smallcap-demo", persist=False)
    ev = evaluate.evaluate_ranking(r, p)
    h252 = ev["by_horizon"][252]
    check("long-short at 252d includes at least one delisted name",
          h252.get("n_delisted_included", 0) >= 1, str(h252.get("n_delisted_included")))
    check("delisted names are NOT dropped from evaluation",
          "INCLUDED" in ev["delisting_treatment"])
    # Costs reduce net long-short below gross across a schedule.
    dates = [str(d.date()) for d in pd.date_range("2020-06-01", "2022-06-01", freq="MS")]
    free = evaluate.evaluate_schedule(dates, horizon=20, cost_bps=0.0)
    costed = evaluate.evaluate_schedule(dates, horizon=20, cost_bps=25.0)
    check("transaction costs reduce evaluated long-short performance",
          costed["long_short_net_mean_pct"] < free["long_short_net_mean_pct"],
          f"free={free['long_short_net_mean_pct']} costed={costed['long_short_net_mean_pct']}")
    check("schedule reports rank IC, turnover, and dSR (ExperimentRegistry)",
          free.get("mean_rank_ic") is not None and free.get("avg_turnover") is not None
          and "pre-registered" in free.get("n_trials_basis", ""))


if __name__ == "__main__":
    test_survivorship_membership()
    test_security_identity()
    test_pit_features_no_lookahead()
    test_estimates_rejected()
    test_normalization()
    test_data_quality_gates()
    test_weights_preregistered()
    test_determinism_and_immutability()
    test_no_llm_in_numeric_path()
    test_evaluation_delisting_and_costs()
    print("\n" + "=" * 80)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — survivorship-safe, PIT, deterministic cross-sectional ranking")
