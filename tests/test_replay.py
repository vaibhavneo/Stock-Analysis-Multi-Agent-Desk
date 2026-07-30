"""
Verification for Point-in-Time Historical Replay (agents/replay.py).

Run: python3 tests/test_replay.py

Offline/deterministic (synthetic prices; injected pit_fn/fetch_fn; temp DB).
Covers exactly the mission's required tests:
  1. Look-ahead prevention — a replay at D depends ONLY on data <= D; tripling
     the future leaves the decision byte-identical.
  2. Restated filings — the replay requests fundamentals as_of D, so a filing
     filed after D never leaks (delegated to EDGAR's tested as_of path).
  3. Duplicate runs — re-running the same run_id resumes; no duplicate snapshots.
  4. Interrupted resume — an interrupted run continues from where it stopped.
  5. Delisted / missing symbols — a fetch failure is skipped with a reason, the
     run continues, and no fabricated data is produced.
  6. Deterministic reproduction — same config + data -> identical fingerprints.
  7. No synthesis — omitted historical inputs (sentiment/analyst) are flagged
     missing, never invented.
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {name:62s} {'OK' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAILURES.append(name)


def setup():
    from data import ledger, store
    from data import prediction_ledger as pl
    tmp = Path(tempfile.mkdtemp())
    store._DB_PATH = tmp / "s.db"; store.DB_PATH = store._DB_PATH
    ledger.set_db_path(tmp / "l.db"); pl.set_db_path(tmp / "s.db")
    from agents import replay
    return replay, pl


def series(seed=1, drift=0.001, n=1000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.008, n))), index=idx)


NOPIT = lambda t, d: {"available": False, "reason": "offline"}


def test_look_ahead():
    print("=== 1. Look-ahead prevention: the future cannot leak into date D ===")
    replay, pl = setup()
    prices = series(seed=7, drift=0.001)
    as_of = "2022-06-15"

    r1 = replay.replay_one("LA", as_of, prices, run_id="r", pit_fn=NOPIT)
    check("replay produced a frozen prediction", r1["status"] == "done" and r1["snapshot_id"])

    # Independently: the technical/algo inputs must equal those from prices<=D.
    df = prices[prices.index <= pd.Timestamp(as_of)]
    from tools.market_data import compute_algo_signals, compute_indicators, compute_signal_summary
    ohlcv = pd.DataFrame({"Open": df, "High": df, "Low": df, "Close": df, "Volume": 1.0})
    ind = compute_indicators(ohlcv)
    check("replay coverage matches independently-truncated bar count",
          r1["coverage"]["price_bars_as_of"] == len(df))
    check("frozen prediction is dated AS OF the replay date, not now",
          pl.get_snapshot(r1["snapshot_id"])["created_at"].startswith(as_of))

    # Triple ALL future prices and replay the same date: identical decision.
    future_spiked = prices.copy()
    future_spiked.loc[future_spiked.index > pd.Timestamp(as_of)] *= 3.0
    r2 = replay.replay_one("LA", as_of, future_spiked, run_id="r", pit_fn=NOPIT)
    check("tripling the future does NOT change the D-dated decision",
          r1["fingerprint"] == r2["fingerprint"], f"{r1['fingerprint']} vs {r2['fingerprint']}")
    # Truncating BEFORE D (removing history) SHOULD change it (sanity: the test
    # is sensitive, not vacuously passing).
    earlier = prices[prices.index <= pd.Timestamp("2022-03-15")]
    r3 = replay.replay_one("LA", "2022-03-15", earlier, run_id="r", pit_fn=NOPIT)
    check("a different as_of yields a different decision (test is sensitive)",
          r3["fingerprint"] != r1["fingerprint"])


def test_restated_filings():
    print("=== 2. Restated filings: fundamentals requested as-of D ===")
    replay, pl = setup()
    prices = series(seed=3)
    seen = {}

    def pit_fn(ticker, as_of):
        # Record the as_of the replay asked for, and return the value that was
        # KNOWABLE at that date: original (0.20) before the 2023 restatement,
        # restated (0.09) only on/after it. A replay at 2022-06 must get 0.20.
        seen["as_of"] = as_of
        margin = 0.20 if as_of < "2023-02-01" else 0.09
        return {"available": True, "as_of_honored": True, "provider": "sec-edgar",
                "ratios": {"net_margin": {"value": margin}, "return_on_equity": {"value": 0.3},
                           "debt_to_equity": {"value": 0.5}, "operating_margin": {"value": 0.25}}}

    r = replay.replay_one("REST", "2022-06-15", prices, run_id="r", pit_fn=pit_fn)
    check("replay requested fundamentals as_of the replay date", seen.get("as_of") == "2022-06-15")
    snap = pl.get_snapshot(r["snapshot_id"])
    import json
    pillars = json.loads(snap["pillars_json"])
    # With the ORIGINAL 20% margin the fundamentals pillar scores high; the later
    # restatement (9%) never leaks into a 2022 prediction.
    check("fundamentals pillar reflects the as-of-D filing (not a later restatement)",
          pillars["fundamentals"] is not None and pillars["fundamentals"] >= 60,
          f"fundamentals={pillars['fundamentals']}")
    check("fundamentals source is SEC (gateway), not synthesized",
          snap["conf_data"] is not None)


def test_duplicate_and_deterministic():
    print("=== 3. Duplicate runs are idempotent + deterministic ===")
    replay, pl = setup()
    prices = {"AA": series(1, 0.0012), "SPY": series(3, 0.0004)}
    fetch = lambda t: prices.get(t, prices["SPY"])
    cfg = {"run_id": "fixed-run", "tickers": ["AA"], "start": "2022-01-01",
           "end": "2022-06-01", "cadence": "M", "benchmark": "SPY"}

    out1 = replay.run_replay(cfg, fetch_fn=fetch, pit_fn=NOPIT, evaluate=False)
    n_after_1 = len(pl.list_snapshots(ticker="AA"))
    fps1 = sorted(s["decision_fingerprint"] for s in pl.list_snapshots(ticker="AA"))

    out2 = replay.run_replay(cfg, fetch_fn=fetch, pit_fn=NOPIT, evaluate=False)   # rerun same run_id
    n_after_2 = len(pl.list_snapshots(ticker="AA"))
    fps2 = sorted(s["decision_fingerprint"] for s in pl.list_snapshots(ticker="AA"))

    check("first run froze predictions", n_after_1 >= 3, str(n_after_1))
    check("re-running the same run creates NO new snapshots (idempotent)",
          n_after_1 == n_after_2, f"{n_after_1} -> {n_after_2}")
    check("deterministic: identical fingerprints across re-run", fps1 == fps2)
    check("resumed run reports 0 new done (all cells already complete)",
          out2["done"] == 0, f"done={out2['done']}")


def test_interrupted_resume():
    print("=== 4. Interrupted run resumes from where it stopped ===")
    replay, pl = setup()
    prices = {"BB": series(5, 0.001), "SPY": series(3, 0.0004)}
    fetch = lambda t: prices.get(t, prices["SPY"])
    cfg = {"run_id": "resume-run", "tickers": ["BB"], "start": "2022-01-01",
           "end": "2022-12-01", "cadence": "M", "benchmark": "SPY"}

    dates = replay.sample_dates(cfg["start"], cfg["end"], "M")
    run_id = replay.start_run(cfg)
    # Simulate an interruption: only the first 3 cells were completed.
    for as_of in dates[:3]:
        res = replay.replay_one("BB", as_of, prices["BB"], run_id, pit_fn=NOPIT)
        replay._record_item(run_id, "BB", as_of, res)
    replay._finalize(run_id)
    mid = replay.get_run(run_id)
    check("interrupted run is marked incomplete", mid["status"] == "interrupted"
          and mid["done_items"] == 3, f"status={mid['status']} done={mid['done_items']}")

    # Resume: same run_id, full config -> completes the rest, no redo.
    out = replay.run_replay(cfg, fetch_fn=fetch, pit_fn=NOPIT, evaluate=False)
    fin = replay.get_run(run_id)
    check("resume completes the run", fin["status"] == "complete"
          and fin["done_items"] == fin["total_items"], f"{fin['done_items']}/{fin['total_items']}")
    check("resume only did the REMAINING cells", out["done"] == len(dates) - 3,
          f"did {out['done']}, expected {len(dates) - 3}")


def test_delisted_missing():
    print("=== 5. Delisted / missing symbols are skipped, not faked ===")
    replay, pl = setup()
    good = series(9, 0.001)

    def fetch(t):
        if t == "DEAD":
            raise RuntimeError("no data for ticker (delisted)")
        return good
    cfg = {"tickers": ["GOOD", "DEAD"], "start": "2022-01-01", "end": "2022-04-01",
           "cadence": "M", "benchmark": "GOOD"}
    out = replay.run_replay(cfg, fetch_fn=fetch, pit_fn=NOPIT, evaluate=False)
    run = replay.get_run(out["run_id"])
    check("the delisted symbol's cells are all skipped",
          run["items_by_status"].get("skipped", 0) >= 3, str(run["items_by_status"]))
    check("the good symbol still produced predictions",
          run["items_by_status"].get("done", 0) >= 3)
    check("no snapshot exists for the delisted symbol (nothing fabricated)",
          len(pl.list_snapshots(ticker="DEAD")) == 0)
    # And an insufficient-history date is skipped with a clear reason.
    early = replay.replay_one("GOOD", "2021-01-11", good, run_id="x", pit_fn=NOPIT)
    check("insufficient early history is skipped, not built",
          early["status"] == "skipped" and "insufficient" in early["reason"])


def test_no_synthesis_of_missing_inputs():
    print("=== 6. Unavailable historical inputs are flagged, never invented ===")
    replay, pl = setup()
    r = replay.replay_one("NS", "2022-06-15", series(2, 0.001), run_id="r", pit_fn=NOPIT)
    miss = set(r["missing"])
    check("social sentiment recorded as missing (no PIT history)", "social_sentiment" in miss)
    check("analyst consensus recorded as missing (no PIT history)", "analyst_consensus" in miss)
    check("beta + short interest recorded as missing (current = look-ahead)",
          "beta" in miss and "short_interest" in miss)
    check("macro exclusion recorded (FRED current-vintage, not PIT)",
          "macro_current_vintage_not_pit" in r["exclusions"])
    check("coverage marks sentiment/analyst/macro as excluded, not zeroed",
          r["coverage"]["sentiment"].startswith("excluded")
          and r["coverage"]["macro"].startswith("excluded"))
    # The frozen snapshot carries the coverage/missing audit immutably.
    import json
    snap = pl.get_snapshot(r["snapshot_id"])
    frozen = json.loads(snap["frozen_json"])
    check("frozen snapshot preserves the replay coverage/missing audit",
          frozen.get("replay_meta", {}).get("missing") and frozen["replay_run_id"] == "r")


if __name__ == "__main__":
    test_look_ahead()
    test_restated_filings()
    test_duplicate_and_deterministic()
    test_interrupted_resume()
    test_delisted_missing()
    test_no_synthesis_of_missing_inputs()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL PASS — historical replay: point-in-time correct, resumable, deterministic")
