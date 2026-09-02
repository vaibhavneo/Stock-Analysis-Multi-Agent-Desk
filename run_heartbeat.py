#!/usr/bin/env python3
"""
Daily heartbeat runner — the cron entry point.

    python3 run_heartbeat.py --tickers AAPL,MSFT,NVDA
    python3 run_heartbeat.py --file watchlist.txt
    python3 run_heartbeat.py --tickers AAPL --status-only

Exit codes: 0 = every ticker frozen, 1 = some failed, 2 = nothing ran.
A non-zero exit is what makes a silent cron failure visible.

Suggested crontab — after the US close, weekdays only:
    30 17 * * 1-5  cd /path/to/stock_agent && /usr/bin/python3 run_heartbeat.py \
                     --file watchlist.txt >> logs/heartbeat.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_tickers(args) -> list:
    """Symbols from --file or --tickers, de-duplicated, order preserved.

    Comments are stripped BEFORE splitting on commas. The original order was
    the other way round, which meant a comment containing a comma — e.g.
    "(p50 3.4s, p90 19.8s, max 42.0s)" — was split into fragments where only
    the first still began with '#'. The rest were treated as ticker symbols
    and produced five bogus lookup errors on the first real cron run.
    """
    if args.file:
        tokens = []
        with open(args.file) as f:
            for line in f:
                line = line.split("#", 1)[0]          # comments first
                tokens.extend(line.replace(",", " ").split())
    else:
        tokens = (args.tickers or "").replace(",", " ").split()

    seen, out = set(), []
    for t in tokens:
        t = t.strip().upper()
        # A real symbol is letters, digits, dot or dash. Anything else is
        # almost certainly prose that leaked in, and is better dropped loudly
        # at parse time than sent to a data provider as a lookup.
        if not t or t in seen:
            continue
        if not all(c.isalnum() or c in ".-" for c in t):
            print(f"  skipping unparseable symbol: {t!r}", file=sys.stderr)
            continue
        seen.add(t)
        out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily prediction heartbeat")
    ap.add_argument("--tickers", help="comma-separated symbols")
    ap.add_argument("--file", help="file with one symbol per line (# comments ok)")
    ap.add_argument("--as-of", help="date to stamp predictions with (default: now)")
    ap.add_argument("--no-grade", action="store_true", help="skip outcome refresh")
    ap.add_argument("--no-refit", action="store_true", help="skip calibration refit")
    ap.add_argument("--force", action="store_true",
                    help="re-predict even if this ticker already has a call today")
    ap.add_argument("--status-only", action="store_true",
                    help="print the independence report and exit, changing nothing")
    ap.add_argument("--json", action="store_true", help="emit the full result as JSON")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the verified ledger backup after the run")
    ap.add_argument("--health-only", action="store_true",
                    help="print the flywheel health report and exit, changing nothing")
    args = ap.parse_args()

    from agents.heartbeat import independence_report, run_daily

    if args.health_only:
        from agents.flywheel_health import format_report, health_report
        print(format_report(health_report()))
        return 0

    if args.status_only:
        rep = independence_report()
        print("Independent evidence per horizon (what actually gates calibration):")
        for h, r in sorted(rep.items()):
            mark = "OK " if r.get("gate_met") else "-- "
            print(f"  {mark} h={h:>3}d  rows={r.get('rows', 0):<6} "
                  f"effective_n={r.get('effective_n', 0):<6} "
                  f"needed={r.get('needed', '?')}  shortfall={r.get('shortfall', '?')}")
        return 0

    tickers = _load_tickers(args)
    if not tickers:
        print("No tickers given. Use --tickers or --file.", file=sys.stderr)
        return 2

    def progress(r):
        if r["status"] == "done":
            p1 = (r.get("p_up") or {}).get(1)
            print(f"  {r['ticker']:<6} {str(r.get('action')):<10} "
                  f"composite={r.get('composite')}  p_up(1d)={p1}")
        else:
            print(f"  {r['ticker']:<6} {r['status'].upper()}: {r.get('reason', '')}")

    res = run_daily(tickers, grade=not args.no_grade, refit=not args.no_refit,
                    as_of=args.as_of, force=args.force, progress_cb=progress)

    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(f"\nfrozen={res['frozen']} skipped={res['skipped']} "
              f"errors={res['errors']} in {res['elapsed_s']}s")
        if res.get("graded"):
            print(f"graded: {res['graded']['matured']} matured outcomes")
        active = res.get("calibration_active_horizons")
        if active is not None:
            print(f"calibration active at: {active or 'no horizon yet (gate not met)'}")

    # Back up the ledger AFTER the run, so the snapshot includes what was just
    # frozen. Canonical only: a secondary deployment's rows are not evidence,
    # so backing them up would imply they were.
    if not args.no_backup:
        try:
            from data import prediction_ledger as pl
            if pl.is_canonical_ledger():
                from data.backup import backup_ledger
                b = backup_ledger()
                if b.get("ok"):
                    print(f"backup: verified {b['path']} "
                          f"({b.get('bytes', 0) // 1024} KB)")
                else:
                    print(f"backup: FAILED — {b.get('error')}", file=sys.stderr)
            else:
                print("backup: skipped (secondary ledger is not evidence)")
        except Exception as e:
            print(f"backup: FAILED — {e}", file=sys.stderr)

    # Health report: monitoring only, never influences scoring or calibration.
    try:
        from agents.flywheel_health import format_report, health_report
        print()
        print(format_report(health_report()))
    except Exception as e:
        print(f"health report unavailable: {e}", file=sys.stderr)

    # A run where every ticker was already predicted today is a SUCCESS - that
    # is the idempotency guard working, not a failed cron.
    ok = res["errors"] == 0 and (res["frozen"] > 0 or res["skipped"] == len(tickers))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
