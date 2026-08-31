"""
Stock Agent Web Server — Flask port 5051
Powered by DeepSeek LLM (free, OpenAI-compatible)
"""
from __future__ import annotations

import json
import os
import sys
import queue
import threading
from pathlib import Path

# Load .env from project root
_root = Path(__file__).parent.parent
_env  = _root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        # Manual parse if dotenv not installed
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(_root))

from datetime import datetime

from flask import Flask, Response, jsonify, request, stream_with_context
from agents.orchestrator import analyze_stock
from data.store import get_history, check_outcome, get_historical_hit_rate
from backtest.strategies import STRATEGY_REGISTRY

app = Flask(__name__, static_folder="static")

# Only used for the broker OAuth connect/callback round-trip (one short-lived
# {state, code_verifier} value per flask.session — see /api/broker/connect
# below). Nothing else in this app uses sessions. Falls back to a per-process
# random key when unset so local dev works without configuring it, at the
# cost of every restart invalidating any in-flight connect attempt — fine
# locally, but Railway MUST set a real FLASK_SECRET_KEY (see M0), since a
# random-per-process key across two gunicorn workers would make roughly half
# of all callback requests fail signature verification.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)


def _get_api_key() -> str:
    return (
        os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("ANTHROPIC_API_KEY", "")
    )


@app.route("/")
def index():
    return (Path(__file__).parent / "static" / "index.html").read_text()


@app.route("/api/status")
def status():
    key = _get_api_key()
    return jsonify({
        "ok": bool(key),
        "key_set": bool(key),
        "key_preview": (key[:8] + "...") if key else None,
        "model": "deepseek-v4-pro",
    })


@app.route("/api/analyze/stream", methods=["POST"])
def analyze_stream():
    data   = request.json or {}
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    api_key = _get_api_key()
    if not api_key:
        def err_gen():
            yield 'event: error\ndata: {"error": "No API key. Create .env file in stock_agent/ with: DEEPSEEK_API_KEY=sk-..."}\n\n'
            yield 'event: done\ndata: {}\n\n'
        return Response(stream_with_context(err_gen()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache"})

    result_queue: queue.Queue   = queue.Queue()
    progress_queue: queue.Queue = queue.Queue()

    def run():
        def on_progress(stage: str, msg: str):
            progress_queue.put(("progress", {"stage": stage, "msg": msg}))
        try:
            result = analyze_stock(ticker, api_key, verbose=False, on_progress=on_progress)
            result_queue.put(("result", result))
        except Exception as e:
            result_queue.put(("error", {"error": str(e)}))
        finally:
            result_queue.put(("done", {}))

    threading.Thread(target=run, daemon=True).start()

    def generate():
        while True:
            try:
                while True:
                    evt, d = progress_queue.get_nowait()
                    yield f"event: {evt}\ndata: {json.dumps(d)}\n\n"
            except queue.Empty:
                pass
            try:
                evt, d = result_queue.get(timeout=0.2)
                if evt == "done":
                    try:
                        while True:
                            pe, pd = progress_queue.get_nowait()
                            yield f"event: {pe}\ndata: {json.dumps(pd)}\n\n"
                    except queue.Empty:
                        pass
                    yield "event: done\ndata: {}\n\n"
                    break
                else:
                    yield f"event: {evt}\ndata: {json.dumps(d, default=str)}\n\n"
            except queue.Empty:
                continue

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/quick", methods=["POST"])
def quick_data():
    ticker = (request.json or {}).get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        from tools.market_data import (
            fetch_price_history, fetch_fundamentals, fetch_recent_news,
            compute_indicators, compute_signal_summary, compute_algo_signals,
        )
        df   = fetch_price_history(ticker, "3mo")
        fund = fetch_fundamentals(ticker)
        news = fetch_recent_news(ticker, 5)
        ind  = compute_indicators(df)
        sig  = compute_signal_summary(ind)
        algo = compute_algo_signals(df, ind)
        closes = df["Close"].tail(60).tolist()
        dates = [d.strftime("%b %d") for d in df.index[-60:]]
        return jsonify({
            "ticker":         ticker,
            "company_name":   fund.get("longName", ticker),
            "sector":         fund.get("sector", ""),
            "current_price":  ind.get("current_price"),
            "change_pct":     ind.get("price_change_pct"),
            "indicators":     ind,
            "signal_summary": sig,
            "algo_signals":   algo,
            "news":           news,
            "sparkline":      [round(c, 2) for c in closes],
            "sparkline_dates": dates,
            "fundamentals": {
                "pe":             fund.get("trailingPE"),
                "forward_pe":     fund.get("forwardPE"),
                "market_cap":     fund.get("marketCap"),
                "target_price":   fund.get("targetMeanPrice"),
                "analyst_count":  fund.get("numberOfAnalystOpinions"),
                "52w_high":       fund.get("fiftyTwoWeekHigh"),
                "52w_low":        fund.get("fiftyTwoWeekLow"),
                "beta":           fund.get("beta"),
                "dividend":       fund.get("dividendYield"),
                "short_ratio":    fund.get("shortRatio"),
                "profit_margin":  fund.get("profitMargins"),
                "roe":            fund.get("returnOnEquity"),
                "revenue_growth": fund.get("revenueGrowth"),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategies")
def list_strategies():
    return jsonify({"strategies": list(STRATEGY_REGISTRY.keys())})


@app.route("/api/backtest", methods=["POST"])
def run_backtest():
    data     = request.json or {}
    ticker   = data.get("ticker", "").upper()
    strategy = data.get("strategy", "sma_crossover")
    period   = data.get("period", "5y")
    realistic_costs = bool(data.get("realistic_costs", True))   # honest by default
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    if strategy not in STRATEGY_REGISTRY:
        return jsonify({"error": f"Unknown strategy. Choose from: {list(STRATEGY_REGISTRY)}"}), 400
    try:
        import warnings; warnings.filterwarnings("ignore")
        from tools.market_data import fetch_price_history
        from backtest.engine import run_vectorized_backtest, compute_performance_metrics, deflated_sharpe_ratio
        from backtest.strategies import STRATEGIES_NEEDING_FULL_DF
        from backtest.risk import safe_kelly_fraction
        from backtest.costs import DEFAULT_COST_MODEL
        from backtest import experiments
        df     = fetch_price_history(ticker, period=period)
        prices = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else None
        fn     = STRATEGY_REGISTRY[strategy]
        signal = fn(df) if strategy in STRATEGIES_NEEDING_FULL_DF else fn(prices)
        signal = signal.fillna(0)

        cost_model = DEFAULT_COST_MODEL if realistic_costs else None
        result = run_vectorized_backtest(prices, signal, cost_model=cost_model, volume=volume)
        m      = compute_performance_metrics(result.strategy_returns)

        # HONEST dSR: deflate against the IMMUTABLE pre-registered variant count
        # (fixed = 8), NOT a per-click count. Log the execution idempotently for
        # the audit; it never changes the denominator. This is why re-running a
        # backtest can no longer change its own dSR.
        n_trials = experiments.deflation_n()
        dsr = deflated_sharpe_ratio(
            m["sharpe_ratio"], n_trials=n_trials,
            skewness=m["skewness"], kurtosis=m["kurtosis"], n_obs=m["n_observations"],
        )
        experiments.record_execution(ticker, strategy, sharpe=m["sharpe_ratio"],
                                     dsr=round(dsr, 4), cost_model=result.cost_model)
        kelly = safe_kelly_fraction(result.strategy_returns)
        return jsonify({
            "ticker": ticker, "strategy": strategy, "period": period,
            "current_signal": int(signal.iloc[-1]),
            "sharpe":   round(m["sharpe_ratio"], 3),
            "sortino":  round(m["sortino_ratio"], 3),
            "max_dd":   round(m["max_drawdown"], 3),
            "calmar":   round(m["calmar_ratio"], 3),
            "win_rate": round(m["win_rate"], 3),
            "dsr":      round(dsr, 3),
            "n_trials": n_trials,             # surfaced: the honest denominator
            "n_trades": result.n_trades,
            "kelly_pct": round(kelly * 100, 1),
            "cost_model": result.cost_model,   # "CostModel" or "flat"
            "total_cost_pct": round(result.total_cost * 100, 2),
            "data_flags": {"survivorship_safe": False, "pit_fundamentals": False,
                           "cost_model": result.cost_model},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fil")
def fil_page():
    return (Path(__file__).parent / "static" / "fil.html").read_text()


@app.route("/api/fundamentals_pit", methods=["POST"])
def fundamentals_pit():
    """Evidence-backed point-in-time fundamentals from EDGAR (FIL)."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper()
    as_of  = data.get("as_of") or None
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        from agents.fundamentals_pit import analyze_fundamentals_pit
        return jsonify(analyze_fundamentals_pit(ticker, as_of=as_of, run_id="web"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evidence/<claim_id>")
def evidence(claim_id):
    """Full audit trail for one numeric claim: formula + datums + filing accessions."""
    try:
        from data import ledger
        return jsonify(ledger.explain(claim_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/all", methods=["POST"])
def backtest_all():
    """Backtest EVERY registered strategy on one ticker with ONE price fetch —
    the Backtest Lab's compare table. Honest by construction: every strategy is
    registered as a trial BEFORE any is scored, so each dSR is deflated against
    a count that includes its siblings (the M2a two-pass discipline), and costs
    default to the realistic CostModel."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    period = data.get("period", "5y")
    realistic_costs = bool(data.get("realistic_costs", True))
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        import warnings; warnings.filterwarnings("ignore")
        import pandas as pd
        from tools.market_data import fetch_price_history
        from backtest.engine import (run_vectorized_backtest, compute_performance_metrics,
                                     deflated_sharpe_ratio)
        from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
        from backtest.costs import DEFAULT_COST_MODEL
        from backtest.risk import safe_kelly_fraction
        from backtest import experiments

        df     = fetch_price_history(ticker, period=period)
        prices = df["Close"]
        volume = df["Volume"] if "Volume" in df.columns else None
        cost_model = DEFAULT_COST_MODEL if realistic_costs else None

        # Benchmark: buy & hold, zero costs (the bar every strategy must beat).
        bh = compute_performance_metrics(
            run_vectorized_backtest(prices, pd.Series(1.0, index=prices.index),
                                    transaction_cost_bps=0).strategy_returns)

        # dSR deflates against the FIXED pre-registered count — the same for
        # every strategy and every click. Running Compare-All 100 times cannot
        # move any dSR. Executions are logged idempotently per (ticker, variant).
        n_trials = experiments.deflation_n()
        rows = []
        for name, fn in STRATEGY_REGISTRY.items():
            try:
                sig = (fn(df) if name in STRATEGIES_NEEDING_FULL_DF else fn(prices)).fillna(0)
                res = run_vectorized_backtest(prices, sig, cost_model=cost_model, volume=volume)
                m = compute_performance_metrics(res.strategy_returns)
            except Exception as e:
                rows.append({"strategy": name, "error": str(e)})
                continue
            dsr = deflated_sharpe_ratio(m["sharpe_ratio"], n_trials=n_trials,
                                        skewness=m["skewness"], kurtosis=m["kurtosis"],
                                        n_obs=m["n_observations"])
            alive = dsr >= 0.5 and m["annualized_return"] > 0
            experiments.record_execution(ticker, name,
                                         outcome="alive" if alive else "dead",
                                         sharpe=m["sharpe_ratio"], dsr=round(dsr, 4),
                                         cost_model=res.cost_model)
            rows.append({
                "strategy": name,
                "sharpe": round(m["sharpe_ratio"], 3),
                "dsr": round(dsr, 3),
                "annualized_return_pct": round(m["annualized_return"] * 100, 1),
                "max_dd_pct": round(m["max_drawdown"] * 100, 1),
                "win_rate_pct": round(m["win_rate"] * 100, 1),
                "n_trades": res.n_trades,
                "total_cost_pct": round(res.total_cost * 100, 2),
                "kelly_pct": round(safe_kelly_fraction(res.strategy_returns) * 100, 1),
                "current_signal": int(sig.iloc[-1]) if len(sig) else 0,
                "beats_hold": m["sharpe_ratio"] > bh["sharpe_ratio"],
            })
        rows.sort(key=lambda r: r.get("sharpe", -99), reverse=True)
        return jsonify({
            "ticker": ticker, "period": period, "n_trials": n_trials,
            "cost_model": "CostModel" if realistic_costs else "flat_10bps",
            "buy_hold": {"sharpe": round(bh["sharpe_ratio"], 3),
                         "annualized_return_pct": round(bh["annualized_return"] * 100, 1),
                         "max_dd_pct": round(bh["max_drawdown"] * 100, 1)},
            "rows": rows,
            "data_flags": {"survivorship_safe": False, "pit_fundamentals": False},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def predictions():
    """Frozen prediction snapshots + their matured outcomes (the ledger view)."""
    try:
        from data import prediction_ledger as pl
        import sqlite3
        ticker = request.args.get("ticker", "").upper() or None
        snaps = pl.list_snapshots(ticker=ticker, limit=200)
        conn = sqlite3.connect(str(pl._db())); conn.row_factory = sqlite3.Row
        try:
            out = []
            for s in snaps:
                orows = conn.execute(
                    "SELECT horizon_days, matured, raw_return_pct, excess_return_pct, "
                    "direction_correct, as_of_date FROM prediction_outcomes "
                    "WHERE snapshot_id=? ORDER BY horizon_days", (s["snapshot_id"],)).fetchall()
                outcomes = {r["horizon_days"]: dict(r) for r in orows}
                fully = outcomes.get(252, {}).get("matured") == 1
                out.append({
                    "snapshot_id": s["snapshot_id"], "ticker": s["ticker"],
                    "created_at": s["created_at"], "action": s["action"],
                    "price_at_call": s["price_at_call"],
                    "confidence": {"thesis": s["conf_thesis"], "data": s["conf_data"],
                                   "statistical_edge": s["conf_statistical_edge"],
                                   "allocation": s["conf_allocation"]},
                    "sector": s["sector"], "regime": s["regime"],
                    "strategy_version": s["strategy_version"],
                    "decision_fingerprint": s["decision_fingerprint"],
                    "outcomes": outcomes, "fully_matured": fully,
                })
        finally:
            conn.close()
        return jsonify({"predictions": out, "summary": pl.summary()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions/refresh", methods=["POST"])
def predictions_refresh():
    """Manually trigger outcome evaluation (also runnable on a schedule via
    `python3 data/prediction_ledger.py refresh [TICKER]`). Idempotent."""
    try:
        from data import prediction_ledger as pl
        ticker = (request.json or {}).get("ticker", "").upper() or None
        return jsonify(pl.refresh_outcomes(ticker=ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/calibration")
def calibration():
    """Win rate, calibration error, agent (pillar) attribution, and confidence
    reliability over matured outcomes at a horizon (default 20 trading days).
    `source` = all | live | replay (historical) for the historical-vs-live view.
    `all_horizons=1` additionally returns the same report at EVERY tracked
    horizon under `by_horizon` - "performance by horizon", which needs the
    horizons side by side to be readable at all. Opt-in rather than always-on
    because it is N times the query work of the single-horizon default."""
    try:
        from data import prediction_ledger as pl
        horizon = int(request.args.get("horizon", 20))
        source = request.args.get("source", "all")
        rep = pl.calibration_report(horizon=horizon, source=source)
        rep["summary"] = pl.summary(source=source)
        if request.args.get("all_horizons") in ("1", "true", "yes"):
            rep["by_horizon"] = {
                str(h): {"overall": r.get("overall"),
                         "calibration_error_ece": r.get("calibration_error_ece")}
                for h, r in pl.calibration_report_all_horizons(source=source).items()
            }
        return jsonify(rep)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay", methods=["POST"])
def replay_start():
    """Start (or resume) a point-in-time historical replay in the background:
    replay the recommendation pipeline on historical dates with only as-of data,
    freeze the predictions, and evaluate outcomes. Returns a run_id to poll."""
    try:
        from agents import replay as rp
        cfg = request.json or {}
        tickers = cfg.get("tickers")
        if isinstance(tickers, str):
            tickers = [t.strip() for t in tickers.replace(",", " ").split() if t.strip()]
        if not tickers or not cfg.get("start") or not cfg.get("end"):
            return jsonify({"error": "tickers, start, end are required"}), 400
        config = {"tickers": tickers, "start": cfg["start"], "end": cfg["end"],
                  "cadence": cfg.get("cadence", "M"), "benchmark": cfg.get("benchmark", "SPY"),
                  "run_id": cfg.get("run_id")}
        run_id = rp.start_run(config)     # register up-front so progress is pollable
        config["run_id"] = run_id

        def worker():
            try:
                rp.run_replay(config)
            except Exception:
                pass
        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"started": True, "run_id": run_id, "run": rp.get_run(run_id)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay/<run_id>")
def replay_status(run_id):
    """Progress of a replay run (done/total, status, recent items)."""
    try:
        from agents import replay as rp
        return jsonify(rp.get_run(run_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/replay")
def replay_list():
    """List recent replay runs."""
    try:
        from agents import replay as rp
        return jsonify({"runs": rp.list_runs()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/universes")
def universes():
    """Available point-in-time universes + survivorship status (mission §10)."""
    try:
        from xsection.universe import list_universes
        return jsonify({"universes": list_universes()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings/run", methods=["POST"])
def rankings_run():
    """Produce (and immutably freeze) a survivorship-safe, point-in-time
    cross-sectional ranking for an as_of date. Idempotent."""
    try:
        from xsection import ranking as rk
        d = request.json or {}
        as_of = d.get("as_of")
        if not as_of:
            return jsonify({"error": "as_of (YYYY-MM-DD) is required"}), 400
        res = rk.run_ranking(as_of, universe_id=d.get("universe_id", "reference-smallcap-demo"),
                             config_version=d.get("config_version", "xsec-v1"), persist=True)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings/<ranking_run_id>")
def rankings_get(ranking_run_id):
    try:
        from xsection import ranking as rk
        res = rk.get_ranking(ranking_run_id)
        return jsonify(res or {"error": "not found"}), (200 if res else 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings/latest")
def rankings_latest():
    try:
        from xsection import ranking as rk
        res = rk.latest_ranking(request.args.get("universe_id"))
        return jsonify(res or {"error": "no rankings yet"}), (200 if res else 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings/history")
def rankings_history():
    try:
        from xsection import ranking as rk
        return jsonify({"runs": rk.list_rankings(limit=int(request.args.get("limit", 50)),
                                                 universe_id=request.args.get("universe_id"))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings/evaluate", methods=["POST"])
def rankings_evaluate():
    """Historical, delisting-inclusive evaluation across a schedule of dates
    (rank IC, long-short net of costs, turnover, dSR)."""
    try:
        import pandas as pd
        from xsection import evaluate
        d = request.json or {}
        start, end = d.get("start"), d.get("end")
        if not start or not end:
            return jsonify({"error": "start and end are required"}), 400
        cadence = {"M": "MS", "W": "W-FRI"}.get(d.get("cadence", "M"), "MS")
        dates = [str(x.date()) for x in pd.date_range(start, end, freq=cadence)]
        return jsonify(evaluate.evaluate_schedule(
            dates, universe_id=d.get("universe_id", "reference-smallcap-demo"),
            config_version=d.get("config_version", "xsec-v1"),
            horizon=int(d.get("horizon", 20)), cost_bps=float(d.get("cost_bps", 10.0))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/datasources")
def datasources():
    """Provider registry + live configuration status — the honest answer to
    'what data is this system actually running on right now?'."""
    try:
        from financial_data.gateway import load_registry
        from financial_data.keys import has_key
        out = []
        for pid, p in load_registry()["providers"].items():
            req = p.get("requires_key")
            configured = True if not req else has_key(req)
            out.append({
                "id": pid, "name": p.get("name"), "kinds": p.get("kinds", []),
                "pit_capable": p.get("pit_capable", []),
                "reliability": p.get("reliability"),
                "cost": p.get("cost"), "requires_key": req or None,
                "configured": configured,
                "status": "active" if configured else f"needs {req} in .env",
            })
        # The pre-gateway scrapers are data sources too — reported honestly.
        for sid, note in (("reddit-scraper", "public JSON API, keyless"),
                          ("stocktwits-scraper", "public API, keyless"),
                          ("duckduckgo-search", "web forum sentiment, keyless")):
            out.append({"id": sid, "name": sid, "kinds": ["sentiment"],
                        "pit_capable": [], "reliability": 0.4, "cost": "free",
                        "requires_key": None, "configured": True,
                        "status": f"active (legacy scraper — {note})"})
        return jsonify({"providers": out,
                        "n_active": sum(1 for p in out if p["configured"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro")
def macro():
    """Live macro snapshot (FRED + CBOE, keyless): the market context strip."""
    try:
        import warnings; warnings.filterwarnings("ignore")
        from datetime import date, timedelta
        from financial_data import get
        # 120-day window: monthly series (UNRATE, CPI) publish with a lag and a
        # 45-day window can miss their latest observation entirely.
        start = (date.today() - timedelta(days=120)).isoformat()
        r = get("macro", ["DGS10", "T10Y2Y", "UNRATE", "VIXCLS"], start=start)
        latest = {}
        for d in r["data"]:
            cur = latest.get(d["symbol"])
            if cur is None or d["available_at"] > cur["as_of"]:
                latest[d["symbol"]] = {"value": d["value"], "as_of": d["available_at"],
                                       "concept": d["concept"], "unit": d["unit"]}
        return jsonify({"series": latest, "provider": r["provider"],
                        "note": "current vintage (not PIT) — see registry caveats"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendation", methods=["POST"])
def recommendation():
    """The 7-pillar composite recommendation — FULLY KEYLESS (like /api/quick):
    every number is computed, so no LLM is needed. Social fetch failures degrade
    to a flagged-neutral pillar, never a 500 — a missing scrape is a data-quality
    fact the pillar reports, not an outage."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    period = data.get("period", "5y")
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        import warnings; warnings.filterwarnings("ignore")
        from tools.market_data import (
            fetch_price_history, fetch_fundamentals,
            fetch_reddit_sentiment, fetch_stocktwits_sentiment,
            compute_indicators, compute_signal_summary, compute_algo_signals,
        )
        from agents.recommendation import build_recommendation, log_composite_recommendation
        from agents.fundamentals_pit import analyze_fundamentals_pit

        df   = fetch_price_history(ticker, period=period)
        fund = fetch_fundamentals(ticker)   # market data (beta/52w/analyst) — NOT the fundamentals pillar
        ind  = compute_indicators(df)
        ss   = compute_signal_summary(ind)
        algo = compute_algo_signals(df, ind)
        # Fundamentals pillar is SEC-sourced: fetch point-in-time fundamentals
        # through the FinancialDataGateway (EDGAR), never legacy yfinance ratios.
        try:
            pit = analyze_fundamentals_pit(ticker, run_id="web-rec")
        except Exception:
            pit = None
        # Social is best-effort: the pillar itself flags absence honestly.
        try:
            reddit = fetch_reddit_sentiment(ticker)
        except Exception:
            reddit = None
        try:
            stocktwits = fetch_stocktwits_sentiment(ticker)
        except Exception:
            stocktwits = None

        rec = build_recommendation(ticker, df, ind, ss, algo, fund, pit=pit,
                                   reddit=reddit, stocktwits=stocktwits, run_id="web")
        rec["recommendation_id"] = log_composite_recommendation(rec)
        return jsonify(rec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_full_recommendation(ticker: str, period: str = "5y"):
    """Fetch data + run the keyless 7-pillar engine. Shared by /api/decision and the
    fallback path of /api/decision-brief so both endpoints build the recommendation
    the same way instead of drifting apart."""
    import warnings as _w; _w.filterwarnings("ignore")
    from tools.market_data import (
        fetch_price_history, fetch_fundamentals,
        fetch_reddit_sentiment, fetch_stocktwits_sentiment,
        compute_indicators, compute_signal_summary, compute_algo_signals,
    )
    from agents.recommendation import build_recommendation
    from agents.fundamentals_pit import analyze_fundamentals_pit

    df   = fetch_price_history(ticker, period=period)
    fund = fetch_fundamentals(ticker)
    ind  = compute_indicators(df)
    ss   = compute_signal_summary(ind)
    algo = compute_algo_signals(df, ind)
    try:
        pit = analyze_fundamentals_pit(ticker, run_id="decision")
    except Exception:
        pit = None
    try:
        reddit = fetch_reddit_sentiment(ticker)
    except Exception:
        reddit = None
    try:
        stocktwits = fetch_stocktwits_sentiment(ticker)
    except Exception:
        stocktwits = None

    rec = build_recommendation(ticker, df, ind, ss, algo, fund, pit=pit,
                               reddit=reddit, stocktwits=stocktwits, run_id="decision")
    return rec, df


def _build_backtest_all(ticker: str, df):
    """Race the full strategy library net of realistic costs. Best-effort — a
    failure here degrades the report, it never blocks it."""
    try:
        from backtest.engine import run_vectorized_backtest, compute_performance_metrics
        from backtest.costs import DEFAULT_COST_MODEL
        from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
        from backtest import experiments
        from backtest.engine import deflated_sharpe_ratio
        rows = []
        bh_ret = df["Close"].pct_change().dropna()
        bh_m = compute_performance_metrics(bh_ret)
        volume = df["Volume"] if "Volume" in df.columns else None
        for name, fn in STRATEGY_REGISTRY.items():
            try:
                sig = (fn(df) if name in STRATEGIES_NEEDING_FULL_DF
                       else fn(df["Close"])).fillna(0)
                bt = run_vectorized_backtest(df["Close"], sig,
                                             cost_model=DEFAULT_COST_MODEL, volume=volume)
                m = compute_performance_metrics(bt.strategy_returns)
                dsr = deflated_sharpe_ratio(m["sharpe_ratio"],
                                            n_trials=experiments.deflation_n(),
                                            skewness=m["skewness"],
                                            kurtosis=m["kurtosis"],
                                            n_obs=m["n_observations"])
                rows.append({
                    "strategy": name,
                    "sharpe": round(m["sharpe_ratio"], 3),
                    "dsr": round(dsr, 3),
                    "annualized_return_pct": round(m["annualized_return"] * 100, 2),
                    "max_dd_pct": round(m["max_drawdown"] * 100, 1),
                    "win_rate_pct": round(m["win_rate"] * 100, 1),
                    "n_trades": bt.n_trades,
                    "total_cost_pct": round(bt.total_cost * 100, 2),
                    "beats_hold": m["sharpe_ratio"] > bh_m["sharpe_ratio"],
                })
            except Exception:
                pass
        return {"ticker": ticker, "rows": rows,
                "buy_hold": {"sharpe": round(bh_m["sharpe_ratio"], 3)}}
    except Exception:
        return None


def _gather_cheap_enrichments(rec: dict, ticker: str):
    """Calibration + cross-sectional rank — no LLM, no heavy backtest, safe to call
    on every request. Best-effort: any failure degrades to 'unavailable', never 500s."""
    calibration = None
    prediction_summary = None
    try:
        from data import prediction_ledger as pl
        calibration = pl.calibration_report(horizon=20)
        prediction_summary = pl.summary()
    except Exception:
        pass

    xsec_ranking = None
    try:
        from xsection import ranking as xr
        xsec_ranking = xr.run_ranking(
            rec.get("data_asof") or datetime.now().strftime("%Y-%m-%d"),
            universe_id="production-pilot", persist=False)
    except Exception:
        pass

    return calibration, prediction_summary, xsec_ranking


@app.route("/api/decision", methods=["POST"])
def decision_report():
    """Composite Decision Report — one BUY/HOLD/SELL from all existing outputs.
    Fully keyless: calls /api/recommendation internally, optionally enriches with
    backtest-all, calibration, and cross-sectional ranking."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    period = data.get("period", "5y")
    owns_position = bool(data.get("owns_position", False))
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        from agents.decision_synthesis import synthesize_decision

        rec, df = _build_full_recommendation(ticker, period)
        backtest_all = _build_backtest_all(ticker, df)
        calibration, prediction_summary, xsec_ranking = _gather_cheap_enrichments(rec, ticker)

        report = synthesize_decision(
            ticker, rec,
            backtest_all=backtest_all,
            calibration=calibration,
            xsec_ranking=xsec_ranking,
            prediction_summary=prediction_summary,
            owns_position=owns_position,
        )
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/intelligence", methods=["POST"])
def intelligence_endpoint():
    """Evidence-driven decision engine (the intelligence/ package): market
    regime, multi-horizon historical context, historical analogs, a
    probabilistic forecast, risk/cost-basis analysis, and a reliability-
    weighted evidence ledger with named contradictions — folded into the
    SAME Decision Report synthesize_decision() already produces for
    /api/decision, not a second, competing action authority.

    Adaptive (item 10): `sections` selects which of the 6 intelligence
    blocks actually compute - a preset name ("valuation" | "full" |
    "price_action" | "recovery"), an explicit list, or omitted (auto-
    selects "recovery" when avg_cost is supplied, else "full"). Body:
    {ticker, avg_cost?, shares?, sections?}."""
    data = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    avg_cost = data.get("avg_cost")
    shares = data.get("shares")
    requested_sections = data.get("sections")
    owns_position = bool(avg_cost)

    try:
        from intelligence.orchestration import plan_sections, run_selected
        from agents.decision_synthesis import synthesize_decision

        sections = plan_sections(requested_sections, has_position=owns_position)
        intel = run_selected(ticker, sections, avg_cost=avg_cost, shares=shares)

        # The recommendation itself is always needed - synthesize_decision()
        # is built around it. But the two enrichment steps are genuinely
        # expensive (backtest_all races the whole strategy library;
        # _gather_cheap_enrichments runs a cross-sectional ranking over an
        # entire universe, which dominates this endpoint's latency), so they
        # run only for the "full" preset. Skipping them for a narrow preset
        # is the same adaptive principle plan_sections() applies to the
        # intelligence modules - measured at ~80s vs ~10s on a live request.
        rec, df = _build_full_recommendation(ticker)
        deep = "evidence" in sections and "analog" in sections
        backtest_all = _build_backtest_all(ticker, df) if deep else None
        if deep:
            calibration, prediction_summary, xsec_ranking = _gather_cheap_enrichments(rec, ticker)
        else:
            calibration = prediction_summary = xsec_ranking = None

        report = synthesize_decision(
            ticker, rec,
            backtest_all=backtest_all,
            calibration=calibration,
            xsec_ranking=xsec_ranking,
            prediction_summary=prediction_summary,
            owns_position=owns_position,
            regime=intel.get("regime"),
            historical_context=intel.get("historical_context"),
            analog=intel.get("analog"),
            forecast=intel.get("forecast"),
            risk_profile=intel.get("risk_profile"),
            evidence=intel.get("evidence"),
        )
        report["sections_computed"] = sections

        # Item 11 reaching the user: pair "here is my forecast at each
        # horizon" with "here is how accurate this engine has actually BEEN
        # at each horizon." A forecast shown without its own track record is
        # the exact false confidence this upgrade is meant to remove. Local
        # SQLite only (no network), but still gated to the deep preset so a
        # narrow request stays narrow.
        if deep:
            try:
                from data import prediction_ledger as _pl
                report["calibration_by_horizon"] = {
                    str(h): {"n": (r.get("overall") or {}).get("n"),
                             "win_rate": (r.get("overall") or {}).get("win_rate"),
                             "avg_return_pct": (r.get("overall") or {}).get("avg_raw_return_pct"),
                             "brier": (r.get("overall") or {}).get("brier")}
                    for h, r in _pl.calibration_report_all_horizons().items()
                }
            except Exception:
                report["calibration_by_horizon"] = None

        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decision-brief", methods=["POST"])
def decision_brief_endpoint():
    """Decision Brief v2 — one 20-second-readable action from all existing outputs.
    Fast path: pass an already-computed `recommendation` (e.g. from a completed full
    analysis) and this only does the cheap calibration/cross-sectional lookups.
    Slow path: pass just a ticker and it builds the recommendation from scratch,
    same as /api/decision. Ownership: pass owns_position true/false, or omit it for
    unknown (treated as not-owned — AVOID, never SELL)."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper().strip()
    period = data.get("period", "5y")
    owns_position = data.get("owns_position", None)
    if owns_position is not None:
        owns_position = bool(owns_position)

    # Position context (avg cost + optional shares) — the user's own entry price,
    # combined with the objective seven-agent verdict, to produce a position-aware
    # decision (agents/decision_brief.py::_apply_position_awareness). Supplying this
    # also implies ownership unless owns_position was explicitly set above.
    position = None
    avg_cost = data.get("avg_cost")
    if avg_cost is not None:
        try:
            avg_cost = float(avg_cost)
            if avg_cost > 0:
                position = {"avg_cost": avg_cost}
                shares = data.get("shares")
                if shares is not None:
                    position["shares"] = float(shares)
        except (TypeError, ValueError):
            position = None

    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        from agents.decision_brief import build_decision_brief

        rec = data.get("recommendation")
        if rec:
            calibration, prediction_summary, xsec_ranking = _gather_cheap_enrichments(rec, ticker)
            backtest_all = None  # already have a recommendation; skip the heavy re-backtest
        else:
            rec, df = _build_full_recommendation(ticker, period)
            backtest_all = _build_backtest_all(ticker, df)
            calibration, prediction_summary, xsec_ranking = _gather_cheap_enrichments(rec, ticker)

        brief = build_decision_brief(
            ticker, rec,
            backtest_all=backtest_all,
            calibration=calibration,
            xsec_ranking=xsec_ranking,
            prediction_summary=prediction_summary,
            owns_position=owns_position,
            position=position,
        )
        return jsonify(brief)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/portfolio-brief", methods=["POST"])
def portfolio_brief_endpoint():
    """Portfolio Decision Brief v2 — a position- and weight-aware action plan per
    holding plus a structural portfolio stance.

    Request: {"holdings": [{"ticker","shares","avg_cost","recommendation"?}, ...],
              "max_weight_pct"?: 25}. If a holding omits "recommendation", it is
      built server-side (keyless), same as /api/recommendation. Supplying the
      recommendation (e.g. cached from an earlier per-ticker analysis) is the fast
      path and avoids re-fetching."""
    data = request.json or {}
    raw = data.get("holdings") or []
    max_weight = data.get("max_weight_pct", 25.0)
    try:
        max_weight = float(max_weight)
    except (TypeError, ValueError):
        max_weight = 25.0
    if not raw:
        return jsonify({"error": "No holdings"}), 400

    try:
        from agents.portfolio_brief import build_portfolio_brief

        # Cross-sectional ranking and calibration are PORTFOLIO-WIDE (one universe,
        # one as-of date, one prediction ledger) — computing them once and reusing
        # across all holdings turns O(holdings) redundant rankings into one. Doing
        # this per holding is what made a 13-name portfolio hang for minutes.
        shared_calibration = shared_prediction_summary = shared_xsec = None
        try:
            from data import prediction_ledger as _pl
            shared_calibration = _pl.calibration_report(horizon=20)
            shared_prediction_summary = _pl.summary()
        except Exception:
            pass
        try:
            from xsection import ranking as xr
            _asof = datetime.now().strftime("%Y-%m-%d")
            shared_xsec = xr.run_ranking(_asof, universe_id="production-pilot", persist=False)
        except Exception:
            pass

        enriched = []
        for h in raw:
            ticker = str(h.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            rec = h.get("recommendation")
            if not rec:
                # Slow path: build the recommendation from scratch (keyless).
                rec, _df = _build_full_recommendation(ticker)
            enriched.append({
                "ticker": ticker,
                "shares": h.get("shares"),
                "avg_cost": h.get("avg_cost"),
                "recommendation": rec,
                "calibration": shared_calibration,
                "prediction_summary": shared_prediction_summary,
                "xsec_ranking": shared_xsec,   # same ranking; each holding finds its own row
            })

        # Cross-position correlation inputs. Fetched HERE, not inside
        # portfolio_brief: that module is a pure consumer that creates no
        # providers. A failure (or a thin/short series) simply means no
        # correlation adjustment - never a failed brief.
        from tools.market_data import fetch_price_history

        returns = {}
        for h in enriched:
            try:
                _hist = fetch_price_history(h["ticker"], period="1y")
                if _hist is not None and not _hist.empty:
                    returns[h["ticker"]] = _hist["Close"].astype(float).pct_change().dropna()
            except Exception:
                continue

        result = build_portfolio_brief(enriched, max_weight_pct=max_weight,
                                       returns=returns or None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Robinhood broker connection (read-only: positions in, nothing out) ─────
#
# No order-placement route exists here, deliberately — the user chose
# "recommendations + a manual trade queue", not automatic execution, even
# though Robinhood's own MCP server supports placing real orders. The
# frontend still POSTs whatever /api/broker/positions returns straight to
# the existing /api/portfolio-brief above — this section's only job is
# getting real holdings into that exact {ticker, shares, avg_cost} shape.

@app.route("/api/broker/status")
def broker_status():
    from broker import oauth
    try:
        token = None
        connected = oauth.is_connected()
        if connected:
            from broker import token_store
            token = token_store.load()
        return jsonify({
            "connected": connected,
            "obtained_at": (token or {}).get("obtained_at"),
            "expires_at": (token or {}).get("expires_at"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _broker_redirect_uri() -> str:
    """The exact redirect_uri registered with Robinhood for this app
    (broker/register_client.py). request.url_root's scheme can't be trusted
    here: Railway terminates TLS at its edge and forwards to this container
    over plain HTTP, so Flask sees every request as http:// regardless of
    what the browser actually used — confirmed live, this produced an
    authorize URL with redirect_uri=http://...railway.app/... which does
    not exactly match the https:// URI registered with Robinhood and would
    have failed the OAuth handshake outright. request.host reflects the
    Host header the browser actually sent, which IS trustworthy here, so
    scheme is derived from that instead: localhost stays http (matching
    what was registered for local dev), everything else is forced https
    (the only other registered redirect_uri)."""
    host = request.host
    scheme = "http" if host.startswith("localhost") or host.startswith("127.0.0.1") else "https"
    return f"{scheme}://{host}/api/broker/callback"


@app.route("/api/broker/connect")
def broker_connect():
    from flask import redirect, session
    from broker import oauth
    from broker.keys import get_key, NotConfiguredError

    try:
        client_id = get_key("ROBINHOOD_CLIENT_ID", "broker_connect")
    except NotConfiguredError as e:
        return jsonify({"error": str(e)}), 503

    code_verifier, code_challenge = oauth.generate_pkce_pair()
    state = oauth.generate_state()
    session["broker_oauth"] = {"state": state, "code_verifier": code_verifier}

    redirect_uri = _broker_redirect_uri()
    authorize_url = oauth.build_authorize_url(client_id, redirect_uri, state, code_challenge)
    return redirect(authorize_url)


@app.route("/api/broker/callback")
def broker_callback():
    from flask import redirect, session
    from broker import oauth

    pending = session.pop("broker_oauth", None)
    error = request.args.get("error")
    code = request.args.get("code")
    state = request.args.get("state")

    if error:
        return redirect(f"/?robinhood=error&msg={error}")
    if not pending or state != pending.get("state"):
        return redirect("/?robinhood=error&msg=state_mismatch")
    if not code:
        return redirect("/?robinhood=error&msg=no_code")

    try:
        redirect_uri = _broker_redirect_uri()
        oauth.complete_authorization(code, redirect_uri, pending["code_verifier"])
    except Exception as e:
        return redirect(f"/?robinhood=error&msg={e}")

    return redirect("/?robinhood=connected")


@app.route("/api/broker/disconnect", methods=["POST"])
def broker_disconnect():
    from broker import token_store
    try:
        token_store.clear()
        return jsonify({"connected": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/broker/positions")
def broker_positions():
    from broker import oauth
    from broker.providers.robinhood import get_default_holdings, RobinhoodError

    if not oauth.is_connected():
        return jsonify({"error": "not connected — visit /api/broker/connect first"}), 401
    try:
        return jsonify(get_default_holdings())
    except oauth.NotConnectedError as e:
        return jsonify({"error": str(e)}), 401
    except RobinhoodError as e:
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trials")
def trials():
    """TrialRegistry stats — the true n_trials behind every dSR (M2a)."""
    try:
        from data import ledger
        ticker = request.args.get("ticker", "").upper() or None
        stats = ledger.trial_stats()
        out = {"stats": stats}
        if ticker:
            out["ticker"] = ticker
            out["n_trials_for_ticker"] = ledger.n_trials(family=ledger.strategy_family(ticker))
            out["recent"] = ledger.list_trials(family=ledger.strategy_family(ticker), limit=20)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/validate", methods=["POST"])
def validate_overfitting():
    """Purged walk-forward CV + PBO across the whole strategy library for a ticker."""
    data   = request.json or {}
    ticker = data.get("ticker", "").upper()
    period = data.get("period", "5y")
    if not ticker:
        return jsonify({"error": "No ticker"}), 400
    try:
        import warnings; warnings.filterwarnings("ignore")
        import pandas as pd
        from tools.market_data import fetch_price_history
        from backtest.engine import run_vectorized_backtest
        from backtest.strategies import STRATEGY_REGISTRY, STRATEGIES_NEEDING_FULL_DF
        from backtest.validation import walk_forward_cv, probability_of_backtest_overfitting
        df     = fetch_price_history(ticker, period=period)
        prices = df["Close"]

        # Per-strategy returns matrix (for PBO) + walk-forward on each.
        returns = {}
        wf = {}
        for name, fn in STRATEGY_REGISTRY.items():
            try:
                sig = (fn(df) if name in STRATEGIES_NEEDING_FULL_DF else fn(prices)).fillna(0)
                r = run_vectorized_backtest(prices, sig)
                returns[name] = r.strategy_returns
                # walk-forward needs a signal fn over a price slice
                def make(fn_=fn, needs_df=(name in STRATEGIES_NEEDING_FULL_DF)):
                    return lambda p: (fn_(df.loc[p.index]) if needs_df else fn_(p)).fillna(0)
                wfres = walk_forward_cv(prices, make(), n_folds=5)
                wf[name] = {"mean_test_sharpe": round(wfres.mean_test_sharpe, 3),
                            "consistency": round(wfres.consistency, 2),
                            "oos_return_pct": round(wfres.oos_return * 100, 1),
                            "n_folds": wfres.n_folds}
            except Exception:
                continue

        rmat = pd.DataFrame(returns).dropna(how="any")
        pbo = probability_of_backtest_overfitting(rmat, n_splits=8) if rmat.shape[1] >= 2 else None
        return jsonify({
            "ticker": ticker, "period": period,
            "pbo": round(pbo.pbo, 3) if pbo else None,
            "is_overfit": pbo.is_overfit if pbo else None,
            "pbo_interpretation": (
                None if not pbo else
                ("HIGH — selecting the best backtest here tends to pick an out-of-sample loser (overfitting)"
                 if pbo.is_overfit else
                 "LOW — the best in-sample strategy tends to hold up out-of-sample")),
            "walk_forward": wf,
            "n_strategies": rmat.shape[1],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/track/history")
def track_history():
    ticker = request.args.get("ticker", "").upper() or None
    limit  = int(request.args.get("limit", 50))
    try:
        rows = get_history(ticker=ticker, limit=limit)
        return jsonify({"rows": rows, "count": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/track/check", methods=["POST"])
def track_check():
    data = request.json or {}
    rec_id = data.get("recommendation_id")
    if not rec_id:
        return jsonify({"error": "recommendation_id required"}), 400
    try:
        result = check_outcome(int(rec_id))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/track/hit_rate")
def track_hit_rate():
    strategy = request.args.get("strategy") or None
    ticker   = request.args.get("ticker", "").upper() or None
    try:
        result = get_historical_hit_rate(strategy_source=strategy, ticker=ticker)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


PRICE_HISTORY_PERIODS = {"1mo", "3mo", "6mo", "ytd", "1y", "5y", "max"}


@app.route("/api/price-history", methods=["GET"])
def price_history_endpoint():
    ticker = (request.args.get("ticker") or "").strip().upper()
    period = (request.args.get("period") or "3mo").strip().lower()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    if period not in PRICE_HISTORY_PERIODS:
        return jsonify({
            "error": "unsupported period '%s' - use one of %s" % (period, sorted(PRICE_HISTORY_PERIODS))
        }), 400
    try:
        from tools.market_data import fetch_price_history
        df = fetch_price_history(ticker, period)
    except ValueError as exc:
        return jsonify({"error": "no data for ticker '%s': %s" % (ticker, exc)}), 404
    except Exception as exc:
        return jsonify({"error": "provider failure: %s" % exc}), 502

    if df is None or df.empty:
        return jsonify({"error": "no historical data available for %s" % ticker}), 404

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    def _num(v):
        try:
            fv = float(v)
            return fv if fv == fv else None
        except (TypeError, ValueError):
            return None

    has_ohlc = all(c in df.columns for c in ("Open", "High", "Low"))
    has_volume = "Volume" in df.columns

    points = []
    for ts, row in df.tail(2000).iterrows():
        points.append({
            "timestamp": ts.strftime("%Y-%m-%dT00:00:00"),
            "open": _num(row["Open"]) if has_ohlc else None,
            "high": _num(row["High"]) if has_ohlc else None,
            "low": _num(row["Low"]) if has_ohlc else None,
            "close": _num(row["Close"]),
            "volume": int(row["Volume"]) if has_volume and row["Volume"] == row["Volume"] else None,
        })
    points = [p for p in points if p["close"] is not None]
    if not points:
        return jsonify({"error": "no usable closing prices for %s" % ticker}), 404

    closes = [p["close"] for p in points]
    start_price = closes[0]
    latest_price = closes[-1]
    change = latest_price - start_price
    change_percent = (change / start_price * 100.0) if start_price else 0.0
    highs = [p["high"] for p in points if p["high"] is not None] or closes
    lows = [p["low"] for p in points if p["low"] is not None] or closes
    period_high = max(highs)
    period_low = min(lows)

    running_max = closes[0]
    max_drawdown = 0.0
    for c in closes:
        running_max = max(running_max, c)
        if running_max:
            dd = (c - running_max) / running_max * 100.0
            max_drawdown = min(max_drawdown, dd)

    provider_name = "market-data-provider"
    try:
        provider_name = df.attrs.get("provider", provider_name)
    except Exception:
        pass

    return jsonify({
        "ticker": ticker,
        "period": period,
        "requested_interval": "1d",
        "actual_interval": "1d",
        "timezone": "America/New_York",
        "currency": "USD",
        "source": provider_name,
        "as_of": points[-1]["timestamp"],
        "points": points,
        "summary": {
            "start_price": round(start_price, 4),
            "latest_price": round(latest_price, 4),
            "change": round(change, 4),
            "change_percent": round(change_percent, 4),
            "period_high": round(period_high, 4),
            "period_low": round(period_low, 4),
            "maximum_drawdown_percent": round(max_drawdown, 4),
        },
    })



if __name__ == "__main__":
    port = int(os.getenv("PORT", 5051))
    key  = _get_api_key()
    print(f"\n  Stock Agent AI  →  http://localhost:{port}")
    print(f"  Model: DeepSeek deepseek-v4-pro (OpenAI-compatible)")
    print(f"  API Key: {'SET (' + key[:8] + '...)' if key else 'NOT SET — create stock_agent/.env with DEEPSEEK_API_KEY=sk-...'}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
