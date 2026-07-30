"""
Stock Analysis Orchestrator — 7-Agent Pipeline (DeepSeek LLM)
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Run: pip3 install openai")

from tools.market_data import (
    fetch_price_history, fetch_fundamentals, fetch_recent_news,
    fetch_earnings, fetch_analyst_ratings,
    fetch_reddit_sentiment, fetch_stocktwits_sentiment, fetch_web_forum_sentiment,
    compute_indicators, compute_signal_summary, compute_algo_signals,
)
from agents.stock_agents import (
    run_research_agent, run_fundamentals_agent, run_technical_agent,
    run_risk_agent, run_social_agent, run_algo_agent, run_prediction_agent,
    _get_client,
)
from agents.synthesis import ground_prediction
from data.store import log_recommendation


def _web_search(query: str, num_results: int = 5) -> str:
    try:
        enc = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={enc}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8", errors="ignore")
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        titles   = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
        results  = []
        for t, s in zip(titles, snippets):
            t_clean = re.sub(r"<[^>]+>", "", t).strip()
            s_clean = re.sub(r"<[^>]+>", "", s).strip()
            results.append(f"• {t_clean}: {s_clean}")
            if len(results) >= num_results:
                break
        return "\n".join(results) or "No results."
    except Exception as e:
        return f"Search error: {e}"


def _price_history_summary(df) -> str:
    if df is None or df.empty:
        return "No price history available."
    close   = df["Close"]
    current = float(close.iloc[-1])
    week_ago  = float(close.iloc[-6])   if len(close) > 5   else current
    month_ago = float(close.iloc[-22])  if len(close) > 21  else current
    q_ago     = float(close.iloc[-66])  if len(close) > 65  else current
    y_ago     = float(close.iloc[-252]) if len(close) > 251 else current

    def pct(a, b): return f"{(b-a)/a*100:+.1f}%"
    recent_closes = ", ".join(f"${v:.2f}" for v in close.tail(5).tolist())
    return (
        f"Current: ${current:.2f}\n"
        f"1W: {pct(week_ago, current)} | 1M: {pct(month_ago, current)} | "
        f"3M: {pct(q_ago, current)} | 1Y: {pct(y_ago, current)}\n"
        f"Last 5 closes: {recent_closes}\n"
        f"Avg vol (20d): {df['Volume'].tail(20).mean():,.0f}"
    )


def analyze_stock(
    ticker: str,
    api_key: str,
    verbose: bool = True,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict:
    """Full 7-agent pipeline using DeepSeek LLM."""
    t0 = time.time()
    ticker = ticker.upper().strip()

    # Resolve API key: arg → env DEEPSEEK_API_KEY → env ANTHROPIC_API_KEY
    resolved_key = (
        api_key
        or os.getenv("DEEPSEEK_API_KEY", "")
        or os.getenv("ANTHROPIC_API_KEY", "")
    )
    if not resolved_key:
        return {"error": "No API key found. Set DEEPSEEK_API_KEY in .env or environment.", "ticker": ticker}

    client = _get_client(resolved_key)

    def progress(stage: str, msg: str):
        if verbose: print(f"  {msg}")
        if on_progress: on_progress(stage, msg)

    # ── Market data ────────────────────────────────────────────────────
    progress("data", f"Fetching market data for {ticker}...")
    try:
        df = fetch_price_history(ticker, period="1y")
    except ValueError as e:
        return {"error": str(e), "ticker": ticker}

    fundamentals     = fetch_fundamentals(ticker)
    news             = fetch_recent_news(ticker, max_items=10)
    earnings         = fetch_earnings(ticker)
    analyst_ratings  = fetch_analyst_ratings(ticker)
    indicators       = compute_indicators(df)
    signal_summary   = compute_signal_summary(indicators)
    algo_signals     = compute_algo_signals(df, indicators)

    company_name  = fundamentals.get("longName", ticker)
    sector        = fundamentals.get("sector", "Unknown")
    current_price = indicators.get("current_price", float(df["Close"].iloc[-1]))
    price_hist_summary = _price_history_summary(df)
    progress("data", f"Data ready: {company_name} | ${current_price:.2f} | {sector}")

    # ── Social data ────────────────────────────────────────────────────
    progress("social_data", "Fetching Reddit & StockTwits sentiment...")
    reddit_data     = fetch_reddit_sentiment(ticker)
    stocktwits_data = fetch_stocktwits_sentiment(ticker)
    web_forum_data  = fetch_web_forum_sentiment(ticker, company_name)
    progress("social_data", f"Social: Reddit={reddit_data.get('mention_count', 0)} mentions, StockTwits={stocktwits_data.get('total', 0)} msgs")

    # ── AI Agents ──────────────────────────────────────────────────────
    progress("research", "Research agent: analyzing news & sentiment...")
    research = run_research_agent(client, ticker, company_name, sector, news, _web_search)

    progress("fundamentals", "Fundamentals agent: analyzing valuation...")
    fundamentals_analysis = run_fundamentals_agent(client, ticker, fundamentals, earnings, analyst_ratings)

    progress("technical", "Technical agent: reading price action & signals...")
    technical_analysis = run_technical_agent(client, ticker, indicators, signal_summary, price_hist_summary)

    progress("risk", "Risk agent: assessing downside scenarios...")
    risk_analysis = run_risk_agent(client, ticker, fundamentals, indicators, research)

    progress("social", "Social agent: scanning Reddit, StockTwits & forums...")
    social_analysis = run_social_agent(client, ticker, company_name, reddit_data, stocktwits_data, web_forum_data)

    progress("algo", "Algo agent: running quantitative models...")
    algo_analysis = run_algo_agent(client, ticker, current_price, algo_signals, indicators)

    progress("prediction", "Prediction agent: generating final verdict...")
    prediction = run_prediction_agent(
        client, ticker, company_name, current_price,
        research, fundamentals_analysis, technical_analysis, risk_analysis,
        signal_summary, social_analysis, algo_analysis,
    )

    progress("grounding", "Grounding prediction with backtested strategies...")
    df_long = None
    try:
        # Use 5y history for grounding so strategies have enough warmup bars
        df_long = fetch_price_history(ticker, period="5y")
        prediction = ground_prediction(ticker, current_price, prediction, df_long, indicators)
    except Exception as _e:
        prediction["grounding"] = f"error: {_e}"

    # ── 7-pillar composite recommendation (deterministic; the numbers) ─────
    # Reuses the already-fetched data + the prediction agent's prose as the
    # labelled narrative thesis. This is the SINGLE persistence point for the
    # run: the composite logger replaces the old bare log_recommendation so
    # one analysis = one tracked row (tagged seven_pillar_composite).
    progress("recommendation", "Building 7-pillar composite recommendation...")
    recommendation = None
    try:
        from agents.recommendation import (build_recommendation,
                                           log_composite_recommendation)
        # Fundamentals pillar is SEC-sourced (EDGAR via the gateway), not the
        # yfinance `fundamentals` dict (which is market data + analyst consensus).
        try:
            from agents.fundamentals_pit import analyze_fundamentals_pit
            pit_fund = analyze_fundamentals_pit(ticker, run_id="pipeline-rec")
        except Exception:
            pit_fund = None
        recommendation = build_recommendation(
            ticker, df_long if df_long is not None else df,
            indicators, signal_summary, algo_signals, fundamentals, pit=pit_fund,
            reddit=reddit_data, stocktwits=stocktwits_data,
            llm_prose=prediction, run_id="pipeline")
        recommendation["recommendation_id"] = log_composite_recommendation(recommendation)
    except Exception as _e:
        # The composite failing must not sink the 7-agent analysis; fall back
        # to the legacy logging path so the run is still tracked.
        recommendation = {"error": str(_e)}
        try:
            log_recommendation({
                "ticker":        ticker,
                "current_price": current_price,
                "prediction":    prediction,
                "grounding": {
                    "grounding_strategy": prediction.get("grounding_strategy"),
                    "position_size":      prediction.get("position_size"),
                },
            })
        except Exception:
            pass

    elapsed = round(time.time() - t0, 1)
    progress("done", f"Analysis complete in {elapsed}s")

    return {
        "ticker":                ticker,
        "company_name":          company_name,
        "sector":                sector,
        "current_price":         current_price,
        "analyzed_at":           datetime.now().strftime("%Y-%m-%d %H:%M"),
        "elapsed_s":             elapsed,
        "fundamentals":          fundamentals,
        "indicators":            indicators,
        "signal_summary":        signal_summary,
        "algo_signals":          algo_signals,
        "news":                  news,
        "reddit":                reddit_data,
        "stocktwits":            stocktwits_data,
        "research":              research,
        "fundamentals_analysis": fundamentals_analysis,
        "technical_analysis":    technical_analysis,
        "risk_analysis":         risk_analysis,
        "social_analysis":       social_analysis,
        "algo_analysis":         algo_analysis,
        "prediction":            prediction,
        "recommendation":        recommendation,
    }
