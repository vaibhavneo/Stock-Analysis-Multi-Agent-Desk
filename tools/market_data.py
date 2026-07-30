"""
Market Data Tool — yfinance + pandas + ta + Reddit + StockTwits + Algo Signals
"""
from __future__ import annotations

import json
import math
import random
import re
import urllib.parse
import urllib.request
import warnings
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    import ta
    TA_OK = True
except ImportError:
    TA_OK = False


# ── Raw data fetch ─────────────────────────────────────────────────────────

def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """OHLCV history for the live pipeline.

    Routed through the FinancialDataGateway (FIL M2b) rather than calling
    yfinance directly: the price source is now a registry choice, so the same
    swap-to-a-paid-provider that FIL gives backtests also covers this live path
    — a config edit, not a code change here. Behavior is preserved exactly
    (yfinance provider, auto_adjust=True, raise on empty). Provenance rides in
    df.attrs. Import is local so the rest of this module (fundamentals/news,
    still on yfinance — see FIL.md gap 1) never depends on FIL loading.
    """
    from financial_data import get_bars_df
    df = get_bars_df(ticker, period=period)
    if df.empty:
        raise ValueError(f"No data for ticker '{ticker}'. Check the symbol.")
    return df


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    t = yf.Ticker(ticker)
    info = t.info or {}
    keys = [
        "longName", "sector", "industry", "country", "currency",
        "marketCap", "enterpriseValue",
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        "enterpriseToEbitda", "enterpriseToRevenue",
        "trailingEps", "forwardEps", "bookValue",
        "profitMargins", "grossMargins", "operatingMargins",
        "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
        "returnOnEquity", "returnOnAssets",
        "totalDebt", "totalCash", "freeCashflow", "operatingCashflow",
        "totalRevenue", "grossProfits",
        "dividendYield", "payoutRatio", "beta",
        "52WeekChange", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "fiftyDayAverage", "twoHundredDayAverage",
        "averageVolume", "averageVolume10days",
        "shortRatio", "shortPercentOfFloat",
        "recommendationMean", "numberOfAnalystOpinions",
        "targetMeanPrice", "targetHighPrice", "targetLowPrice",
        "currentPrice", "previousClose", "open", "dayHigh", "dayLow",
        "volume", "regularMarketVolume",
    ]
    return {k: info.get(k) for k in keys if info.get(k) is not None}


def fetch_recent_news(ticker: str, max_items: int = 8) -> list[dict]:
    t = yf.Ticker(ticker)
    news = t.news or []
    out = []
    for item in news[:max_items]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        publisher = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
        link = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
        pub_time = content.get("pubDate", "") or ""
        if not pub_time:
            ts = item.get("providerPublishTime", 0)
            pub_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
        if title:
            out.append({"title": title, "publisher": publisher, "link": link, "published": pub_time})
    return out


def _sanitize(obj):
    """Recursively convert non-JSON-serializable keys/values to strings."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def fetch_earnings(ticker: str) -> dict:
    import json as _json
    t = yf.Ticker(ticker)
    result = {}
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            result["next_earnings"] = str(cal.iloc[0].get("Earnings Date", "Unknown"))
    except Exception:
        pass
    try:
        hist = t.earnings_history
        if hist is not None and not hist.empty:
            result["recent_earnings"] = _sanitize(hist.tail(4).to_dict(orient="records"))
    except Exception:
        pass
    try:
        fin = t.quarterly_financials
        if fin is not None and not fin.empty and "Total Revenue" in fin.index:
            result["quarterly_revenue"] = _sanitize(fin.loc["Total Revenue"].head(4).to_dict())
    except Exception:
        pass
    return result


def fetch_analyst_ratings(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    result = {}
    try:
        recs = t.recommendations
        if recs is not None and not recs.empty:
            result["recent_recommendations"] = _sanitize(recs.tail(5).reset_index().to_dict(orient="records"))
    except Exception:
        pass
    try:
        upgrades = t.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            result["recent_upgrades_downgrades"] = _sanitize(upgrades.head(5).reset_index().to_dict(orient="records"))
    except Exception:
        pass
    return result


# ── Social Sentiment ───────────────────────────────────────────────────────

def fetch_reddit_sentiment(ticker: str) -> dict:
    """Fetch Reddit posts from WSB, stocks, investing subreddits via public JSON API."""
    results = {"posts": [], "sentiment_score": 0, "mention_count": 0, "subreddits": []}
    subreddits = ["wallstreetbets", "stocks", "investing", "SecurityAnalysis", "StockMarket"]
    all_posts = []

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/search.json?q={ticker}&sort=hot&limit=5&t=week"
            req = urllib.request.Request(url, headers={
                "User-Agent": "StockAgent/1.0 (research bot)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode())
            posts = data.get("data", {}).get("children", [])
            for p in posts:
                d = p.get("data", {})
                title = d.get("title", "")
                score = d.get("score", 0)
                comments = d.get("num_comments", 0)
                selftext = d.get("selftext", "")[:300]
                if ticker.upper() in title.upper() or ticker.upper() in selftext.upper():
                    all_posts.append({
                        "subreddit": sub,
                        "title": title,
                        "score": score,
                        "comments": comments,
                        "preview": selftext[:200],
                        "created": datetime.fromtimestamp(d.get("created_utc", 0)).strftime("%Y-%m-%d"),
                    })
                    if sub not in results["subreddits"]:
                        results["subreddits"].append(sub)
        except Exception:
            continue

    results["posts"] = sorted(all_posts, key=lambda x: x["score"], reverse=True)[:10]
    results["mention_count"] = len(all_posts)

    # Simple sentiment: upvote score weighted
    if all_posts:
        total_score = sum(p["score"] for p in all_posts)
        # Normalize to -100 to +100: positive scores = bullish sentiment
        results["sentiment_score"] = min(100, max(-100, int(total_score / 100)))
        results["top_title"] = all_posts[0]["title"] if all_posts else ""

    return results


def fetch_stocktwits_sentiment(ticker: str) -> dict:
    """Fetch StockTwits messages and sentiment (free public API)."""
    result = {"bullish": 0, "bearish": 0, "total": 0, "messages": [], "sentiment_ratio": 0}
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        req = urllib.request.Request(url, headers={"User-Agent": "StockAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())

        messages = data.get("messages", [])
        for m in messages[:20]:
            sentiment = m.get("entities", {}).get("sentiment", {})
            if sentiment:
                basic = sentiment.get("basic", "")
                if basic == "Bullish":
                    result["bullish"] += 1
                elif basic == "Bearish":
                    result["bearish"] += 1
            result["messages"].append({
                "text": m.get("body", "")[:150],
                "sentiment": sentiment.get("basic", "Neutral") if sentiment else "Neutral",
                "created": m.get("created_at", "")[:10],
                "likes": m.get("likes", {}).get("total", 0),
            })

        result["total"] = result["bullish"] + result["bearish"]
        if result["total"] > 0:
            result["sentiment_ratio"] = round(result["bullish"] / result["total"] * 100, 1)

    except Exception:
        pass
    return result


def fetch_web_forum_sentiment(ticker: str, company_name: str) -> dict:
    """Scrape DuckDuckGo for financial forum mentions from X, Reddit, financial sites."""
    result = {"web_results": [], "social_buzz": ""}
    queries = [
        f"${ticker} stock analysis site:twitter.com OR site:x.com",
        f"{ticker} {company_name} stock forecast forum discussion",
        f"${ticker} wallstreetbets OR stocktwits OR seekingalpha",
    ]
    all_snippets = []
    for q in queries[:2]:
        try:
            enc = urllib.parse.quote_plus(q)
            url = f"https://html.duckduckgo.com/html/?q={enc}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                html = r.read().decode("utf-8", errors="ignore")
            snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            for t, s in zip(titles[:3], snippets[:3]):
                t_clean = re.sub(r"<[^>]+>", "", t).strip()
                s_clean = re.sub(r"<[^>]+>", "", s).strip()
                all_snippets.append(f"• {t_clean}: {s_clean}")
        except Exception:
            continue

    result["web_results"] = all_snippets[:6]
    result["social_buzz"] = "\n".join(all_snippets[:6])
    return result


# ── Technical Indicators ───────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty or len(df) < 20:
        return {}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    result: dict[str, Any] = {}

    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2])
    result["current_price"]    = round(current, 4)
    result["price_change_1d"]  = round(current - prev, 4)
    result["price_change_pct"] = round((current - prev) / prev * 100, 2)

    year_data = df.tail(252) if len(df) >= 252 else df
    result["52w_high"] = round(float(year_data["High"].max()), 4)
    result["52w_low"]  = round(float(year_data["Low"].min()), 4)
    result["pct_from_52w_high"] = round((current - result["52w_high"]) / result["52w_high"] * 100, 2)

    if not TA_OK:
        return result

    result["sma_20"]  = _last(ta.trend.sma_indicator(close, 20))
    result["sma_50"]  = _last(ta.trend.sma_indicator(close, 50))
    result["sma_200"] = _last(ta.trend.sma_indicator(close, 200)) if len(df) >= 200 else None
    result["ema_12"]  = _last(ta.trend.ema_indicator(close, 12))
    result["ema_26"]  = _last(ta.trend.ema_indicator(close, 26))

    if result["sma_50"] and result["sma_200"]:
        result["ma_cross"] = "golden" if result["sma_50"] > result["sma_200"] else "death"

    result["above_sma_20"]  = current > result["sma_20"]  if result["sma_20"]  else None
    result["above_sma_50"]  = current > result["sma_50"]  if result["sma_50"]  else None
    result["above_sma_200"] = current > result["sma_200"] if result["sma_200"] else None

    rsi = ta.momentum.RSIIndicator(close, 14).rsi()
    result["rsi_14"] = _last(rsi)
    if result["rsi_14"]:
        if result["rsi_14"] >= 70:   result["rsi_signal"] = "overbought"
        elif result["rsi_14"] <= 30: result["rsi_signal"] = "oversold"
        else:                         result["rsi_signal"] = "neutral"

    try:
        stoch = ta.momentum.StochRSIIndicator(close, 14, 3, 3)
        result["stoch_rsi_k"] = _last(stoch.stochrsi_k())
        result["stoch_rsi_d"] = _last(stoch.stochrsi_d())
    except Exception:
        pass

    macd_ind = ta.trend.MACD(close, 26, 12, 9)
    result["macd"]        = _last(macd_ind.macd())
    result["macd_signal"] = _last(macd_ind.macd_signal())
    result["macd_hist"]   = _last(macd_ind.macd_diff())
    if result["macd"] and result["macd_signal"]:
        result["macd_cross"] = "bullish" if result["macd"] > result["macd_signal"] else "bearish"

    bb = ta.volatility.BollingerBands(close, 20, 2)
    result["bb_upper"]  = _last(bb.bollinger_hband())
    result["bb_middle"] = _last(bb.bollinger_mavg())
    result["bb_lower"]  = _last(bb.bollinger_lband())
    result["bb_width"]  = _last(bb.bollinger_wband())
    if result["bb_upper"] and result["bb_lower"]:
        bb_range = result["bb_upper"] - result["bb_lower"]
        result["bb_pct"] = round((current - result["bb_lower"]) / bb_range * 100, 1) if bb_range else None

    result["atr_14"] = _last(ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range())

    vol_sma = volume.rolling(20).mean()
    result["volume_current"] = int(volume.iloc[-1])
    result["volume_avg_20d"] = int(vol_sma.iloc[-1]) if not pd.isna(vol_sma.iloc[-1]) else None
    if result["volume_avg_20d"]:
        result["volume_ratio"] = round(result["volume_current"] / result["volume_avg_20d"], 2)

    result["obv"] = _last(ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume())

    try:
        adx = ta.trend.ADXIndicator(high, low, close, 14)
        result["adx"]      = _last(adx.adx())
        result["adx_pos"]  = _last(adx.adx_pos())
        result["adx_neg"]  = _last(adx.adx_neg())
        if result["adx"]:
            result["trend_strength"] = (
                "strong"   if result["adx"] > 25 else
                "moderate" if result["adx"] > 20 else "weak"
            )
    except Exception:
        pass

    try:
        result["williams_r"] = _last(ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r())
    except Exception:
        pass

    try:
        result["cci"] = _last(ta.trend.CCIIndicator(high, low, close, 20).cci())
    except Exception:
        pass

    recent = df.tail(20)
    result["support"]    = round(float(recent["Low"].min()), 4)
    result["resistance"] = round(float(recent["High"].max()), 4)

    return {k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in result.items() if v is not None}


def _last(series) -> float | None:
    try:
        val = series.dropna().iloc[-1]
        return round(float(val), 4)
    except Exception:
        return None


# ── Algorithmic Trading Signals ────────────────────────────────────────────

def compute_algo_signals(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Mathematical / quantitative trading signals:
    - Mean reversion (Z-score)
    - Momentum factor (multi-period)
    - Trend following
    - Volatility regime
    - Monte Carlo price simulation (GBM)
    - Linear regression trend & R²
    - Volume-price divergence
    - Candlestick pattern detection
    """
    result: dict[str, Any] = {}
    if df.empty or len(df) < 30:
        return result

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    current = float(close.iloc[-1])

    # ── Mean Reversion: Z-score (20-day)
    window = 20
    roll_mean = close.rolling(window).mean()
    roll_std  = close.rolling(window).std()
    z_score = (close - roll_mean) / roll_std
    z_now = float(z_score.iloc[-1]) if not pd.isna(z_score.iloc[-1]) else 0
    result["mean_reversion_zscore"] = round(z_now, 3)
    if z_now < -1.5:
        result["mean_reversion_signal"] = "STRONG_BUY"
        result["mean_reversion_note"]   = f"Z-score {z_now:.2f} — significantly below mean, reversion expected"
    elif z_now < -0.8:
        result["mean_reversion_signal"] = "BUY"
        result["mean_reversion_note"]   = f"Z-score {z_now:.2f} — below mean, mild reversion opportunity"
    elif z_now > 1.5:
        result["mean_reversion_signal"] = "STRONG_SELL"
        result["mean_reversion_note"]   = f"Z-score {z_now:.2f} — significantly above mean, pullback likely"
    elif z_now > 0.8:
        result["mean_reversion_signal"] = "SELL"
        result["mean_reversion_note"]   = f"Z-score {z_now:.2f} — above mean, mild pullback risk"
    else:
        result["mean_reversion_signal"] = "NEUTRAL"
        result["mean_reversion_note"]   = f"Z-score {z_now:.2f} — near mean, no reversion signal"

    # ── Momentum Factor (multi-period returns)
    def pct_return(n):
        if len(close) > n:
            return round((current - float(close.iloc[-n-1])) / float(close.iloc[-n-1]) * 100, 2)
        return None

    result["momentum_1w"]  = pct_return(5)
    result["momentum_1m"]  = pct_return(21)
    result["momentum_3m"]  = pct_return(63)
    result["momentum_6m"]  = pct_return(126)
    result["momentum_1y"]  = pct_return(252)

    # Momentum composite score
    mom_scores = [v for v in [result["momentum_1m"], result["momentum_3m"], result["momentum_6m"]] if v is not None]
    if mom_scores:
        avg_mom = sum(mom_scores) / len(mom_scores)
        result["momentum_composite"] = round(avg_mom, 2)
        result["momentum_signal"] = (
            "STRONG_BULLISH" if avg_mom > 10 else
            "BULLISH"        if avg_mom > 3  else
            "BEARISH"        if avg_mom < -3 else
            "STRONG_BEARISH" if avg_mom < -10 else "NEUTRAL"
        )

    # ── Linear Regression Trend (30-day)
    n = min(30, len(close))
    x = list(range(n))
    y = [float(close.iloc[-n+i]) for i in range(n)]
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    ss_xy = sum((x[i]-x_mean)*(y[i]-y_mean) for i in range(n))
    ss_xx = sum((x[i]-x_mean)**2 for i in range(n))
    slope = ss_xy / ss_xx if ss_xx else 0
    intercept = y_mean - slope * x_mean
    y_pred = [slope*xi + intercept for xi in x]
    ss_res = sum((y[i]-y_pred[i])**2 for i in range(n))
    ss_tot = sum((y[i]-y_mean)**2 for i in range(n))
    r_squared = 1 - ss_res/ss_tot if ss_tot else 0
    daily_slope_pct = (slope / y_mean * 100) if y_mean else 0

    result["linreg_slope_pct_per_day"] = round(daily_slope_pct, 4)
    result["linreg_r_squared"]         = round(r_squared, 4)
    result["linreg_signal"] = (
        "STRONG_UPTREND"   if daily_slope_pct > 0.3 and r_squared > 0.7 else
        "UPTREND"          if daily_slope_pct > 0.1 else
        "STRONG_DOWNTREND" if daily_slope_pct < -0.3 and r_squared > 0.7 else
        "DOWNTREND"        if daily_slope_pct < -0.1 else "SIDEWAYS"
    )

    # ── Volatility Regime (historical volatility)
    returns = close.pct_change().dropna()
    hv_20 = float(returns.tail(20).std() * math.sqrt(252) * 100) if len(returns) >= 20 else 0
    hv_60 = float(returns.tail(60).std() * math.sqrt(252) * 100) if len(returns) >= 60 else hv_20
    result["historical_volatility_20d"] = round(hv_20, 2)
    result["historical_volatility_60d"] = round(hv_60, 2)
    result["vol_regime"] = (
        "HIGH"   if hv_20 > 40 else
        "MEDIUM" if hv_20 > 20 else "LOW"
    )
    result["vol_expanding"] = hv_20 > hv_60 * 1.1

    # ── Monte Carlo Simulation (Geometric Brownian Motion, 1000 paths, 30 days)
    mc_result = _monte_carlo_simulation(close, days=30, simulations=1000)
    result.update(mc_result)

    # ── Volume-Price Divergence
    if len(close) >= 5 and len(volume) >= 5:
        price_chg_5d = (float(close.iloc[-1]) - float(close.iloc[-6])) / float(close.iloc[-6]) if len(close) > 5 else 0
        vol_chg_5d   = (float(volume.iloc[-1]) - float(volume.rolling(20).mean().iloc[-1])) / float(volume.rolling(20).mean().iloc[-1]) if not pd.isna(volume.rolling(20).mean().iloc[-1]) else 0
        if price_chg_5d > 0.02 and vol_chg_5d > 0.5:
            result["volume_price_signal"] = "CONFIRMED_BREAKOUT"
        elif price_chg_5d > 0.02 and vol_chg_5d < -0.2:
            result["volume_price_signal"] = "WEAK_RALLY"
        elif price_chg_5d < -0.02 and vol_chg_5d > 0.5:
            result["volume_price_signal"] = "CONFIRMED_BREAKDOWN"
        elif price_chg_5d < -0.02 and vol_chg_5d < -0.2:
            result["volume_price_signal"] = "WEAK_SELLOFF"
        else:
            result["volume_price_signal"] = "NEUTRAL"

    # ── Candlestick Pattern Detection (simplified)
    patterns = _detect_candlestick_patterns(df)
    result["candlestick_patterns"] = patterns

    # ── Composite Algo Score (0-100)
    algo_bull = 0
    algo_bear = 0
    if result.get("mean_reversion_signal") in ("STRONG_BUY", "BUY"):      algo_bull += 2
    if result.get("mean_reversion_signal") in ("STRONG_SELL", "SELL"):    algo_bear += 2
    if result.get("momentum_signal") in ("STRONG_BULLISH", "BULLISH"):    algo_bull += 2
    if result.get("momentum_signal") in ("STRONG_BEARISH", "BEARISH"):    algo_bear += 2
    if result.get("linreg_signal") in ("STRONG_UPTREND", "UPTREND"):      algo_bull += 1
    if result.get("linreg_signal") in ("STRONG_DOWNTREND", "DOWNTREND"):  algo_bear += 1
    if result.get("volume_price_signal") == "CONFIRMED_BREAKOUT":         algo_bull += 2
    if result.get("volume_price_signal") == "CONFIRMED_BREAKDOWN":        algo_bear += 2

    total = algo_bull + algo_bear
    result["algo_score"] = int(algo_bull / total * 100) if total else 50
    result["algo_direction"] = (
        "BULLISH" if result["algo_score"] >= 60 else
        "BEARISH" if result["algo_score"] <= 40 else "NEUTRAL"
    )

    return result


def _monte_carlo_simulation(close: pd.Series, days: int = 30, simulations: int = 1000) -> dict:
    """Geometric Brownian Motion Monte Carlo for price distribution."""
    try:
        returns = close.pct_change().dropna()
        if len(returns) < 20:
            return {}

        mu    = float(returns.mean())
        sigma = float(returns.std())
        S0    = float(close.iloc[-1])

        # Seed for reproducibility
        random.seed(42)
        endings = []
        for _ in range(simulations):
            price = S0
            for _ in range(days):
                z = random.gauss(0, 1)
                price *= math.exp((mu - 0.5 * sigma**2) + sigma * z)
            endings.append(price)

        endings.sort()
        pct5  = endings[int(0.05 * simulations)]
        pct25 = endings[int(0.25 * simulations)]
        pct50 = endings[int(0.50 * simulations)]
        pct75 = endings[int(0.75 * simulations)]
        pct95 = endings[int(0.95 * simulations)]

        return {
            "monte_carlo_days":     days,
            "monte_carlo_sims":     simulations,
            "mc_price_5pct":   round(pct5, 2),
            "mc_price_25pct":  round(pct25, 2),
            "mc_price_median": round(pct50, 2),
            "mc_price_75pct":  round(pct75, 2),
            "mc_price_95pct":  round(pct95, 2),
            "mc_upside_pct":   round((pct75 - S0) / S0 * 100, 2),
            "mc_downside_pct": round((pct25 - S0) / S0 * 100, 2),
            "mc_expected_pct": round((pct50 - S0) / S0 * 100, 2),
        }
    except Exception:
        return {}


def _detect_candlestick_patterns(df: pd.DataFrame) -> list[str]:
    """Detect common candlestick reversal/continuation patterns."""
    patterns = []
    if len(df) < 3:
        return patterns

    def body(i): return abs(float(df["Close"].iloc[i]) - float(df["Open"].iloc[i]))
    def upper_shadow(i): return float(df["High"].iloc[i]) - max(float(df["Close"].iloc[i]), float(df["Open"].iloc[i]))
    def lower_shadow(i): return min(float(df["Close"].iloc[i]), float(df["Open"].iloc[i])) - float(df["Low"].iloc[i])
    def is_bullish(i): return float(df["Close"].iloc[i]) > float(df["Open"].iloc[i])
    def is_bearish(i): return float(df["Close"].iloc[i]) < float(df["Open"].iloc[i])

    i = -1  # Latest candle

    # Doji (small body relative to range)
    rng = float(df["High"].iloc[i]) - float(df["Low"].iloc[i])
    if rng > 0 and body(i) / rng < 0.1:
        patterns.append("Doji — indecision, potential reversal")

    # Hammer / Hanging Man
    if body(i) > 0 and lower_shadow(i) >= 2 * body(i) and upper_shadow(i) < body(i):
        if is_bullish(i):
            patterns.append("Hammer — bullish reversal signal")
        else:
            patterns.append("Hanging Man — bearish reversal signal")

    # Shooting Star / Inverted Hammer
    if body(i) > 0 and upper_shadow(i) >= 2 * body(i) and lower_shadow(i) < body(i):
        if is_bearish(i):
            patterns.append("Shooting Star — bearish reversal signal")
        else:
            patterns.append("Inverted Hammer — potential bullish reversal")

    # Engulfing (2-candle)
    if len(df) >= 2:
        prev_body = body(-2)
        curr_body = body(-1)
        if is_bullish(-1) and is_bearish(-2) and curr_body > prev_body:
            patterns.append("Bullish Engulfing — strong reversal signal")
        elif is_bearish(-1) and is_bullish(-2) and curr_body > prev_body:
            patterns.append("Bearish Engulfing — strong reversal signal")

    # Morning Star / Evening Star (3-candle)
    if len(df) >= 3:
        if (is_bearish(-3) and body(-2) < body(-3) * 0.3 and is_bullish(-1)
                and float(df["Close"].iloc[-1]) > float(df["Open"].iloc[-3])):
            patterns.append("Morning Star — bullish reversal pattern")
        elif (is_bullish(-3) and body(-2) < body(-3) * 0.3 and is_bearish(-1)
                and float(df["Close"].iloc[-1]) < float(df["Open"].iloc[-3])):
            patterns.append("Evening Star — bearish reversal pattern")

    return patterns if patterns else ["No major pattern detected"]


def detect_candlestick_signal(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized, whole-history sibling of _detect_candlestick_patterns() (which
    only inspects the latest candle, for the live dashboard). Returns a
    {-1, 0, 1} signal per day for backtesting — reuses the same geometric
    pattern definitions (body/shadow ratios, engulfing) but computed across
    every row via pandas vectorized ops instead of a fixed "latest candle"
    index, so a strategy can be backtested over history.
    Only Bullish/Bearish Engulfing are used here (the strongest, least
    ambiguous 2-candle pattern) — Doji/Hammer/Star patterns are more
    subjective single/triple-candle shapes better suited to the prose
    interpretation _detect_candlestick_patterns() already provides.
    """
    o, c = df["Open"].astype(float), df["Close"].astype(float)
    body = (c - o).abs()
    is_bullish = c > o
    is_bearish = c < o

    prev_body = body.shift(1)
    prev_bullish = is_bullish.shift(1)
    prev_bearish = is_bearish.shift(1)

    bullish_engulf = is_bullish & prev_bearish.fillna(False) & (body > prev_body)
    bearish_engulf = is_bearish & prev_bullish.fillna(False) & (body > prev_body)

    signal = pd.Series(0, index=df.index)
    signal[bullish_engulf] = 1
    signal[bearish_engulf] = -1
    return signal


# ── Composite signals summary ──────────────────────────────────────────────

def compute_signal_summary(indicators: dict) -> dict:
    bull = 0
    bear = 0
    signals = []

    rsi = indicators.get("rsi_14")
    if rsi:
        if rsi < 30:   bull += 2; signals.append(f"RSI={rsi:.1f} — oversold (bullish)")
        elif rsi > 70: bear += 2; signals.append(f"RSI={rsi:.1f} — overbought (bearish)")
        elif rsi > 55: bull += 1; signals.append(f"RSI={rsi:.1f} — bullish momentum")
        elif rsi < 45: bear += 1; signals.append(f"RSI={rsi:.1f} — bearish momentum")

    macd_cross = indicators.get("macd_cross")
    macd_hist  = indicators.get("macd_hist")
    if macd_cross == "bullish":  bull += 2; signals.append("MACD above signal line (bullish)")
    elif macd_cross == "bearish": bear += 2; signals.append("MACD below signal line (bearish)")
    if macd_hist and macd_hist > 0:  bull += 1; signals.append(f"MACD histogram positive ({macd_hist:.3f})")
    elif macd_hist and macd_hist < 0: bear += 1; signals.append(f"MACD histogram negative ({macd_hist:.3f})")

    if indicators.get("above_sma_20"):  bull += 1; signals.append("Price above 20-day SMA")
    else:                               bear += 1; signals.append("Price below 20-day SMA")
    if indicators.get("above_sma_50"):  bull += 1; signals.append("Price above 50-day SMA")
    else:                               bear += 1; signals.append("Price below 50-day SMA")
    if indicators.get("above_sma_200"): bull += 2; signals.append("Price above 200-day SMA (long-term bullish)")
    elif indicators.get("above_sma_200") is False:
        bear += 2; signals.append("Price below 200-day SMA (long-term bearish)")

    ma_cross = indicators.get("ma_cross")
    if ma_cross == "golden": bull += 2; signals.append("Golden cross (SMA50 > SMA200)")
    elif ma_cross == "death": bear += 2; signals.append("Death cross (SMA50 < SMA200)")

    bb_pct = indicators.get("bb_pct")
    if bb_pct is not None:
        if bb_pct < 10:   bull += 1; signals.append(f"Price near lower Bollinger Band ({bb_pct:.0f}%)")
        elif bb_pct > 90: bear += 1; signals.append(f"Price near upper Bollinger Band ({bb_pct:.0f}%)")

    vol_ratio = indicators.get("volume_ratio")
    if vol_ratio and vol_ratio > 1.5:
        signals.append(f"High volume day ({vol_ratio:.1f}x average) — confirms move")

    adx = indicators.get("adx")
    trend_str = indicators.get("trend_strength", "")
    if adx:
        signals.append(f"ADX={adx:.1f} — {trend_str} trend")

    total = bull + bear
    score = int(bull / total * 100) if total else 50

    if score >= 65:   direction = "BULLISH"; strength = "strong" if score >= 75 else "moderate"
    elif score <= 35: direction = "BEARISH"; strength = "strong" if score <= 25 else "moderate"
    else:             direction = "NEUTRAL";  strength = "weak"

    return {
        "score": score,
        "direction": direction,
        "strength": strength,
        "bull_signals": bull,
        "bear_signals": bear,
        "signals": signals,
    }
