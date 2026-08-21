"""
Hand-curated eval cases for tools/eval_agents.py.

The prediction ledger (data/prediction_ledger.py) is the natural place to
source real, matured cases from - but it currently has fewer than 10
matured predictions (per STOCK_AGENT_V1.md's own INSUFFICIENT_HISTORY
threshold), so these are hand-built instead. Each case uses a real,
recognizable ticker with figures that match that company's well-known
general financial/technical *character* (a quality mega-cap, a richly
valued grower, a value name in decline, a meme-stock-style extreme) -
they are illustrative fixtures constructed to exercise the judge rubrics
across a spread of "obviously good/bad/ambiguous" scenarios, NOT a
precise, verified point-in-time snapshot of any real filing or trading
session. Swap in real prediction-ledger cases here once enough of them
mature; every dict below matches the exact keyword arguments its
agent's `_build_X_prompt()` function takes in agents/stock_agents.py.
"""
from __future__ import annotations

FUNDAMENTALS_CASES = [
    {
        "notes": "Quality mega-cap: strong margins/ROE, slow growth, premium multiple - "
                 "tests whether the agent correctly reads 'expensive but high quality' "
                 "rather than defaulting to a lazy 'PE is high = sell'.",
        "ticker": "AAPL",
        "fundamentals": {
            "longName": "Apple Inc.", "sector": "Technology",
            "trailingPE": 29.4, "forwardPE": 27.1, "priceToBook": 46.2,
            "enterpriseToEbitda": 21.8, "profitMargins": 0.262,
            "returnOnEquity": 1.52, "revenueGrowth": 0.02,
            "debtToEquity": 145.0, "freeCashflow": 99_000_000_000,
        },
        "earnings": {"epsCurrentYear": 6.68, "epsForwardYear": 7.35, "earningsGrowth": 0.10},
        "analyst_ratings": {"recommendationMean": 2.0, "targetMeanPrice": 240.0,
                             "numberOfAnalystOpinions": 38},
    },
    {
        "notes": "Richly-valued, unprofitable-on-GAAP grower - tests whether the agent "
                 "notices growth quality (accelerating, real) without being scared off "
                 "purely by a negative/undefined PE, and without hand-waving away the "
                 "genuine valuation risk either.",
        "ticker": "CRWD",
        "fundamentals": {
            "longName": "CrowdStrike Holdings", "sector": "Technology",
            "trailingPE": None, "forwardPE": 68.5, "priceToBook": 20.1,
            "enterpriseToEbitda": 95.0, "profitMargins": -0.02,
            "returnOnEquity": -0.03, "revenueGrowth": 0.29,
            "debtToEquity": 40.0, "freeCashflow": 1_000_000_000,
        },
        "earnings": {"epsCurrentYear": 0.30, "epsForwardYear": 0.95, "earningsGrowth": 0.45},
        "analyst_ratings": {"recommendationMean": 1.8, "targetMeanPrice": 420.0,
                             "numberOfAnalystOpinions": 42},
    },
    {
        "notes": "Declining value/legacy name - cheap multiple but shrinking revenue and "
                 "thin margins; tests whether 'low PE' correctly reads as a value trap "
                 "warning rather than being scored bullish on cheapness alone.",
        "ticker": "INTC",
        "fundamentals": {
            "longName": "Intel Corporation", "sector": "Technology",
            "trailingPE": None, "forwardPE": 22.0, "priceToBook": 0.9,
            "enterpriseToEbitda": 11.5, "profitMargins": -0.05,
            "returnOnEquity": -0.04, "revenueGrowth": -0.08,
            "debtToEquity": 55.0, "freeCashflow": -8_000_000_000,
        },
        "earnings": {"epsCurrentYear": -0.15, "epsForwardYear": 0.40, "earningsGrowth": -1.20},
        "analyst_ratings": {"recommendationMean": 2.9, "targetMeanPrice": 24.0,
                             "numberOfAnalystOpinions": 30},
    },
    {
        "notes": "Missing/sparse data edge case (thin coverage small-cap) - tests whether "
                 "the agent hedges appropriately on 'N/A' fields instead of inventing "
                 "precision it doesn't have.",
        "ticker": "SMCX",
        "fundamentals": {"longName": "Example Small Cap Co.", "sector": "Industrials",
                          "trailingPE": None, "profitMargins": None},
        "earnings": {},
        "analyst_ratings": {"numberOfAnalystOpinions": 1},
    },
]

TECHNICAL_CASES = [
    {
        "notes": "Clean, textbook uptrend with confirming volume - the 'easy' case a "
                 "good technical agent should read unambiguously bullish.",
        "ticker": "NVDA",
        "indicators": {
            "current_price": 138.50, "sma20": 132.0, "sma50": 121.0, "sma200": 98.0,
            "rsi": 64.0, "macd": 2.1, "macd_signal": 1.4, "adx": 31.0,
            "stoch_rsi": 0.72, "bb_upper": 145.0, "bb_lower": 118.0,
        },
        "signal_summary": {"score": 78, "direction": "BULLISH"},
        "price_history_summary": (
            "Current: $138.50\n1W: +3.1% | 1M: +9.4% | 3M: +22.0% | 1Y: +180.0%\n"
            "Last 5 closes: $132.10, $134.80, $135.90, $137.20, $138.50\n"
            "Avg vol (20d): 245,000,000"
        ),
    },
    {
        "notes": "Overbought-into-resistance - tests whether the agent flags exhaustion "
                 "risk (high RSI against a resistance level) instead of chasing pure trend.",
        "ticker": "TSLA",
        "indicators": {
            "current_price": 265.0, "sma20": 240.0, "sma50": 220.0, "sma200": 210.0,
            "rsi": 81.0, "macd": 4.5, "macd_signal": 3.9, "adx": 22.0,
            "stoch_rsi": 0.95, "bb_upper": 266.0, "bb_lower": 225.0,
        },
        "signal_summary": {"score": 58, "direction": "MIXED"},
        "price_history_summary": (
            "Current: $265.00\n1W: +6.5% | 1M: +18.0% | 3M: +25.0% | 1Y: +40.0%\n"
            "Last 5 closes: $248.00, $253.50, $258.00, $262.00, $265.00\n"
            "Avg vol (20d): 95,000,000"
        ),
    },
    {
        "notes": "Range-bound consolidation, no clear trend - tests whether the agent "
                 "correctly says 'no edge here' instead of forcing a directional call.",
        "ticker": "KO",
        "indicators": {
            "current_price": 62.0, "sma20": 61.8, "sma50": 62.2, "sma200": 61.5,
            "rsi": 51.0, "macd": 0.05, "macd_signal": 0.03, "adx": 12.0,
            "stoch_rsi": 0.48, "bb_upper": 64.0, "bb_lower": 60.0,
        },
        "signal_summary": {"score": 50, "direction": "NEUTRAL"},
        "price_history_summary": (
            "Current: $62.00\n1W: +0.2% | 1M: -0.5% | 3M: +1.1% | 1Y: +3.0%\n"
            "Last 5 closes: $61.70, $61.90, $62.10, $61.95, $62.00\n"
            "Avg vol (20d): 12,000,000"
        ),
    },
    {
        "notes": "Sharp breakdown below all moving averages - tests bearish-case symmetry, "
                 "since these agents' rubrics/examples skew bullish by default.",
        "ticker": "SNAP",
        "indicators": {
            "current_price": 9.20, "sma20": 11.50, "sma50": 12.80, "sma200": 13.50,
            "rsi": 24.0, "macd": -0.8, "macd_signal": -0.5, "adx": 34.0,
            "stoch_rsi": 0.08, "bb_upper": 12.5, "bb_lower": 9.0,
        },
        "signal_summary": {"score": 18, "direction": "BEARISH"},
        "price_history_summary": (
            "Current: $9.20\n1W: -8.0% | 1M: -22.0% | 3M: -30.0% | 1Y: -35.0%\n"
            "Last 5 closes: $10.80, $10.20, $9.90, $9.50, $9.20\n"
            "Avg vol (20d): 28,000,000"
        ),
    },
]

SOCIAL_CASES = [
    {
        "notes": "Genuine meme-stock/short-squeeze pattern (GME 2021-style) - tests "
                 "whether the agent identifies the crowd dynamic and flags it as a "
                 "contrarian-risk signal, not just 'sentiment is bullish, buy.'",
        "ticker": "GME", "company_name": "GameStop Corp.",
        "reddit_data": {
            "mention_count": 4200, "subreddits": ["wallstreetbets", "Superstonk"],
            "sentiment_score": 88,
            "posts": [
                {"subreddit": "wallstreetbets", "title": "GME to the moon, short interest still huge", "score": 15000, "comments": 3200},
                {"subreddit": "Superstonk", "title": "DRS numbers update", "score": 8000, "comments": 1500},
            ],
        },
        "stocktwits_data": {"bullish": 8400, "bearish": 900, "sentiment_ratio": 90, "total": 9300,
                             "messages": [{"sentiment": "Bullish", "text": "This is the way"},
                                          {"sentiment": "Bullish", "text": "Loaded up more shares"}]},
        "web_forum_data": {"web_results": ["Reddit forum: massive coordinated buying discussion",
                                            "Short interest reported near 20% of float"]},
    },
    {
        "notes": "Low-attention blue chip with sparse/neutral chatter - tests that the "
                 "agent doesn't manufacture a strong signal out of near-nothing.",
        "ticker": "JNJ", "company_name": "Johnson & Johnson",
        "reddit_data": {"mention_count": 3, "subreddits": [], "sentiment_score": 2, "posts": []},
        "stocktwits_data": {"bullish": 5, "bearish": 4, "sentiment_ratio": 55, "total": 9, "messages": []},
        "web_forum_data": {"web_results": []},
    },
    {
        "notes": "Strongly bearish retail sentiment on bad news - tests bearish-case "
                 "coverage and whether the agent can call a real negative crowd read.",
        "ticker": "BBBY", "company_name": "Example Retailer Co.",
        "reddit_data": {
            "mention_count": 1800, "subreddits": ["wallstreetbets", "stocks"],
            "sentiment_score": -70,
            "posts": [{"subreddit": "wallstreetbets", "title": "Bankruptcy filing looks imminent", "score": 6000, "comments": 2100}],
        },
        "stocktwits_data": {"bullish": 300, "bearish": 2200, "sentiment_ratio": 12, "total": 2500,
                             "messages": [{"sentiment": "Bearish", "text": "Selling everything, this is done"}]},
        "web_forum_data": {"web_results": ["Multiple reports of store closures and vendor pullback"]},
    },
    {
        "notes": "Contradictory signal - loud Reddit bull chatter but StockTwits skewed "
                 "bearish; tests whether the agent notices and reports the divergence "
                 "instead of averaging it away silently.",
        "ticker": "PLTR", "company_name": "Palantir Technologies",
        "reddit_data": {
            "mention_count": 900, "subreddits": ["wallstreetbets"], "sentiment_score": 65,
            "posts": [{"subreddit": "wallstreetbets", "title": "PLTR government contracts thread", "score": 4000, "comments": 900}],
        },
        "stocktwits_data": {"bullish": 400, "bearish": 950, "sentiment_ratio": 30, "total": 1350,
                             "messages": [{"sentiment": "Bearish", "text": "Valuation is way ahead of fundamentals"}]},
        "web_forum_data": {"web_results": ["Mixed opinions on forward contract pipeline"]},
    },
]

ALGO_CASES = [
    {
        "notes": "Strong momentum + trend agreement, low vol - the clean, high-confidence "
                 "quant case.",
        "ticker": "AAPL", "current_price": 227.50,
        "algo_signals": {
            "mean_reversion_zscore": 0.4, "mean_reversion_signal": "NEUTRAL", "mean_reversion_note": "within normal band",
            "momentum_1w": 1.8, "momentum_1m": 6.2, "momentum_3m": 14.5, "momentum_6m": 22.0,
            "momentum_signal": "BULLISH",
            "linreg_slope_pct_per_day": 0.18, "linreg_r_squared": 0.82, "linreg_signal": "BULLISH",
            "historical_volatility_20d": 18.0, "historical_volatility_60d": 20.5,
            "vol_regime": "LOW", "vol_expanding": False,
            "mc_price_5pct": 205.0, "mc_downside_pct": -9.9, "mc_price_median": 232.0,
            "mc_expected_pct": 2.0, "mc_price_95pct": 258.0, "mc_upside_pct": 13.4,
            "volume_price_signal": "CONFIRMING", "candlestick_patterns": ["bullish engulfing"],
            "algo_score": 74, "algo_direction": "BULLISH",
        },
        "indicators": {"current_price": 227.50},
    },
    {
        "notes": "Mean-reversion vs momentum conflict (extended z-score against a still- "
                 "positive trend) - the genuinely hard case that tests whether the agent "
                 "actually reasons about the tension instead of picking one signal blindly.",
        "ticker": "NVDA", "current_price": 138.50,
        "algo_signals": {
            "mean_reversion_zscore": 2.6, "mean_reversion_signal": "OVERBOUGHT", "mean_reversion_note": "2.6 std above 20d mean",
            "momentum_1w": 3.1, "momentum_1m": 9.4, "momentum_3m": 22.0, "momentum_6m": 55.0,
            "momentum_signal": "BULLISH",
            "linreg_slope_pct_per_day": 0.35, "linreg_r_squared": 0.71, "linreg_signal": "BULLISH",
            "historical_volatility_20d": 42.0, "historical_volatility_60d": 38.0,
            "vol_regime": "HIGH", "vol_expanding": True,
            "mc_price_5pct": 115.0, "mc_downside_pct": -17.0, "mc_price_median": 141.0,
            "mc_expected_pct": 1.8, "mc_price_95pct": 172.0, "mc_upside_pct": 24.2,
            "volume_price_signal": "CONFIRMING", "candlestick_patterns": [],
            "algo_score": 62, "algo_direction": "BULLISH",
        },
        "indicators": {"current_price": 138.50},
    },
    {
        "notes": "High-vol expanding regime with a bearish linear-regression slope - tests "
                 "whether the agent correctly treats expanding HIGH vol as a reason for "
                 "caution/smaller sizing, not just background noise.",
        "ticker": "SNAP", "current_price": 9.20,
        "algo_signals": {
            "mean_reversion_zscore": -2.1, "mean_reversion_signal": "OVERSOLD", "mean_reversion_note": "2.1 std below 20d mean",
            "momentum_1w": -8.0, "momentum_1m": -22.0, "momentum_3m": -30.0, "momentum_6m": -35.0,
            "momentum_signal": "BEARISH",
            "linreg_slope_pct_per_day": -0.55, "linreg_r_squared": 0.65, "linreg_signal": "BEARISH",
            "historical_volatility_20d": 68.0, "historical_volatility_60d": 52.0,
            "vol_regime": "HIGH", "vol_expanding": True,
            "mc_price_5pct": 6.80, "mc_downside_pct": -26.1, "mc_price_median": 9.00,
            "mc_expected_pct": -2.2, "mc_price_95pct": 11.90, "mc_upside_pct": 29.3,
            "volume_price_signal": "CONFIRMING", "candlestick_patterns": ["bearish engulfing"],
            "algo_score": 22, "algo_direction": "BEARISH",
        },
        "indicators": {"current_price": 9.20},
    },
    {
        "notes": "Flat/neutral across the board - tests that the agent is willing to say "
                 "'no statistical edge' rather than manufacturing a lean from noise.",
        "ticker": "KO", "current_price": 62.0,
        "algo_signals": {
            "mean_reversion_zscore": 0.1, "mean_reversion_signal": "NEUTRAL", "mean_reversion_note": "within normal band",
            "momentum_1w": 0.2, "momentum_1m": -0.5, "momentum_3m": 1.1, "momentum_6m": 2.0,
            "momentum_signal": "NEUTRAL",
            "linreg_slope_pct_per_day": 0.01, "linreg_r_squared": 0.10, "linreg_signal": "NEUTRAL",
            "historical_volatility_20d": 11.0, "historical_volatility_60d": 12.0,
            "vol_regime": "LOW", "vol_expanding": False,
            "mc_price_5pct": 58.0, "mc_downside_pct": -6.5, "mc_price_median": 62.2,
            "mc_expected_pct": 0.3, "mc_price_95pct": 66.5, "mc_upside_pct": 7.3,
            "volume_price_signal": "NEUTRAL", "candlestick_patterns": [],
            "algo_score": 50, "algo_direction": "NEUTRAL",
        },
        "indicators": {"current_price": 62.0},
    },
]

PREDICTION_CASES = [
    {
        "notes": "All 4 source analyses agree bullish - tests that the combiner's JSON "
                 "action actually reflects unanimous input rather than hedging to HOLD.",
        "ticker": "AAPL", "company_name": "Apple Inc.", "current_price": 227.50,
        "fundamentals_analysis": "Quality compounder, premium but justified multiple given margin "
                                  "and cash generation. Fundamental Score: 8/10.",
        "technical_analysis": "Clean uptrend, confirming volume, RSI healthy not overbought. "
                               "Entry $225-228, target $250, stop $210. Technical Score: 8/10.",
        "social_analysis": "Low-key positive sentiment, no unusual retail activity. Social Score: 6/10.",
        "algo_analysis": "Momentum and trend both bullish, low vol regime, positive expected value. "
                          "Algo Score: 7/10.",
        "signal_summary": {"score": 78, "direction": "BULLISH"},
    },
    {
        "notes": "All 4 source analyses agree bearish - the mirror case, tests the same "
                 "consistency check on the sell side.",
        "ticker": "SNAP", "company_name": "Snap Inc.", "current_price": 9.20,
        "fundamentals_analysis": "Deteriorating unit economics, no clear path to profitability. "
                                  "Fundamental Score: 3/10.",
        "technical_analysis": "Breakdown below all moving averages, oversold but no reversal signal "
                               "yet. Technical Score: 2/10.",
        "social_analysis": "Sentiment sharply negative, bearish StockTwits skew. Social Score: 2/10.",
        "algo_analysis": "Bearish momentum and trend, high expanding volatility - caution warranted. "
                          "Algo Score: 3/10.",
        "signal_summary": {"score": 18, "direction": "BEARISH"},
    },
    {
        "notes": "Genuinely mixed inputs (bullish fundamentals, bearish technicals) - the "
                 "case most likely to expose whether the combiner is actually reasoning "
                 "about the disagreement, which is exactly what a self-critique pass "
                 "(Phase 6) would be designed to catch if the JSON quietly ignores it.",
        "ticker": "INTC", "company_name": "Intel Corporation", "current_price": 21.50,
        "fundamentals_analysis": "Deeply cheap on book value with a credible multi-year turnaround "
                                  "narrative (foundry ramp), but currently GAAP-unprofitable and "
                                  "burning cash. Fundamental Score: 5/10.",
        "technical_analysis": "Still in a multi-quarter downtrend, price below all major moving "
                               "averages, no confirmed reversal. Technical Score: 3/10.",
        "social_analysis": "Sentiment mildly negative, turnaround skeptics outnumber bulls. "
                            "Social Score: 4/10.",
        "algo_analysis": "Momentum bearish but z-score deeply oversold - could be a reversion "
                          "setup or a continuing value trap. Algo Score: 4/10.",
        "signal_summary": {"score": 38, "direction": "MIXED"},
    },
    {
        "notes": "Degraded input case - one source analysis is an '[Agent error: ...]' string "
                  "(simulating a failed upstream agent, per Phase 1's failure-isolation "
                  "behavior) - tests that the prediction agent still produces a sane, "
                  "appropriately-hedged verdict instead of treating the error string as content.",
        "ticker": "PLTR", "company_name": "Palantir Technologies", "current_price": 42.0,
        "fundamentals_analysis": "[Agent error: DeepSeek timeout after 180s]",
        "technical_analysis": "Choppy, range-bound, no clear trend. Technical Score: 5/10.",
        "social_analysis": "Divergent signal: loud Reddit bulls, cautious StockTwits crowd. "
                            "Social Score: 5/10.",
        "algo_analysis": "Neutral momentum, average volatility, no statistical edge either way. "
                          "Algo Score: 5/10.",
        "signal_summary": {"score": 50, "direction": "NEUTRAL"},
    },
]

EVAL_CASES = {
    "fundamentals": FUNDAMENTALS_CASES,
    "technical": TECHNICAL_CASES,
    "social": SOCIAL_CASES,
    "algo": ALGO_CASES,
    "prediction": PREDICTION_CASES,
}
