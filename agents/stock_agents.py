"""
Stock Agent Specialists — powered by DeepSeek LLM (OpenAI-compatible API)
5 specialist agents: Fundamentals, Technical, Social, Algo, Prediction
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Run: pip3 install openai")

from agents.prediction_schema import validate_prediction


def _get_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        # deepseek-v4-pro reasons before it answers, and a measured call on this
        # key took 115s. At the previous 45s every substantial agent timed out,
        # retried, and still failed — which is what produced the empty
        # completions behind "Could not generate prediction".
        timeout=180.0,
        max_retries=1,
    )


def _call(client: OpenAI, system: str, user: str,
          model: str = "deepseek-v4-pro", max_tokens: int = 8000) -> str:
    # deepseek-v4-pro reasons before answering; reasoning tokens count against
    # max_tokens. Too low a budget (was 2000) lets it exhaust the whole thing
    # mid-thought (finish_reason="length") and return empty content.
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"[Agent error: {e}]"


def _fmt(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def _pillar_block(label: str, pillar: dict | None) -> str:
    """Format the deterministic pillar score (backtest/pillars.py) for a
    given domain as an appended prompt block, or "" if none was supplied.

    This is grounding, not instruction: the agent isn't told what to
    conclude, just given the number the system will actually act on, so
    its prose can explain that number instead of independently inventing
    a different one for the same domain.
    """
    if not pillar:
        return ""
    return (
        f"\n\nDETERMINISTIC {label} PILLAR SCORE (this app's own quant scoring "
        f"engine, not a guess): {pillar.get('score')}/100\n"
        f"Formula: {pillar.get('formula')}\n"
        f"This is the number the system will actually act on - your analysis "
        f"should explain and be consistent with it, not silently contradict it."
    )


# ══════════════════════════════════════════════════════════════════════════
# 1. Fundamentals Agent
# ══════════════════════════════════════════════════════════════════════════

def _build_fundamentals_prompt(
    ticker: str,
    fundamentals: dict,
    earnings: dict,
    analyst_ratings: dict,
    pillar: dict | None = None,
) -> tuple[str, str]:
    system = """You are a CFA-level fundamental analyst. Evaluate stocks on
valuation multiples, earnings quality, growth trajectory, balance sheet health,
and analyst consensus. Compare to sector norms and flag outliers."""

    user = f"""Perform fundamental analysis of {ticker}.

FUNDAMENTALS:
{_fmt(fundamentals)}

EARNINGS DATA:
{_fmt(earnings)}

ANALYST RATINGS:
{_fmt(analyst_ratings)}

Analyze:
1. **Valuation** — cheap, fair, or expensive? (PE, PB, EV/EBITDA vs sector)
2. **Growth Quality** — revenue/earnings trend; accelerating or decelerating?
3. **Financial Health** — debt, cash, free cash flow
4. **Profitability** — margins, ROE, ROA vs industry
5. **Analyst Consensus** — price target upside/downside
6. **Fundamental Score** — rate 1-10
7. **Fair Value Estimate** — rough intrinsic value range

Be specific with numbers. 300 words max.{_pillar_block("FUNDAMENTALS", pillar)}"""

    return system, user


def run_fundamentals_agent(
    client: OpenAI,
    ticker: str,
    fundamentals: dict,
    earnings: dict,
    analyst_ratings: dict,
    pillar: dict | None = None,
    verbose: bool = False,
) -> str:
    system, user = _build_fundamentals_prompt(ticker, fundamentals, earnings, analyst_ratings, pillar)
    return _call(client, system, user)


# ══════════════════════════════════════════════════════════════════════════
# 2. Technical Agent
# ══════════════════════════════════════════════════════════════════════════

def _build_technical_prompt(
    ticker: str,
    indicators: dict,
    signal_summary: dict,
    price_history_summary: str,
    pillar: dict | None = None,
) -> tuple[str, str]:
    system = """You are an expert technical analyst with 20 years of experience.
Read price action, trends, momentum, and volume to determine optimal
entry and exit points. Set specific price targets and stop-losses."""

    user = f"""Perform technical analysis of {ticker}.

TECHNICAL INDICATORS:
{_fmt(indicators)}

SIGNAL SUMMARY:
{_fmt(signal_summary)}

PRICE HISTORY:
{price_history_summary}

Analyze:
1. **Trend** — uptrend, downtrend, or consolidation? (use MAs, ADX)
2. **Momentum** — RSI, MACD, Stoch RSI status
3. **Volume** — confirming or diverging from price?
4. **Support & Resistance** — key levels to watch
5. **Entry Points** — specific BUY price levels
6. **Exit Points** — price target(s) and stop-loss
7. **Technical Score** — rate 1-10

Give specific price levels. 300 words max.{_pillar_block("TECHNICAL", pillar)}"""

    return system, user


def run_technical_agent(
    client: OpenAI,
    ticker: str,
    indicators: dict,
    signal_summary: dict,
    price_history_summary: str,
    pillar: dict | None = None,
    verbose: bool = False,
) -> str:
    system, user = _build_technical_prompt(ticker, indicators, signal_summary, price_history_summary, pillar)
    return _call(client, system, user)


# ══════════════════════════════════════════════════════════════════════════
# 3. Social Sentiment Agent
# ══════════════════════════════════════════════════════════════════════════

def _build_social_prompt(
    ticker: str,
    company_name: str,
    reddit_data: dict,
    stocktwits_data: dict,
    web_forum_data: dict,
    pillar: dict | None = None,
) -> tuple[str, str]:
    reddit_posts = "\n".join(
        f"- [r/{p['subreddit']}] {p['title']} (score:{p['score']}, comments:{p['comments']})"
        for p in reddit_data.get("posts", [])[:8]
    ) or "No Reddit posts found."

    st_msgs = "\n".join(
        f"- [{m['sentiment']}] {m['text']}"
        for m in stocktwits_data.get("messages", [])[:6]
    ) or "No StockTwits messages."

    system = """You are a social sentiment analyst specializing in retail investor behavior,
Reddit momentum plays, and social media signals. Identify crowd psychology,
unusual activity, and sentiment shifts before they move prices."""

    user = f"""Analyze social media sentiment for {company_name} ({ticker}).

REDDIT DATA:
- Mentions this week: {reddit_data.get('mention_count', 0)}
- Subreddits: {', '.join(reddit_data.get('subreddits', []))}
- Sentiment score: {reddit_data.get('sentiment_score', 0)} (-100 to +100)
Top posts:
{reddit_posts}

STOCKTWITS:
- Bullish: {stocktwits_data.get('bullish', 0)} | Bearish: {stocktwits_data.get('bearish', 0)}
- Bullish ratio: {stocktwits_data.get('sentiment_ratio', 0)}%
Recent:
{st_msgs}

WEB BUZZ:
{chr(10).join(web_forum_data.get('web_results', [])[:4])}

Analyze:
1. **Retail Sentiment** — crowd bullish/bearish/divided? Unusual activity?
2. **Reddit Activity** — trending? Short squeeze potential? Meme stock?
3. **StockTwits Mood** — community sentiment?
4. **Contrarian Signal** — extreme retail sentiment as contrarian indicator?
5. **Social Score** — rate 1-10 (10 = very bullish crowd)

250 words max.{_pillar_block("SOCIAL", pillar)}"""

    return system, user


def run_social_agent(
    client: OpenAI,
    ticker: str,
    company_name: str,
    reddit_data: dict,
    stocktwits_data: dict,
    web_forum_data: dict,
    pillar: dict | None = None,
    verbose: bool = False,
) -> str:
    system, user = _build_social_prompt(ticker, company_name, reddit_data, stocktwits_data, web_forum_data, pillar)
    return _call(client, system, user)


# ══════════════════════════════════════════════════════════════════════════
# 4. Algo Trading Agent
# ══════════════════════════════════════════════════════════════════════════

def _build_algo_prompt(
    ticker: str,
    current_price: float,
    algo_signals: dict,
    indicators: dict,
    pillar: dict | None = None,
) -> tuple[str, str]:
    mc = {k: v for k, v in algo_signals.items() if "mc_" in k}
    patterns = algo_signals.get("candlestick_patterns", [])

    system = """You are a quantitative analyst at a systematic hedge fund.
Interpret mathematical trading signals, statistical models, and algorithmic
indicators for precise probability-weighted recommendations.
Think in expected value, Sharpe ratios, and statistical edge."""

    user = f"""Interpret quantitative signals for {ticker} at ${current_price:.2f}.

MEAN REVERSION:
- Z-Score (20d): {algo_signals.get('mean_reversion_zscore', 'N/A')}
- Signal: {algo_signals.get('mean_reversion_signal', 'N/A')}
- Note: {algo_signals.get('mean_reversion_note', '')}

MOMENTUM FACTOR:
- 1W: {algo_signals.get('momentum_1w', 'N/A')}% | 1M: {algo_signals.get('momentum_1m', 'N/A')}%
- 3M: {algo_signals.get('momentum_3m', 'N/A')}% | 6M: {algo_signals.get('momentum_6m', 'N/A')}%
- Signal: {algo_signals.get('momentum_signal', 'N/A')}

LINEAR REGRESSION (30d):
- Slope: {algo_signals.get('linreg_slope_pct_per_day', 'N/A')}%/day | R²: {algo_signals.get('linreg_r_squared', 'N/A')}
- Signal: {algo_signals.get('linreg_signal', 'N/A')}

VOLATILITY:
- HV 20d: {algo_signals.get('historical_volatility_20d', 'N/A')}% | HV 60d: {algo_signals.get('historical_volatility_60d', 'N/A')}%
- Regime: {algo_signals.get('vol_regime', 'N/A')} | Expanding: {algo_signals.get('vol_expanding', 'N/A')}

MONTE CARLO (30 days, 1000 paths):
- Bear (5th %ile): ${mc.get('mc_price_5pct', 'N/A')} ({mc.get('mc_downside_pct', 'N/A')}%)
- Median: ${mc.get('mc_price_median', 'N/A')} ({mc.get('mc_expected_pct', 'N/A')}%)
- Bull (95th %ile): ${mc.get('mc_price_95pct', 'N/A')} ({mc.get('mc_upside_pct', 'N/A')}%)

VOLUME-PRICE: {algo_signals.get('volume_price_signal', 'N/A')}
PATTERNS: {', '.join(patterns)}
COMPOSITE ALGO SCORE: {algo_signals.get('algo_score', 50)}/100 ({algo_signals.get('algo_direction', 'NEUTRAL')})

Analyze:
1. **Statistical Edge** — clear quant signal? Statistical significance?
2. **Mean Reversion vs Momentum** — reversion play or trend-following?
3. **Monte Carlo Outlook** — 30-day probability distribution?
4. **Volatility Assessment** — how does vol regime affect strategy?
5. **Optimal Strategy** — mathematically optimal approach?
6. **Algo Score** — rate 1-10

300 words max.{_pillar_block("ALGO", pillar)}"""

    return system, user


def run_algo_agent(
    client: OpenAI,
    ticker: str,
    current_price: float,
    algo_signals: dict,
    indicators: dict,
    pillar: dict | None = None,
    verbose: bool = False,
) -> str:
    system, user = _build_algo_prompt(ticker, current_price, algo_signals, indicators, pillar)
    return _call(client, system, user)


# ══════════════════════════════════════════════════════════════════════════
# 5. Prediction Agent — final BUY / SELL / HOLD
# ══════════════════════════════════════════════════════════════════════════

PREDICTION_SCHEMA = {
    "action":        "BUY | SELL | HOLD",
    "conviction":    "HIGH | MEDIUM | LOW",
    "time_horizon":  "short-term (days-weeks) | medium-term (1-3 months) | long-term (6-12 months)",
    "time_horizon_days": 90,
    "entry_price":   "specific price or range like $150-155",
    "target_price":  "price target",
    "stop_loss":     "stop-loss price",
    "upside_pct":    15.5,
    "downside_pct":  -7.2,
    "risk_reward":   "2.1:1",
    "summary":       "2-3 sentence synthesis",
    "bull_case":     "what needs to happen for bull case",
    "bear_case":     "what could go wrong",
    "key_catalysts": ["catalyst 1", "catalyst 2"],
    "watch_levels":  {"support": "price", "resistance": "price"},
    "scores": {"fundamentals": 6, "technical": 5, "social": 6, "algo": 7, "overall": 6},
}


def _build_prediction_prompt(
    ticker: str,
    company_name: str,
    current_price: float,
    fundamentals_analysis: str,
    technical_analysis: str,
    signal_summary: dict,
    social_analysis: str = "",
    algo_analysis: str = "",
) -> tuple[str, str]:
    system = """You are the head of investment strategy at a hedge fund.
Synthesize fundamentals, technicals, social sentiment, and quant signals
into a precise, actionable investing decision.
Output ONLY valid JSON — no markdown fences, no prose, no explanation."""

    user = f"""Generate an investing decision for {company_name} ({ticker}) at ${current_price:.2f}.

FUNDAMENTAL ANALYSIS:
{fundamentals_analysis}

TECHNICAL ANALYSIS:
{technical_analysis}

SOCIAL SENTIMENT:
{social_analysis}

ALGO/QUANT SIGNALS:
{algo_analysis}

TECHNICAL SIGNAL SCORE: {signal_summary.get('score', 'N/A')}/100 ({signal_summary.get('direction', '')})

Output ONLY valid JSON matching this schema exactly:
{json.dumps(PREDICTION_SCHEMA, indent=2)}

Replace all placeholder strings with actual values. Use real numbers not strings for upside_pct, downside_pct, and scores."""

    return system, user


def run_prediction_agent(
    client: OpenAI,
    ticker: str,
    company_name: str,
    current_price: float,
    fundamentals_analysis: str,
    technical_analysis: str,
    signal_summary: dict,
    social_analysis: str = "",
    algo_analysis: str = "",
    verbose: bool = False,
) -> dict:
    system, user = _build_prediction_prompt(
        ticker, company_name, current_price,
        fundamentals_analysis, technical_analysis,
        signal_summary, social_analysis, algo_analysis,
    )

    raw = ""
    parsed = None
    validation = None
    # Up to 2 attempts total (1 initial + 1 retry) - the same call budget as
    # before, but now a structurally-invalid-JSON response (parses fine, but
    # fails validate_prediction - e.g. action="Maybe") earns the same retry
    # an empty/error response already did, instead of falling straight to
    # the hardcoded fallback below.
    for _attempt in range(2):
        raw = _call(client, system, user)
        if not raw or raw.startswith("[Agent error:"):
            continue
        parsed = _parse_prediction_json(raw)
        if parsed is None:
            continue
        validation = validate_prediction(parsed)
        if validation.valid:
            return parsed
        # structurally invalid but syntactically valid JSON - retry once.

    fallback = {
        "action": "HOLD",
        "conviction": "LOW",
        "time_horizon": "N/A",
        "time_horizon_days": 0,
        "summary": raw[:500] if raw else "Could not generate prediction.",
        "bull_case": "—",
        "bear_case": "—",
        "key_catalysts": [],
        "entry_price": f"${current_price:.2f}",
        "target_price": "N/A",
        "stop_loss": "N/A",
        "upside_pct": 0,
        "downside_pct": 0,
        "risk_reward": "N/A",
        "watch_levels": {},
        "scores": {},
    }
    if parsed is not None and validation is not None and not validation.valid:
        # Diagnosable rather than opaque: the model DID return something
        # JSON-shaped, it just didn't pass validation - keep why.
        fallback["validation_errors"] = validation.errors
    return fallback


def _parse_prediction_json(raw: str) -> dict | None:
    """Extract+parse a JSON object from a raw LLM response. None on failure."""
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        # Try to clean common issues (stray embedded newlines).
        cleaned = m.group().replace('\n', ' ').strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return None
