# AI Stock Analysis and Trading Tool (MVP)

This is a local-first MVP for stock analysis and paper trading.

It currently supports:
- Loading OHLCV CSV data (`symbol,date,open,high,low,close,volume`)
- Computing basic technical indicators (SMA, momentum, RSI, volatility)
- Generating AI-style trade signals (`BUY`, `HOLD`, `SELL`) via a transparent scoring model
- Executing simulated paper trades with a broker abstraction
- Running a multi-agent pipeline for AI + quantum stock selection with 5 actions:
  - sector analysis
  - fundamental analysis
  - technical analysis
  - stock recommendation
  - call/put option recommendation

## Quick Start

Run a market scan:

```bash
python3 -m stock_ai_tool.cli scan sample_data/prices.csv
```

Run a paper-trading cycle:

```bash
python3 -m stock_ai_tool.cli trade sample_data/prices.csv --capital 10000
```

Filter to specific symbols:

```bash
python3 -m stock_ai_tool.cli trade sample_data/prices.csv --symbols ALFA,GAMA --capital 5000
```

Run the multi-agent AI + quantum workflow:

```bash
python3 -m stock_ai_tool.cli multiagent \
  sample_data/thematic_prices.csv \
  sample_data/thematic_profiles.csv \
  sample_data/thematic_fundamentals.csv \
  --top 5
```

Launch the browser UI:

```bash
python3 -m stock_ai_tool.web_server --port 5052
```

Then open `http://127.0.0.1:5052`.

Run the full AI + quantum universe:

```bash
python3 -m stock_ai_tool.cli multiagent \
  sample_data/thematic_prices.csv \
  sample_data/thematic_profiles.csv \
  sample_data/thematic_fundamentals.csv \
  --top 40
```

Local analysis APIs:

```bash
GET /api/analyze?symbols=NVDA,IONQ,RGTI&top=20
GET /api/deep-analysis?symbol=NVDA
GET /api/benchmarks
GET /api/universe
```

The browser trade desk now supports three data-feed modes:
- `Sample`: local CSV plus generated history for a stable demo universe.
- `Yahoo Live`: backend attempts Yahoo Finance/Stooq live history and falls back to sample data.
- `Alpha Vantage`: browser fetches `OVERVIEW` and `TIME_SERIES_DAILY_ADJUSTED` directly using the API key entered in the desk.

## Output

- `scan` ranks symbols by score and shows signal + confidence.
- `trade` applies the strategy and executes paper orders on the latest close.
- `multiagent` returns all 5 agent outputs and top picks grouped by `AI` and `QUANTUM`.
- `web_server` serves a local dashboard backed by the same multi-agent pipeline.
- `deep-analysis` explains valuation, sales scale, margins, industry benchmarks, technical setup, recommendation rationale, option bias, and risk flags.

## Notes

- This is not connected to live broker APIs yet.
- No real money is traded.
- Use this as a foundation for adding live data feeds, LLM summarization, and broker integrations.
