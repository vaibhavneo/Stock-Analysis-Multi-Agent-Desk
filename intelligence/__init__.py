"""
Evidence-driven decision engine - deterministic Python modules that add
market-regime awareness, multi-horizon historical context, historical-analog
matching, probabilistic forecasting, risk/cost-basis analysis, and a
reliability-weighted, contradiction-aware evidence ledger on top of the
existing 5-agent Stock Agent pipeline.

No LLM calls live in this package. Every module here computes from data the
rest of the app already fetches (or a handful of new keyless fetches - SPY,
QQQ, VIX - through the same financial_data gateway everything else uses),
and degrades honestly (explicit confidence + flags, never a fabricated
number) when data is missing, stale, or insufficient. See intelligence/common.py
for the shared honesty-pattern helpers every module here builds on.
"""
