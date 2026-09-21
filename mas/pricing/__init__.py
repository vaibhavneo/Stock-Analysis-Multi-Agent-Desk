"""The local derivatives engine.

OptionsPilot prices from REAL CHAINS over its own 31-name equity universe.
That is strictly better information and this package never competes with it:
when OptionsPilot can answer, it answers.

This exists for everything else — the other ~thousands of equities, and every
crypto, FX, index and commodity underlying, none of which OptionsPilot covers.
It prices from a MODEL (Black-Scholes / Cox-Ross-Rubinstein) driven by
realized volatility, which is a categorically weaker claim than a quoted
market, and every number it emits is labelled as such. The two must never
appear in one undifferentiated list — `decision/levels.py` already draws
exactly this line between TRADED_PRICE and MODEL, and this package reuses it.
"""
from .black_scholes import (
    price, greeks, implied_volatility, BSResult, MODEL_BASIS,
)
from .structures import build_structure, STRUCTURES
from .payoff import evaluate

__all__ = ["price", "greeks", "implied_volatility", "BSResult", "MODEL_BASIS",
           "build_structure", "STRUCTURES", "evaluate"]
