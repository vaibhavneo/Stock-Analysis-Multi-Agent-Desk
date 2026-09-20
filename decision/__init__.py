"""
Decision Intelligence — the synthesis layer above the research engine.

This package is a PURE CONSUMER of agents/, backtest/, intelligence/ and data/.
It creates no agents, no strategies, no factors, no databases and no data
providers. Its job is to take what those already compute and answer the one
question the research output does not: *what should be done with it, and under
what conditions would that change?*

Two rules hold everywhere in here:

  1. No module in this package may originate a number. Every quantity it emits
     must be traceable to a value some existing module computed, and the
     provenance field says which one.
  2. The LLM explains; it never decides. decision/narrative.py builds a prompt
     from the finished structured object — the object is complete and correct
     before any prose exists, and prose can never write back into it.
"""
