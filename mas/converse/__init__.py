"""The conversational front door to the multi-agent desk.

    corpus.py    held-out questions, frozen BEFORE the router existed
    symbols.py   getting the symbol out of a sentence without inventing one
    intent.py    utterance -> which specialists to ask (deterministic)
    session.py   what carries between turns (the subject; never the verdict)
    reply.py     specialist results -> something a person would say back
    engine.py    one turn, end to end
"""
from .engine import turn

__all__ = ["turn"]
