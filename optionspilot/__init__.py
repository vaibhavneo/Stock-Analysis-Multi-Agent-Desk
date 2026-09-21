"""
The OptionsPilot link — a DEPENDENCY, not a component.

OptionsPilot is a separate deployed service with its own release cycle, its own
research freeze (V6.0.0) and its own journal. Importing its code would fork it;
calling it over HTTP keeps one copy of the options logic. This mirrors the
contract `desk/client.py` uses in the other direction, where OptionsPilot reads
this desk — deliberately, so the two halves of the link behave the same way.
"""
