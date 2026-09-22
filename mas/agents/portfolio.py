"""Adapter module for the `portfolio` sub-agent. See desk_capabilities.py."""
from .desk_capabilities import Portfolio as _A

AGENT_ID = _A.AGENT_ID
available = _A.available
run = _A.run
