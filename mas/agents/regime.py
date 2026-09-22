"""Adapter module for the `regime` sub-agent. See desk_capabilities.py."""
from .desk_capabilities import Regime as _A

AGENT_ID = _A.AGENT_ID
available = _A.available
run = _A.run
