"""Adapter module for the `track_record` sub-agent. See desk_capabilities.py."""
from .desk_capabilities import TrackRecord as _A

AGENT_ID = _A.AGENT_ID
available = _A.available
run = _A.run
