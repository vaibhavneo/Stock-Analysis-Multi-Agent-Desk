"""Sub-agent adapters. One module per entry in mas/registry.json.

Each module exposes:
    available(symbol, asset_class) -> (bool, reason)
    run(request: AgentRequest)     -> AgentResult      # never raises
"""
