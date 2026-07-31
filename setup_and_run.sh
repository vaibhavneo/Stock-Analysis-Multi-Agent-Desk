#!/bin/bash
# Run this script from inside your local Stock-Analysis-Multi-Agent-Desk folder
# It starts the multiagent stock desk at http://127.0.0.1:8765
#
# Note: this is a DIFFERENT app from ~/Desktop/Agentic AI/stock_agent (the
# 7-agent "Stock Agent AI" app), which runs on port 5051. Keep them on
# separate ports so they don't collide.

set -e
cd "$(dirname "$0")"

echo "==> Checking Python..."
python3 --version

echo "==> Starting the Multiagent Stock Desk at http://127.0.0.1:8765 ..."
python3 -m stock_ai_tool.web_server --host 127.0.0.1 --port 8765
