#!/bin/bash
# Run this script from inside your local Stock-Analysis-Multi-Agent-Desk folder
# It applies the Claude Code additions and starts the server at http://127.0.0.1:5051

set -e
cd "$(dirname "$0")"

echo "==> Checking Python..."
python3 --version

echo "==> Starting Claude Code stock desk at http://127.0.0.1:5051 ..."
python3 -m stock_ai_tool.web_server --host 127.0.0.1 --port 5051
