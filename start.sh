#!/bin/bash
cd "$(dirname "$0")"

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  echo ""
  echo "Run this instead:"
  echo "  ANTHROPIC_API_KEY=sk-ant-... python3 web/app.py"
  echo ""
  exit 1
fi

echo ""
echo "Starting Stock Agent AI..."
echo "Open: http://localhost:5051"
echo "Press Ctrl+C to stop"
echo ""
python3 web/app.py
