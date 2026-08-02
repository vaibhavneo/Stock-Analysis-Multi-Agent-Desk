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
echo "Starting Stock Agent AI (gunicorn production server)..."
echo "Open: http://localhost:5051"
echo "Press Ctrl+C to stop"
echo ""
python3 -m gunicorn --bind 0.0.0.0:5051 --workers 2 --threads 4 --timeout 120 web.app:app
