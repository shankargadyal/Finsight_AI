#!/bin/bash
echo ""
echo "========================================="
echo "  FinSight AI v2 — Starting..."
echo "========================================="

[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -q

[ ! -f .env ] && [ -f .env.example ] && cp .env.example .env && echo "Created .env from .env.example"

echo ""
echo "  http://localhost:5001"
echo "  Roadmap: http://localhost:5001/future"
echo ""
python3 app.py
