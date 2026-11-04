#!/usr/bin/env bash
set -euo pipefail
if [[ ! -d .venv ]]; then echo "Run ./install.sh first." >&2; exit 1; fi
source .venv/bin/activate
exec python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
