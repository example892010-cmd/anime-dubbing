#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 is required. Install it, then run: PYTHON_BIN=python3.10 ./install.sh" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg and ffprobe are required. Install FFmpeg and add both commands to PATH." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
# Install PyTorch separately so users can choose the matching CPU/CUDA wheel.
python -m pip install torch torchaudio
python -m pip install git+https://github.com/ai4bharat/IndicF5.git
cp -n .env.example .env || true

echo
echo "Install finished. Add HF_TOKEN to .env after accepting the IndicF5 model terms."
echo "Run: source .venv/bin/activate && python -m uvicorn app.main:app --host 127.0.0.1 --port 7860"
