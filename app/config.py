from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Load the repository's .env before reading environment-backed settings.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

DATA_DIR = Path(os.getenv("DUB_DATA_DIR", ROOT / "data")).expanduser().resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
MAX_VIDEO_MINUTES = int(os.getenv("MAX_VIDEO_MINUTES", "25"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_REFERENCE_MB = int(os.getenv("MAX_REFERENCE_MB", "100"))
ASR_MODEL = os.getenv("ASR_MODEL", "small")
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
TTS_MODEL = os.getenv("TTS_MODEL", "ai4bharat/IndicF5")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

for directory in (DATA_DIR, UPLOAD_DIR, JOBS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
