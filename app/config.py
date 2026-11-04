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
def _resolve_asr_settings(env: dict[str, str]) -> tuple[str, str, str, int, int]:
    """Choose a safe ASR profile for Render's small CPU instances.

    Render always sets RENDER=true. The general ASR_MODEL variable may have been
    set to large-v3 from a desktop setup; do not let that silently exhaust a
    small hosted instance. RENDER_ASR_MODEL is the explicit hosted override.
    """
    on_render = env.get("RENDER", "").strip().lower() == "true"
    if on_render:
        model = env.get("RENDER_ASR_MODEL", "tiny").strip() or "tiny"
        device = env.get("RENDER_ASR_DEVICE", "cpu").strip() or "cpu"
        compute_type = env.get("RENDER_ASR_COMPUTE_TYPE", "int8").strip() or "int8"
        try:
            cpu_count = float(env.get("RENDER_CPU_COUNT", "1"))
        except (TypeError, ValueError):
            cpu_count = 1.0
        cpu_threads = max(1, min(2, int(cpu_count)))
        beam_size = 1
        return model, device, compute_type, cpu_threads, beam_size

    model = env.get("ASR_MODEL", "small").strip() or "small"
    device = env.get("ASR_DEVICE", "auto").strip() or "auto"
    compute_type = env.get("ASR_COMPUTE_TYPE", "int8").strip() or "int8"
    try:
        cpu_threads = max(1, int(env.get("ASR_CPU_THREADS", "4")))
    except (TypeError, ValueError):
        cpu_threads = 4
    try:
        beam_size = max(1, int(env.get("ASR_BEAM_SIZE", "5")))
    except (TypeError, ValueError):
        beam_size = 5
    return model, device, compute_type, cpu_threads, beam_size


IS_RENDER = os.getenv("RENDER", "").strip().lower() == "true"
ASR_MODEL, ASR_DEVICE, ASR_COMPUTE_TYPE, ASR_CPU_THREADS, ASR_BEAM_SIZE = _resolve_asr_settings(os.environ)
ASR_CACHE_DIR = Path(os.getenv("WHISPER_CACHE_DIR", DATA_DIR / "models" / "faster-whisper")).expanduser()
ASR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
TTS_MODEL = os.getenv("TTS_MODEL", "ai4bharat/IndicF5")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

for directory in (DATA_DIR, UPLOAD_DIR, JOBS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
