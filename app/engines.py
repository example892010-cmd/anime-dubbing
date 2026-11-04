from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .config import (
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_MODEL,
    HF_TOKEN,
    OLLAMA_MODEL,
    OLLAMA_URL,
    TRANSLATION_MODEL,
    TTS_MODEL,
)
from .models import SpeechSegment


class FasterWhisperTranscriber:
    """Japanese ASR with real Whisper weights; no fixture/demo mode is used."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        device = ASR_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute_type = ASR_COMPUTE_TYPE
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        self.model = WhisperModel(ASR_MODEL, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, progress=None) -> list[SpeechSegment]:
        segments, _info = self.model.transcribe(
            str(audio_path),
            language="ja",
            task="transcribe",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 450},
            beam_size=5,
            condition_on_previous_text=True,
        )
        result: list[SpeechSegment] = []
        duration = max(float(getattr(_info, "duration", 0.0) or 0.0), 1.0)
        last_reported = 0
        for segment in segments:
            text = re.sub(r"\s+", " ", segment.text).strip()
            if text and float(segment.end) > float(segment.start):
                result.append(SpeechSegment(float(segment.start), float(segment.end), text))
            if progress is not None:
                # Report actual audio position, not a fabricated time-based animation.
                current = min(40, 15 + int(25 * min(float(segment.end), duration) / duration))
                if current > last_reported:
                    last_reported = current
                    progress(current, f"Transcribing Japanese dialogue… {min(100, int(100 * min(float(segment.end), duration) / duration))}% of audio scanned")
        if not result:
            raise RuntimeError("Whisper could not find any Japanese speech in this video.")
        return result


class OllamaTamilTranslator:
    """Local LLM translation; prompt is tuned for spoken, casual Tamil."""

    def __init__(self) -> None:
        import requests
        self.requests = requests
        self.url = OLLAMA_URL
        self.model = OLLAMA_MODEL
        self.timeout = (4, 180)

    def is_available(self) -> bool:
        try:
            response = self.requests.get(f"{self.url}/api/tags", timeout=(2, 4))
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
            return self.model in names or any(
                name.split(":", 1)[0] == self.model.split(":", 1)[0] for name in names
            )
        except Exception:
            return False

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        prompt = (
            "Translate the Japanese dialogue into natural, casual spoken Tamil written in Tamil script. "
            "Use everyday conversational Tamil (neutral contemporary Tamil, not stiff literary Tamil). "
            "Keep the character's meaning, emotion, names, and level of politeness. "
            "Make it sound like a real person speaking in a dubbed film. "
            "Do not explain, add facts, transliterate Japanese, or include quotation marks. "
            "Return only the Tamil dialogue.\n\nJapanese dialogue:\n" + text
        )
        response = self.requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are an expert Japanese-to-Tamil dubbing adapter."},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.25},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        translated = response.json().get("message", {}).get("content", "").strip()
        translated = re.sub(r"^```[^\n]*\n|\n```$", "", translated).strip().strip('"“”')
        if not translated:
            raise RuntimeError("Ollama returned an empty Tamil translation.")
        return translated


class NllbJapaneseTamilTranslator:
    """Offline fallback. NLLB supports Japanese (jpn_Jpan) and Tamil (tam_Taml)."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL, src_lang="jpn_Jpan", token=HF_TOKEN or None
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL, token=HF_TOKEN or None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.target_id = self.tokenizer.convert_tokens_to_ids("tam_Taml")
        if self.target_id is None or self.target_id < 0:
            raise RuntimeError("The NLLB tokenizer does not expose the Tamil target token tam_Taml.")

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                forced_bos_token_id=self.target_id,
                max_new_tokens=min(220, max(48, len(text) * 4)),
                num_beams=4,
                length_penalty=1.0,
            )
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not result:
            raise RuntimeError("NLLB returned an empty Tamil translation.")
        return result


def make_translator(mode: str):
    mode = (mode or "auto").lower()
    if mode not in {"auto", "ollama", "nllb"}:
        raise ValueError("translator must be auto, ollama, or nllb")
    if mode in {"auto", "ollama"}:
        ollama = OllamaTamilTranslator()
        if ollama.is_available():
            return ollama, "ollama"
        if mode == "ollama":
            raise RuntimeError(
                f"Ollama model '{OLLAMA_MODEL}' is not available. Install Ollama and run: ollama pull {OLLAMA_MODEL}"
            )
    return NllbJapaneseTamilTranslator(), "nllb"


class IndicF5VoiceCloner:
    """Actual IndicF5 zero-shot voice cloning for Tamil speech."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(
            TTS_MODEL, trust_remote_code=True, token=HF_TOKEN or None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def synthesize(
        self, text: str, reference_audio: Path, reference_text: str, output_path: Path
    ) -> None:
        audio: Any = self.model(
            text,
            ref_audio_path=str(reference_audio),
            ref_text=reference_text,
        )
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        waveform = np.asarray(audio, dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError("IndicF5 returned an empty or invalid audio waveform.")
        if not np.isfinite(waveform).all():
            raise RuntimeError("IndicF5 returned NaN or infinite audio samples.")
        peak = float(np.max(np.abs(waveform)))
        if peak > 0.99:
            waveform = waveform / peak * 0.95
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, 24000, subtype="PCM_16")


@lru_cache(maxsize=1)
def get_transcriber() -> FasterWhisperTranscriber:
    return FasterWhisperTranscriber()


@lru_cache(maxsize=1)
def get_voice_cloner() -> IndicF5VoiceCloner:
    return IndicF5VoiceCloner()


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("Could not determine the media duration. Is FFmpeg installed?") from exc
    if duration <= 0:
        raise RuntimeError("The uploaded media has no usable duration.")
    return duration
