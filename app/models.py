from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SpeechSegment:
    start: float
    end: float
    source_text: str
    tamil_text: str = ""
    speaker_id: str = "speaker_0"

    def to_dict(self) -> dict:
        return asdict(self)


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[SpeechSegment]: ...


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class VoiceCloner(Protocol):
    def synthesize(
        self, text: str, reference_audio: Path, reference_text: str, output_path: Path
    ) -> None: ...
