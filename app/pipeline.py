from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import HF_TOKEN
from .engines import probe_duration
from .models import SpeechSegment

Progress = Callable[[int, str], None]


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "No FFmpeg error details available")[-3000:]
        raise RuntimeError(f"FFmpeg failed: {detail}")


def extract_audio(video_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", str(output_path),
    ])


def assign_speakers(audio_path: Path, segments: list[SpeechSegment], enabled: bool) -> list[SpeechSegment]:
    """Optional local pyannote diarization; without it, the project is single-speaker mode."""
    if not enabled:
        for segment in segments:
            segment.speaker_id = "speaker_0"
        return segments
    if not HF_TOKEN:
        raise RuntimeError(
            "Multi-speaker detection needs HF_TOKEN and access to pyannote/speaker-diarization-community-1. "
            "Add your Hugging Face token to .env, accept that model's conditions, or turn off multi-speaker mode."
        )
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Multi-speaker mode is an optional extra. Install it with: pip install -r requirements-diarization.txt"
        ) from exc
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    diarized = pipeline(str(audio_path))
    turns = list(diarized.speaker_diarization)
    for segment in segments:
        best_speaker = "speaker_0"
        best_overlap = 0.0
        for turn, speaker in turns:
            overlap = max(0.0, min(segment.end, float(turn.end)) - max(segment.start, float(turn.start)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(speaker)
        segment.speaker_id = best_speaker
    return segments


def _write_reference_from_segments(
    audio_path: Path,
    candidates: list[SpeechSegment],
    work_dir: Path,
    speaker_id: str,
) -> tuple[Path, str]:
    # Pick several clean utterances from the same speaker and concatenate them, avoiding
    # snippets from other speakers. IndicF5 expects both reference audio and its transcript.
    ordered = sorted(
        (s for s in candidates if s.speaker_id == speaker_id and s.source_text.strip() and s.end > s.start),
        key=lambda s: s.start,
    )
    if not ordered:
        raise RuntimeError(f"No speech was detected for {speaker_id} to use as a voice reference.")

    selected: list[SpeechSegment] = []
    if speaker_id != "speaker_0":
        # pyannote has identified these turns as the same speaker, so short lines across
        # the video can be concatenated without pulling in other characters.
        candidate: list[SpeechSegment] = []
        total = 0.0
        for item in sorted(ordered, key=lambda s: s.end - s.start, reverse=True):
            spoken = item.end - item.start
            if len(candidate) >= 8 or spoken > 18.0 or total + spoken > 18.0:
                continue
            candidate.append(item)
            total += spoken
            if total >= 7.0:
                break
        if total >= 4.0:
            selected = candidate
    else:
        # Without diarization, only stitch a contiguous speech bout: don't accidentally
        # concatenate samples from different characters into one cloned voice.
        groups: list[list[SpeechSegment]] = []
        for item in ordered:
            if not groups or item.start - groups[-1][-1].end > 1.0:
                groups.append([item])
            else:
                groups[-1].append(item)
        for group in sorted(groups, key=lambda items: sum(s.end - s.start for s in items), reverse=True):
            candidate = []
            total = 0.0
            for item in group:
                spoken = item.end - item.start
                if spoken > 18.0 or total + spoken > 18.0:
                    break
                candidate.append(item)
                total += spoken
            if total >= 4.0:
                selected = candidate
                break
    if not selected:
        raise RuntimeError(
            f"Not enough continuous clean speech was found for {speaker_id}. Upload a 4-30 second voice reference with its exact transcript."
        )
    selected.sort(key=lambda s: s.start)
    ref_path = work_dir / f"reference_{speaker_id}.wav"
    filters: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(selected):
        label = f"r{index}"
        filters.append(
            f"[0:a]atrim=start={max(0.0, item.start):.3f}:end={item.end:.3f},"
            f"asetpts=PTS-STARTPTS[{label}]"
        )
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}aresample=24000,atrim=0:18[outa]")
    else:
        filters.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,aresample=24000,atrim=0:18[outa]")
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(audio_path), "-filter_complex", ";".join(filters),
        "-map", "[outa]", "-ac", "1", "-ar", "24000", str(ref_path),
    ])
    transcript = " ".join(item.source_text.strip() for item in selected)
    return ref_path, transcript


def ensure_references(
    audio_path: Path,
    segments: list[SpeechSegment],
    work_dir: Path,
    uploaded_reference: Path | None,
    provided_reference_text: str,
) -> dict[str, tuple[Path, str]]:
    speakers = sorted({segment.speaker_id for segment in segments}) or ["speaker_0"]
    if uploaded_reference:
        if not provided_reference_text.strip():
            raise ValueError("Enter the exact transcript spoken in your uploaded voice reference.")
        ref_out = work_dir / "voice_reference.wav"
        reference_duration = probe_duration(uploaded_reference)
        if not 4.0 <= reference_duration <= 30.0:
            raise ValueError("Voice reference must be between 4 and 30 seconds long.")
        run_ffmpeg([
            "ffmpeg", "-y", "-i", str(uploaded_reference), "-vn", "-ac", "1",
            "-ar", "24000", str(ref_out),
        ])
        return {speaker: (ref_out, provided_reference_text.strip()) for speaker in speakers}

    return {
        speaker: _write_reference_from_segments(audio_path, segments, work_dir, speaker)
        for speaker in speakers
    }


def separate_background(audio_path: Path, work_dir: Path) -> Path | None:
    """Optional Demucs speech/vocal removal. The default fallback ducks source audio."""
    executable = shutil.which("demucs")
    if not executable:
        return None
    out_dir = work_dir / "demucs"
    command = [executable, "--two-stems=vocals", "-n", "htdemucs", "-o", str(out_dir), str(audio_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    candidates = list(out_dir.glob("**/no_vocals.wav"))
    return candidates[0] if candidates else None


def _build_mix_filter(
    segments: list[SpeechSegment],
    clip_durations: dict[int, float],
    video_duration: float,
    background_is_separated: bool,
) -> tuple[list[str], list[str]]:
    filters: list[str] = []
    labels: list[str] = []
    windows: list[tuple[float, float]] = []
    planned: list[tuple[int, SpeechSegment, float, float]] = []

    # Determine each dub's available slot before making the background ducking envelope.
    for segment_index, segment in enumerate(segments):
        if not segment.tamil_text.strip() or segment_index not in clip_durations:
            continue
        later_starts = [item.start for item in segments[segment_index + 1:] if item.start > segment.start]
        slot_end = min(later_starts) if later_starts else video_duration
        slot = max(0.65, min(30.0, slot_end - segment.start))
        clip_duration = clip_durations[segment_index]
        tempo = max(0.80, min(1.90, clip_duration / slot))
        audible_duration = min(slot, clip_duration / tempo)
        windows.append((max(0.0, segment.start), min(video_duration, segment.start + audible_duration)))
        planned.append((segment_index, segment, slot, tempo))

    if background_is_separated:
        filters.append("[1:a]aresample=48000,volume=0.62[bg]")
    else:
        # Keep ambience/music where possible and duck the original Japanese dialogue while
        # Tamil speech is audible. Without Demucs, a little source voice may remain.
        expression = "0.62"
        if windows:
            duck = "+".join(f"between(t,{start:.3f},{end:.3f})" for start, end in windows)
            expression = f"if(gt({duck},0),0.09,0.62)"
        filters.append(f"[1:a]aresample=48000,volume='{expression}':eval=frame[bg]")
    labels.append("[bg]")

    for clip_number, (segment_index, segment, slot, tempo) in enumerate(planned):
        input_index = clip_number + 2  # input 0 = video; input 1 = background/source audio
        label = f"dub{clip_number}"
        delay_ms = max(0, int(segment.start * 1000))
        filters.append(
            f"[{input_index}:a]aresample=48000,atempo={tempo:.4f},atrim=0:{slot:.3f},"
            f"adelay={delay_ms}|{delay_ms},volume=1.35[{label}]"
        )
        labels.append(f"[{label}]")
    return filters, labels


def render_dub(
    video_path: Path,
    source_audio: Path,
    segments: list[SpeechSegment],
    reference_map: dict[str, tuple[Path, str]],
    cloner,
    work_dir: Path,
    output_path: Path,
    preserve_background: bool,
    progress: Progress,
) -> None:
    duration = probe_duration(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip_paths: dict[int, Path] = {}
    clip_durations: dict[int, float] = {}
    work_dir.joinpath("tts").mkdir(parents=True, exist_ok=True)
    for index, segment in enumerate(segments):
        tamil = segment.tamil_text.strip()
        if not tamil:
            continue
        if segment.speaker_id not in reference_map:
            raise RuntimeError(f"There is no voice reference configured for {segment.speaker_id}.")
        ref_audio, ref_text = reference_map[segment.speaker_id]
        clip_path = work_dir / "tts" / f"line_{index:04d}.wav"
        cloner.synthesize(tamil, ref_audio, ref_text, clip_path)
        clip_paths[index] = clip_path
        clip_durations[index] = probe_duration(clip_path)
        progress(25 + int(50 * len(clip_paths) / max(1, sum(bool(s.tamil_text.strip()) for s in segments))),
                 f"Generating Tamil voice ({len(clip_paths)}/{sum(bool(s.tamil_text.strip()) for s in segments)})")

    if not clip_paths:
        raise RuntimeError("There are no Tamil dialogue lines to render.")

    background_path = separate_background(source_audio, work_dir) if preserve_background else None
    selected_background = background_path or source_audio
    filters, audio_labels = _build_mix_filter(segments, clip_durations, duration, background_path is not None)

    # Use the original video as input zero so its video stream is explicitly mapped.
    command = [
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(selected_background),
    ]
    for index in sorted(clip_paths):
        command.extend(["-i", str(clip_paths[index])])
    filter_complex = ";".join(filters)
    filter_complex += (
        f";{''.join(audio_labels)}amix=inputs={len(audio_labels)}:duration=longest:"
        f"dropout_transition=0:normalize=0,alimiter=limit=0.96,"
        f"atrim=0:{duration:.3f},apad=whole_dur={duration:.3f}[mix]"
    )
    filter_script = work_dir / "audio_mix_filter.txt"
    filter_script.write_text(filter_complex, encoding="utf-8")
    command.extend([
        "-filter_complex_script", str(filter_script),
        "-map", "0:v:0", "-map", "[mix]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output_path),
    ])
    run_ffmpeg(command)
    progress(100, "Tamil dubbed video is ready")


def save_segments(path: Path, segments: list[SpeechSegment]) -> None:
    path.write_text(
        json.dumps([segment.to_dict() for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_segments(path: Path) -> list[SpeechSegment]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [SpeechSegment(**item) for item in raw]
