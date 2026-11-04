from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main, store
from app.models import SpeechSegment
from app.pipeline import (
    _build_mix_filter,
    assign_speakers,
    extract_audio,
    load_segments,
    render_dub,
    save_segments,
)


def test_segment_round_trip_keeps_tamil_unicode_and_speaker(tmp_path: Path) -> None:
    source = [SpeechSegment(1.25, 2.75, "こんにちは", "元気？", "speaker_1")]
    path = tmp_path / "segments.json"
    save_segments(path, source)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw[0]["tamil_text"] == "元気？"
    restored = load_segments(path)
    assert restored == source


def test_disabled_diarization_is_single_speaker() -> None:
    segments = [
        SpeechSegment(0.0, 1.0, "おはよう", speaker_id="unknown"),
        SpeechSegment(1.5, 2.5, "ありがとう", speaker_id="unknown"),
    ]
    result = assign_speakers(Path("does-not-need-to-exist.wav"), segments, False)
    assert {item.speaker_id for item in result} == {"speaker_0"}


def test_mix_filter_uses_video_plus_background_then_dub_input() -> None:
    segments = [
        SpeechSegment(0.4, 1.4, "こんにちは", "வணக்கம்", "speaker_0"),
        SpeechSegment(1.6, 2.1, "ありがとう", "நன்றி", "speaker_0"),
    ]
    filters, labels = _build_mix_filter(segments, {0: 1.0, 1: 0.8}, 5.0, False)
    joined = ";".join(filters)
    assert "[1:a]" in joined  # source/background audio
    assert "[2:a]" in joined  # first synthesized clip, video is input zero
    assert "[3:a]" in joined  # second synthesized clip
    assert labels == ["[bg]", "[dub0]", "[dub1]"]


def test_edit_payload_only_changes_tamil_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(main, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(store, "JOBS_DIR", jobs_dir)
    job_id = "ab" * 16
    folder = jobs_dir / job_id
    folder.mkdir(parents=True)
    video_path = folder / "source.mp4"
    video_path.write_bytes(b"video")
    put_value = {
        "id": job_id, "status": "ready_for_review", "job_dir": str(folder),
        "video_path": str(video_path), "progress": 88, "segments": [],
    }
    store.put(job_id, put_value)
    save_segments(folder / "segments.json", [SpeechSegment(0.0, 1.0, "やあ", "வணக்கம்")])
    with TestClient(main.app) as client:
        response = client.put(f"/api/jobs/{job_id}/segments", json={"segments": [{"tamil_text": "ஹலோ!"}]})
    assert response.status_code == 200
    updated = load_segments(folder / "segments.json")
    assert updated[0].source_text == "やあ"
    assert updated[0].start == 0.0
    assert updated[0].tamil_text == "ஹலோ!"


def test_api_requires_voice_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(store, "JOBS_DIR", tmp_path / "jobs")
    with TestClient(main.app) as client:
        response = client.post(
            "/api/jobs",
            data={"voice_source": "video", "voice_consent": "false"},
            files={"video": ("clip.mp4", b"not-a-real-video", "video/mp4")},
        )
    assert response.status_code == 400
    assert "permission" in response.json()["detail"].lower()


def test_api_rejects_video_over_25_minutes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(main, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(store, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(main, "probe_duration", lambda _path: 25 * 60 + 0.5)
    monkeypatch.setattr(main, "_process_transcription", lambda _job_id: None)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/jobs",
            data={"voice_source": "video", "voice_consent": "true"},
            files={"video": ("clip.mp4", b"fake-container-for-validation-test", "video/mp4")},
        )
    assert response.status_code == 413
    assert "25 minutes" in response.json()["detail"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg not installed")
def test_real_ffmpeg_video_audio_render(tmp_path: Path) -> None:
    """Run the actual mux/filter path with real FFmpeg and a deterministic test voice."""
    video = tmp_path / "source.mp4"
    audio = tmp_path / "source.wav"
    output = tmp_path / "dubbed.mp4"
    work = tmp_path / "work"
    work.mkdir()
    create = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=24:d=2.5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2.5",
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(video),
        ],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        pytest.skip(f"Local FFmpeg build lacks the test encoders: {create.stderr[-500:]}")
    extract_audio(video, audio)

    class TestVoice:
        def synthesize(self, text, reference_audio, reference_text, output_path):
            sr = 24000
            t = np.arange(int(sr * 0.9), dtype=np.float32) / sr
            wave = (0.06 * np.sin(2 * np.pi * 235 * t)).astype(np.float32)
            sf.write(output_path, wave, sr)

    segment = SpeechSegment(0.4, 1.4, "こんにちは", "வணக்கம்", "speaker_0")
    fake_reference = tmp_path / "reference.wav"
    sf.write(fake_reference, np.zeros(24000, dtype=np.float32), 24000)
    render_dub(
        video,
        audio,
        [segment],
        {"speaker_0": (fake_reference, "வணக்கம்")},
        TestVoice(),
        work,
        output,
        preserve_background=False,
        progress=lambda _pct, _message: None,
    )
    assert output.is_file() and output.stat().st_size > 1000
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    streams = set(probe.stdout.split())
    assert "video" in streams
    assert "audio" in streams


def test_api_accepts_under_limit_video_and_queues_real_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(main, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(store, "JOBS_DIR", jobs_dir)
    monkeypatch.setattr(main, "probe_duration", lambda _path: 12.0)

    def fake_worker(job_id: str) -> None:
        store.update(job_id, status="ready_for_review", progress=88, message="Test worker finished")

    monkeypatch.setattr(main, "_process_transcription", fake_worker)
    with TestClient(main.app) as client:
        created = client.post(
            "/api/jobs",
            data={"voice_source": "video", "voice_consent": "true", "translator": "auto"},
            files={"video": ("short.mp4", b"valid-looking-upload-bytes", "video/mp4")},
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        current = client.get(f"/api/jobs/{job_id}")
    assert current.status_code == 200
    assert current.json()["status"] == "ready_for_review"
    assert current.json()["duration_seconds"] == 12.0
    assert "video_path" not in current.json()
