# NammaDub — Japanese → Tamil video dubbing

A **working, self-hosted web application** for dubbing Japanese videos into conversational Tamil. It is not a mock UI: the server runs actual ASR, translation, reference-conditioned Tamil speech generation, and FFmpeg video/audio rendering.

- **25-minute maximum** per source video (configurable in `.env`).
- Japanese speech recognition: `faster-whisper`.
- Casual Tamil translation: local Ollama (`qwen2.5:7b`) when available; offline NLLB-200 fallback.
- Tamil zero-shot voice cloning: `ai4bharat/IndicF5`.
- Speaker separation: optional local `pyannote/speaker-diarization-community-1`.
- Audio/video muxing: FFmpeg. Demucs background/vocal separation is optional.
- Review and edit each Tamil line before generating the final MP4.

## Important reality check

This app is local. There is no per-minute API charge and the app enforces a 25-minute source limit, but it is **not a hosted free cloud service**. First use downloads large AI models and a GPU is strongly recommended. CPU inference may be slow.

IndicF5 is gated on Hugging Face. You must sign in, accept the model's access conditions, and set a Hugging Face token in `.env`. The model card specifically requires explicit permission to clone a voice. IndicF5 supports Tamil and uses a reference audio clip **plus the exact words spoken in that clip**. An uploaded Tamil reference normally gives better Tamil pronunciation; extracting a Japanese actor's sample is supported experimentally, but cross-language voice similarity can vary.

The fallback `facebook/nllb-200-distilled-600M` has non-commercial/research-oriented terms. Use the Ollama translator or another appropriately licensed translator for commercial workflows, and review each model's terms yourself. Model terms are separate from this repository's code license.

## 1. Requirements

- Windows 10/11, macOS, or Linux.
- Python **3.10** (recommended for the IndicF5 package).
- FFmpeg + ffprobe on `PATH`.
- NVIDIA GPU + compatible CUDA/PyTorch is strongly recommended for reasonable speed. CPU can work but will be slow, especially on long videos.
- Internet access for first-time model downloads.
- A Hugging Face account/token with access to `ai4bharat/IndicF5`.

## 2. Install

### Windows

1. Install Python 3.10 and FFmpeg; make sure `ffmpeg` and `ffprobe` both work in PowerShell.
2. Open PowerShell in this repository folder.
3. Run:

   ```powershell
   .\install.ps1
   ```

4. Open `.env` and set `HF_TOKEN=hf_...` after accepting the IndicF5 model conditions on Hugging Face.
5. Run:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 7860
   ```

### Linux / macOS

1. Install Python 3.10, FFmpeg and ffprobe.
2. Run:

   ```bash
   chmod +x install.sh run.sh
   ./install.sh
   ```
3. Add `HF_TOKEN=hf_...` to `.env` after accepting the IndicF5 conditions.
4. Start the site:

   ```bash
   ./run.sh
   ```

Open **http://127.0.0.1:7860** in your browser.

### PyTorch GPU note

`install.sh`/`install.ps1` installs the default PyTorch wheel. If you need a particular CUDA build, use the official selector at <https://pytorch.org/get-started/locally/> and install the matching `torch` and `torchaudio` pair in `.venv` before running IndicF5.

## 3. Improve casual Tamil translation (recommended)

Install Ollama from <https://ollama.com>, then run:

```bash
ollama pull qwen2.5:7b
```

Keep Ollama running. In NammaDub, choose **Auto** or **Ollama**. The prompt requests contemporary spoken Tamil, not literary Tamil. You can edit every translated line before TTS.

If Ollama is not running, **Auto** falls back to NLLB-200. NLLB may produce more formal wording and uses the model's own license terms.

## 4. Using the website

1. Upload an MP4/MKV/MOV/WebM/AVI/M4V video no longer than 25 minutes.
2. Choose the voice reference source:
   - **Extract from this video:** NammaDub takes several recognized utterances and their Japanese transcript as the reference. Cross-language matching is experimental. The selected speaker needs at least four seconds of detected speech.
   - **Upload a voice sample:** provide a clean 4–30 second recording and type the exact transcript. A Tamil reference is recommended for natural Tamil pronunciation.
3. Select Auto, Ollama, or NLLB translation.
4. For multiple characters, enable speaker detection. Install `requirements-diarization.txt`, accept the `pyannote/speaker-diarization-community-1` model conditions, and set `HF_TOKEN`. With diarization and **Extract from this video**, each detected speaker gets their own sample. An uploaded voice sample is intentionally reused for all detected speakers. Without diarization, all lines are cloned from one source reference.
5. Confirm voice permission, then click **Transcribe & translate**.
6. Review/edit the Tamil dialogue, then click **Save edits & generate dubbed MP4**.
7. Download the rendered MP4.

## 5. Optional add-ons

### Multi-speaker diarization

Activate `.venv`, then install:

```bash
pip install -r requirements-diarization.txt
```

Accept the access conditions for <https://huggingface.co/pyannote/speaker-diarization-community-1> and set `HF_TOKEN` in `.env`. This runs locally. Diarization can still make mistakes on overlapping or noisy voices.

### Background music/effects separation

Activate `.venv`, then install:

```bash
pip install -r requirements-demucs.txt
```

With Demucs installed and the background option enabled, NammaDub tries to create a `no_vocals` stem. If Demucs is not installed or fails, the fallback keeps the source track quieter and ducks it during spoken lines; faint Japanese speech may remain and music/effects may be reduced along with it.

## 6. Configuration

Copy `.env.example` to `.env` and edit values as needed:

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | empty | Hugging Face token for gated models / diarization |
| `MAX_VIDEO_MINUTES` | `25` | Source video duration cap |
| `MAX_UPLOAD_MB` | `2048` | Video upload size cap |
| `ASR_MODEL` | `small` | faster-whisper model (`small`, `medium`, `large-v3`) |
| `ASR_DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `ASR_COMPUTE_TYPE` | `int8` | faster-whisper compute type |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local translation model |
| `TRANSLATION_MODEL` | `facebook/nllb-200-distilled-600M` | Offline translation model |
| `TTS_MODEL` | `ai4bharat/IndicF5` | Tamil voice-cloning model |
| `DUB_DATA_DIR` | `./data` | Uploads and job outputs location |

## 7. Checks and tests

Run the environment checks:

```bash
.venv/bin/python scripts/preflight.py        # Windows: .venv\Scripts\python.exe scripts\preflight.py
```

Run the automated test suite:

```bash
.venv/bin/python -m pytest                       # Windows: .venv\Scripts\python.exe -m pytest
```

The test suite covers API validation, segment persistence, speaker fallback, FFmpeg filter construction, and a real short MP4 mux/render using a deterministic test waveform. It does **not** download or benchmark the production Whisper/NLLB/IndicF5 weights; run a short real clip after setup to validate your GPU, model access and voice sample.

## Security and privacy

- The web server binds to `127.0.0.1` by default. Do not expose it publicly without adding authentication and transport security.
- Uploaded videos, voice references and outputs are stored under `data/jobs/` and are not automatically deleted. Remove job folders when finished.
- The voice-consent confirmation is mandatory. Clone only voices you own or have explicit permission to use, and only process videos you are authorized to use.
- Do not use the app to impersonate someone or deceive viewers.

## Project layout

```text
namma-dub/
├── app/
│   ├── main.py          # FastAPI routes and background jobs
│   ├── engines.py       # Whisper, Ollama/NLLB, IndicF5 adapters
│   ├── pipeline.py      # extraction, diarization, reference prep, FFmpeg render
│   ├── models.py        # speech segment contracts
│   ├── store.py         # local job metadata
│   └── static/index.html
├── scripts/preflight.py
├── tests/
├── requirements*.txt
├── install.ps1 / install.sh
└── run.sh
```

## References

- IndicF5 model and reference-audio API: <https://github.com/AI4Bharat/IndicF5>
- faster-whisper Japanese ASR: <https://github.com/SYSTRAN/faster-whisper>
- NLLB translation documentation: <https://huggingface.co/docs/transformers/model_doc/nllb>
- pyannote Community-1 diarization: <https://huggingface.co/pyannote/speaker-diarization-community-1>
- Ollama API: <https://github.com/ollama/ollama/blob/main/docs/api.md>

## License

The NammaDub application code is MIT-licensed. The downloaded AI models, datasets and optional dependencies have separate licenses and terms; those terms take precedence for their own components.
