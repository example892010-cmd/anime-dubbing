$ErrorActionPreference = "Stop"
$Python = Get-Command py -ErrorAction SilentlyContinue
if (-not $Python) { throw "Install Python 3.10 and the Python Launcher first." }
ffmpeg -version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Install FFmpeg and add it to PATH first." }
ffprobe -version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "ffprobe was not found. Reinstall FFmpeg with ffprobe." }
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# PyTorch will select the available Windows wheel. For a specific CUDA build, follow pytorch.org.
.\.venv\Scripts\python.exe -m pip install torch torchaudio
.\.venv\Scripts\python.exe -m pip install git+https://github.com/ai4bharat/IndicF5.git
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Write-Host "Install finished. Add HF_TOKEN to .env after accepting the IndicF5 model terms."
Write-Host "Run: .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 7860"
