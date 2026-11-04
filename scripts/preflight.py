from __future__ import annotations

import importlib.util
import shutil
import sys


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK" if ok else "MISSING"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark}] {label}{suffix}")
    return ok


def main() -> int:
    print("NammaDub environment preflight\n")
    supported_python = (3, 10) <= sys.version_info[:2] < (3, 12)
    all_ok = check("Python 3.10 or 3.11", supported_python, sys.version.split()[0]) and True
    all_ok &= check("ffmpeg", shutil.which("ffmpeg") is not None, "install FFmpeg and add it to PATH")
    all_ok &= check("ffprobe", shutil.which("ffprobe") is not None, "comes with FFmpeg")
    all_ok &= check("FastAPI", importlib.util.find_spec("fastapi") is not None)
    all_ok &= check("faster-whisper", importlib.util.find_spec("faster_whisper") is not None)
    all_ok &= check("Transformers", importlib.util.find_spec("transformers") is not None)
    all_ok &= check("PyTorch", importlib.util.find_spec("torch") is not None)
    all_ok &= check("soundfile", importlib.util.find_spec("soundfile") is not None)
    all_ok &= check("IndicF5 package", importlib.util.find_spec("f5_tts") is not None,
                    "install git+https://github.com/ai4bharat/IndicF5.git")
    if not all_ok:
        print("\nResolve the missing items, then run this check again.")
        return 1
    print("\nBase environment checks passed. Model downloads and a real dubbing run still need to be tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
