"""
Convert voice recordings in any format (.mp4, .aac, .ogg, .m4a, .opus, .mp3 ...)
to 16 kHz mono WAV, the format src/preprocess_audio.py expects.

Uses the ffmpeg binary bundled with the `imageio-ffmpeg` pip package, so
nothing else needs to be installed:
    pip install imageio-ffmpeg

Run from the repository root:
    python src/convert_audio.py --input path/to/original_recordings
    python src/convert_audio.py --input path/to/outsider_recordings --output data/raw/audio/unauthorized

Name the originals first (member01_yes_approve.mp4, member01_confirm_transaction.ogg ...);
the WAV keeps the same name, e.g. member01_yes_approve.wav.
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

SR = 16_000
SKIP_EXTS = {".wav", ".txt", ".md", ".ini", ".db"}


def convert(src: Path, dst: Path, ffmpeg: str) -> None:
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-vn",              # drop the video track of .mp4 files
         "-ac", "1",         # mono
         "-ar", str(SR),     # 16 kHz
         str(dst)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert recordings to 16 kHz mono WAV")
    parser.add_argument("--input", type=Path, required=True, help="folder with the original files")
    parser.add_argument("--output", type=Path, default=Path("data/raw/audio"))
    args = parser.parse_args()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(args.input.iterdir())
             if p.is_file() and p.suffix.lower() not in SKIP_EXTS]
    if not files:
        raise SystemExit(f"No recordings found in {args.input}")

    for src in files:
        dst = args.output / f"{src.stem.lower()}.wav"
        try:
            convert(src, dst, ffmpeg)
            print(f"OK    {src.name} -> {dst}")
        except subprocess.CalledProcessError:
            print(f"FAIL  {src.name} (not a readable audio/video file)")


if __name__ == "__main__":
    main()
