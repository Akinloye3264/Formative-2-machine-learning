"""
Convert voice recordings in any format (.mp4, .aac, .ogg, .m4a, .opus, .mp3 ...)
to 16 kHz mono WAV, the format src/preprocess_audio.py expects.

Uses the ffmpeg binary bundled with the `imageio-ffmpeg` pip package, so
nothing else needs to be installed:
    pip install imageio-ffmpeg

Run from the repository root:
    python src/convert_audio.py --input path/to/original_recordings

Name the originals first (member01_yes_approve.mp4, member01_yes_approve_2.ogg,
outsider01_confirm_transaction.m4a ...). The WAV keeps the same name and is
filed in the per-person layout that src/preprocess_audio.py reads:
    member01_yes_approve.mp4        -> data/raw/audio/member01/member01_yes_approve.wav
    outsider01_yes_approve.m4a      -> data/raw/audio/unauthorized/outsider01/outsider01_yes_approve.wav
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


def destination(src: Path, root: Path) -> Path:
    """<root>/<person>/<name>.wav, or <root>/unauthorized/<person>/ for outsiders."""
    name = src.stem.lower()
    person = name.split("_")[0]
    folder = root / "unauthorized" / person if person.startswith("outsider") else root / person
    return folder / f"{name}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert recordings to 16 kHz mono WAV")
    parser.add_argument("--input", type=Path, required=True, help="folder with the original files")
    parser.add_argument("--output", type=Path, default=Path("data/raw/audio"),
                        help="audio root; files go to <output>/<person>/ "
                             "(outsiders: <output>/unauthorized/<person>/)")
    args = parser.parse_args()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    files = [p for p in sorted(args.input.iterdir())
             if p.is_file() and p.suffix.lower() not in SKIP_EXTS]
    if not files:
        raise SystemExit(f"No recordings found in {args.input}")

    for src in files:
        dst = destination(src, args.output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            convert(src, dst, ffmpeg)
            print(f"OK    {src.name} -> {dst}")
        except subprocess.CalledProcessError:
            print(f"FAIL  {src.name} (not a readable audio/video file)")


if __name__ == "__main__":
    main()
