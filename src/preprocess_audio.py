"""
Task 3 - Sound data collection and processing.

Loads each member's voice recordings, applies augmentations, extracts
features and saves them to audio_features.csv.

Expected recordings (WAV preferred), one file per phrase per member, named
with the same member IDs as the image pipeline (member01 ... member04):
    data/raw/audio/member01_yes_approve.wav
    data/raw/audio/member01_confirm_transaction.wav
Extra takes are allowed: member01_yes_approve_2.wav

Recordings from a non-member (for the unauthorized-attempt demo):
    data/raw/audio/unauthorized/<anything>.wav

Raw recordings are biometric data: keep them out of Git (see README).

Run from the repository root:
    python src/preprocess_audio.py
    python src/preprocess_audio.py --input path/to/private/audio --save-augmented

The feature function `extract_features` is reused by the voiceprint
verification model and the command-line app, so a new voice sample is
described exactly like the training samples.
"""
from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

SR = 16_000            # resample everything to 16 kHz mono (enough for speech)
N_MFCC = 13
TOP_DB = 30            # silence-trimming threshold
SEED = 42

PHRASES = {
    "yes_approve": "Yes, approve",
    "confirm_transaction": "Confirm transaction",
}
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "audio"
UNAUTH_DIR = RAW_DIR / "unauthorized"
AUG_DIR = ROOT / "data" / "processed" / "audio_augmented"
FEATURES_CSV = ROOT / "data" / "processed" / "audio_features.csv"
UNAUTH_CSV = ROOT / "data" / "processed" / "audio_features_unauthorized.csv"

FILENAME_RE = re.compile(
    r"^(?P<member_id>[a-z0-9]+)_(?P<phrase>" + "|".join(PHRASES) + r")(?:_(?P<take>\d+))?$"
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_filename(path: Path) -> dict | None:
    """'member01_yes_approve_2.wav' -> {member_id, phrase, take}. None if the name is wrong."""
    m = FILENAME_RE.match(path.stem.lower())
    if not m:
        return None
    return {"member_id": m["member_id"], "phrase": m["phrase"], "take": int(m["take"] or 1)}


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Load as mono at `sr`, trim leading/trailing silence and peak-normalise."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    peak = np.max(np.abs(y)) if y.size else 0.0
    return y / peak if peak > 0 else y


def list_recordings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Table of correctly named recordings. Badly named files are reported, not used."""
    rows, skipped = [], []
    for p in sorted(raw_dir.glob("*")):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        info = parse_filename(p)
        if info is None:
            skipped.append(p.name)
            continue
        rows.append({"source_file": p.name, "path": str(p), **info})
    if skipped:
        print(f"Skipped (name must be <member_id>_<phrase>.wav): {skipped}")
    return pd.DataFrame(rows, columns=["source_file", "path", "member_id", "phrase", "take"])


def check_coverage(recordings: pd.DataFrame) -> pd.DataFrame:
    """Members x phrases count table; raises if any member is missing a phrase."""
    table = (recordings.groupby(["member_id", "phrase"]).size()
             .unstack(fill_value=0).reindex(columns=list(PHRASES), fill_value=0))
    missing = table[(table == 0).any(axis=1)]
    if not missing.empty:
        raise ValueError(f"Members missing a phrase:\n{missing}")
    return table


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def pitch_shift(y, sr=SR, n_steps=2.0):
    """Raise pitch by 2 semitones: simulates a slightly different speaking pitch."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate=1.15):
    """Speak 15% faster without changing pitch."""
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y, snr_db=20.0, seed=SEED):
    """Add white background noise at a 20 dB signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return y + rng.normal(0.0, np.sqrt(noise_power), size=y.shape)


AUGMENTATIONS = {
    "pitch_shift": lambda y, sr: pitch_shift(y, sr),
    "time_stretch": lambda y, sr: time_stretch(y),
    "background_noise": lambda y, sr: add_noise(y),
}


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(y: np.ndarray, sr: int = SR) -> dict:
    """
    Fixed-length feature vector for one clip (summary statistics over frames):
      - MFCC 1-13 mean and std     -> timbre of the voice (the "voiceprint")
      - spectral roll-off mean/std -> frequency below which 85% of energy lies
      - spectral centroid mean/std -> brightness of the voice
      - spectral bandwidth mean
      - RMS energy mean/std        -> loudness
      - zero-crossing rate mean    -> noisiness / voicing
      - duration in seconds
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    feats = {"duration_s": len(y) / sr}
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())
    feats.update({
        "rolloff_mean": float(rolloff.mean()),
        "rolloff_std": float(rolloff.std()),
        "centroid_mean": float(centroid.mean()),
        "centroid_std": float(centroid.std()),
        "bandwidth_mean": float(bandwidth.mean()),
        "rms_energy_mean": float(rms.mean()),
        "rms_energy_std": float(rms.std()),
        "zcr_mean": float(zcr.mean()),
    })
    return feats


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def process_recordings(recordings: pd.DataFrame, save_audio: bool = False) -> pd.DataFrame:
    """
    Original + every augmentation of every recording -> one feature row each.

    Columns match what src/train_auth_models.py expects: `member_id` is the label
    (same IDs as the face model, so the voice can be checked against the face),
    `source_file` keeps a recording and its augmented copies on the same side of
    the train/test split, and `phrase` / `augmentation` are ignored as features.
    """
    if save_audio:
        AUG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in recordings.itertuples(index=False):
        y = load_audio(rec.path)
        versions = {"original": y}
        versions.update({name: fn(y, SR) for name, fn in AUGMENTATIONS.items()})
        for aug_name, y_aug in versions.items():
            if save_audio and aug_name != "original":
                out = AUG_DIR / f"{Path(rec.source_file).stem}_{aug_name}.wav"
                sf.write(out, y_aug, SR)
            rows.append({
                "sample_id": f"{Path(rec.source_file).stem}_{aug_name}",
                "member_id": rec.member_id,
                "phrase": rec.phrase,
                "source_file": rec.source_file,
                "augmentation": aug_name,
                **extract_features(y_aug, SR),
            })
    return pd.DataFrame(rows)


def process_unauthorized(unauth_dir: Path = UNAUTH_DIR) -> pd.DataFrame:
    """Features of non-member recordings, kept apart from training data for the demo."""
    rows = []
    for p in sorted(unauth_dir.glob("*")):
        if p.suffix.lower() in AUDIO_EXTS:
            rows.append({"source_file": p.name, "member_id": "unauthorized",
                         **extract_features(load_audio(p), SR)})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Audio augmentation and feature extraction")
    parser.add_argument("--input", type=Path, default=RAW_DIR, help="folder of member recordings")
    parser.add_argument("--unauthorized", type=Path, default=None,
                        help="folder of non-member recordings (default: <input>/unauthorized)")
    parser.add_argument("--output", type=Path, default=FEATURES_CSV)
    parser.add_argument("--save-augmented", action="store_true",
                        help=f"also write augmented WAVs to {AUG_DIR.relative_to(ROOT)}")
    args = parser.parse_args()

    recordings = list_recordings(args.input)
    if recordings.empty:
        raise SystemExit(f"No recordings found in {args.input}")
    print(check_coverage(recordings), "\n")

    features = process_recordings(recordings, save_audio=args.save_augmented)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved {features.shape[0]} rows x {features.shape[1]} columns -> {args.output}")

    unauth_dir = args.unauthorized or args.input / "unauthorized"
    if unauth_dir.exists():
        unauth = process_unauthorized(unauth_dir)
        if not unauth.empty:
            unauth.to_csv(UNAUTH_CSV, index=False)
            print(f"Saved {len(unauth)} unauthorized samples -> {UNAUTH_CSV}")


if __name__ == "__main__":
    main()
