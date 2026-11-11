"""
Task 3 - Sound data collection and processing.

Loads each member's voice recordings, applies augmentations, extracts
features, assigns train / validation / test splits and saves everything to
data/processed/.

Raw recordings (16 kHz mono WAV), one folder per person, named with the same
member IDs as the image pipeline (member01 ... member04):

    data/raw/audio/member01/member01_yes_approve.wav          take 1
    data/raw/audio/member01/member01_yes_approve_2.wav        take 2
    data/raw/audio/member01/member01_yes_approve_3.wav        take 3
    data/raw/audio/member01/member01_confirm_transaction.wav  ...

Recordings from people outside the group (never used to train anything):

    data/raw/audio/unauthorized/outsider01/outsider01_yes_approve.wav
    data/raw/audio/unauthorized/outsider01/outsider01_confirm_transaction.wav

The raw recordings are committed to the repository so the pipeline can be
reproduced; a recording may only be added with its speaker's consent (see the
README privacy note). Augmented WAVs are regenerated on demand and stay out
of Git.

Splits (fixed, saved to data/processed/audio_splits.csv):
  * members:   take 1 -> train, take 2 -> validation, take 3 -> test.
               A recording and all of its augmented copies share one split.
  * outsiders: split BY SPEAKER (seed 42): half of the outsiders -> validation,
               the other half -> test. Outsider clips get the same
               augmentations as member clips but never train the classifier.

Run from the repository root:
    python src/preprocess_audio.py
    python src/preprocess_audio.py --input path/to/audio --save-augmented

The feature function `extract_features` is reused by the voice verification
model and the command-line app, so a new voice sample is described exactly like
the training samples.
"""
from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

try:  # works both as `import preprocess_audio` (src on sys.path) and as a script
    import speaker_embeddings
except ImportError:  # pragma: no cover
    from . import speaker_embeddings

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
SPLITS_CSV = ROOT / "data" / "processed" / "audio_splits.csv"

# Member takes -> split. Extra takes (4, 5, ...) would be added to enrolment (train).
TAKE_TO_SPLIT = {1: "train", 2: "validation", 3: "test"}
SPLIT_ORDER = ["train", "validation", "test"]

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


def _scan(folder: Path, skip: set[str] = frozenset()) -> tuple[list[dict], list[str]]:
    """Audio files in `folder/<person>/` (and, for older layouts, directly in `folder`)."""
    rows, skipped = [], []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXTS or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if rel.parts[0] in skip or len(rel.parts) > 2:
            continue
        info = parse_filename(p)
        person_dir = rel.parts[0] if len(rel.parts) == 2 else None
        if info is None or (person_dir is not None and person_dir.lower() != info["member_id"]):
            skipped.append(str(rel))
            continue
        rows.append({"source_file": p.name, "path": str(p), **info})
    return rows, skipped


def list_recordings(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Table of correctly named member recordings in raw_dir/<member_id>/.

    The `unauthorized/` sub-folder is ignored. Badly named files, or files in
    another person's folder, are reported and not used.
    """
    rows, skipped = _scan(Path(raw_dir), skip={UNAUTH_DIR.name})
    if skipped:
        print(f"Skipped (must be <member_id>/<member_id>_<phrase>[_<take>].wav): {skipped}")
    df = pd.DataFrame(rows, columns=["source_file", "path", "member_id", "phrase", "take"])
    df.insert(3, "speaker_id", df["member_id"])
    return df


def list_unauthorized(unauth_dir: Path = UNAUTH_DIR) -> pd.DataFrame:
    """Table of outsider recordings in unauth_dir/<outsider_id>/.

    `speaker_id` is the outsider (outsider01 ...); `member_id` is always
    "unauthorized" so these rows can never be mistaken for a member's label.
    """
    rows, skipped = _scan(Path(unauth_dir))
    if skipped:
        print(f"Skipped (must be <outsider_id>/<outsider_id>_<phrase>.wav): {skipped}")
    df = pd.DataFrame(rows, columns=["source_file", "path", "member_id", "phrase", "take"])
    df.insert(3, "speaker_id", df["member_id"])
    df["member_id"] = "unauthorized"
    return df


def assign_splits(members: pd.DataFrame, outsiders: pd.DataFrame | None = None,
                  seed: int = SEED) -> pd.DataFrame:
    """One row per ORIGINAL recording with its split (see the module docstring).

    Members are split by take. Outsiders are split by speaker with a fixed
    seed, so every clip of one outsider is either validation or test, never both.
    """
    m = members.copy()
    m["role"] = "member"
    m["split"] = m["take"].map(lambda t: TAKE_TO_SPLIT.get(int(t), "train"))
    parts = [m]
    if outsiders is not None and not outsiders.empty:
        o = outsiders.copy()
        o["role"] = "outsider"
        speakers = np.array(sorted(o["speaker_id"].unique()))
        order = np.random.default_rng(seed).permutation(speakers)
        n_val = len(order) // 2
        val_speakers = set(order[:n_val])
        o["split"] = o["speaker_id"].map(lambda s: "validation" if s in val_speakers else "test")
        parts.append(o)
    table = pd.concat(parts, ignore_index=True)
    table["path"] = table["path"].map(lambda p: Path(p).resolve().relative_to(ROOT).as_posix()
                                      if Path(p).resolve().is_relative_to(ROOT) else str(p))
    cols = ["source_file", "role", "speaker_id", "member_id", "phrase", "take", "split", "path"]
    return table[cols].sort_values(["role", "speaker_id", "phrase", "take"]).reset_index(drop=True)


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
    """Speak 15% faster without changing pitch.

    librosa 1.0 builds the output frame times with np.arange(0, n_frames, rate),
    which can overshoot the last frame by a rounding error and raise a
    ParameterError for some clip lengths (it happened on one outsider clip).
    In that case the same phase vocoder is run with the frame times clipped.
    """
    try:
        return librosa.effects.time_stretch(y, rate=rate)
    except librosa.util.exceptions.ParameterError:
        stft = librosa.stft(y)
        n_frames = stft.shape[-1]
        t_out = np.arange(0.0, n_frames, rate)
        t_out = t_out[t_out < n_frames]
        stretched = librosa.phase_vocoder(stft, t_out=t_out)
        return librosa.istft(stretched, dtype=y.dtype, length=round(len(y) / rate))


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


def augment_versions(y: np.ndarray, sr: int = SR) -> dict:
    """{'original': y, 'pitch_shift': ..., 'time_stretch': ..., 'background_noise': ...}.

    Deterministic (fixed noise seed), so any script that needs the waveforms
    again (e.g. speaker embeddings) gets exactly the same clips.
    """
    versions = {"original": y}
    versions.update({name: fn(y, sr) for name, fn in AUGMENTATIONS.items()})
    return versions


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_features(y: np.ndarray, sr: int = SR, embedding: bool | None = None) -> dict:
    """
    Fixed-length feature vector for one clip.

    Hand-crafted summary statistics over frames (always):
      - MFCC 1-13 mean and std     -> timbre of the voice (the "voiceprint")
      - spectral roll-off mean/std -> frequency below which 85% of energy lies
      - spectral centroid mean/std -> brightness of the voice
      - spectral bandwidth mean
      - RMS energy mean/std        -> loudness
      - zero-crossing rate mean    -> noisiness / voicing
      - duration in seconds

    Pretrained speaker embedding (ecapa_0 ... ecapa_191, see
    src/speaker_embeddings.py): added when `embedding` is True, or when it is
    None (default) and SpeechBrain is installed. The saved voice model uses
    these columns, so the app needs SpeechBrain to verify a new recording.
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
    if embedding is None:
        embedding = speaker_embeddings.available()
    if embedding:
        if sr != speaker_embeddings.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=speaker_embeddings.SR)
        feats.update(speaker_embeddings.embedding_features(y))
    return feats



# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def process_recordings(recordings: pd.DataFrame, save_audio: bool = False) -> pd.DataFrame:
    """
    Original + every augmentation of every recording -> one feature row each.

    `member_id` is the label (same IDs as the face model, so the voice can be
    checked against the face; "unauthorized" for outsiders), `speaker_id` names
    the person who spoke, `source_file` ties a recording to its augmented
    copies, and `split` (when the input has it) is copied to every copy, so an
    original and its augmentations always land in the same split.
    `phrase`, `augmentation` and `split` are metadata, never model inputs.
    """
    rows = []
    for rec in recordings.itertuples(index=False):
        y = load_audio(rec.path)
        speaker = getattr(rec, "speaker_id", rec.member_id)
        for aug_name, y_aug in augment_versions(y, SR).items():
            if save_audio and aug_name != "original":
                sub = AUG_DIR / ("unauthorized" if rec.member_id == "unauthorized" else "") / speaker
                sub.mkdir(parents=True, exist_ok=True)
                sf.write(sub / f"{Path(rec.source_file).stem}_{aug_name}.wav", y_aug, SR)
            row = {
                "sample_id": f"{Path(rec.source_file).stem}_{aug_name}",
                "member_id": rec.member_id,
                "speaker_id": speaker,
                "phrase": rec.phrase,
                "source_file": rec.source_file,
                "augmentation": aug_name,
            }
            if hasattr(rec, "split"):
                row["split"] = rec.split
            rows.append({**row, **extract_features(y_aug, SR)})
    return pd.DataFrame(rows)


def process_unauthorized(unauth_dir: Path = UNAUTH_DIR, splits: pd.DataFrame | None = None,
                         save_audio: bool = False) -> pd.DataFrame:
    """Features of outsider recordings (original + the same augmentations as members).

    Saved separately from the member features: outsiders are only used to tune
    (validation speakers) and measure (test speakers) unknown-voice rejection.
    """
    outsiders = list_unauthorized(unauth_dir)
    if outsiders.empty:
        return pd.DataFrame()
    if splits is None:
        splits = assign_splits(list_recordings(), outsiders)
    outsiders = outsiders.merge(splits[["source_file", "split"]], on="source_file", how="left")
    return process_recordings(outsiders, save_audio=save_audio)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Audio augmentation, splits and feature extraction")
    parser.add_argument("--input", type=Path, default=RAW_DIR,
                        help="folder with one sub-folder per member (member01/ ...)")
    parser.add_argument("--unauthorized", type=Path, default=None,
                        help="folder with one sub-folder per outsider (default: <input>/unauthorized)")
    parser.add_argument("--output", type=Path, default=FEATURES_CSV)
    parser.add_argument("--save-augmented", action="store_true",
                        help=f"also write augmented WAVs to {AUG_DIR.relative_to(ROOT)}")
    args = parser.parse_args()

    recordings = list_recordings(args.input)
    if recordings.empty:
        raise SystemExit(f"No recordings found in {args.input}")
    print(check_coverage(recordings), "\n")

    unauth_dir = args.unauthorized or args.input / UNAUTH_DIR.name
    outsiders = list_unauthorized(unauth_dir) if unauth_dir.exists() else pd.DataFrame()
    splits = assign_splits(recordings, outsiders)
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_CSV, index=False)
    print(splits.groupby(["role", "split"]).agg(recordings=("source_file", "size"),
                                                speakers=("speaker_id", "nunique")), "\n")
    print(f"Saved split table -> {SPLITS_CSV}")

    members = recordings.merge(splits[["source_file", "split"]], on="source_file")
    if speaker_embeddings.available():
        print("SpeechBrain found: adding ECAPA speaker embeddings (ecapa_0 ... ecapa_191).")
    else:
        print("WARNING: SpeechBrain is not installed, so the ECAPA speaker-embedding columns "
              "used by the saved voice model will be missing. Install with:\n  "
              + speaker_embeddings.INSTALL_HINT)
    features = process_recordings(members, save_audio=args.save_augmented)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved {features.shape[0]} rows x {features.shape[1]} columns -> {args.output}")

    if not outsiders.empty:
        unauth = process_unauthorized(unauth_dir, splits, save_audio=args.save_augmented)
        unauth.to_csv(UNAUTH_CSV, index=False)
        print(f"Saved {len(unauth)} unauthorized rows "
              f"({unauth.speaker_id.nunique()} outsiders) -> {UNAUTH_CSV}")


if __name__ == "__main__":
    main()
