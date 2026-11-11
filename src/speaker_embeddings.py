"""Pretrained ECAPA-TDNN speaker embeddings (optional dependency).

`preprocess_audio.extract_features` adds 192 columns ecapa_0 ... ecapa_191 to
every clip when SpeechBrain is installed, so the training CSVs, the voice
model and the command-line app all describe a clip the same way. The voice
model chosen on validation data uses these columns.

Install (CPU only is enough, about one second per clip):

    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install speechbrain

The model (speechbrain/spkrec-ecapa-voxceleb, trained on the VoxCeleb
speaker-recognition data) is downloaded once from Hugging Face into
pretrained_models/ (git-ignored).
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"
EMB_PREFIX = "ecapa_"
EMB_DIM = 192
SR = 16_000          # the ECAPA model expects 16 kHz mono audio
INSTALL_HINT = ("pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu "
                "&& pip install speechbrain")


def available() -> bool:
    """True when SpeechBrain (and therefore torch) can be imported."""
    return all(importlib.util.find_spec(m) is not None for m in ("speechbrain", "torch"))


@lru_cache(maxsize=1)
def load_encoder():
    """Load the ECAPA encoder once per process."""
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=str(MODEL_DIR),
                                          run_opts={"device": "cpu"})


def embed(y: np.ndarray) -> np.ndarray:
    """192-dim speaker embedding of one 16 kHz mono clip."""
    import torch
    with torch.no_grad():
        emb = load_encoder().encode_batch(torch.tensor(np.asarray(y), dtype=torch.float32)[None])
    return emb.squeeze().cpu().numpy().astype(float)


def embedding_features(y: np.ndarray) -> dict:
    """{'ecapa_0': ..., 'ecapa_191': ...} for one clip."""
    return {f"{EMB_PREFIX}{i}": float(v) for i, v in enumerate(embed(y))}
