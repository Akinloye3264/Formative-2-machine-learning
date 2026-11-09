"""Voiceprint verification model (Part 4). Run from the repo root:

    python src/train_voice_model.py
    python src/train_voice_model.py --data data/processed/audio_features.csv

Reads audio_features.csv, trains logistic regression and random forest,
keeps the better one, rejects unknown voices, and saves
models/voice_model.joblib and outputs/metrics/voice_metrics.json.
"""
import sys

from train_auth_models import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data" not in args:
        args += ["--data", "data/processed/audio_features.csv"]
    sys.argv = [sys.argv[0], "voice"] + args
    main()
