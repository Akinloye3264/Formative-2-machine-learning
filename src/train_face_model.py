"""Face recognition model (Part 4). Run from the repo root:

    python src/train_face_model.py
    python src/train_face_model.py --data data/processed/image_features.csv

Reads image_features.csv, trains logistic regression and random forest,
keeps the better one, rejects unknown faces, and saves
models/face_model.joblib and outputs/metrics/face_metrics.json.
"""
import sys

from train_auth_models import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--data" not in args:
        args += ["--data", "data/processed/image_features.csv"]
    sys.argv = [sys.argv[0], "face"] + args
    main()
