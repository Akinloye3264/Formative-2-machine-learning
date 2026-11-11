"""Part 4 - Model Creation: face, voiceprint and product recommendation models.

Run from the repository root. Examples:

    python src/train_auth_models.py face    --data data/image_features.csv
    python src/train_auth_models.py voice   --data data/audio_features.csv
    python src/train_auth_models.py product --data data/merged_dataset.csv

    # Use a trained model on new rows (shows "ACCEPTED" or "REJECTED"):
    python src/train_auth_models.py predict --model models/face_model.joblib --data new_face_row.csv

Each training command:
  1. loads the CSV and works out which columns are the label and the features,
  2. splits the rows into train and test (augmented copies of the same photo or
     recording are never split across train and test, so the scores are honest),
  3. trains Logistic Regression and Random Forest, and keeps the one with the
     better cross-validated weighted F1 on the TRAIN rows only,
  4. reports accuracy, weighted F1 and log loss on the untouched TEST rows,
  5. saves the model to models/<name>_model.joblib and the scores to
     outputs/metrics/<name>_metrics.json.

If your CSV uses different column names, pass them explicitly, e.g.
    --label member --group-col source_image
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("outputs/metrics")

# Column names we look for when --label / --group-col are not given.
LABEL_GUESSES = {
    "face": ["member_id", "member", "person", "user", "name", "label", "identity"],
    "voice": ["member_id", "member", "speaker", "person", "user", "name", "label", "identity"],
    "product": ["product_category", "product", "category", "target", "label"],
}
GROUP_GUESSES = {
    "face": ["source_file", "source_image", "original_image", "image_name", "image_id", "image", "file", "filename"],
    "voice": ["source_file", "source_audio", "original_audio", "audio_name", "audio_id", "audio", "file", "filename"],
    "product": ["customer_id_new", "customer_id", "customer_id_legacy", "customer"],
}
# Columns that describe a row but must never be used as model inputs.
NON_FEATURE_HINTS = ["augment", "phrase", "expression", "source", "filename", "filepath"]


# ----------------------------------------------------------------------------- helpers
def find_column(df, wanted, guesses, what):
    """Return the column the user asked for, or the first guess that exists."""
    if wanted:
        if wanted not in df.columns:
            raise SystemExit(f"--{what} '{wanted}' is not a column. Columns are: {list(df.columns)}")
        return wanted
    lowered = {c.lower(): c for c in df.columns}
    for guess in guesses:
        if guess in lowered:
            return lowered[guess]
    return None


def guess_group_column(df, kind, label):
    """For face/voice: a text column naming the original photo or recording."""
    if kind not in ("face", "voice"):
        return None
    tokens = ["source", "original", "file", "path", "image", "audio", "recording", "clip"]
    for col in df.columns:
        is_text = not pd.api.types.is_numeric_dtype(df[col])
        if col != label and is_text and any(t in col.lower() for t in tokens):
            return col
    return None


def looks_like_id(name):
    # matches "id", "customer_id", "customer_id_new", "transaction_id" - but not "hist_idx_3"
    return re.search(r"(^|_)id($|_)", name.lower()) is not None


def prepare_columns(df, kind, label, group):
    """Decide which columns are numeric features and which are categorical features."""
    df = df.copy()
    # Turn date columns into simple numbers a model can use.
    for col in [c for c in df.columns if "date" in c.lower()]:
        dates = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_month"] = dates.dt.month
        df[f"{col}_dayofweek"] = dates.dt.dayofweek
        df = df.drop(columns=[col])

    skip = {label}
    if group:
        skip.add(group)
    numeric, categorical = [], []
    for col in df.columns:
        if col in skip or looks_like_id(col):
            continue
        if kind in ("face", "voice") and any(h in col.lower() for h in NON_FEATURE_HINTS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif kind == "product" and df[col].nunique() <= 30:
            categorical.append(col)
    if not numeric and not categorical:
        raise SystemExit("No usable feature columns found. Check the CSV.")
    return df, numeric, categorical


def make_preprocessor(numeric, categorical):
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(parts)


def split_rows(y, groups, test_size=0.25):
    """Return (train_index, test_index). Rows in the same group stay together."""
    idx = np.arange(len(y))
    if groups is not None:
        # Hold out whole groups (e.g. one photo plus all its augmented copies) but keep
        # every class (person / product) represented in the test set where possible.
        n = group_folds(y, groups, cap=int(round(1 / test_size)))
        if n >= 2:
            splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
            return next(splitter.split(idx, y, groups))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        return next(splitter.split(idx, y, groups))
    try:
        return train_test_split(idx, test_size=test_size, random_state=SEED, stratify=y)
    except ValueError:
        warnings.warn("Too few rows per class to stratify; using a plain random split.")
        return train_test_split(idx, test_size=test_size, random_state=SEED)


def group_folds(y, groups, cap):
    """How many folds we can make so every class has at least one group per fold."""
    per_class = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)}).groupby("y")["g"].nunique()
    return int(min(cap, per_class.min()))


def cv_splitter(y_train, groups_train):
    """Cross-validation that respects groups and small classes; None if too little data."""
    if groups_train is not None:
        n = group_folds(y_train, groups_train, cap=5)
        if n >= 2:
            return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        return None  # some class has only one original recording/photo: CV would be meaningless
    n = min(5, int(pd.Series(y_train).value_counts().min()))
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED) if n >= 2 else None


def candidates(preprocessor):
    return {
        "logistic_regression": Pipeline([("prep", preprocessor),
                                         ("model", LogisticRegression(max_iter=3000))]),
        "random_forest": Pipeline([("prep", preprocessor),
                                   ("model", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
    }


def to_dense(matrix):
    return np.asarray(matrix.todense()) if hasattr(matrix, "todense") else np.asarray(matrix)


def safe_log_loss(model, X, y):
    proba = model.predict_proba(X)
    return float(log_loss(y, proba, labels=list(model.classes_)))


# ----------------------------------------------------------------------------- training
def train(kind, args):
    df = pd.read_csv(args.data)
    label = find_column(df, args.label, LABEL_GUESSES[kind], "label")
    if label is None:
        raise SystemExit(f"Could not find the label column. Use --label. Columns: {list(df.columns)}")
    group = (find_column(df, args.group_col, GROUP_GUESSES[kind], "group-col")
             or guess_group_column(df, kind, label))
    df = df.dropna(subset=[label])
    if group is None and kind in ("face", "voice"):
        print("WARNING: no column identifies the ORIGINAL photo/recording. Augmented copies of "
              "the same file can end up in both train and test, which makes the scores look "
              "better than they really are. Add --group-col <column>.")

    print(f"\n=== {kind.upper()} MODEL ===")
    print(f"Rows: {len(df)} | label column: '{label}' | group column: '{group}'")
    print("Class counts:\n" + df[label].value_counts().to_string())

    df, numeric, categorical = prepare_columns(df, kind, label, group)
    features = numeric + categorical
    print(f"Features used: {len(numeric)} numeric, {len(categorical)} categorical")

    X, y = df[features], df[label].astype(str)
    groups = df[group].astype(str).values if group else None
    if groups is not None and group_folds(y.values, groups, cap=2) < 2:
        print("WARNING: some class has only ONE original photo/recording, so rows cannot be "
              "split by original file. Falling back to a normal split; augmented copies of the "
              "same file may appear in both train and test, so treat the scores as optimistic. "
              "Collecting at least 2 originals per person fixes this.")
        groups = None
    train_idx, test_idx = split_rows(y.values, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = groups[train_idx] if groups is not None else None
    print(f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)}")

    # Choose the better model using ONLY the training rows (cross-validation).
    models = candidates(make_preprocessor(numeric, categorical))
    cv = cv_splitter(y_train.values, g_train)
    cv_scores = {}
    for name, model in models.items():
        if cv is None:
            cv_scores[name] = float("nan")
            continue
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted",
                                 groups=g_train)
        cv_scores[name] = float(scores.mean())
        print(f"  {name}: cross-validated weighted F1 = {scores.mean():.3f}")
    if cv is None:
        print("  (too little data for cross-validation; using logistic_regression, "
              "which is the steadier choice for very small datasets)")
        best_name = "logistic_regression"
    else:
        best_name = max(cv_scores, key=cv_scores.get)
    best = models[best_name].fit(X_train, y_train)
    print(f"Chosen model: {best_name}")

    # Final scores on the test rows the model has never seen.
    pred = best.predict(X_test)
    metrics = {
        "model_type": best_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "log_loss": safe_log_loss(best, X_test, y_test),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "cv_f1_weighted": {k: (None if v != v else v) for k, v in cv_scores.items()},
    }
    print(f"\nTEST RESULTS  accuracy={metrics['accuracy']:.3f}  "
          f"F1={metrics['f1_weighted']:.3f}  loss={metrics['log_loss']:.3f}")
    print(classification_report(y_test, pred, zero_division=0))

    # Refit on ALL rows so the saved model knows every person / product.
    final = candidates(make_preprocessor(numeric, categorical))[best_name].fit(X, y)
    bundle = {"model": final, "features": features, "label": label, "kind": kind,
              "classes": list(final.classes_), "threshold": args.threshold, "metrics": metrics,
              "centroids": None, "limits": None}
    if kind in ("face", "voice"):
        # "Unknown person" guard: how far may a new sample be from a person's usual samples?
        Z = to_dense(final.named_steps["prep"].transform(X))
        labels = y.values
        bundle["centroids"], bundle["limits"] = {}, {}
        for person in final.classes_:
            pts = Z[labels == person]
            centre = pts.mean(axis=0)
            bundle["centroids"][person] = centre
            # factor x the farthest training sample: genuine new photos/recordings fall inside
            bundle["limits"][person] = float(args.distance_factor *
                                             np.linalg.norm(pts - centre, axis=1).max())
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / f"{kind}_model.joblib")
    (REPORT_DIR / f"{kind}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved {MODEL_DIR / f'{kind}_model.joblib'} and {REPORT_DIR / f'{kind}_metrics.json'}")


# ----------------------------------------------------------------------------- predicting
def predict_rows(bundle, df):
    """Return a list of (best_guess, confidence, accepted) for each row of df.

    A face/voice row is 'accepted' only when BOTH checks pass:
      1. the model is at least `threshold` sure about who it is, and
      2. the sample is not unusually far from that person's training samples.
    Anything else is treated as an UNKNOWN person (unauthorized attempt).

    Voice models saved by src/train_voice_model.py (bundle with a "method") use
    claimed-identity verification instead; each row is checked against its own
    best-matching member here.
    """
    if bundle.get("method"):
        import voice_verification as vv
        return [vv.verify(bundle, row) for row in df.to_dict("records")]
    df = df.copy()
    for col in [c for c in df.columns if "date" in c.lower()]:
        dates = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_month"] = dates.dt.month
        df[f"{col}_dayofweek"] = dates.dt.dayofweek
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"The new data is missing these columns: {missing}")
    model = bundle["model"]
    X = df[bundle["features"]]
    proba = model.predict_proba(X)
    Z = to_dense(model.named_steps["prep"].transform(X)) if bundle.get("centroids") else None
    out = []
    for n, row in enumerate(proba):
        i = int(np.argmax(row))
        person = model.classes_[i]
        accepted = bool(row[i] >= bundle["threshold"])
        if Z is not None:  # also reject samples that are far from this person's usual samples
            accepted = accepted and bool(
                np.linalg.norm(Z[n] - bundle["centroids"][person]) <= bundle["limits"][person])
        out.append((person, float(row[i]), accepted))
    return out


def verify_features(model_path, features, expected_member=None):
    """One-sample check for the CLI app (Member 4).

    features: dict {feature_name: value} from the preprocessing step.
    expected_member: for the voice check, the member the FACE model recognised.
    Returns (member_id, confidence, authorized). If expected_member is given,
    authorized is True only when the model hears that same person.

    For the voice model saved by src/train_voice_model.py, `authorized` is the
    verification decision for expected_member (cosine similarity to that
    member's enrolment centroid >= the validation-tuned threshold, and that
    member is the best match); member_id is the best-matching member and
    confidence its similarity score. `features` comes from
    preprocess_audio.extract_features (with SpeechBrain installed, so the
    ecapa_* columns are present).
    """
    bundle = joblib.load(model_path)
    if bundle.get("method"):
        import voice_verification as vv
        return vv.verify(bundle, features, expected_member)
    guess, conf, ok = predict_rows(bundle, pd.DataFrame([features]))[0]
    if expected_member is not None:
        ok = ok and guess == expected_member
    return guess, conf, ok


def predict(args):
    bundle = joblib.load(args.model)
    df = pd.read_csv(args.data)
    for i, (guess, conf, ok) in enumerate(predict_rows(bundle, df)):
        status = "ACCEPTED" if ok else "REJECTED (unknown)"
        print(f"row {i}: {guess}  confidence={conf:.2f}  -> {status}")


# ----------------------------------------------------------------------------- command line
def main():
    parser = argparse.ArgumentParser(description="Part 4: train / use the three models")
    sub = parser.add_subparsers(dest="command", required=True)
    for kind in ("face", "voice", "product"):
        p = sub.add_parser(kind, help=f"train the {kind} model")
        p.add_argument("--data", required=True, help="path to the CSV file")
        p.add_argument("--label", help="label column (auto-detected if omitted)")
        p.add_argument("--group-col", help="column that identifies the original photo/recording/customer")
        p.add_argument("--threshold", type=float, default=0.6,
                       help="minimum confidence to accept a face/voice match (default 0.6)")
        p.add_argument("--distance-factor", type=float, default=2.0,
                       help="how far from a person's usual samples is still accepted, as a "
                            "multiple of their farthest training sample (default 2.0; "
                            "raise it if real users are rejected, lower it if strangers get in)")
    p = sub.add_parser("predict", help="use a saved model on new rows")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    args = parser.parse_args()
    if args.command == "predict":
        predict(args)
    else:
        train(args.command, args)


if __name__ == "__main__":
    main()
