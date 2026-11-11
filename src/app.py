"""Command-line demo: face check -> hidden recommendation -> voice check -> show.

Owner: Member 4 (integration).

Run from the repository root::

    python src/app.py
    python src/app.py --face path/to/photo.jpg --voice path/to/recording.wav

The flow follows the assignment diagram:

1. Ask for a face image and run face recognition.
2. If the face is rejected, deny access immediately (the voice is never asked for).
3. If the face is accepted, run the product model but keep the answer hidden.
4. Ask for a voice recording and check that it matches the SAME member the face
   model recognised (a known voice that belongs to someone else is rejected).
5. Show the recommendation only when both checks pass.

Every problem (missing file, wrong format, model error) prints a message and
denies access instead of crashing.

Models used (all already in the repository):
    models/face_recognition_bundle.joblib   face model (Member 2)
    models/voice_model.joblib               voice model (Member 3 / Part 4 code)
    models/product_model.joblib              product model (Member 1)

Demo assumption: the product model needs a customer profile, and the project has
no table linking members to customers. Each member is therefore linked to one
customer from ``merged_dataset.csv`` (member01 -> the first customer id, member02
-> the second, ...). Use ``--customer`` to choose another customer id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

FACE_MODEL = ROOT / "models" / "face_recognition_bundle.joblib"
VOICE_MODEL = ROOT / "models" / "voice_model.joblib"
MERGED_DATA = ROOT / "data" / "processed" / "merged_dataset.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}

# Columns of merged_dataset.csv that describe a customer for the product model.
PROFILE_COLUMNS = [
    "social_media_platform", "engagement_score", "purchase_interest_score",
    "review_sentiment", "n_social_records", "n_prior_transactions",
    "prior_avg_amount", "prior_avg_rating", "prior_mode_category",
    "days_since_prev", "month", "day_of_week",
]


class AppError(Exception):
    """A problem the user can understand (missing file, wrong format, ...)."""


# ----------------------------------------------------------------------------- checks
def _check_file(path, allowed, kind):
    path = Path(str(path).strip().strip('"'))
    if not path.exists():
        raise AppError(f"{kind} file not found: {path}")
    if path.suffix.lower() not in allowed:
        raise AppError(f"Unsupported {kind} format '{path.suffix}'. Use one of: "
                       f"{', '.join(sorted(allowed))}")
    return path


def check_face(image_path, model_path=FACE_MODEL):
    """Return {'member', 'confidence', 'accepted'} for one face image."""
    path = _check_file(image_path, IMAGE_EXTENSIONS, "image")
    if not Path(model_path).exists():
        raise AppError(f"Face model not found: {model_path}")
    from preprocess_images import verify_face  # heavy import (torch), so only when needed

    result = verify_face(path, model_path=model_path)
    return {"member": result.predicted_member,
            "confidence": float(result.classifier_confidence),
            "accepted": bool(result.accepted)}


def check_voice(audio_path, expected_member, model_path=VOICE_MODEL):
    """Return {'member', 'confidence', 'accepted'}; accepted only if it matches expected_member."""
    path = _check_file(audio_path, AUDIO_EXTENSIONS, "audio")
    if not Path(model_path).exists():
        raise AppError(f"Voice model not found: {model_path}")
    import joblib
    from preprocess_audio import extract_features, load_audio
    from train_auth_models import verify_features

    samples = load_audio(path)
    if samples.size == 0:
        raise AppError("The recording is silent or empty.")
    features = extract_features(samples)
    needed = joblib.load(model_path)["features"]
    row = {name: features[name] for name in needed if name in features}
    member, confidence, accepted = verify_features(model_path, row, expected_member)
    return {"member": member, "confidence": float(confidence), "accepted": bool(accepted)}


# ----------------------------------------------------------------------------- recommendation
def customer_for(member_id, merged_path=MERGED_DATA):
    """Customer id linked to a member for the demo (see module docstring)."""
    import pandas as pd

    ids = sorted(pd.read_csv(merged_path)["customer_id"].unique())
    number = int("".join(ch for ch in str(member_id) if ch.isdigit()) or 1)
    return ids[(number - 1) % len(ids)]


def recommend(member_id, customer_id=None, merged_path=MERGED_DATA, top_k=3):
    """Top product categories for the customer linked to this member."""
    import pandas as pd
    from train_product_model import predict_product

    data = pd.read_csv(merged_path)
    customer = customer_id if customer_id is not None else customer_for(member_id, merged_path)
    rows = data[data["customer_id"].astype(str) == str(customer)]
    if rows.empty:
        raise AppError(f"Customer {customer} is not in {merged_path.name}")
    latest = rows.sort_values("purchase_date").iloc[-1]
    return predict_product({c: latest[c] for c in PROFILE_COLUMNS if c in latest.index}, top_k=top_k)


# ----------------------------------------------------------------------------- the flow
def run_transaction(face_input=None, voice_input=None, customer_id=None,
                    face_checker=check_face, voice_checker=check_voice,
                    recommender=recommend, ask=input, say=print):
    """Run the full flow once. Returns a dict with 'outcome' and 'reason'.

    The checkers, recommender, ask and say can be replaced, which is how the
    evaluation notebook tests the four scenarios without needing raw files.
    """
    say("=== Multimodal authentication and product recommendation ===")
    result = {"outcome": "DENIED", "reason": "", "member": None, "recommendation": None}

    # Step 1 and 2: face
    try:
        face_input = face_input or ask("Path to a face image: ")
        face = face_checker(face_input)
    except AppError as error:
        result["reason"] = str(error)
        say(f"ACCESS DENIED: {error}")
        return result
    except Exception as error:  # a model or library problem must not crash the demo
        result["reason"] = f"face check failed ({error})"
        say(f"ACCESS DENIED: face check failed ({error})")
        return result
    if not face["accepted"]:
        result["reason"] = "face not recognised"
        say(f"ACCESS DENIED: face not recognised (closest match {face['member']}, "
            f"confidence {face['confidence']:.2f}).")
        return result
    member = face["member"]
    result["member"] = member
    say(f"Face recognised as {member} (confidence {face['confidence']:.2f}).")

    # Step 3: run the product model but keep the answer hidden
    try:
        hidden = recommender(member, customer_id)
    except Exception as error:
        result["reason"] = f"recommendation failed ({error})"
        say(f"ACCESS DENIED: recommendation failed ({error})")
        return result
    say("A recommendation is ready. Voice verification is required to display it.")

    # Step 4: voice, must match the recognised member
    try:
        voice_input = voice_input or ask("Path to a voice recording (say 'Yes, approve'): ")
        voice = voice_checker(voice_input, member)
    except AppError as error:
        result["reason"] = str(error)
        say(f"ACCESS DENIED: {error}")
        return result
    except Exception as error:
        result["reason"] = f"voice check failed ({error})"
        say(f"ACCESS DENIED: voice check failed ({error})")
        return result
    if not voice["accepted"]:
        if voice["member"] != member:
            why = f"voice sounds like {voice['member']}, not {member}"
        else:
            why = "voice not recognised as an approved voice"
        result["reason"] = why
        say(f"ACCESS DENIED: {why} (confidence {voice['confidence']:.2f}).")
        return result

    # Step 5: both checks passed
    result.update(outcome="GRANTED", reason="face and voice verified", recommendation=hidden)
    say(f"Voice verified as {member} (confidence {voice['confidence']:.2f}).")
    say("ACCESS GRANTED. Recommended products:")
    for rank, (category, probability) in enumerate(hidden, start=1):
        say(f"  {rank}. {category} ({probability:.0%})")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Face + voice protected product recommendation")
    parser.add_argument("--face", help="path to a face image (.jpg, .jpeg, .png)")
    parser.add_argument("--voice", help="path to a voice recording (.wav, .flac, .ogg, .mp3)")
    parser.add_argument("--customer", help="customer id for the recommendation (default: the member's linked customer)")
    args = parser.parse_args(argv)
    result = run_transaction(args.face, args.voice, args.customer)
    return 0 if result["outcome"] == "GRANTED" else 1


if __name__ == "__main__":
    sys.exit(main())
