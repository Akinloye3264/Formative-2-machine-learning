# Multimodal User Authentication and Product Recommendation

This project combines facial recognition, voice verification, and product
recommendation in a multimodal access-control workflow. A recommendation is
shown only after an authorized face is recognized and the supplied voice
matches the recognized identity.

## System workflow

1. Process a facial image and extract a FaceNet embedding.
2. Predict the closest authorized identity.
3. Reject the face when its centroid similarity is below the fixed threshold.
4. Generate a product recommendation from the tabular model.
5. Verify that the voice belongs to the identity recognized from the face.
6. Display the recommendation only when both authentication checks pass.

## Current project status

| Component | Main implementation | Status |
|---|---|---|
| Product recommendation | `notebooks/01_tabular_eda_merge.ipynb` and `src/train_product_model.py` | Implemented |
| Face recognition | `notebooks/02_image_processing.ipynb` | Implemented and evaluated |
| Voice verification | `notebooks/03_audio_processing.ipynb` and `src/train_voice_model.py` | Implemented and evaluated (validation-tuned threshold, separate test) |
| Combined application | `src/app.py` and `notebooks/04_model_evaluation.ipynb` | Pending final integration |

## Facial-recognition pipeline

The image pipeline processes three expressions for each of four authorized
members: neutral, smile, and surprised.

- MTCNN detects, aligns, crops, and resizes each face to 160 x 160 pixels.
- Each aligned image produces an original plus horizontal-flip, eight-degree
  rotation, and brightness variants.
- A VGGFace2-pretrained FaceNet `InceptionResnetV1` model extracts normalized
  512-dimensional embeddings.
- Logistic Regression predicts the authorized identity.
- Cosine similarity to the predicted identity centroid provides unknown-person
  rejection.

Evaluation uses three-fold leave-one-expression-out validation. In each fold,
two expressions per member and their augmentations are used for training, while
the held-out original expression is used for testing. Augmented versions of a
test image never enter its training fold.

### Face-model results

| Measurement | Result |
|---|---:|
| Authorized originals evaluated | 12 |
| Cross-validation accuracy | 1.000 |
| Macro F1-score | 1.000 |
| Weighted F1-score | 1.000 |
| Log loss | 0.005542 |
| Minimum authorized leave-one-out similarity | 0.847625 |
| Maximum unauthorized-validation similarity | 0.605112 |
| Selected rejection threshold | 0.726368 |
| Validation false rejection rate | 0.000 |
| Validation false acceptance rate | 0.000 |
| Final unauthorized-test false acceptance rate | 0.000 |

The threshold is the midpoint between the minimum authorized leave-one-out
similarity and maximum unauthorized-validation similarity. It was fixed before
the separate unauthorized test was evaluated. It is a validation-derived
experimental threshold, not a production-security threshold, because only one
unauthorized identity was available for validation and one different identity
was available for final testing.

## Voice-verification pipeline

Each of the four members recorded "Yes, approve" and "Confirm transaction"
three times (24 recordings); four people outside the group recorded both
phrases once (8 recordings). All audio is 16 kHz mono WAV.

- `src/preprocess_audio.py` trims silence, peak-normalises, adds three
  augmentations (pitch shift, time stretch, background noise) to member and
  outsider clips, and extracts 35 hand-crafted features (MFCC, spectral,
  energy) plus a 192-dimensional pretrained ECAPA-TDNN speaker embedding
  (SpeechBrain, `src/speaker_embeddings.py`).
- Splits (`data/processed/audio_splits.csv`, fixed seed 42): members' take 1 is
  enrolment/training, take 2 is validation and take 3 is test. An original and
  its augmentations always share a split. Outsiders are split by speaker: two
  for validation, two for test. Outsiders never train any model.
- The voice is verified against the identity the face model recognised:
  accept when the cosine similarity to that member's enrolment centroid is at
  least the threshold and that member is the best match.
- Three methods were compared on validation only: the previous logistic
  regression with confidence + distance checks (retuned), cosine similarity on
  standardised hand-crafted features, and cosine similarity on ECAPA
  embeddings. ECAPA was chosen (validation FRR 6.2% vs 53-88% for the others at
  FAR <= 5%).
- Threshold rule: lowest validation FRR with outsider FAR <= 5% and mismatch
  <= 5%, middle of the tied range. Chosen threshold: 0.345. The test split was
  evaluated once, after the choice.

### Voice-model results (real recordings; validation alongside)

| Measurement | Validation | Test |
|---|---:|---:|
| FAR: outsider accepted (clips x claimed members) | 0.0% (0/16) | 6.2% (1/16) |
| FRR: member rejected for own identity | 0.0% (0/8) | 0.0% (0/8) |
| Mismatch: member accepted as another member | 0.0% (0/24) | 0.0% (0/24) |
| Identification accuracy / weighted F1 | 1.000 / 1.000 | 1.000 / 1.000 |
| Log loss | 0.076 | 0.133 |

Including the augmented copies, test FAR is 10.9% (7/64), FRR 15.6% (5/32)
and mismatch 0% (0/96). Every accepted test outsider trial was a claim of the
same member (member03). Full numbers, 95% confidence intervals and the method
comparison are in `outputs/metrics/voice_metrics.json`; every trial is in
`outputs/metrics/voice_verification_trials.csv`. The previous model's
unknown-voice FAR was 59.4% (notebook 05 leave-one-member-out test).

These are small-sample numbers: two test outsiders and eight test recordings.
The 95% interval for the test FAR on real recordings is 0.2%-30%, so the
model is far better than before but its true error rates are not yet known
precisely.

## Repository structure

```text
data/
  raw/                         # Original tabular datasets
    audio/member01/ ...        # Voice recordings, one folder per member
    audio/unauthorized/outsider01/ ...  # Recordings of non-members
  processed/                   # Generated feature, split and merged CSV files
models/                        # Serialized trained-model artifacts
notebooks/
  01_tabular_eda_merge.ipynb
  02_image_processing.ipynb
  03_audio_processing.ipynb
  04_model_evaluation.ipynb
outputs/
  figures/                     # Evaluation visualizations
  metrics/                     # JSON and CSV evaluation results
src/
  preprocess_tabular.py
  preprocess_images.py
  convert_audio.py
  preprocess_audio.py
  speaker_embeddings.py
  voice_verification.py
  train_product_model.py
  train_auth_models.py
  train_face_model.py
  train_voice_model.py
  app.py
requirements.txt
README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The image notebook was developed in Google Colab with a T4 GPU. Open
`notebooks/02_image_processing.ipynb`, select a GPU runtime, connect the expected
Google Drive project directory, and run the notebook from top to bottom.

The standalone model-training entry points can be run from the repository root
after their feature CSV files are available:

```powershell
python src/preprocess_images.py extract --input path/to/private/image-folders
python src/preprocess_images.py verify --image path/to/test-face.jpg
python src/train_product_model.py
python src/train_face_model.py --data data/processed/image_features.csv
python src/preprocess_audio.py
python src/train_voice_model.py
```

The voice model needs SpeechBrain for the ECAPA speaker embedding (CPU is
enough):

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain
```

`src/preprocess_audio.py` reads `data/raw/audio/<member_id>/` and
`data/raw/audio/unauthorized/<outsider_id>/`, writes the feature CSVs and
`data/processed/audio_splits.csv`. `src/train_voice_model.py --validation-only`
compares the methods without touching test data; without the flag it also
evaluates the chosen model once on the test split and saves
`models/voice_model.joblib`, `outputs/metrics/voice_metrics.json`,
`outputs/metrics/voice_verification_trials.csv` and
`outputs/figures/voice_far_frr_threshold.png`.

The image preprocessing command expects `member01` through `member04`
subdirectories containing one neutral, smile, and surprised image per member.
It writes aligned and augmented images under `data/processed/images/`, plus the
augmentation manifest and 512-dimensional FaceNet feature table. The verification
command loads `models/face_recognition_bundle.joblib` by default and prints an
`AUTHORIZED` or `REJECTED` decision using the saved experimental threshold.

## Expected artifacts

The completed pipelines should provide:

- `data/processed/merged_dataset.csv`
- `data/processed/image_features.csv`
- `data/processed/audio_features.csv`
- `models/product_model.joblib`
- `models/face_recognition_bundle.joblib` or the integration model equivalent
- `models/voice_model.joblib`
- `outputs/metrics/product_model_metrics.json`
- `outputs/metrics/face_model_metrics.json`
- `outputs/metrics/unauthorized_face_test_results.csv`
- `outputs/metrics/voice_metrics.json`
- figures used in the evaluation and report

Large generated artifacts may be supplied through the agreed project storage
when they are intentionally excluded from Git. Feature schemas, model paths,
and retrieval instructions must remain documented so another team member can
reproduce the application.

## Data privacy

Raw facial images and voice recordings are biometric data. They must only be
collected and shared with participant consent.

The raw voice recordings in `data/raw/audio/` (members and outsiders) are
committed so the audio pipeline can be reproduced and graded. A recording
may only be in this repository if the person speaking (member or outsider)
has agreed to it being stored here, and to the repository's visibility. A voice recording can identify a person, so a
recording must be removed (and the history rewritten) if its owner withdraws
consent, and new recordings must not be added without the speaker's
agreement. Augmented audio is regenerated by the scripts and is not committed.
Raw facial images remain outside the repository.

## Team responsibilities

| Member | Responsibility |
|---|---|
| Member 1 | Tabular preprocessing and product recommendation |
| Member 2 | Image preprocessing and facial recognition |
| Member 3 | Audio preprocessing and voice verification |
| Member 4 | Integration, combined evaluation, report, and demonstration |

## Limitations

- The face dataset contains only three original expressions per authorized
  member.
- Face-threshold validation and final testing each use only one unauthorized
  identity with three images.
- Voice verification is tuned on two outsiders and tested on two others, with
  one recording per phrase per member per split. Noisy rooms, other phrases,
  other devices and replayed recordings were not tested.
- Perfect results on this small controlled dataset do not establish real-world
  biometric-security performance.
- More identities, recording devices, poses, lighting conditions, background
  noise, and independent test samples are required for production evaluation.
