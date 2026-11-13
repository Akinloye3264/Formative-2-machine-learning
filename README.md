# Multimodal Authentication and Product Recommendation

Group 25's Formative 2 project implements an end-to-end machine-learning
pipeline that combines:

1. product-category recommendation from tabular customer and transaction data;
2. facial recognition with unknown-face rejection; and
3. speaker verification against the identity predicted from the face.

The application prepares a recommendation after a face is accepted but reveals
it only when the supplied voice belongs to the same authorized member. An
unknown face, an outsider voice, or a voice from a different authorized member
results in `ACCESS DENIED`.

## Project and submission links

- [Contribution tracker and task sheet](https://docs.google.com/spreadsheets/d/1Q8PtWaS_K_qOS2p1XhSQ_1ya45wTa9oqK_xMN-mO44M/edit?usp=sharing)
- [GitHub repository](https://github.com/Akinloye3264/Formative-2-machine-learning)
- [CLI demonstration video](https://youtu.be/_WRzMGv5p7s)
- [Final Group 25 report](report/Formative%202_%20Multimodal%20Data%20Preprocessing%20Assignment_Report_Group25.pdf)

## Team

| Team member | Primary contribution |
|---|---|
| Akinloye Emmanuel | Image preprocessing and augmentation, facial-feature extraction, face classification and unknown-face rejection, repository coordination, CLI compatibility testing, and demonstration video |
| Kevin Nizeyimana | Tabular-data audit, EDA, merge validation, feature engineering, and product-recommendation model |
| Henriette Biziyaremye | Audio preprocessing and augmentation, audio-feature extraction, and voice-verification model |
| Dorian Sheja Bitega | Combined evaluation, multimodal integration, CLI application, and end-to-end testing |

## System workflow

```text
Face image
    |
    v
MTCNN alignment -> FaceNet embedding -> identity classifier
    |
    +-- similarity below face threshold --> ACCESS DENIED
    |
    v
Prepare product recommendation (keep hidden)
    |
    v
Voice recording -> ECAPA-TDNN embedding -> claimed-identity verification
    |
    +-- voice rejected or identity mismatch --> ACCESS DENIED
    |
    v
ACCESS GRANTED -> display top-three product categories
```

The face and voice thresholds are validation-derived experimental thresholds.
They are appropriate for this academic prototype but are not production
security guarantees.

## Final results

| Component | Accuracy | Macro F1 | Log loss | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| Product recommendation | 0.167 | 0.142 | 2.751 | N/A | N/A |
| Face recognition | 1.000 | 1.000 | 0.005542 | 0.000 (0/3 final unauthorized images) | 0.000 (0/12 authorized predictions) |
| Voice verification | 1.000 | 1.000 | 0.133 | 0.063 (1/16 outsider claimed-identity attempts) | 0.000 (0/8 genuine attempts) |

The product model is below the 0.200 majority-class accuracy baseline. This is
reported as a limitation: the small tabular dataset contains weak predictive
relationships between the available customer features and product category.
The perfect authorized face and voice identification scores also come from
small, controlled test sets and must not be interpreted as real-world biometric
performance.

### Product recommendation

- The two supplied tabular datasets are normalized and joined by customer ID.
- Social records are aggregated before the join to avoid many-to-many row
  inflation.
- Historical transaction features use only purchases preceding the row being
  predicted.
- The split is grouped by customer to prevent the same customer appearing in
  training and testing data.
- A scikit-learn Logistic Regression pipeline produces ranked product-category
  probabilities.

The final merged dataset contains 117 transactions from 61 customers. The
grouped hold-out contains 30 transactions from 16 customers.

### Face recognition

- Four authorized members supplied neutral, smiling, and surprised images: 12
  authorized originals in total.
- MTCNN detects, aligns, crops, and resizes every face to 160 x 160 pixels.
- Each aligned image produces four variants: original, horizontal flip,
  eight-degree rotation, and 115% brightness.
- A VGGFace2-pretrained FaceNet `InceptionResnetV1` extracts a normalized
  512-dimensional embedding.
- Logistic Regression predicts the authorized identity.
- Cosine similarity to the predicted identity centroid rejects unknown faces.

Evaluation uses three-fold leave-one-expression-out validation. Augmented
copies of a held-out image never enter its training fold. The selected face
threshold is `0.726368`, the midpoint between the minimum authorized
leave-one-out similarity (`0.847625`) and maximum unauthorized-validation
similarity (`0.605112`). It rejected all three images from a separate unseen
unauthorized identity.

### Voice verification

- Four authorized members recorded two phrases three times, producing 24
  original member recordings.
- Four outsiders recorded both phrases once, producing eight unauthorized
  recordings.
- Audio is converted to 16 kHz mono WAV, trimmed, peak-normalized, and augmented
  with pitch shift, time stretch, and background noise.
- Feature extraction produces 35 hand-crafted audio features and a
  192-dimensional pretrained ECAPA-TDNN speaker embedding.
- Member take 1 is used for enrollment/training, take 2 for validation, and
  take 3 for testing. Outsiders are split by speaker, with two validation and
  two test speakers. Outsiders never train the model.
- Verification uses cosine similarity to the face-predicted member's enrollment
  centroid and also requires that member to be the closest match.

The voice threshold `0.345` was chosen on validation data and fixed before the
test set was evaluated. On original test recordings, the model rejected all
genuine-member mismatches, accepted all eight genuine attempts, and falsely
accepted one of 16 outsider claimed-identity attempts.

## Repository structure

```text
data/
  raw/                         Supplied tabular data and consented audio
    audio/member01...04/       Authorized-member recordings
    audio/unauthorized/        Outsider recordings
  processed/                   Merged data, features, manifests, and splits
models/
  product_model.joblib
  face_recognition_bundle.joblib
  voice_model.joblib
notebooks/
  01_tabular_eda_merge.ipynb
  02_image_processing.ipynb
  03_audio_processing.ipynb
  04_model_evaluation.ipynb
  05_face_voice_models.ipynb
outputs/
  figures/                     Figures used for evaluation and reporting
  metrics/                     JSON/CSV evaluation summaries and trials
report/
  Formative 2_ Multimodal Data Preprocessing Assignment_Report_Group25.pdf
src/
  app.py                       End-to-end command-line application
  preprocess_tabular.py
  preprocess_images.py
  convert_audio.py
  preprocess_audio.py
  speaker_embeddings.py
  train_product_model.py
  train_face_model.py
  train_voice_model.py
  train_auth_models.py
  voice_verification.py
requirements.txt
README.md
```

Raw facial images are intentionally excluded from version control. Downloaded
pretrained weights, virtual environments, caches, and generated augmented media
are also excluded.

## Environment setup

### Prerequisites

- Windows, Linux, or macOS
- Python 3.11 (recommended and tested)
- Git
- Internet access on the first run to download pretrained FaceNet and ECAPA
  weights

Python 3.13 should not be used for this project because the pinned FaceNet and
PyTorch stack targets Python 3.11-compatible releases.

### Clone and install

```bash
git clone https://github.com/Akinloye3264/Formative-2-machine-learning.git
cd Formative-2-machine-learning
py -3.11 -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin compatible versions of NumPy, Pillow, PyTorch,
Torchvision, Torchaudio, SpeechBrain, and Hugging Face Hub. SpeechBrain uses a
local copy strategy on Windows, so Administrator privileges or Developer Mode
are not required for the ECAPA model.

Verify the main dependencies:

```bash
python -c "import numpy, PIL, torch, torchaudio, torchvision, facenet_pytorch, speechbrain; print('Dependencies ready')"
```

## Reproducing the pipelines

Run commands from the repository root. The notebooks document the complete
analysis and can be executed in numerical order.

### 1. Tabular pipeline

```bash
python src/preprocess_tabular.py
python src/train_product_model.py
```

Primary outputs:

- `data/processed/merged_dataset.csv`
- `models/product_model.joblib`
- `outputs/metrics/product_model_metrics.json`

### 2. Image pipeline

The raw face directories are private and must be supplied locally. Each member
directory should contain neutral, smile, and surprised images.

```bash
python src/preprocess_images.py extract --input path/to/private/image-folders
python src/train_face_model.py --data data/processed/image_features.csv
python src/preprocess_images.py verify --image path/to/test-face.jpg
```

Primary outputs:

- `data/processed/image_features.csv`
- `models/face_recognition_bundle.joblib`
- `outputs/metrics/face_model_metrics.json`
- `outputs/metrics/unauthorized_face_test_results.csv`

`notebooks/02_image_processing.ipynb` was developed in Google Colab with a T4
GPU, although the saved model can be used locally on CPU.

### 3. Audio pipeline

```bash
python src/preprocess_audio.py
python src/train_voice_model.py
```

Primary outputs:

- `data/processed/audio_features.csv`
- `data/processed/audio_features_unauthorized.csv`
- `data/processed/audio_splits.csv`
- `models/voice_model.joblib`
- `outputs/metrics/voice_metrics.json`
- `outputs/metrics/voice_verification_trials.csv`
- `outputs/figures/voice_far_frr_threshold.png`

Use `python src/train_voice_model.py --validation-only` to compare candidate
methods without evaluating the held-out test set.

### 4. Combined evaluation

Open and run:

```text
notebooks/04_model_evaluation.ipynb
```

This notebook validates successful authentication, unknown-face rejection,
outsider-voice rejection, and authorized-member identity mismatch.

## Running the CLI application

Show the available arguments:

```bash
python src/app.py --help
```

Run the complete flow:

```bash
python src/app.py --face path/to/authorized-face.jpeg --voice path/to/voice.wav
```

The optional `--customer` argument selects a customer ID. When it is omitted,
the demonstration maps each member to a customer from the processed dataset
because the supplied data contains no real member-to-customer relationship.

Example outcomes:

- authorized face + matching authorized voice: `ACCESS GRANTED`, followed by
  the top-three recommendations;
- unauthorized face: immediate `ACCESS DENIED`, without running voice
  verification;
- authorized face + outsider voice: `ACCESS DENIED`;
- authorized face + a different authorized member's voice: `ACCESS DENIED`.

The raw face files used in the demonstration video are not included in this
repository.

## Data privacy and responsible use

Facial images and voice recordings are biometric data. Raw facial images are
kept outside version control. Audio is present only for academic
reproducibility and must remain in the repository only while every participant
has consented to its storage and to the repository's visibility. If consent is
withdrawn, the affected recordings must be removed from the repository and its
history.

This prototype does not include liveness detection, replay-attack prevention,
encryption, access-control infrastructure, demographic bias analysis, or
production monitoring. It must not be deployed as a real authentication
system without substantially broader data, independent security evaluation,
and appropriate legal and privacy controls.

## Known limitations

- Only four authorized identities were evaluated.
- Each member supplied three original face expressions and six original audio
  recordings.
- Face threshold validation and final testing each used only one unauthorized
  identity.
- Voice threshold validation used two outsiders and final testing used two
  different outsiders.
- Results were obtained under limited cameras, microphones, lighting,
  expressions, phrases, accents, and environmental conditions.
- The product model performs near chance because of the small dataset and weak
  relationship between available features and the target.
- Authentication thresholds are experimental and are not production-security
  guarantees.

## Key artifacts

- Final report: `report/Formative 2_ Multimodal Data Preprocessing Assignment_Report_Group25.pdf`
- CLI: `src/app.py`
- Product model: `models/product_model.joblib`
- Face model: `models/face_recognition_bundle.joblib`
- Voice model: `models/voice_model.joblib`
- Product metrics: `outputs/metrics/product_model_metrics.json`
- Face metrics: `outputs/metrics/face_model_metrics.json`
- Voice metrics: `outputs/metrics/voice_metrics.json`
- Voice trial details: `outputs/metrics/voice_verification_trials.csv`

For the full methodology, evaluation tables, figures, ethical considerations,
and references, see the [final report](report/Formative%202_%20Multimodal%20Data%20Preprocessing%20Assignment_Report_Group25.pdf).
