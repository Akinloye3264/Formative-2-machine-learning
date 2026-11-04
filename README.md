# Formative 2: Multimodal User Authentication and Product Recommendation

> **Temporary team working guide:** This README is intentionally detailed so every
> group member can set up the project and complete their assigned work. Replace it
> with a shorter, results-focused README when the project is complete.

## 1. Project overview

This project builds a system that recognizes an authorized user from a face,
generates a product recommendation from merged customer data, and verifies the
user's voice before displaying the recommendation. It must reject an unknown
face, an unknown voice, or a voice that does not match the recognized person.

The submission must contain three working models:

- Facial recognition model
- Voiceprint verification model
- Product recommendation model

It must also include exploratory analysis, cleaned and merged tabular data, image
and audio feature files, model evaluation, a command-line demonstration, a
report, a GitHub link, a video link, and individual contributions.

## 2. Team roles

| Team member | Primary ownership | Main deliverables |
|---|---|---|
| Member 1 | Tabular data and product model | EDA, merge, `merged_dataset.csv`, product model |
| Member 2 | Image pipeline and face model | Image processing, `image_features.csv`, face model |
| Member 3 | Audio pipeline and voice model | Audio processing, `audio_features.csv`, voice model |
| Member 4 | Integration, testing, report, and video | CLI app, combined evaluation, unauthorized tests, report/video |

Primary ownership does not remove the individual data-collection requirement.
Every member must provide their own facial images and audio recordings.

## 3. Repository structure

```text
Formative-2-machine-learning/
├── data/
│   ├── raw/
│   │   ├── customer_social_profiles.csv
│   │   └── customer_transactions.csv
│   ├── processed/
│   │   ├── merged_dataset.csv
│   │   ├── image_features.csv
│   │   └── audio_features.csv
│   ├── images/
│   │   ├── member01/
│   │   ├── member02/
│   │   ├── member03/
│   │   ├── member04/
│   │   └── unauthorized/
│   └── audio/
│       ├── member01/
│       ├── member02/
│       ├── member03/
│       ├── member04/
│       └── unauthorized/
├── notebooks/
│   ├── 01_tabular_eda_merge.ipynb
│   ├── 02_image_processing.ipynb
│   ├── 03_audio_processing.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── preprocess_tabular.py
│   ├── preprocess_images.py
│   ├── preprocess_audio.py
│   ├── train_product_model.py
│   ├── train_face_model.py
│   ├── train_voice_model.py
│   └── app.py
├── models/
├── outputs/
│   ├── figures/
│   └── metrics/
├── report/
├── requirements.txt
└── README.md
```

Do not rename shared folders or output files without discussing the change with
the group. Other notebooks and scripts will depend on these paths.

## 4. Required software and accounts

Each member needs:

- A GitHub account and access to the team repository
- A Google account for Google Colab
- Git installed locally
- Optional: Python 3.10 or newer for running scripts locally
- Optional: VS Code for editing Markdown and Python files

The team will use **Google Colab** for notebooks. Local Jupyter is not required.
The `.ipynb` format works in Colab and Jupyter.

## 5. One-time setup for every member

### 5.1 Accept repository access

The repository owner must invite Members 2, 3, and 4 as GitHub collaborators.
Every member must accept the invitation before attempting to push.

### 5.2 Configure Git identity

Run once, replacing the example values:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

### 5.3 Clone the repository

```powershell
Set-Location "$env:USERPROFILE\Documents"
git clone https://github.com/TEAM-ACCOUNT/REPOSITORY-NAME.git
Set-Location "REPOSITORY-NAME"
```

Replace the example URL and folder name with the real repository information.

### 5.4 Optional local Python environment

Colab is the main notebook environment, but a local environment is useful for
testing the command-line application:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `.venv` folder is ignored by Git and must never be committed.

### 5.5 Verify the clone

```powershell
git status
git remote -v
Get-ChildItem
```

`git status` should initially report a clean working tree.

## 6. Google Colab workflow

### 6.1 Open a notebook from GitHub

1. Go to <https://colab.research.google.com/>.
2. Select **File > Open notebook**.
3. Select the **GitHub** tab.
4. Enter the repository URL.
5. Select the notebook assigned to you.

If the repository is private, authorize Colab to access private repositories.
Never paste a GitHub token into a saved notebook cell.

### 6.2 Make repository data available in Colab

Opening a notebook from GitHub does not automatically download the datasets into
the runtime. A public repository can be cloned with:

```python
import os

REPO_URL = "https://github.com/TEAM-ACCOUNT/REPOSITORY-NAME.git"
REPO_NAME = "REPOSITORY-NAME"

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}

%cd /content/{REPO_NAME}
```

For a private repository, use Colab's authorized GitHub integration or upload the
needed files temporarily. Do not store passwords or access tokens in notebooks.

### 6.3 Install packages

Colab includes many data-science packages. Install project requirements with:

```python
!pip install -q -r requirements.txt
```

If Colab asks to restart after changing a core package, restart the runtime and
rerun the notebook from the beginning.

### 6.4 Verify data paths

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
SOCIAL_DATA = PROJECT_ROOT / "data/raw/customer_social_profiles.csv"
TRANSACTION_DATA = PROJECT_ROOT / "data/raw/customer_transactions.csv"

assert SOCIAL_DATA.exists(), f"Missing: {SOCIAL_DATA}"
assert TRANSACTION_DATA.exists(), f"Missing: {TRANSACTION_DATA}"
```

Use repository-relative paths. Do not save personal Windows or Google Drive paths
in the final notebooks.

### 6.5 Save Colab work safely

Before closing Colab:

1. Select **Runtime > Run all** and confirm every cell succeeds.
2. Select **File > Download > Download .ipynb**.
3. Move the file into the local `notebooks/` folder.
4. Ensure its name has not become `filename (1).ipynb`.
5. Commit and push it from the local repository.

The recommended workflow is: Colab, download notebook, local Git commit, push.

## 7. Git collaboration workflow

Each member must work on a separate branch. Do not develop directly on `main`.

### 7.1 Create a branch

```powershell
git switch main
git pull origin main
git switch -c member-1-tabular
```

Use these branch names:

- `member-1-tabular`
- `member-2-images`
- `member-3-audio`
- `member-4-integration`

### 7.2 Before each work session

```powershell
git switch YOUR-BRANCH-NAME
git pull origin main
```

If Git reports conflicts, resolve them with the relevant teammate. Do not delete
another member's work merely to remove a conflict.

### 7.3 Commit focused work

```powershell
git status
git add PATH-TO-FILES-YOU-CHANGED
git commit -m "Describe the completed task"
git push -u origin YOUR-BRANCH-NAME
```

Useful commit messages include:

```text
Complete tabular EDA and merge validation
Add image augmentation and embedding extraction
Extract MFCC and spectral audio features
Integrate authentication and recommendation flow
```

Check `git status` before staging. Do not accidentally commit credentials,
temporary downloads, unrelated files, or a virtual environment.

### 7.4 Merge through pull requests

1. Push the member branch.
2. Open a GitHub pull request into `main`.
3. Ask another member to review it.
4. Confirm the notebook runs from beginning to end.
5. Merge only when paths and output formats match this guide.

## 8. Shared data-collection rules

### 8.1 Participant identifiers

Use only these identifiers in filenames and feature tables:

```text
member01
member02
member03
member04
unauthorized01
```

Do not alternate between real names, initials, and member IDs.

### 8.2 Consent and privacy

- Obtain permission before storing anyone's face or voice.
- Explain that the files will be used in a student GitHub project.
- Do not collect unrelated personal information.
- Do not use an online stranger's photo or voice without permission.
- Before using a public repository, confirm that participants accept public
  storage or ask the instructor how raw biometric samples should be submitted.

### 8.3 Facial images required from every member

Each member must provide at least one neutral, one smiling, and one surprised
image. Three originals per expression are recommended, producing nine images per
member and making evaluation more credible.

Use JPG or PNG, one clearly visible face, reasonable lighting, no beauty filter,
and some natural variation in angle or lighting.

```text
member01_neutral_01.jpg
member01_smile_01.jpg
member01_surprised_01.jpg
```

### 8.4 Audio required from every member

Each member records:

1. "Yes, approve."
2. "Confirm transaction."

Record three takes per phrase when possible. Use WAV, mono, 16 kHz, a quiet room,
a normal voice, and a microphone approximately 15–25 cm away.

```text
member01_yes_approve_01.wav
member01_confirm_transaction_01.wav
```

Do not alter originals. Augmentations must be created programmatically and labeled.

### 8.5 Unauthorized demonstration data

With consent, collect at least one face and one voice recording from someone
outside the group. Keep these samples out of training and use them as final unseen
tests.

## 9. Member 1: tabular data and product recommendation

### Goal and owned files

Create a clean merged dataset and a model that predicts the product category a
customer is likely to purchase.

- `notebooks/01_tabular_eda_merge.ipynb`
- `src/preprocess_tabular.py`
- `src/train_product_model.py`
- `data/processed/merged_dataset.csv`
- Product-model artifact, figures, and metrics

### Detailed tasks

1. Load both raw CSVs without modifying them.
2. Display shapes, columns, types, samples, and summary statistics.
3. Count missing values, exact duplicates, and duplicate identifiers.
4. Create at least three labeled EDA plots for distributions, outliers, category
   balance, or correlations, with a written interpretation under each.
5. Normalize the join key. Social IDs resemble `A178`, while transaction IDs
   resemble `178`; create one consistently typed customer ID.
6. Validate relationships before merging. Both files have multiple rows per
   customer, so a raw many-to-many merge can incorrectly multiply observations.
7. Choose and justify an aggregation strategy. A likely approach is one aggregated
   social record per customer joined to transaction-level prediction rows.
8. Report unmatched customer IDs and all pre/post-merge row counts.
9. Engineer features such as mean engagement, mean purchase interest, platform
   activity, transaction frequency, average spending, recency, and ratings.
10. Prevent target leakage when creating historical transaction features.
11. Save `data/processed/merged_dataset.csv`.
12. Use `product_category` as the target unless analysis establishes otherwise.
13. Build a `ColumnTransformer` for imputation, categorical encoding, and scaling
    when appropriate.
14. Train a baseline and comparison model, such as Logistic Regression and Random
    Forest.
15. Consider grouping train/test splits by customer so one person does not appear
    in both sets.
16. Evaluate accuracy, precision, recall, macro/weighted F1, confusion matrix, and
    log loss when probabilities are available.
17. Save the complete preprocessing/model pipeline with `joblib`.
18. Move reusable logic from the notebook into the owned scripts.

### Handoff to Member 4

Provide the exact input columns/types, saved pipeline path, a one-prediction
function, product-label meanings, metrics, and known limitations.

## 10. Member 2: image preprocessing and facial recognition

### Goal and owned files

Convert facial images into features and recognize group members while rejecting
an unknown face.

- `notebooks/02_image_processing.ipynb`
- `src/preprocess_images.py`
- `src/train_face_model.py`
- `data/processed/image_features.csv`
- Face-model artifact, figures, and metrics

### Detailed tasks

1. Confirm every member supplied all three expressions.
2. Verify filenames, formats, readability, and participant IDs.
3. Display labeled samples for all four members.
4. Detect/crop the face, resize consistently, and normalize it.
5. Split by original image before augmentation.
6. Apply at least two training augmentations such as small rotation, horizontal
   flip, brightness adjustment, or appropriate grayscale conversion.
7. Never put augmented versions of a test image into training.
8. Extract meaningful pretrained facial embeddings where possible.
9. Save metadata (`sample_id`, `member_id`, `expression`, `augmentation`, and
   `source_file`) followed by numeric features in `image_features.csv`.
10. Train Logistic Regression, SVM, Random Forest, or another justified classifier.
11. Add a confidence or embedding-distance rejection threshold so an unknown face
    is not automatically labeled as a member.
12. Select the threshold on validation data, not the final unauthorized sample.
13. Evaluate accuracy, precision, recall, F1, confusion matrix, false acceptance
    rate, and false rejection rate.
14. Test the unauthorized face only after training and threshold selection.
15. Save the classifier, labels, feature configuration, and threshold.
16. Move reusable preprocessing and inference into the owned scripts.

### Handoff to Member 4

Provide model paths, threshold, input format, metrics, and a function like:

```python
member_id, confidence, authorized = verify_face(image_path)
```

## 11. Member 3: audio preprocessing and voice verification

### Goal and owned files

Extract speaker-related features and verify that a voice matches the member
identified by the facial model.

- `notebooks/03_audio_processing.ipynb`
- `src/preprocess_audio.py`
- `src/train_voice_model.py`
- `data/processed/audio_features.csv`
- Voice-model artifact, figures, and metrics

### Detailed tasks

1. Confirm every member recorded both phrases with correct filenames.
2. Check duration, sample rate, channel count, clipping, silence, and readability.
3. Convert consistently to mono 16 kHz, trim excess silence, and normalize audio.
4. Display and label a waveform and spectrogram for every member, explaining
   differences in duration, energy, and frequency content.
5. Split by original recording before augmentation.
6. Apply at least two training augmentations such as pitch shift, time stretch,
   or low-level background noise.
7. Never put an augmented version of a test recording into training.
8. Extract MFCC means/standard deviations, spectral roll-off, RMS energy, spectral
   centroid, and zero-crossing rate.
9. Save metadata (`sample_id`, `member_id`, `phrase`, `augmentation`, and
   `source_file`) followed by numeric features in `audio_features.csv`.
10. Train a speaker classifier or verification model.
11. Verify against the face model's expected identity instead of accepting any
    known voice.
12. Choose a confidence/distance threshold using validation data.
13. Evaluate accuracy, precision, recall, F1, confusion matrix, false acceptance
    rate, and false rejection rate.
14. Test the unauthorized voice and an identity mismatch such as Member 1's face
    followed by Member 2's voice.
15. Save preprocessing objects, classifier, labels, and threshold.
16. Move reusable preprocessing and inference into the owned scripts.

### Handoff to Member 4

Provide model paths, threshold, WAV requirements, metrics, and a function like:

```python
confidence, authorized = verify_voice(audio_path, expected_member_id)
```

## 12. Member 4: integration, evaluation, report, and demonstration

### Goal and owned files

Combine all models into a command-line flow, independently verify their outputs,
and coordinate the final submission.

- `notebooks/04_model_evaluation.ipynb`
- `src/app.py`
- Combined figures and metrics
- Report material and demonstration plan

### Detailed tasks

1. Review the three handoff contracts and confirm model artifacts load with
   repository-relative paths.
2. Build one evaluation table containing accuracy, F1, applicable loss,
   authentication error rates, and limitations.
3. Implement this command-line sequence:
   - Ask for a facial image and run facial verification.
   - Deny access immediately if the face is rejected.
   - If accepted, run the product recommendation but keep it hidden.
   - Ask for an audio recording.
   - Verify that the voice matches the recognized face identity.
   - Display the recommendation only if both checks pass.
4. Handle missing files, unsupported formats, and model errors without crashing.
5. Test four flows: matching authorized face/voice, unauthorized face,
   authorized face/unauthorized voice, and authorized face/different member voice.
6. Verify that final test samples were not used in training or threshold choice.
7. Run every notebook from a fresh Colab runtime.
8. Check feature CSV metadata and column consistency.
9. Coordinate report sections: problem, data, preprocessing, EDA, features,
   models, evaluation, multimodal flow, limitations, ethics, and contributions.
10. Prepare the video script showing a successful transaction and rejected attempts.
11. Confirm GitHub and video sharing permissions.
12. Perform the final repository audit.

## 13. Notebook standards

Every notebook must:

- Begin with its title, owner, purpose, inputs, and outputs.
- Use repository-relative paths.
- Explain decisions in Markdown cells.
- Give every plot a title, axis labels, legend when needed, and interpretation.
- Set random seeds for reproducibility.
- Avoid hidden manual steps, credentials, and personal paths.
- Run top to bottom without errors in a fresh Colab runtime.
- End with findings, limitations, and generated artifact paths.

```python
import random
import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
```

## 14. Shared output contracts

`merged_dataset.csv` must contain the cleaned product-model data, target, and a
normalized identifier for validation. Member 1 must document columns excluded
from modeling.

`image_features.csv` must begin with:

```text
sample_id, member_id, expression, augmentation, source_file
```

`audio_features.csv` must begin with:

```text
sample_id, member_id, phrase, augmentation, source_file
```

Numeric features follow the metadata columns in both feature files.

## 15. Order of work

### Phase 1: setup and collection

1. Repository owner makes the setup commit.
2. Everyone clones the repository and creates a branch.
3. Everyone provides images and recordings.
4. The group verifies consent, filenames, and completeness.

### Phase 2: parallel pipelines

1. Member 1 performs tabular EDA, merge validation, and product modeling.
2. Member 2 performs image processing, feature extraction, and face modeling.
3. Member 3 performs audio processing, feature extraction, and voice modeling.
4. Member 4 defines integration interfaces and reviews progress.

### Phase 3: integration

1. Members 1–3 complete their handoffs.
2. Member 4 integrates the pipelines into `src/app.py`.
3. Everyone tests success, unauthorized, and identity-mismatch flows.

### Phase 4: submission

1. Rerun notebooks from clean Colab runtimes.
2. Confirm all outputs exist.
3. Complete and proofread the report.
4. Record and review the video.
5. Add repository and video links.
6. Replace this temporary guide with a concise final README covering setup,
   usage, results, architecture, limitations, and actual contributions.

## 16. Final checklist

- [ ] Both raw datasets are included.
- [ ] Cleaning, nulls, duplicates, types, and join validation are documented.
- [ ] At least three labeled tabular EDA plots are interpreted.
- [ ] Every member supplied the three required expressions.
- [ ] At least two image augmentations are demonstrated.
- [ ] `image_features.csv` contains metadata and features.
- [ ] Every member recorded both phrases.
- [ ] Waveforms and spectrograms are shown and interpreted.
- [ ] At least two audio augmentations are demonstrated.
- [ ] `audio_features.csv` contains MFCC and spectral/energy features.
- [ ] All three models are trained, evaluated, and saved.
- [ ] Accuracy, F1, and applicable loss are reported.
- [ ] False acceptance and false rejection are discussed.
- [ ] Authorized transaction succeeds.
- [ ] Unauthorized face and voice are rejected.
- [ ] Mismatched known face and voice are rejected.
- [ ] Recommendation appears only after both authentication checks.
- [ ] Notebooks run from top to bottom without errors.
- [ ] No passwords, tokens, `.env` files, or virtual environments are committed.
- [ ] Report, GitHub link, video link, and contributions are complete.

## 17. Current known data notes

Initial inspection found:

- `customer_social_profiles.csv`: 155 rows and 84 unique social customer IDs.
- `customer_transactions.csv`: 150 rows and 75 unique transaction customer IDs.
- Social identifiers have an `A` prefix; legacy transaction identifiers are numeric.
- Removing the prefix gives 61 overlapping unique customer IDs.
- Both datasets contain repeated customers, so an unvalidated direct merge may
  cause a many-to-many row explosion.

Member 1 must reproduce and explain these observations in the notebook rather
than relying only on this README.
