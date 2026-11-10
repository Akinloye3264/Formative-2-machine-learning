"""Reusable face preprocessing, embedding extraction, and verification.

This module mirrors the validated image pipeline in
``notebooks/02_image_processing.ipynb``:

* MTCNN detects, aligns, crops, and resizes one face to 160 x 160 pixels.
* Deterministic flip, rotation, and brightness variants are generated.
* VGGFace2-pretrained FaceNet produces normalized 512-value embeddings.
* Saved face-model bundles can verify a new image and reject unknown people.

Examples, run from the repository root::

    python src/preprocess_images.py extract --input data/images
    python src/preprocess_images.py verify --image path/to/photo.jpg

The pretrained FaceNet weights are downloaded by ``facenet-pytorch`` on first
use. Raw facial images are biometric data and should only be processed or
shared with participant consent.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_EXPRESSIONS = ("neutral", "smile", "surprised")
DEFAULT_MEMBERS = ("member01", "member02", "member03", "member04")
IMAGE_SIZE = 160
MARGIN = 20
FILENAME_PATTERN = re.compile(
    r"^(?P<member>member\d+)_(?P<expression>neutral|smile|surprised)_"
    r"(?P<number>\d+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceImage:
    """Metadata parsed from one authorized source-image path."""

    path: Path
    member_id: str
    expression: str


@dataclass(frozen=True)
class FaceVerificationResult:
    """Decision returned when a saved face bundle verifies one image."""

    source_file: str
    predicted_member: str
    classifier_confidence: float
    nearest_centroid: str | None
    similarity_to_prediction: float | None
    threshold: float
    accepted: bool
    decision: str


def select_device(requested: str = "auto") -> Any:
    """Return a torch device, preferring CUDA when ``requested`` is ``auto``."""

    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def load_rgb_image(path: str | Path) -> Image.Image:
    """Open an image, correct EXIF orientation, and return an RGB copy."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported image format '{image_path.suffix}'. Use {allowed}.")
    with Image.open(image_path) as opened_image:
        return ImageOps.exif_transpose(opened_image).convert("RGB").copy()


def parse_source_image(path: str | Path) -> SourceImage:
    """Parse ``memberNN_expression_number`` metadata from an image filename."""

    image_path = Path(path)
    match = FILENAME_PATTERN.fullmatch(image_path.stem)
    if match is None:
        raise ValueError(
            "Invalid authorized-image filename "
            f"'{image_path.name}'. Expected memberNN_expression_number.jpg."
        )
    return SourceImage(
        path=image_path,
        member_id=match.group("member").lower(),
        expression=match.group("expression").lower(),
    )


def discover_source_images(
    image_root: str | Path,
    members: Sequence[str] = DEFAULT_MEMBERS,
    expressions: Sequence[str] = DEFAULT_EXPRESSIONS,
) -> list[SourceImage]:
    """Find and validate one authorized source image per member/expression."""

    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    records: list[SourceImage] = []
    problems: list[str] = []
    for member_id in members:
        member_id = member_id.lower()
        member_folder = root / member_id
        if not member_folder.is_dir():
            problems.append(f"Missing member directory: {member_folder}")
            continue
        candidates = [
            path
            for path in member_folder.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ]
        parsed: list[SourceImage] = []
        for path in candidates:
            try:
                parsed.append(parse_source_image(path))
            except ValueError as error:
                problems.append(str(error))

        for expression in expressions:
            matches = [
                record
                for record in parsed
                if record.member_id == member_id
                and record.expression == expression.lower()
            ]
            if len(matches) != 1:
                problems.append(
                    f"Expected one {member_id} {expression} image, found {len(matches)}."
                )
            else:
                records.append(matches[0])

    if problems:
        raise ValueError("Image collection validation failed:\n- " + "\n- ".join(problems))
    return sorted(records, key=lambda item: (item.member_id, item.expression))


def deterministic_augmentations(image: Image.Image) -> dict[str, Image.Image]:
    """Return the four variants used by the validated notebook pipeline."""

    rgb_image = image.convert("RGB")
    rotation_fill = rgb_image.getpixel((0, 0))
    return {
        "original": rgb_image.copy(),
        "horizontal_flip": ImageOps.mirror(rgb_image),
        "rotation_8deg": rgb_image.rotate(
            8,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=rotation_fill,
        ),
        "brightness_115": ImageEnhance.Brightness(rgb_image).enhance(1.15),
    }


class FaceEmbeddingPipeline:
    """Load MTCNN and FaceNet once and reuse them for multiple images."""

    def __init__(self, device: str = "auto") -> None:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN

        self.torch = torch
        self.device = select_device(device)
        self.detector = MTCNN(
            image_size=IMAGE_SIZE,
            margin=MARGIN,
            keep_all=True,
            post_process=True,
            device=self.device,
        )
        self.embedding_model = (
            InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        )

    def align_one_face(self, image: Image.Image) -> tuple[Image.Image, Any]:
        """Detect exactly one face and return its aligned PIL image and tensor."""

        face_batch = self.detector(image.convert("RGB"))
        if face_batch is None:
            raise ValueError("No face was detected in the image.")
        if face_batch.ndim == 3:
            face_batch = face_batch.unsqueeze(0)
        face_count = int(face_batch.shape[0])
        if face_count != 1:
            raise ValueError(f"Expected exactly one face, but detected {face_count}.")

        face_tensor = face_batch[0].detach().cpu()
        face_array = face_tensor.permute(1, 2, 0).numpy()
        face_array = np.clip(face_array * 128.0 + 127.5, 0, 255).astype(np.uint8)
        return Image.fromarray(face_array), face_batch

    def embed_face_batch(self, face_batch: Any) -> np.ndarray:
        """Return L2-normalized FaceNet embeddings for MTCNN face tensors."""

        import torch.nn.functional as functional

        with self.torch.inference_mode():
            embeddings = self.embedding_model(face_batch.to(self.device))
            embeddings = functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()

    def embed_aligned_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        """Embed already aligned 160 x 160 RGB images."""

        from facenet_pytorch import fixed_image_standardization

        tensors = []
        for image in images:
            resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            array = np.asarray(resized, dtype=np.float32).copy()
            tensor = self.torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(fixed_image_standardization(tensor))
        if not tensors:
            return np.empty((0, 512), dtype=np.float32)
        return self.embed_face_batch(self.torch.stack(tensors))

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Detect one face in a source image and return its normalized embedding."""

        image = load_rgb_image(image_path)
        _, face_batch = self.align_one_face(image)
        return self.embed_face_batch(face_batch)[0]


def extract_feature_table(
    image_root: str | Path,
    processed_root: str | Path = "data/processed/images",
    manifest_path: str | Path = "data/processed/augmentation_manifest.csv",
    features_path: str | Path = "data/processed/image_features.csv",
    members: Sequence[str] = DEFAULT_MEMBERS,
    expressions: Sequence[str] = DEFAULT_EXPRESSIONS,
    device: str = "auto",
) -> pd.DataFrame:
    """Run the complete authorized-image pipeline and save its CSV artifacts."""

    sources = discover_source_images(image_root, members, expressions)
    processed = Path(processed_root)
    aligned_root = processed / "aligned"
    augmented_root = processed / "augmented"
    aligned_root.mkdir(parents=True, exist_ok=True)
    augmented_root.mkdir(parents=True, exist_ok=True)

    pipeline = FaceEmbeddingPipeline(device=device)
    manifest_records: list[dict[str, Any]] = []
    feature_metadata: list[dict[str, Any]] = []
    aligned_variants: list[Image.Image] = []

    for source in sources:
        source_image = load_rgb_image(source.path)
        aligned_image, _ = pipeline.align_one_face(source_image)

        aligned_folder = aligned_root / source.member_id
        augmented_folder = augmented_root / source.member_id
        aligned_folder.mkdir(parents=True, exist_ok=True)
        augmented_folder.mkdir(parents=True, exist_ok=True)

        aligned_name = f"{source.path.stem}_aligned.png"
        aligned_image.save(aligned_folder / aligned_name)

        for augmentation, variant in deterministic_augmentations(aligned_image).items():
            augmented_name = f"{source.path.stem}_{augmentation}.png"
            variant.save(augmented_folder / augmented_name)
            manifest_records.append(
                {
                    "member_id": source.member_id,
                    "expression": source.expression,
                    "source_file": source.path.name,
                    "augmentation": augmentation,
                    "augmented_file": augmented_name,
                    "width": variant.width,
                    "height": variant.height,
                }
            )
            feature_metadata.append(
                {
                    "sample_id": f"{source.path.stem}__{augmentation}",
                    "member_id": source.member_id,
                    "expression": source.expression,
                    "source_file": source.path.name,
                    "augmentation": augmentation,
                    "augmented_file": augmented_name,
                }
            )
            aligned_variants.append(variant)

    embeddings = pipeline.embed_aligned_images(aligned_variants)
    embedding_columns = [f"embedding_{index:03d}" for index in range(embeddings.shape[1])]
    manifest = pd.DataFrame(manifest_records)
    features = pd.concat(
        [
            pd.DataFrame(feature_metadata),
            pd.DataFrame(embeddings, columns=embedding_columns),
        ],
        axis=1,
    )

    manifest_output = Path(manifest_path)
    features_output = Path(features_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    features_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False)
    features.to_csv(features_output, index=False)
    return features


def _verify_notebook_bundle(
    bundle: dict[str, Any], embedding: np.ndarray, source_file: str
) -> FaceVerificationResult:
    feature_columns = list(bundle["embedding_columns"])
    feature_row = pd.DataFrame([embedding], columns=feature_columns)
    classifier = bundle["classifier"]
    predicted_member = str(classifier.predict(feature_row)[0])
    probabilities = classifier.predict_proba(feature_row)[0]
    confidence = float(probabilities.max())
    similarities = {
        str(member): float(embedding @ np.asarray(centroid))
        for member, centroid in bundle["authorized_centroids"].items()
    }
    nearest_member = max(similarities, key=similarities.get)
    similarity = similarities[predicted_member]
    threshold = float(bundle["rejection_threshold"])
    accepted = bool(similarity >= threshold)
    return FaceVerificationResult(
        source_file=source_file,
        predicted_member=predicted_member,
        classifier_confidence=confidence,
        nearest_centroid=nearest_member,
        similarity_to_prediction=similarity,
        threshold=threshold,
        accepted=accepted,
        decision="AUTHORIZED" if accepted else "REJECTED",
    )


def _verify_team_bundle(
    bundle: dict[str, Any], embedding: np.ndarray, source_file: str
) -> FaceVerificationResult:
    feature_row = pd.DataFrame([embedding], columns=bundle["features"])
    model = bundle["model"]
    probabilities = model.predict_proba(feature_row)[0]
    best_index = int(np.argmax(probabilities))
    predicted_member = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    threshold = float(bundle["threshold"])
    accepted = bool(confidence >= threshold)

    distance = None
    if bundle.get("centroids"):
        transformed = model.named_steps["prep"].transform(feature_row)
        if hasattr(transformed, "todense"):
            transformed = np.asarray(transformed.todense())
        transformed = np.asarray(transformed)[0]
        distance = float(
            np.linalg.norm(transformed - bundle["centroids"][predicted_member])
        )
        accepted = accepted and bool(distance <= bundle["limits"][predicted_member])

    return FaceVerificationResult(
        source_file=source_file,
        predicted_member=predicted_member,
        classifier_confidence=confidence,
        nearest_centroid=predicted_member,
        similarity_to_prediction=None if distance is None else -distance,
        threshold=threshold,
        accepted=accepted,
        decision="AUTHORIZED" if accepted else "REJECTED",
    )


def verify_face(
    image_path: str | Path,
    model_path: str | Path = "models/face_recognition_bundle.joblib",
    device: str = "auto",
) -> FaceVerificationResult:
    """Verify one image using either supported saved face-model bundle schema."""

    path = Path(image_path)
    embedding = FaceEmbeddingPipeline(device=device).embed_image(path)
    bundle = joblib.load(model_path)

    notebook_keys = {
        "classifier",
        "authorized_centroids",
        "rejection_threshold",
        "embedding_columns",
    }
    if notebook_keys.issubset(bundle):
        return _verify_notebook_bundle(bundle, embedding, path.name)
    if {"model", "features", "threshold"}.issubset(bundle):
        return _verify_team_bundle(bundle, embedding, path.name)
    raise ValueError(f"Unsupported face-model bundle keys: {sorted(bundle)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess facial images, extract FaceNet features, or verify a face."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract", help="build aligned images, augmentations, and image_features.csv"
    )
    extract_parser.add_argument("--input", required=True, help="root containing member folders")
    extract_parser.add_argument(
        "--processed-root", default="data/processed/images", help="aligned/augmented output root"
    )
    extract_parser.add_argument(
        "--manifest", default="data/processed/augmentation_manifest.csv"
    )
    extract_parser.add_argument("--features", default="data/processed/image_features.csv")
    extract_parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    extract_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    verify_parser = subparsers.add_parser("verify", help="verify one face image")
    verify_parser.add_argument("--image", required=True)
    verify_parser.add_argument(
        "--model", default="models/face_recognition_bundle.joblib"
    )
    verify_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        features = extract_feature_table(
            image_root=args.input,
            processed_root=args.processed_root,
            manifest_path=args.manifest,
            features_path=args.features,
            members=args.members,
            device=args.device,
        )
        print(f"Saved {len(features)} feature rows to {args.features}")
        print(f"Saved augmentation manifest to {args.manifest}")
        return

    result = verify_face(args.image, args.model, args.device)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
