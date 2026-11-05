"""Product-category model: training, evaluation and the Member 4 handoff.

Owner: Member 1.

Trains a baseline and two comparison models on
``data/processed/merged_dataset.csv``, evaluates them on a customer-grouped
hold-out split, and saves the complete preprocessing/model pipeline with joblib.

Run from the repository root::

    python src/train_product_model.py

-----------------------------
Handoff contract for Member 4
-----------------------------
Input columns (all read from ``data/processed/merged_dataset.csv``).  A caller
only needs the social profile of the authenticated customer; the history columns
may be left unknown and are imputed.

=====================================  ==========================================
column                                 type / meaning
=====================================  ==========================================
``social_media_platform``              str, one of Facebook, Instagram, LinkedIn,
                                       TikTok, Twitter
``engagement_score``                   float, 50-99 social engagement
``purchase_interest_score``            float, 1.0-5.0 purchase interest
``review_sentiment``                   str, Negative | Neutral | Positive
``n_social_records``                   int, how many social rows were averaged
``n_prior_transactions``               int, purchases before this one (0 if new)
``prior_avg_amount``                   float, mean spend before this purchase (NaN if new)
``prior_avg_rating``                   float, mean rating before this purchase (NaN if new)
``prior_mode_category``                str, most frequent previous category (NaN if new)
``days_since_prev``                    float, days since previous purchase (NaN if new)
``month``                              int, 1-12
``day_of_week``                        int, 0=Monday .. 6=Sunday
=====================================  ==========================================

Saved pipeline: ``models/product_model.joblib`` (imputation, encoding, scaling
and the classifier, so it accepts the raw columns above).

One-prediction function::

    from train_product_model import predict_product
    predict_product({"social_media_platform": "Twitter", "engagement_score": 82,
                     "purchase_interest_score": 4.8,
                     "review_sentiment": "Neutral"})
    # -> [("Electronics", 0.28), ("Sports", 0.22), ("Books", 0.19)]

Product-label meanings: the five raw ``product_category`` values
(Books, Clothing, Electronics, Groceries, Sports) are kept unchanged, so the
label space is the same one Member 4 reads from the raw data.

Known limitations are printed by ``main()`` and written to
``outputs/metrics/product_model_metrics.json``.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Works both when this file is run directly (`python src/train_product_model.py`
# and `python src/app.py`) and when it is imported as `src.train_product_model`
# from the repository root, which is how Member 4 is likely to consume it.
try:
    from preprocess_tabular import (
        EXCLUDED_FROM_MODELING,
        ID_COLUMN,
        MERGED_PATH,
        REPO_ROOT,
        TARGET,
        build_merged_dataset,
    )
except ImportError:  # imported as part of the `src` package
    from src.preprocess_tabular import (
        EXCLUDED_FROM_MODELING,
        ID_COLUMN,
        MERGED_PATH,
        REPO_ROOT,
        TARGET,
        build_merged_dataset,
    )

# Use a non-interactive backend when run as a script, but never override the
# notebook's inline backend (importing this module from a notebook must not
# break figure display).
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "engagement_score",
    "purchase_interest_score",
    "n_social_records",
    "n_prior_transactions",
    "prior_avg_amount",
    "prior_avg_rating",
    "days_since_prev",
    "month",
    "day_of_week",
]
CATEGORICAL_FEATURES = [
    "social_media_platform",
    "review_sentiment",
    "prior_mode_category",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_PATH = REPO_ROOT / "models" / "product_model.joblib"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
METRICS_PATH = METRICS_DIR / "product_model_metrics.json"

COMPARISON_MODELS = ("logistic_regression", "random_forest")
MISSING_CATEGORY = "__missing__"


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the imputation/encoding/scaling steps.

    Imputation is mandatory, not cosmetic: the first transaction of every
    customer has ``NaN`` history values, and estimators such as Logistic
    Regression raise ``ValueError: Input X contains NaN`` without it.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False keeps the transformed matrix dense, which is
            # required by the default Random Forest and easier to inspect.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def build_models() -> dict[str, Pipeline]:
    """The comparison models requested by the README (task 14)."""
    return {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                # Single-threaded on purpose: with 117 rows parallelism buys
                # nothing, and the parallel tree-vote summation makes log loss
                # differ in the last decimal, so reruns would not be byte-stable.
                n_jobs=1,
            )
        ),
    }


def build_baseline() -> Pipeline:
    """Predict the most frequent training category for every row."""
    return Pipeline(
        steps=[("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))]
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def aligned_probabilities(model, features: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Expand ``predict_proba`` output to the full label set, in ``classes`` order."""
    raw = model.predict_proba(features)
    expanded = np.zeros((raw.shape[0], len(classes)), dtype=float)
    for position, label in enumerate(model.classes_):
        expanded[:, classes.index(label)] = raw[:, position]
    return expanded


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    classes: list[str],
) -> dict:
    """Accuracy, precision, recall, macro/weighted F1, confusion matrix, log loss."""
    metrics = {
        "n_rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)
        ),
        "labels": list(classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
        "log_loss": None,
    }
    if probabilities is not None:
        # Clip so a zero probability (always the case for the majority baseline)
        # stays finite instead of raising or returning inf.
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-15, None)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        metrics["log_loss"] = float(log_loss(y_true, clipped, labels=list(classes)))
    return metrics


def split_diagnostics(groups: pd.Series, train_idx, test_idx) -> dict:
    """Row and customer counts for one split, including any customer overlap."""
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    return {
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_train_customers": int(len(train_groups)),
        "n_test_customers": int(len(test_groups)),
        "customers_in_both_sets": int(len(train_groups & test_groups)),
    }


def evaluate_split(features, target, groups, train_idx, test_idx, classes) -> tuple[dict, dict, dict]:
    """Fit every model on one split and return metrics, split info and fitted models."""
    x_train, y_train = features.iloc[train_idx], target.iloc[train_idx]
    x_test, y_test = features.iloc[test_idx], target.iloc[test_idx]

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    baseline = build_baseline()
    baseline.fit(x_train, y_train)
    fitted["baseline_majority"] = baseline
    results["baseline_majority"] = evaluate_predictions(
        y_test,
        baseline.predict(x_test),
        aligned_probabilities(baseline, x_test, classes),
        classes,
    )

    # Rule baseline: repeat the customer's last observed category, falling back
    # to the most common training category for customers with no history.
    fallback = y_train.mode().iat[0]
    rule_predictions = x_test["prior_mode_category"].fillna(fallback).to_numpy()
    results["baseline_rule_prior_category"] = evaluate_predictions(
        y_test, rule_predictions, None, classes
    )

    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        results[name] = evaluate_predictions(
            y_test,
            pipeline.predict(x_test),
            aligned_probabilities(pipeline, x_test, classes),
            classes,
        )

    return results, split_diagnostics(groups, train_idx, test_idx), fitted


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_figure(figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_model_comparison(results: dict, chance_level: float) -> Path:
    """Macro F1 of every model on the honest, customer-grouped split."""
    names = list(results)
    scores = [results[name]["f1_macro"] for name in names]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(names, scores, color="#4C72B0")
    axis.axhline(
        chance_level,
        color="crimson",
        linestyle="--",
        label=f"random guessing ({chance_level:.2f})",
    )
    axis.set_ylabel("Macro F1")
    axis.set_ylim(0, 1)
    axis.set_title("Product model comparison (customer-grouped hold-out)")
    axis.set_xlabel("Model")
    axis.tick_params(axis="x", rotation=20)
    for bar, score in zip(bars, scores):
        axis.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center")
    axis.legend()
    return _save_figure(figure, "model_comparison_grouped_f1.png")


def plot_confusion_matrix(metrics: dict, model_name: str) -> Path:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["labels"]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted category")
    axis.set_ylabel("Actual category")
    axis.set_title(f"Confusion matrix - {model_name} (grouped hold-out)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    figure.colorbar(image, ax=axis)
    return _save_figure(figure, "confusion_matrix_grouped.png")


def plot_split_comparison(grouped: dict, random: dict) -> Path:
    """Show how a non-grouped split inflates the scores by leaking customers."""
    names = [name for name in COMPARISON_MODELS if name in grouped and name in random]
    positions = np.arange(len(names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        positions - width / 2,
        [grouped[name]["accuracy"] for name in names],
        width,
        label="customer-grouped split (honest)",
        color="#4C72B0",
    )
    axis.bar(
        positions + width / 2,
        [random[name]["accuracy"] for name in names],
        width,
        label="random row split (leaks customers)",
        color="#DD8452",
    )
    axis.set_xticks(positions, names)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Effect of grouping the train/test split by customer")
    axis.legend()
    return _save_figure(figure, "split_leakage_comparison.png")


def plot_permutation_importance(pipeline, features: pd.DataFrame, target: pd.Series) -> tuple[Path, dict]:
    """Model-agnostic feature importance, measured on held-out rows only.

    Shuffling a column is only informative on data the model did not train on:
    on training rows an over-fitted model loses accuracy for *every* column,
    including pure noise.  With 30 hold-out rows these values are noisy, so they
    are reported as a sanity check rather than as a ranking of true drivers.
    """
    importance = permutation_importance(
        pipeline,
        features,
        target,
        scoring="f1_macro",
        n_repeats=30,
        random_state=RANDOM_STATE,
    )
    order = np.argsort(importance.importances_mean)[::-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(
        [features.columns[i] for i in order][::-1],
        importance.importances_mean[order][::-1],
        color="#55A868",
    )
    axis.set_xlabel("Mean decrease in macro F1 when the column is shuffled")
    axis.set_title("Permutation importance on grouped hold-out (noisy, 30 rows)")
    path = _save_figure(figure, "permutation_importance.png")
    values = {
        features.columns[i]: float(importance.importances_mean[i])
        for i in reversed(order)
    }
    return path, values


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Read ``merged_dataset.csv``, rebuilding it if it is missing."""
    if MERGED_PATH.exists():
        return pd.read_csv(MERGED_PATH)
    return build_merged_dataset()


def run_training(merged: pd.DataFrame) -> dict:
    """Train, evaluate, save the pipeline and return everything for the notebook."""
    features = merged[MODEL_FEATURES]
    target = merged[TARGET]
    groups = merged[ID_COLUMN]
    classes = sorted(target.unique().tolist())

    grouped_splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(grouped_splitter.split(features, target, groups))
    grouped_results, grouped_info, grouped_fitted = evaluate_split(
        features, target, groups, train_idx, test_idx, classes
    )
    if grouped_info["customers_in_both_sets"]:
        raise ValueError("Grouped split leaked customers across train and test.")

    random_train_idx, random_test_idx = train_test_split(
        np.arange(len(merged)), test_size=0.25, random_state=RANDOM_STATE, stratify=target
    )
    random_results, random_info, _ = evaluate_split(
        features, target, groups, random_train_idx, random_test_idx, classes
    )

    # Pick the best comparison model on the honest split, then refit it on all
    # rows so the saved artifact uses every customer.  The reported metrics stay
    # the hold-out metrics above.
    best_name = max(COMPARISON_MODELS, key=lambda name: grouped_results[name]["f1_macro"])
    best_pipeline = build_models()[best_name]
    best_pipeline.fit(features, target)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    load_model.cache_clear()

    chance_level = 1.0 / len(classes)
    figure_paths = [
        plot_model_comparison(grouped_results, chance_level),
        plot_confusion_matrix(grouped_results[best_name], best_name),
        plot_split_comparison(grouped_results, random_results),
    ]
    importance_path, importance_values = plot_permutation_importance(
        grouped_fitted[best_name], features.iloc[test_idx], target.iloc[test_idx]
    )
    figure_paths.append(importance_path)

    results = {
        "target": TARGET,
        "classes": classes,
        "chance_level": chance_level,
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": EXCLUDED_FROM_MODELING,
        "dataset": {
            "rows": int(len(merged)),
            "customers": int(groups.nunique()),
            "rows_per_customer": float(len(merged) / groups.nunique()),
            "category_counts": {k: int(v) for k, v in target.value_counts().sort_index().items()},
        },
        "grouped_split": {"diagnostics": grouped_info, "metrics": grouped_results},
        "random_split": {"diagnostics": random_info, "metrics": random_results},
        "best_model": best_name,
        "saved_pipeline": str(MODEL_PATH.relative_to(REPO_ROOT)),
        "figures": [str(path.relative_to(REPO_ROOT)) for path in figure_paths],
        "permutation_importance": importance_values,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(jsonable(results), indent=2), encoding="utf-8")
    return results


def jsonable(value):
    """Convert numpy scalars/arrays so the metrics can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# --------------------------------------------------------------------------- #
# Inference (the Member 4 handoff function)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Load the saved pipeline once per process."""
    return joblib.load(path)


def predict_product(row, top_k: int = 3) -> list[tuple[str, float]]:
    """Rank the product categories a single customer is most likely to buy.

    ``row`` is a mapping, a pandas ``Series`` or a one-row ``DataFrame`` holding
    any of the input columns listed in the module docstring.  Missing columns are
    imputed: unknown numeric history becomes the training median, unknown
    categorical values become an all-zero encoding.

    Returns a list of ``(category, probability)`` pairs, highest probability
    first, so the caller can display the top recommendation or several options.
    """
    pipeline = load_model()
    if isinstance(row, pd.Series):
        frame = row.to_frame().T
    elif isinstance(row, pd.DataFrame):
        frame = row.copy()
    else:
        frame = pd.DataFrame([row])

    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].where(frame[column].notna(), MISSING_CATEGORY)
        else:
            frame[column] = MISSING_CATEGORY
        frame[column] = frame[column].astype(str)

    probabilities = pipeline.predict_proba(frame[MODEL_FEATURES])[0]
    ranking = np.argsort(probabilities)[::-1][: max(1, top_k)]
    return [(str(pipeline.classes_[index]), float(probabilities[index])) for index in ranking]


def main() -> None:
    merged = load_dataset()
    results = run_training(merged)

    print("=== Dataset ===")
    for key, value in results["dataset"].items():
        print(f"{key}: {value}")

    print("\n=== Customer-grouped hold-out (honest evaluation) ===")
    print(results["grouped_split"]["diagnostics"])
    header = f"{'model':34s}{'acc':>8s}{'prec':>8s}{'rec':>8s}{'F1 mac':>9s}{'F1 wtd':>9s}{'logloss':>10s}"
    print(header)
    for name, metrics in results["grouped_split"]["metrics"].items():
        loss = "n/a" if metrics["log_loss"] is None else f"{metrics['log_loss']:.2f}"
        print(
            f"{name:34s}{metrics['accuracy']:8.3f}{metrics['precision_macro']:8.3f}"
            f"{metrics['recall_macro']:8.3f}{metrics['f1_macro']:9.3f}"
            f"{metrics['f1_weighted']:9.3f}{loss:>10s}"
        )

    print("\n=== Random row split (leaks customers, included to show the bias) ===")
    print(results["random_split"]["diagnostics"])
    for name, metrics in results["random_split"]["metrics"].items():
        print(f"{name:34s}{metrics['accuracy']:8.3f}{metrics['f1_macro']:9.3f}")

    print(f"\nChance level for {len(results['classes'])} classes: {results['chance_level']:.2f}")
    print(f"Best comparison model: {results['best_model']}")
    print(f"Saved pipeline: {results['saved_pipeline']}")
    print(f"Metrics: {METRICS_PATH.relative_to(REPO_ROOT)}")
    print(f"Figures: {', '.join(results['figures'])}")

    print("\n=== Known limitations (handoff to Member 4) ===")
    print(
        f"1. Only {results['dataset']['rows']} usable rows for {results['dataset']['customers']} "
        "customers; 33 transactions had no social profile and 23 social customers never bought."
    )
    print(
        "2. Accuracy is close to chance. The social columns carry almost no information about "
        "product category, so the model is a demonstration of the pipeline, not a working recommender."
    )
    print(
        "3. The random-split numbers are higher only because the same customer appears in both "
        "sets; the grouped numbers are the ones to report."
    )
    print(
        "4. Prior-only history features are NaN for a customer's first transaction, so a brand-new "
        "customer gets the weakest predictions."
    )
    print(
        "5. Labels are the five raw categories; the pipeline cannot recommend a product outside them."
    )
    print(
        "6. Probabilities are uncalibrated. With 117 rows a single prediction can look confident "
        "(0.88 in one test) while overall accuracy is at chance, so show the ranked list and the "
        "measured accuracy instead of presenting a probability as a certainty."
    )

    print("\n=== Example prediction ===")
    example = {
        "social_media_platform": "Twitter",
        "engagement_score": 82,
        "purchase_interest_score": 4.8,
        "review_sentiment": "Neutral",
    }
    for category, probability in predict_product(example):
        print(f"  {category:12s}{probability:.3f}")


if __name__ == "__main__":
    main()
