"""Tabular preprocessing and validated merge for the product recommendation model.

Owner: Member 1.

The two raw files describe the same customers at two different grains, so joining
them directly multiplies rows.  This module:

1. normalises the ``A178`` style social identifier and the ``178`` style legacy
   identifier into one integer ``customer_id``;
2. aggregates the repeated social rows to exactly one record per customer;
3. builds *prior-only* transaction history features, so a row can never see its
   own purchase amount, rating or category (target leakage);
4. joins transaction rows to the aggregated social record with
   ``validate="many_to_one"`` and writes ``data/processed/merged_dataset.csv``.

Run from the repository root::

    python src/preprocess_tabular.py

Every function here is importable, so the notebook can reuse the same logic
instead of re-implementing it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_SOCIAL = REPO_ROOT / "data" / "raw" / "customer_social_profiles.csv"
RAW_TRANSACTIONS = REPO_ROOT / "data" / "raw" / "customer_transactions.csv"
MERGED_PATH = REPO_ROOT / "data" / "processed" / "merged_dataset.csv"

TARGET = "product_category"
ID_COLUMN = "customer_id"

#: Columns that stay in ``merged_dataset.csv`` (for auditing and for other
#: members) but must not be used as model inputs.  Documented as required by
#: README sections 9 and 14.
EXCLUDED_FROM_MODELING = {
    "transaction_id": "Row identifier; no predictive meaning.",
    "customer_id": "Identifier only. Used to group train/test splits and to validate the merge.",
    "purchase_date": "Raw date replaced by month, day_of_week and days_since_prev.",
    "purchase_amount": "Known only after the purchase, so using it would leak the target.",
    "customer_rating": "Recorded after the purchase, so using it would leak the target.",
}

MERGED_COLUMNS = [
    ID_COLUMN,
    "transaction_id",
    "purchase_date",
    TARGET,
    "purchase_amount",
    "customer_rating",
    "social_media_platform",
    "engagement_score",
    "purchase_interest_score",
    "review_sentiment",
    "n_social_records",
    "n_prior_transactions",
    "prior_avg_amount",
    "prior_avg_rating",
    "prior_mode_category",
    "days_since_prev",
    "month",
    "day_of_week",
]


def normalize_customer_id(series: pd.Series) -> pd.Series:
    """Return one consistently typed integer customer id.

    ``A178`` and ``178`` both become ``178``.  Leading/trailing whitespace and a
    lower-case ``a`` prefix are handled as well.  Values that are not numeric
    after cleaning raise instead of silently becoming ``NaN``.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"^[Aa]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="raise").astype("int64")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw CSVs without modifying the files on disk."""
    social = pd.read_csv(RAW_SOCIAL)
    transactions = pd.read_csv(RAW_TRANSACTIONS, parse_dates=["purchase_date"])
    return social, transactions


def social_with_customer_id(social: pd.DataFrame) -> pd.DataFrame:
    """Copy of the social table with the normalised ``customer_id`` added."""
    frame = social.copy()
    frame[ID_COLUMN] = normalize_customer_id(frame["customer_id_new"])
    return frame


def transactions_with_customer_id(transactions: pd.DataFrame) -> pd.DataFrame:
    """Copy of the transaction table with the normalised ``customer_id`` added."""
    frame = transactions.copy()
    frame[ID_COLUMN] = normalize_customer_id(frame["customer_id_legacy"])
    return frame


def audit_raw(social: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """Collect the data-quality facts required by README section 9, tasks 2-3."""
    social_ids = normalize_customer_id(social["customer_id_new"])
    txn_ids = normalize_customer_id(transactions["customer_id_legacy"])
    shared = set(social_ids) & set(txn_ids)
    return {
        "social_rows": int(len(social)),
        "social_columns": list(social.columns),
        "social_dtypes": {column: str(dtype) for column, dtype in social.dtypes.items()},
        "social_unique_customers": int(social_ids.nunique()),
        "social_rows_per_customer_max": int(social_ids.value_counts().max()),
        "social_exact_duplicate_rows": int(social.duplicated().sum()),
        "social_duplicate_identifier_rows": int(len(social_ids) - social_ids.nunique()),
        "social_missing_values": {column: int(count) for column, count in social.isna().sum().items()},
        "transaction_rows": int(len(transactions)),
        "transaction_columns": list(transactions.columns),
        "transaction_dtypes": {column: str(dtype) for column, dtype in transactions.dtypes.items()},
        "transaction_unique_customers": int(txn_ids.nunique()),
        "transaction_rows_per_customer_max": int(txn_ids.value_counts().max()),
        "transaction_exact_duplicate_rows": int(transactions.duplicated().sum()),
        "transaction_duplicate_identifier_rows": int(
            len(transactions) - transactions["transaction_id"].nunique()
        ),
        "transaction_missing_values": {
            column: int(count) for column, count in transactions.isna().sum().items()
        },
        "shared_customers": int(len(shared)),
        "transaction_ids_without_social": [int(v) for v in sorted(set(txn_ids) - shared)],
        "social_ids_without_transactions": [int(v) for v in sorted(set(social_ids) - shared)],
    }


def naive_many_to_many_row_count(social: pd.DataFrame, transactions: pd.DataFrame) -> int:
    """Row count produced by the *unvalidated* direct join.

    This is the mistake the README warns about: every social row for a customer
    is matched to every transaction row for that customer, so observations are
    duplicated and the model sees the same purchase more than once.
    """
    left = social_with_customer_id(social)[[ID_COLUMN]]
    right = transactions_with_customer_id(transactions)
    return int(left.merge(right, on=ID_COLUMN, how="inner").shape[0])


def aggregate_social(social: pd.DataFrame) -> pd.DataFrame:
    """Collapse the social table to exactly one row per customer.

    Numeric scores are averaged.  ``social_media_platform`` and
    ``review_sentiment`` are categorical, so the modal value is kept; ties are
    broken by the alphabetically first value, which keeps the result
    deterministic.  ``n_social_records`` records how many raw rows were merged.
    """
    frame = social_with_customer_id(social)
    aggregated = frame.groupby(ID_COLUMN, as_index=False).agg(
        engagement_score=("engagement_score", "mean"),
        purchase_interest_score=("purchase_interest_score", "mean"),
        social_media_platform=("social_media_platform", lambda values: values.mode().iat[0]),
        review_sentiment=("review_sentiment", lambda values: values.mode().iat[0]),
        n_social_records=("customer_id_new", "size"),
    )
    aggregated["n_social_records"] = aggregated["n_social_records"].astype("int64")
    return aggregated


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build transaction-level prediction rows with prior-only history features.

    Rows are sorted by customer and date so that "prior" always means earlier
    purchases.  Features use only information available *before* the current
    transaction:

    * ``n_prior_transactions`` - transaction frequency so far
    * ``prior_avg_amount``     - mean spend before this transaction
    * ``prior_avg_rating``     - mean rating before this transaction
    * ``prior_mode_category``  - most frequent previous category
    * ``days_since_prev``      - recency gap to the previous transaction
    * ``month``, ``day_of_week`` - calendar context of the current transaction

    The first transaction of every customer therefore has ``NaN`` history
    values.  That is correct, not a defect: nothing is known about that customer
    yet, and the model pipeline imputes those values.
    """
    frame = transactions_with_customer_id(transactions)
    frame = frame.sort_values(
        [ID_COLUMN, "purchase_date", "transaction_id"]
    ).reset_index(drop=True)

    grouped = frame.groupby(ID_COLUMN, sort=False)
    frame["n_prior_transactions"] = grouped.cumcount().astype("int64")
    frame["prior_avg_amount"] = grouped["purchase_amount"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    frame["prior_avg_rating"] = grouped["customer_rating"].transform(
        lambda values: values.shift(1).expanding().mean()
    )
    frame["days_since_prev"] = grouped["purchase_date"].diff().dt.days
    frame["month"] = frame["purchase_date"].dt.month.astype("int64")
    frame["day_of_week"] = frame["purchase_date"].dt.dayofweek.astype("int64")

    prior_mode = pd.Series(np.nan, index=frame.index, dtype="object")
    for _, group in frame.groupby(ID_COLUMN, sort=False):
        seen: Counter[str] = Counter()
        values: list[object] = []
        for category in group[TARGET]:
            if seen:
                # Same tie-break as Series.mode(): alphabetically smallest of the
                # most frequent previous categories, so the result is reproducible.
                top_count = max(seen.values())
                values.append(min(name for name, count in seen.items() if count == top_count))
            else:
                values.append(np.nan)
            seen[category] += 1
        prior_mode.loc[group.index] = values
    frame["prior_mode_category"] = prior_mode

    return frame


def build_merged_dataset(
    social: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the analysis-ready dataset: one row per transaction, one social record.

    The ``validate="many_to_one"`` argument is what prevents the row explosion:
    it raises if the aggregated social table is not unique per ``customer_id``.
    """
    if social is None or transactions is None:
        social, transactions = load_raw()

    social_lookup = aggregate_social(social)
    transaction_rows = build_transaction_features(transactions)

    merged = transaction_rows.merge(
        social_lookup,
        on=ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )

    missing = [column for column in MERGED_COLUMNS if column not in merged.columns]
    if missing:
        raise ValueError(f"Merged dataset is missing expected columns: {missing}")
    merged = merged[MERGED_COLUMNS].reset_index(drop=True)

    if not merged["transaction_id"].is_unique:
        raise ValueError("Merged dataset has repeated transaction_id values; merge inflated rows.")
    if merged[TARGET].isna().any():
        raise ValueError("Merged dataset contains missing target values.")
    return merged


def save_merged_dataset(merged: pd.DataFrame, path: Path = MERGED_PATH) -> Path:
    """Write ``merged_dataset.csv``, creating the processed directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return path


def main() -> None:
    social, transactions = load_raw()
    audit = audit_raw(social, transactions)

    social_lookup = aggregate_social(social)
    naive_rows = naive_many_to_many_row_count(social, transactions)
    merged = build_merged_dataset(social, transactions)

    print("=== Raw audit ===")
    for key, value in audit.items():
        if key in {"social_missing_values", "transaction_missing_values",
                   "social_columns", "transaction_columns",
                   "social_dtypes", "transaction_dtypes"}:
            continue
        print(f"{key}: {value}")

    print("\n=== Merge validation ===")
    print(f"transaction rows (pre-merge):        {len(transactions)}")
    print(f"social rows aggregated to customers: {len(social_lookup)}")
    print(f"unvalidated many-to-many row count:  {naive_rows}")
    print(f"validated merged row count:          {len(merged)}")
    print(f"inflation factor avoided:            {naive_rows / len(merged):.2f}x")
    print(f"unique customers in merged data:     {merged[ID_COLUMN].nunique()}")
    print(f"rows per customer:                   {len(merged) / merged[ID_COLUMN].nunique():.2f}")
    print(f"transactions dropped (no social row):{len(transactions) - len(merged)}")
    print(f"social-only customers (unused):      {len(audit['social_ids_without_transactions'])}")

    print("\n=== Columns excluded from modeling ===")
    for column, reason in EXCLUDED_FROM_MODELING.items():
        print(f"- {column}: {reason}")

    path = save_merged_dataset(merged)
    print(f"\nSaved {path.relative_to(REPO_ROOT)} with shape {merged.shape}")


if __name__ == "__main__":
    main()
