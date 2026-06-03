"""Shared helpers for classifier training notebooks (UC naming, metrics, CV)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "label"


def quote_identifier(value: str) -> str:
    escaped = value.replace("`", "``")
    return f"`{escaped}`"


def qualified_table_name(catalog: str, schema: str, table_name: str) -> str:
    return ".".join(quote_identifier(part) for part in (catalog, schema, table_name))


def normalize_model_name(raw_value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", raw_value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        msg = "registered_model_suffix must contain at least one valid character"
        raise ValueError(msg)
    return cleaned


def resolve_experiment_path(raw_value: str) -> str:
    return raw_value if raw_value.startswith("/") else f"/Shared/{raw_value}"


def uc_registered_model_name(catalog: str, schema: str, registered_model_suffix: str, variant: str) -> str:
    normalized = normalize_model_name(registered_model_suffix)
    return f"{catalog}.{schema}.{normalized}_{variant}"


def load_training_frame(
    spark_session: Any,
    source_table: str,
    max_training_rows: int,
    *,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    spark_df = spark_session.table(source_table)
    row_count = spark_df.count()

    if row_count > max_training_rows:
        sampling_fraction = min(1.0, (max_training_rows / row_count) * 1.2)
        spark_df = spark_df.sample(False, sampling_fraction, seed=random_state).limit(max_training_rows)
        print(f"Sampled approximately {max_training_rows:,} rows from {row_count:,} available rows")
    else:
        print(f"Using all {row_count:,} rows from {source_table}")

    frame = spark_df.toPandas()
    if target_column not in frame.columns:
        msg = f"Expected target column '{target_column}' in {source_table}"
        raise ValueError(msg)

    return frame


def feature_columns_from_frame(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    columns = sorted(column for column in frame.columns if column != target_column)
    if not columns:
        msg = "Training data must contain at least one feature column"
        raise ValueError(msg)
    return columns


def prepare_numeric_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    labels = frame[target_column]
    return features, labels


def resolve_effective_cv_folds(requested_cv_folds: int, labels: pd.Series) -> int:
    min_class_count = int(labels.value_counts().min())
    if min_class_count < 2:
        msg = "Need at least 2 samples in each class for stratified evaluation"
        raise ValueError(msg)

    effective_cv_folds = min(requested_cv_folds, min_class_count)
    if effective_cv_folds < 2:
        msg = f"cv_folds must be >= 2 (requested {requested_cv_folds}, effective {effective_cv_folds})"
        raise ValueError(msg)
    if effective_cv_folds != requested_cv_folds:
        print(f"Adjusted cv_folds from {requested_cv_folds} to {effective_cv_folds} based on class balance")
    return effective_cv_folds


def summarize_cv_metrics(cv_results: dict[str, np.ndarray]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name in ("accuracy", "f1", "roc_auc"):
        scores = cv_results[f"test_{metric_name}"]
        summary[f"cv_{metric_name}_mean"] = float(np.mean(scores))
        summary[f"cv_{metric_name}_std"] = float(np.std(scores))
    return summary


def safe_roc_auc_score(labels: pd.Series, probabilities: np.ndarray) -> float:
    unique_labels = np.unique(labels.to_numpy())
    if len(unique_labels) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def evaluate_holdout(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_f1": float(f1_score(labels, predictions)),
        "test_roc_auc": safe_roc_auc_score(labels, probabilities),
    }
