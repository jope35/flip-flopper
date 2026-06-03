"""Tests for skl2onnx pipeline export and ONNX Runtime parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from flip_flopper.sklearn_onnx import (
    build_per_column_onnx_input_example,
    convert_pipeline_to_onnx,
    run_onnx_proba,
    validate_onnx_probabilities,
)


def _build_rf_pipeline(feature_columns: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    classifier = RandomForestClassifier(
        n_estimators=5,
        n_jobs=1,
        random_state=42,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def test_sklearn_onnx_export_and_probability_parity() -> None:
    rng = np.random.default_rng(42)
    n_rows = 200
    features = pd.DataFrame(
        {
            "f0": rng.normal(size=n_rows),
            "f1": rng.normal(size=n_rows),
            "f2": rng.normal(size=n_rows),
        }
    )
    labels = (features["f0"] + features["f1"] > 0).astype(int)
    feature_columns = sorted(features.columns)

    x_train, x_test, y_train, _y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    pipeline = _build_rf_pipeline(feature_columns)
    pipeline.fit(x_train, y_train)

    onnx_model = convert_pipeline_to_onnx(pipeline, feature_columns)
    validation_batch = x_test.head(32).copy()
    example = build_per_column_onnx_input_example(validation_batch, feature_columns)
    assert set(example) == set(feature_columns)

    proba_2d = run_onnx_proba(onnx_model, validation_batch, feature_columns)
    assert proba_2d.shape == (len(validation_batch), 2)

    max_abs_diff, parity_proba = validate_onnx_probabilities(pipeline, onnx_model, validation_batch, feature_columns)
    assert parity_proba.shape == proba_2d.shape
    assert max_abs_diff < 1e-5
