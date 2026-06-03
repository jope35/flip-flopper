"""Tests for CatBoost native ONNX export and probability parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from flip_flopper.catboost_onnx import (
    build_onnx_input_example,
    export_catboost_classifier_to_onnx,
    onnx_input_name,
    onnx_probabilities_output_name,
    probabilities_to_dense_2d,
    run_onnx_proba,
    transform_features,
    validate_onnx_probabilities,
)


def _build_pipeline(feature_columns: list[str]) -> Pipeline:
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
    classifier = CatBoostClassifier(
        iterations=5,
        loss_function="Logloss",
        random_seed=42,
        thread_count=1,
        verbose=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )


def test_catboost_onnx_export_and_probability_parity() -> None:
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

    pipeline = _build_pipeline(feature_columns)
    pipeline.fit(x_train, y_train)

    onnx_model = export_catboost_classifier_to_onnx(pipeline.named_steps["classifier"])
    assert onnx_input_name(onnx_model)

    validation_batch = x_test.head(32).copy()
    preprocessed = transform_features(pipeline, validation_batch)
    example = build_onnx_input_example(preprocessed)
    assert example.ndim == 2
    assert example.shape[0] == len(validation_batch)

    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    assert onnx_probabilities_output_name(session) == "probabilities"

    proba_2d = run_onnx_proba(onnx_model, preprocessed)
    assert proba_2d.shape == (len(validation_batch), 2)

    max_abs_diff, parity_proba = validate_onnx_probabilities(
        pipeline, onnx_model, validation_batch
    )
    assert parity_proba.shape == proba_2d.shape
    assert max_abs_diff < 1e-5


def test_probabilities_to_dense_2d_from_map_sequence() -> None:
    maps = [{0: 0.2, 1: 0.8}, {0: 0.6, 1: 0.4}]
    dense = probabilities_to_dense_2d(maps)
    np.testing.assert_allclose(dense, [[0.2, 0.8], [0.6, 0.4]], rtol=1e-6)


def test_catboost_notebook_uses_native_onnx_export() -> None:
    notebook_path = Path("src/jobs/train_classifier_catboost.ipynb")
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "export_catboost_classifier_to_onnx" in source
    assert "flip_flopper.catboost_onnx" in source
    assert "skl2onnx" not in source
    assert "convert_pipeline_to_onnx" not in source
