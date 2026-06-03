"""CatBoost native ONNX export and validation helpers."""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline


def transform_features(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Apply the fitted sklearn preprocess step (median imputation, column order)."""
    transformed = pipeline.named_steps["preprocess"].transform(frame)
    return np.asarray(transformed, dtype=np.float32)


def export_catboost_classifier_to_onnx(classifier: CatBoostClassifier) -> onnx.ModelProto:
    """Export a fitted CatBoostClassifier to ONNX via a temporary file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.onnx"
        classifier.save_model(
            path,
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.catboost",
                "onnx_model_version": 1,
                "onnx_doc_string": "catboost classifier",
                "onnx_graph_name": "CatBoostClassifier",
            },
        )
        return onnx.load(path)


def onnx_input_name(onnx_model: onnx.ModelProto) -> str:
    return onnx_model.graph.input[0].name


def onnx_probabilities_output_name(session: ort.InferenceSession) -> str:
    for output in session.get_outputs():
        if output.name == "probabilities":
            return output.name
    outputs = session.get_outputs()
    if len(outputs) == 1:
        return outputs[0].name
    msg = "Unable to find ONNX probabilities output"
    raise ValueError(msg)


def probabilities_to_dense_2d(probabilities_output: Any) -> np.ndarray:
    """Normalize CatBoost classifier ONNX probabilities to shape (n_samples, n_classes)."""
    if isinstance(probabilities_output, np.ndarray):
        array = probabilities_output
        if array.ndim == 1:
            return np.column_stack([1.0 - array, array]).astype(np.float32)
        if array.ndim == 2:
            return array.astype(np.float32)
        msg = f"Unexpected probability array shape: {array.shape}"
        raise ValueError(msg)

    if isinstance(probabilities_output, list | tuple):
        rows: list[list[float]] = []
        for row_map in probabilities_output:
            if isinstance(row_map, dict):
                keys = sorted(row_map.keys(), key=lambda key: int(key))
                rows.append([float(row_map[key]) for key in keys])
            else:
                rows.append([float(value) for value in row_map])
        return np.asarray(rows, dtype=np.float32)

    msg = f"Unsupported ONNX probability output type: {type(probabilities_output)!r}"
    raise ValueError(msg)


def run_onnx_proba(
    onnx_model: onnx.ModelProto,
    preprocessed_features: np.ndarray,
) -> np.ndarray:
    """Run ONNX Runtime on preprocessed features; return dense class probabilities."""
    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_key = session.get_inputs()[0].name
    output_key = onnx_probabilities_output_name(session)
    batch = np.asarray(preprocessed_features, dtype=np.float32)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)
    raw = session.run([output_key], {input_key: batch})[0]
    return probabilities_to_dense_2d(raw)


def validate_onnx_probabilities(
    model: Pipeline,
    onnx_model: onnx.ModelProto,
    validation_frame: pd.DataFrame,
) -> tuple[float, np.ndarray]:
    expected = model.predict_proba(validation_frame)[:, 1]
    preprocessed = transform_features(model, validation_frame)
    proba_2d = run_onnx_proba(onnx_model, preprocessed)
    onnx_positive = proba_2d[:, 1]
    max_abs_diff = float(np.max(np.abs(expected - onnx_positive)))
    return max_abs_diff, proba_2d


def build_onnx_input_example(preprocessed_features: np.ndarray) -> np.ndarray:
    """MLflow input example: dense matrix consumed by the CatBoost ONNX graph."""
    batch = np.asarray(preprocessed_features, dtype=np.float32)
    if batch.ndim == 1:
        return batch.reshape(1, -1)
    return batch
