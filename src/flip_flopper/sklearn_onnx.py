"""skl2onnx export and ONNX Runtime parity checks for sklearn pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

DEFAULT_TARGET_OPSET = 17


def convert_pipeline_to_onnx(
    model: Pipeline,
    feature_columns: list[str],
    *,
    target_opset: int | dict[str, int] = DEFAULT_TARGET_OPSET,
) -> Any:
    initial_types = [(column, FloatTensorType([None, 1])) for column in feature_columns]
    return to_onnx(
        model,
        initial_types=initial_types,
        target_opset=target_opset,
        options={id(model.named_steps["classifier"]): {"zipmap": False}},
    )


def run_onnx_proba(
    onnx_model: Any,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    onnx_inputs = {
        column: validation_frame[column].to_numpy(dtype=np.float32).reshape((-1, 1)) for column in feature_columns
    }
    session = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(None, onnx_inputs)

    probability_output = next(
        (
            output
            for output in outputs
            if getattr(output, "ndim", 0) == 2 and output.shape[0] == len(validation_frame) and output.shape[1] >= 2
        ),
        None,
    )
    if probability_output is None:
        msg = "Unable to find probability output from ONNX runtime session"
        raise ValueError(msg)

    return probability_output


def validate_onnx_probabilities(
    model: Pipeline,
    onnx_model: Any,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[float, np.ndarray]:
    expected = model.predict_proba(validation_frame)[:, 1]
    proba_2d = run_onnx_proba(onnx_model, validation_frame, feature_columns)
    onnx_positive = proba_2d[:, 1]
    max_abs_diff = float(np.max(np.abs(expected - onnx_positive)))
    return max_abs_diff, proba_2d


def build_per_column_onnx_input_example(
    validation_frame: pd.DataFrame, feature_columns: list[str]
) -> dict[str, np.ndarray]:
    return {column: validation_frame[[column]].to_numpy(dtype=np.float32) for column in feature_columns}
