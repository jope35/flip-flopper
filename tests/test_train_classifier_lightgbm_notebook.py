"""Regression tests for the classifier LightGBM training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def test_classifier_lightgbm_notebook_has_required_imports_and_settings() -> None:
    notebook_path = Path("src/jobs/train_classifier_lightgbm.ipynb")
    notebook = json.loads(notebook_path.read_text())
    source = "".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "from lightgbm import LGBMClassifier" in source
    assert "update_registered_converter(\n    LGBMClassifier" in source
    assert 'MODEL_VARIANT = "l"' in source
    assert "flip_flopper.classifier_training" in source
    assert "flip_flopper.sklearn_onnx" in source
    assert 'MODEL_ALIAS = "Champion"' in source
