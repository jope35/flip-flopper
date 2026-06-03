"""Regression tests for the classifier A training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def test_classifier_a_notebook_uses_shared_modules() -> None:
    notebook_path = Path("src/jobs/train_classifier_a.ipynb")
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(line for cell in notebook["cells"] for line in cell.get("source", []))

    assert "flip_flopper.classifier_training" in source
    assert "flip_flopper.sklearn_onnx" in source
    assert "def convert_pipeline_to_onnx" not in source
    assert "def quote_identifier" not in source
