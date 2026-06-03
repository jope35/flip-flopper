"""Regression tests for the classifier endpoint smoke notebook."""

from __future__ import annotations

import json
from pathlib import Path


def test_call_classifier_endpoint_notebook_contract() -> None:
    notebook_path = Path("src/jobs/call_classifier_endpoint.ipynb")
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )

    assert "flip_flopper_classifier_endpoint" in source
    assert "dataframe_records" in source
    assert "N_FEATURES = 33" in source
    assert 'f"feature_{i:03d}"' in source
