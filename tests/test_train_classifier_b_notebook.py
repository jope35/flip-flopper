"""Regression tests for the classifier B training notebook."""

from __future__ import annotations

import json
from pathlib import Path


def test_classifier_b_logs_effective_svc_gamma() -> None:
    notebook_path = Path("src/jobs/train_classifier_b.ipynb")
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

    assert "fitted_svc._gamma" in source
    assert "fitted_svc.gamma_" not in source
