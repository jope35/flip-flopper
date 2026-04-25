"""Databricks job entry: deploy two UC ONNX models to one serving endpoint (A/B traffic)."""

from __future__ import annotations

from flip_flopper.serving_deploy import main

if __name__ == "__main__":
    raise SystemExit(main())
