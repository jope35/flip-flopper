"""Databricks job entry: deploy four UC ONNX models to one serving endpoint."""

from __future__ import annotations

from flip_flopper.serving_deploy import main

if __name__ == "__main__":
    code = main()
    if code:
        raise SystemExit(code)
