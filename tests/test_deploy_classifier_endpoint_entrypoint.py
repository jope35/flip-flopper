"""Regression tests for the Databricks deploy job entrypoint."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest


ENTRYPOINT = Path("src/jobs/deploy_classifier_endpoint.py")


def test_deploy_entrypoint_returns_normally_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flip_flopper.serving_deploy.main", lambda: 0)

    runpy.run_path(ENTRYPOINT, run_name="__main__")


def test_deploy_entrypoint_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("flip_flopper.serving_deploy.main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(ENTRYPOINT, run_name="__main__")

    assert exc_info.value.code == 7
