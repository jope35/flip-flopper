# tests/unit/test_train_catboost_demo_cli.py
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_train_catboost_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "src" / "jobs" / "train_catboost_demo.py"
    spec = importlib.util.spec_from_file_location("train_catboost_demo", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_args_required_catalog_schema() -> None:
    mod = _load_train_catboost_demo_module()
    ns = mod.parse_args(["--catalog", "main", "--schema", "serving"])
    assert ns.catalog == "main"
    assert ns.schema == "serving"
    assert ns.experiment_name == "/Shared/flip-flopper/catboost-demo"
    assert ns.registered_model_suffix == "flip_flopper_catboost_demo"


def test_parse_args_custom_experiment() -> None:
    mod = _load_train_catboost_demo_module()
    ns = mod.parse_args(
        [
            "--catalog",
            "c",
            "--schema",
            "s",
            "--experiment-name",
            "/Users/me@example.com/catboost-exp",
        ]
    )
    assert ns.experiment_name == "/Users/me@example.com/catboost-exp"
