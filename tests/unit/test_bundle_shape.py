# tests/unit/test_bundle_shape.py
from pathlib import Path


def test_train_catboost_demo_script_exists() -> None:
    assert Path("src/jobs/train_catboost_demo.py").exists()
