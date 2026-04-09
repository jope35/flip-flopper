from pathlib import Path

from flip_flopper_ab_test.config import load_config


def test_load_root_app_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "config" / "app.yaml")
    assert cfg["endpoint_name"] == "flip-flopper-dev"
    assert cfg["control"]["name"] == "control"
    assert cfg["challenger"]["name"] == "challenger"
