from pathlib import Path

from flip_flopper_ab_test.config import load_config
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


def make_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "config" / "app.yaml")


def test_build_endpoint_core_config_contains_two_routes() -> None:
    payload = build_endpoint_core_config(make_config(), control_percent=90, challenger_percent=10)

    assert [entity["name"] for entity in payload["served_entities"]] == ["control", "challenger"]
    assert payload["traffic_config"]["routes"] == [
        {"served_model_name": "control", "traffic_percentage": 90},
        {"served_model_name": "challenger", "traffic_percentage": 10},
    ]


def test_build_endpoint_core_config_rejects_invalid_total() -> None:
    try:
        build_endpoint_core_config(make_config(), control_percent=80, challenger_percent=30)
    except ValueError as exc:
        assert "sum to 100" in str(exc)
    else:
        raise AssertionError("expected ValueError")
