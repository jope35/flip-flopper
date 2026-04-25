"""Unit tests for serving endpoint config helpers."""

from __future__ import annotations

import pytest
import requests

from flip_flopper.serving_deploy import (
    build_serving_config,
    normalize_registered_model_suffix,
    registered_model_fqn,
    _endpoint_exists,
)


def test_normalize_registered_model_suffix() -> None:
    assert normalize_registered_model_suffix("Flip-Flopper!") == "flip_flopper"


def test_normalize_registered_model_suffix_empty_raises() -> None:
    with pytest.raises(ValueError, match="registered_model_suffix"):
        normalize_registered_model_suffix("   !!!   ")


def test_registered_model_fqn() -> None:
    assert (
        registered_model_fqn("cat", "sch", "flip_flopper_classifier", "a")
        == "cat.sch.flip_flopper_classifier_a"
    )


def test_build_serving_config_shape() -> None:
    cfg = build_serving_config(
        model_a_fqn="c.s.m_a",
        model_b_fqn="c.s.m_b",
        version_a=3,
        version_b=7,
        traffic_pct_a=50,
        traffic_pct_b=50,
    )
    assert len(cfg["served_entities"]) == 2
    assert cfg["served_entities"][0]["entity_version"] == "3"
    assert cfg["served_entities"][1]["entity_version"] == "7"
    assert cfg["served_entities"][0]["name"] == "classifier_champion"
    routes = cfg["traffic_config"]["routes"]
    assert routes[0]["served_model_name"] == "classifier_champion"
    assert routes[0]["traffic_percentage"] == 50


def test_build_serving_config_traffic_must_sum_100() -> None:
    with pytest.raises(ValueError, match="sum to 100"):
        build_serving_config(
            model_a_fqn="a",
            model_b_fqn="b",
            version_a=1,
            version_b=2,
            traffic_pct_a=40,
            traffic_pct_b=50,
        )


def test_endpoint_exists_false_on_404() -> None:
    class FakeClient:
        def get_endpoint(self, name: str) -> None:
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError(response=resp)

    assert _endpoint_exists(FakeClient(), "x") is False  # type: ignore[arg-type]


def test_endpoint_exists_true() -> None:
    class FakeClient:
        def get_endpoint(self, name: str) -> dict[str, str]:
            return {"name": name}

    assert _endpoint_exists(FakeClient(), "ep") is True  # type: ignore[arg-type]
