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


def test_registered_model_fqn_x_for_catboost() -> None:
    assert (
        registered_model_fqn("cat", "sch", "flip_flopper_classifier", "x")
        == "cat.sch.flip_flopper_classifier_x"
    )


def test_registered_model_fqn_l_for_lightgbm() -> None:
    assert (
        registered_model_fqn("cat", "sch", "flip_flopper_classifier", "l")
        == "cat.sch.flip_flopper_classifier_l"
    )


def test_build_serving_config_shape() -> None:
    cfg = build_serving_config(
        model_a_fqn="c.s.m_a",
        model_b_fqn="c.s.m_b",
        model_c_fqn="c.s.m_x",
        model_d_fqn="c.s.m_l",
        version_a=3,
        version_b=7,
        version_c=11,
        version_d=13,
        traffic_pct_a=25,
        traffic_pct_b=25,
        traffic_pct_c=25,
        traffic_pct_d=25,
    )
    assert len(cfg["served_entities"]) == 4
    assert cfg["served_entities"][0]["entity_version"] == "3"
    assert cfg["served_entities"][1]["entity_version"] == "7"
    assert cfg["served_entities"][2]["entity_version"] == "11"
    assert cfg["served_entities"][3]["entity_version"] == "13"
    assert cfg["served_entities"][0]["name"] == "classifier_champion"
    assert cfg["served_entities"][3]["name"] == "classifier_lightgbm"
    routes = cfg["traffic_config"]["routes"]
    assert routes[0]["served_model_name"] == "classifier_champion"
    assert routes[0]["traffic_percentage"] == 25


def test_build_serving_config_four_entities() -> None:
    cfg = build_serving_config(
        model_a_fqn="c.s.m_a",
        model_b_fqn="c.s.m_b",
        model_c_fqn="c.s.m_x",
        model_d_fqn="c.s.m_l",
        version_a=1,
        version_b=2,
        version_c=3,
        version_d=4,
    )
    entities = cfg["served_entities"]
    routes = cfg["traffic_config"]["routes"]
    assert [e["entity_name"] for e in entities] == [
        "c.s.m_a",
        "c.s.m_b",
        "c.s.m_x",
        "c.s.m_l",
    ]
    assert [r["served_model_name"] for r in routes] == [
        "classifier_champion",
        "classifier_challenger",
        "classifier_catboost",
        "classifier_lightgbm",
    ]
    assert sum(r["traffic_percentage"] for r in routes) == 100


def test_build_serving_config_traffic_must_sum_100() -> None:
    with pytest.raises(ValueError, match="sum to 100"):
        build_serving_config(
            model_a_fqn="a",
            model_b_fqn="b",
            model_c_fqn="c",
            model_d_fqn="d",
            version_a=1,
            version_b=2,
            version_c=3,
            version_d=4,
            traffic_pct_a=40,
            traffic_pct_b=40,
            traffic_pct_c=10,
            traffic_pct_d=5,
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
