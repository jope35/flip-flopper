"""Tests for scripts/bundle-run/update_endpoint_split.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/bundle-run/update_endpoint_split.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("update_endpoint_split", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_script_module()


def test_validate_traffic_split_accepts_equal_three_way(mod) -> None:
    split = mod.validate_traffic_split(34, 33, 33)
    assert split.as_tuple() == (34, 33, 33, 0)


def test_validate_traffic_split_rejects_bad_total(mod) -> None:
    with pytest.raises(ValueError, match="sum to 100"):
        mod.validate_traffic_split(50, 25, 20)


def test_validate_traffic_split_rejects_out_of_range(mod) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        mod.validate_traffic_split(101, 0, -1)


def test_format_job_params(mod) -> None:
    split = mod.TrafficSplit(champion=50, challenger=25, catboost=25)
    assert mod.format_job_params(split) == "traffic_a=50,traffic_b=25,traffic_c=25"


def test_build_validate_command(mod) -> None:
    cmd = mod.build_validate_command(target="dev", profile="my-profile")
    assert cmd == ["databricks", "bundle", "validate", "-t", "dev", "-p", "my-profile"]


def test_build_deploy_command(mod) -> None:
    cmd = mod.build_deploy_command(target="prod", profile=None)
    assert cmd == ["databricks", "bundle", "deploy", "--auto-approve", "-t", "prod"]


def test_build_run_command(mod) -> None:
    split = mod.TrafficSplit(champion=100, challenger=0, catboost=0)
    cmd = mod.build_run_command(target="dev", profile=None, split=split)
    assert cmd == [
        "databricks",
        "bundle",
        "run",
        "-t",
        "dev",
        "--params",
        "traffic_a=100,traffic_b=0,traffic_c=0",
        "deploy_classifier_endpoint",
    ]


def test_build_bundle_pipeline_commands_order(mod) -> None:
    split = mod.TrafficSplit(champion=34, challenger=33, catboost=33)
    pipeline = mod.build_bundle_pipeline_commands(target="dev", profile=None, split=split)
    assert len(pipeline) == 3
    assert pipeline[0][2] == "validate"
    assert pipeline[1][2] == "deploy"
    assert pipeline[2][2] == "run"
    assert pipeline[2][-1] == "deploy_classifier_endpoint"
    assert "--no-wait" not in " ".join(" ".join(c) for c in pipeline)


def test_resolve_split_from_args_all_or_none(mod) -> None:
    assert mod.resolve_split_from_args(None, None, None) is None
    split = mod.resolve_split_from_args(50, 25, 25)
    assert split is not None
    assert split.as_tuple() == (50, 25, 25, 0)


def test_resolve_split_from_args_partial_raises(mod) -> None:
    with pytest.raises(ValueError, match="all of"):
        mod.resolve_split_from_args(50, 25, None)


def test_resolve_split_from_env(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLIP_FLOPPER_TRAFFIC_A", "50")
    monkeypatch.setenv("FLIP_FLOPPER_TRAFFIC_B", "25")
    monkeypatch.setenv("FLIP_FLOPPER_TRAFFIC_C", "25")
    split = mod.resolve_split_from_env()
    assert split is not None
    assert split.as_tuple() == (50, 25, 25, 0)
