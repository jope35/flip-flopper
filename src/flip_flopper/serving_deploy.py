"""Build and apply Databricks Model Serving endpoint config for two UC ONNX models."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

import mlflow
import requests
from mlflow.deployments import DatabricksDeploymentClient, get_deploy_client
from mlflow.tracking import MlflowClient


def normalize_registered_model_suffix(raw_value: str) -> str:
    """Match training notebooks: only alnum + underscore, lowercase."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", raw_value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        msg = "registered_model_suffix must contain at least one valid character"
        raise ValueError(msg)
    return cleaned


def registered_model_fqn(catalog: str, schema: str, normalized_suffix: str, letter: str) -> str:
    letter_l = letter.strip().lower()
    if letter_l not in {"a", "b"}:
        msg = "letter must be 'a' or 'b'"
        raise ValueError(msg)
    return f"{catalog.strip()}.{schema.strip()}.{normalized_suffix}_{letter_l}"


def build_serving_config(
    *,
    model_a_fqn: str,
    model_b_fqn: str,
    version_a: int | str,
    version_b: int | str,
    entity_name_a: str = "classifier_champion",
    entity_name_b: str = "classifier_challenger",
    traffic_pct_a: int = 50,
    traffic_pct_b: int = 50,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
) -> dict[str, Any]:
    if traffic_pct_a + traffic_pct_b != 100:
        msg = f"traffic percentages must sum to 100, got {traffic_pct_a} + {traffic_pct_b}"
        raise ValueError(msg)
    va, vb = str(int(version_a)), str(int(version_b))
    return {
        "served_entities": [
            {
                "name": entity_name_a,
                "entity_name": model_a_fqn,
                "entity_version": va,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
            },
            {
                "name": entity_name_b,
                "entity_name": model_b_fqn,
                "entity_version": vb,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
            },
        ],
        "traffic_config": {
            "routes": [
                {
                    "served_model_name": entity_name_a,
                    "traffic_percentage": traffic_pct_a,
                },
                {
                    "served_model_name": entity_name_b,
                    "traffic_percentage": traffic_pct_b,
                },
            ]
        },
    }


def _endpoint_exists(client: DatabricksDeploymentClient, name: str) -> bool:
    try:
        client.get_endpoint(name)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return False
        raise
    return True


def resolve_version_for_alias(
    mlflow_client: MlflowClient,
    model_fqn: str,
    alias: str,
) -> int:
    mv = mlflow_client.get_model_version_by_alias(model_fqn, alias.strip())
    return int(mv.version)


def deploy_endpoint(
    *,
    catalog: str,
    schema: str,
    registered_model_suffix: str,
    endpoint_name: str,
    alias_a: str = "Champion",
    alias_b: str = "Challenger",
    traffic_pct_a: int = 50,
    traffic_pct_b: int = 50,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
    entity_name_a: str = "classifier_champion",
    entity_name_b: str = "classifier_challenger",
) -> dict[str, Any]:
    """Create or update a serving endpoint. Caller must run on Databricks with UC + serving auth."""
    mlflow.set_registry_uri("databricks-uc")
    normalized = normalize_registered_model_suffix(registered_model_suffix)
    model_a = registered_model_fqn(catalog, schema, normalized, "a")
    model_b = registered_model_fqn(catalog, schema, normalized, "b")

    mclient = MlflowClient()
    v_a = resolve_version_for_alias(mclient, model_a, alias_a)
    v_b = resolve_version_for_alias(mclient, model_b, alias_b)

    config = build_serving_config(
        model_a_fqn=model_a,
        model_b_fqn=model_b,
        version_a=v_a,
        version_b=v_b,
        entity_name_a=entity_name_a,
        entity_name_b=entity_name_b,
        traffic_pct_a=traffic_pct_a,
        traffic_pct_b=traffic_pct_b,
        workload_size=workload_size,
        scale_to_zero=scale_to_zero,
    )

    dclient = get_deploy_client("databricks")
    if _endpoint_exists(dclient, endpoint_name):
        dclient.update_endpoint(endpoint_name, config=config)
        action = "updated"
    else:
        dclient.create_endpoint(name=endpoint_name, config=config)
        action = "created"

    return {
        "action": action,
        "endpoint_name": endpoint_name,
        "model_a": model_a,
        "model_b": model_b,
        "version_a": v_a,
        "version_b": v_b,
        "config": config,
    }


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    msg = f"expected boolean string, got {s!r}"
    raise argparse.ArgumentTypeError(msg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Create or update a Databricks serving endpoint for two UC ONNX classifiers."
    )
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--registered-model-suffix", required=True)
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--alias-a", default="Champion")
    p.add_argument("--alias-b", default="Challenger")
    p.add_argument("--traffic-a", type=int, default=50)
    p.add_argument("--traffic-b", type=int, default=50)
    p.add_argument("--workload-size", default="Small")
    p.add_argument("--scale-to-zero", type=_parse_bool, default=True)
    p.add_argument("--entity-name-a", default="classifier_champion")
    p.add_argument("--entity-name-b", default="classifier_challenger")
    ns = p.parse_args(argv)

    try:
        result = deploy_endpoint(
            catalog=ns.catalog,
            schema=ns.schema,
            registered_model_suffix=ns.registered_model_suffix,
            endpoint_name=ns.endpoint_name,
            alias_a=ns.alias_a,
            alias_b=ns.alias_b,
            traffic_pct_a=ns.traffic_a,
            traffic_pct_b=ns.traffic_b,
            workload_size=ns.workload_size,
            scale_to_zero=ns.scale_to_zero,
            entity_name_a=ns.entity_name_a,
            entity_name_b=ns.entity_name_b,
        )
    except Exception as e:  # noqa: BLE001
        print(f"deploy_endpoint failed: {e}", file=sys.stderr)
        return 1

    print(
        f"{result['action']} endpoint {result['endpoint_name']!r}: "
        f"{result['model_a']}@{result['version_a']} ({ns.alias_a}) / "
        f"{result['model_b']}@{result['version_b']} ({ns.alias_b})"
    )
    return 0
