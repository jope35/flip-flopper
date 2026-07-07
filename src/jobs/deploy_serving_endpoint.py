# Databricks notebook source
# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("model_catalog", "")
dbutils.widgets.text("model_schema", "")
dbutils.widgets.text("logistic_model_name", "")
dbutils.widgets.text("lightgbm_model_name", "")
dbutils.widgets.text("xgboost_model_name", "")
dbutils.widgets.text("pytorch_mlp_model_name", "")
dbutils.widgets.text("sparkml_rf_model_name", "")
dbutils.widgets.text("endpoint_name", "")

model_catalog = dbutils.widgets.get("model_catalog").strip()
model_schema = dbutils.widgets.get("model_schema").strip()
logistic_model_name = dbutils.widgets.get("logistic_model_name").strip()
lightgbm_model_name = dbutils.widgets.get("lightgbm_model_name").strip()
xgboost_model_name = dbutils.widgets.get("xgboost_model_name").strip()
pytorch_mlp_model_name = dbutils.widgets.get("pytorch_mlp_model_name").strip()
sparkml_rf_model_name = dbutils.widgets.get("sparkml_rf_model_name").strip()
endpoint_name = dbutils.widgets.get("endpoint_name").strip()

if not all(
    [
        model_catalog,
        model_schema,
        logistic_model_name,
        lightgbm_model_name,
        xgboost_model_name,
        pytorch_mlp_model_name,
        sparkml_rf_model_name,
        endpoint_name,
    ]
):
    raise ValueError(
        "model_catalog, model_schema, logistic_model_name, lightgbm_model_name, "
        "xgboost_model_name, pytorch_mlp_model_name, sparkml_rf_model_name, and endpoint_name must all be non-empty"
    )

MODELS = [
    ("logistic_regression", logistic_model_name),
    ("lightgbm", lightgbm_model_name),
    ("xgboost", xgboost_model_name),
    ("pytorch_mlp", pytorch_mlp_model_name),
    ("random_forest", sparkml_rf_model_name),
]
model_paths = {name: f"{model_catalog}.{model_schema}.{model_name}" for name, model_name in MODELS}

print(f"Endpoint: {endpoint_name}")
for name, path in model_paths.items():
    print(f"{name}: {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve latest model versions

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


def latest_version(model_name: str) -> int:
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for {model_name}")
    return max(int(v.version) for v in versions)


versions = {name: latest_version(path) for name, path in model_paths.items()}
for name, version in versions.items():
    print(f"{name}: version {version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build endpoint config

# COMMAND ----------

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    ServingModelWorkloadType,
    TrafficConfig,
)


def even_traffic_routes(entities):
    n = len(entities)
    base, remainder = divmod(100, n)
    return [
        Route(
            served_entity_name=entity.name,
            traffic_percentage=base + (1 if i < remainder else 0),
        )
        for i, entity in enumerate(entities)
    ]


served_entities = [
    ServedEntityInput(
        name=name,
        entity_name=model_paths[name],
        entity_version=str(versions[name]),
        workload_type=ServingModelWorkloadType.CPU,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
    for name, _ in MODELS
]

traffic_config = TrafficConfig(routes=even_traffic_routes(served_entities))

config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=served_entities,
    traffic_config=traffic_config,
)

print(f"Traffic split: {len(served_entities)} models via even_traffic_routes")
print("Scale-to-zero: enabled (required by workspace)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update endpoint

# COMMAND ----------

import time
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import OperationFailed
from databricks.sdk.errors.platform import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    EndpointStateConfigUpdate,
    EndpointStateReady,
)

w = WorkspaceClient()
DEPLOY_TIMEOUT = timedelta(minutes=30)
POLL_INTERVAL_SECONDS = 5


def endpoint_has_quota_failure(name: str) -> bool:
    try:
        ep = w.serving_endpoints.get(name=name)
    except ResourceDoesNotExist:
        return False
    active_config = ep.pending_config or ep.config
    if active_config is None:
        return False
    for entity in active_config.served_entities or []:
        state = entity.state
        message = (state.deployment_state_message or "").lower() if state else ""
        if "quota exceeded" in message:
            return True
    return False


def wait_for_endpoint_deleted(name: str, timeout: timedelta = timedelta(minutes=2)) -> None:
    deadline = time.time() + timeout.total_seconds()
    while time.time() < deadline:
        try:
            w.serving_endpoints.get(name=name)
        except ResourceDoesNotExist:
            return
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"Endpoint {name} still exists after delete")


def wait_for_endpoint_ready(name: str, timeout: timedelta = DEPLOY_TIMEOUT):
    deadline = time.time() + timeout.total_seconds()
    while time.time() < deadline:
        ep = w.serving_endpoints.get(name=name)
        config_update = ep.state.config_update if ep.state else None
        ready = ep.state.ready if ep.state else None

        if config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
            raise RuntimeError(
                f"Endpoint {name} is in UPDATE_FAILED. Delete it in the Serving UI "
                f"(or `databricks serving-endpoints delete {name}`), recreate it there, "
                "then re-run this job to verify."
            )
        if ready == EndpointStateReady.READY and config_update == EndpointStateConfigUpdate.NOT_UPDATING:
            return ep
        time.sleep(POLL_INTERVAL_SECONDS)
    raise RuntimeError(f"Timed out waiting for endpoint {name} to become READY")


def rollout_entities_incrementally(name: str) -> None:
    # ponytail: 1→N rollout when bulk create hits quota; slower but fits free-tier limits
    first_entities = served_entities[:1]
    first_config = EndpointCoreConfigInput(
        name=name,
        served_entities=first_entities,
        traffic_config=TrafficConfig(routes=even_traffic_routes(first_entities)),
    )
    print(f"Creating endpoint with first model: {first_entities[0].name}")
    w.serving_endpoints.create_and_wait(
        name=name,
        config=first_config,
        timeout=DEPLOY_TIMEOUT,
    )

    for count in range(2, len(served_entities) + 1):
        step_entities = served_entities[:count]
        step_routes = traffic_config.routes if count == len(served_entities) else even_traffic_routes(step_entities)
        print(f"Adding models: {[entity.name for entity in step_entities]}")
        w.serving_endpoints.update_config_and_wait(
            name=name,
            served_entities=step_entities,
            traffic_config=TrafficConfig(routes=step_routes),
            timeout=DEPLOY_TIMEOUT,
        )


def create_endpoint(name: str) -> None:
    """Create all models in one shot (~3x faster than incremental rollout)."""
    print(f"Creating endpoint with all models: {name}")
    try:
        w.serving_endpoints.create_and_wait(
            name=name,
            config=config,
            timeout=DEPLOY_TIMEOUT,
        )
    except OperationFailed as exc:
        message = str(exc).lower()
        if "update_failed" not in message and "quota exceeded" not in message and not endpoint_has_quota_failure(name):
            raise
        print("Bulk create failed; falling back to incremental rollout")
        try:
            failed = w.serving_endpoints.get(name=name)
            if failed.state is not None and (
                failed.state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED
                or failed.state.ready != EndpointStateReady.READY
            ):
                w.serving_endpoints.delete(name=name)
                wait_for_endpoint_deleted(name)
        except ResourceDoesNotExist:
            pass
        rollout_entities_incrementally(name)


existing = None
try:
    existing = w.serving_endpoints.get(name=endpoint_name)
except ResourceDoesNotExist:
    pass

desired_model_versions = {entity.entity_name: entity.entity_version for entity in served_entities}

if existing is not None:
    config_update = existing.state.config_update if existing.state is not None else None
    is_ready = (
        existing.state is not None
        and existing.state.ready == EndpointStateReady.READY
        and config_update == EndpointStateConfigUpdate.NOT_UPDATING
    )

    if config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
        if endpoint_has_quota_failure(endpoint_name):
            print(f"Recovering UPDATE_FAILED endpoint via incremental rollout: {endpoint_name}")
            w.serving_endpoints.delete(name=endpoint_name)
            wait_for_endpoint_deleted(endpoint_name)
            rollout_entities_incrementally(endpoint_name)
            endpoint = w.serving_endpoints.get(name=endpoint_name)
        else:
            raise RuntimeError(
                f"Endpoint {endpoint_name} is UPDATE_FAILED. Delete it in the Serving UI "
                f"(or `databricks serving-endpoints delete {endpoint_name}`), then re-run this job."
            )
    elif not is_ready:
        print(f"Endpoint provisioning in progress, waiting: {endpoint_name}")
        endpoint = wait_for_endpoint_ready(endpoint_name)
    else:
        deployed_model_versions = {
            entity.entity_name: entity.entity_version for entity in (existing.config.served_entities or [])
        }
        if deployed_model_versions == desired_model_versions:
            print(f"Endpoint already READY with latest versions: {endpoint_name}")
            endpoint = existing
        else:
            print(f"Updating versions on READY endpoint: {endpoint_name}")
            endpoint = w.serving_endpoints.update_config_and_wait(
                name=endpoint_name,
                served_entities=served_entities,
                traffic_config=traffic_config,
                timeout=DEPLOY_TIMEOUT,
            )
else:
    create_endpoint(endpoint_name)
    endpoint = w.serving_endpoints.get(name=endpoint_name)

print(f"Endpoint {endpoint.name} ready state: {endpoint.state.ready}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify READY

# COMMAND ----------

from databricks.sdk.service.serving import EndpointStateReady

endpoint = w.serving_endpoints.get(name=endpoint_name)
state = endpoint.state
if state is None or state.ready != EndpointStateReady.READY:
    active = endpoint.pending_config or endpoint.config
    messages = [
        entity.state.deployment_state_message
        for entity in (active.served_entities or [])
        if entity.state and entity.state.deployment_state_message
    ]
    quota = next((m for m in messages if "quota exceeded" in m.lower()), None)
    if quota:
        raise RuntimeError(
            "Serving quota exceeded for 5 scale-to-zero models on this workspace. "
            "Delete failed endpoints and other custom serving endpoints to free capacity, "
            f"or upgrade the workspace. Details: {quota}"
        )
    raise RuntimeError(f"Endpoint not READY: {state}; messages: {messages}")

print(f"Endpoint {endpoint.name} is READY")
for entity in endpoint.config.served_entities:
    print(f"  {entity.name}: {entity.entity_name} v{entity.entity_version}")
for route in endpoint.config.traffic_config.routes:
    route_name = route.served_entity_name or route.served_model_name
    print(f"  {route_name}: {route.traffic_percentage}%")
