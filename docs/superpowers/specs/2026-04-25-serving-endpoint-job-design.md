# Design: Databricks serving endpoint deploy job (2026-04-25)

## Goal

One Databricks Model Serving endpoint serving two ONNX models registered in Unity Catalog, with a 50/50 traffic split. Deployment runs as a Databricks Job provisioned via Databricks Asset Bundle.

## Context

- Training notebooks register models at `${catalog}.${schema}.${normalized_suffix}_a` and `_b` (see `train_classifier_a` / `train_classifier_b`).
- Serving endpoint REST config uses `served_entities[].entity_version` as a concrete version string, not a UC alias string in the entity fields.
- UC model aliases (`Champion`, `Challenger`) are resolved at deploy time via `MlflowClient.get_model_version_by_alias`, then the resolved versions are passed to the serving API (same pattern as Databricks docs for alias-driven endpoint updates).

## Components

1. **Bundle variables** (`databricks.yml`): defaults for endpoint name, classifier suffix, alias names, traffic percentages, workload size, scale-to-zero.
2. **Job resource** (`resources/deploy_classifier_endpoint.yml`): serverless-style job environment (wheel + base env), single `spark_python_task` running `src/jobs/deploy_classifier_endpoint.py` with job parameters mapped to CLI args.
3. **Library code** (`src/flip_flopper/serving_deploy.py`): normalize suffix, build UC model FQNs, build endpoint `config` dict, `deploy_endpoint()` using `mlflow.deployments.get_deploy_client("databricks")` (`get_endpoint` / `create_endpoint` / `update_endpoint`).
4. **Entry script** (`src/jobs/deploy_classifier_endpoint.py`): parse args, call `deploy_endpoint()`.

## Behavior

- Resolve alias A on model `_a` and alias B on model `_b` to integer versions; fail fast if alias missing or model not READY.
- If endpoint missing: `create_endpoint(name=..., config=...)`.
- If endpoint exists: `update_endpoint(endpoint=..., config=...)`.
- `traffic_config.routes[].served_model_name` must match each served entity `name`.
- Document: job identity needs UC privileges (`USE CATALOG`, `USE SCHEMA`, `EXECUTE` on both models); endpoint creator identity is fixed at creation (Databricks serving docs).

## Out of scope

- Auto-polling until endpoint state `READY` (job may finish while endpoint still updating; operators can poll UI or a follow-up check).
- Route optimization, inference tables, provisioned throughput.

## Verification

- Unit tests for config construction and normalization (no live Databricks).
- `ruff check`, `pytest`, `databricks bundle validate`.
