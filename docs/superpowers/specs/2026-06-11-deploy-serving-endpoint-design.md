# Deploy Serving Endpoint Job — Design

**Date:** 2026-06-11
**Status:** Approved

## Summary

Add a Databricks job that deploys all three registered ONNX models from Unity Catalog
(`logistic_regression_onnx`, `lightgbm_onnx`, `xgboost_onnx`) to a single CPU Model
Serving endpoint with a 33/33/34 traffic split and inference tables enabled. The job
resolves the latest version of each model at runtime, creates the endpoint on first run,
and updates it on subsequent runs. The job blocks until the endpoint reaches `READY` state.

Inference table payloads are logged to the same UC schema as the models (`workspace.model`).
The notebook uses a multi-cell layout with section headers for readability in Databricks
and for blog excerpts.

**Prerequisite:** run all three training jobs at least once so
`{catalog}.{model_schema}` contains registered versions of all three models.

## Requirements

| Requirement | Decision |
|---|---|
| Models served | `logistic_regression_onnx`, `lightgbm_onnx`, `xgboost_onnx` |
| Model versions | Latest registered version of each, resolved at job runtime |
| Endpoint compute | CPU, workload size `Small`, scale-to-zero enabled |
| Traffic split | 33% / 33% / 34% (must sum to 100) |
| Inference tables | Enabled via `auto_capture_config` → `{inference_catalog}.{inference_schema}` |
| Default inference location | Same catalog and schema as models (`workspace.model`) |
| Re-run behavior | Idempotent create-or-update; redeploy latest versions, reset traffic split |
| Job completion | Block until endpoint state is `READY` (~2–10 min typical) |
| Task type | Notebook (`src/jobs/deploy_serving_endpoint.ipynb`) |
| Compute | Serverless environment with pip deps, client `"5"` |
| Default endpoint name | `flip_flopper_serving` (new bundle variable `serving_endpoint_name`) |

## Architecture

The bundle-managed `model` schema already holds registered models from the three training
jobs. This job reads model metadata from UC, then creates or updates a Model Serving
endpoint via the Databricks SDK.

```
bundle deploy
  └── resources.schemas.model  →  {catalog}.model

bundle run train_logistic_regression   (prerequisite)
bundle run train_lightgbm              (prerequisite)
bundle run train_xgboost           (prerequisite)
  └── register ONNX models in {catalog}.model

bundle run deploy_serving_endpoint
  └── notebook (serverless)
        ├── resolve latest UC model version for each of 3 models
        ├── build endpoint config:
        │     ├── 3 CPU served entities (Small, scale-to-zero)
        │     ├── traffic split 33 / 33 / 34
        │     └── inference tables → {inference_catalog}.{inference_schema}
        ├── if endpoint missing → create_and_wait
        └── if endpoint exists  → update_config + wait until READY
```

All three models share the same input schema (`feature_000`…`feature_032`), making them
compatible on a single endpoint with traffic routing.

## Alternatives considered

| Approach | Why not chosen |
|---|---|
| Python script job (`spark_python_task`) | Breaks the notebook convention used by all other jobs in this repo |
| Bundle-declared serving endpoint resource | Cannot resolve "latest" model versions at deploy time; poor fit for create-or-update |
| Fixed model version job parameters | User chose runtime resolution of latest versions |
| Separate endpoint per model | User wants a single endpoint with traffic split for A/B/C comparison |
| Inference tables in a dedicated `serving` schema | User chose to colocate with models in `workspace.model` |
| Submit-and-exit (no wait for READY) | User chose to block until endpoint is live |

**Chosen approach:** Serverless notebook job using `MlflowClient` for version resolution
and `WorkspaceClient` for endpoint create-or-update, mirroring the existing training job
pattern.

## Components

### `databricks.yml`

Add one bundle variable:

```yaml
serving_endpoint_name:
  description: Model Serving endpoint name for the 3-model traffic split
  default: flip_flopper_serving
```

Reuse existing model name variables: `model_name`, `lightgbm_model_name`,
`xgboost_model_name`. No new UC schema resources — inference tables land in the
existing bundle-managed `model` schema.

### `resources/deploy_serving_endpoint.yml`

```yaml
resources:
  jobs:
    deploy_serving_endpoint:
      name: deploy_serving_endpoint
      parameters:
        - name: model_catalog
          default: ${var.catalog}
        - name: model_schema
          default: ${resources.schemas.model.name}
        - name: logistic_model_name
          default: ${var.model_name}
        - name: lightgbm_model_name
          default: ${var.lightgbm_model_name}
        - name: xgboost_model_name
          default: ${var.xgboost_model_name}
        - name: endpoint_name
          default: ${var.serving_endpoint_name}
        - name: inference_catalog
          default: ${var.catalog}
        - name: inference_schema
          default: ${resources.schemas.model.name}
      environments:
        - environment_key: deploy_env
          spec:
            client: "5"
            dependencies:
              - databricks-sdk
              - mlflow
      tasks:
        - task_key: deploy_serving_endpoint
          environment_key: deploy_env
          notebook_task:
            notebook_path: ../src/jobs/deploy_serving_endpoint.ipynb
            base_parameters:
              model_catalog: "{{job.parameters.model_catalog}}"
              model_schema: "{{job.parameters.model_schema}}"
              logistic_model_name: "{{job.parameters.logistic_model_name}}"
              lightgbm_model_name: "{{job.parameters.lightgbm_model_name}}"
              xgboost_model_name: "{{job.parameters.xgboost_model_name}}"
              endpoint_name: "{{job.parameters.endpoint_name}}"
              inference_catalog: "{{job.parameters.inference_catalog}}"
              inference_schema: "{{job.parameters.inference_schema}}"
```

### `src/jobs/deploy_serving_endpoint.ipynb`

Five code cells with markdown section headers above each:

#### Cell 1 — Parameters

1. Define widgets for all eight parameters (populated by job `base_parameters`).
2. Validate all parameters are non-empty; raise `ValueError` otherwise.
3. Build full UC model paths:
   - `{model_catalog}.{model_schema}.{logistic_model_name}`
   - `{model_catalog}.{model_schema}.{lightgbm_model_name}`
   - `{model_catalog}.{model_schema}.{xgboost_model_name}`

#### Cell 2 — Resolve latest model versions

1. Call `mlflow.set_registry_uri("databricks-uc")`.
2. For each model, use `MlflowClient.search_model_versions(f"name='{full_name}'")`.
3. Select the version with the highest integer version number.
4. If any model has zero versions, raise `ValueError` naming the missing model.
5. Print resolved model names and versions.

#### Cell 3 — Build endpoint config

Construct an `EndpointCoreConfigInput` with:

**Served entities** (three entries):

| `name` | `entity_name` | `workload_type` | `workload_size` | `scale_to_zero_enabled` |
|---|---|---|---|---|
| `logistic_regression` | `{catalog}.{schema}.logistic_regression_onnx` | `CPU` | `Small` | `true` |
| `lightgbm` | `{catalog}.{schema}.lightgbm_onnx` | `CPU` | `Small` | `true` |
| `xgboost` | `{catalog}.{schema}.xgboost_onnx` | `CPU` | `Small` | `true` |

Each entity uses the latest version resolved in Cell 2.

**Traffic config:**

| `served_model_name` | `traffic_percentage` |
|---|---|
| `logistic_regression` | 33 |
| `lightgbm` | 33 |
| `xgboost` | 34 |

**Auto capture config (inference tables):**

| Field | Value |
|---|---|
| `catalog_name` | `{inference_catalog}` |
| `schema_name` | `{inference_schema}` |
| `table_name_prefix` | `{endpoint_name}` |

Resulting inference table: `{inference_catalog}.{inference_schema}.{endpoint_name}_payload`.

#### Cell 4 — Create or update endpoint

1. Instantiate `WorkspaceClient()`.
2. Attempt `w.serving_endpoints.get(name=endpoint_name)`.
3. If endpoint exists: call `w.serving_endpoints.update_config(name=endpoint_name, served_entities=..., traffic_config=..., auto_capture_config=...).result()`.
4. If endpoint does not exist (`ResourceDoesNotExist`): call `w.serving_endpoints.create_and_wait(name=endpoint_name, config=..., timeout=timedelta(minutes=30))`.
5. Print endpoint name and final state.

Use `databricks.sdk.errors.platform.ResourceDoesNotExist` to detect a missing endpoint.

#### Cell 5 — Verify READY

1. Fetch endpoint via `w.serving_endpoints.get(name=endpoint_name)`.
2. Assert the endpoint ready state is `READY`; raise `RuntimeError` otherwise.
3. Print served entity names, versions, and traffic percentages.
4. Print inference table location (`{inference_catalog}.{inference_schema}.{endpoint_name}_payload`).

Steps 4–5 may be combined if `create_and_wait` / `update_config().result()` already
guarantee READY state; the verify cell serves as an explicit confirmation for blog
readability.

## Data flow

1. Operator runs `databricks bundle deploy --target dev` (or `prod`) to ensure UC schemas
   exist.
2. Operator runs all three training jobs to register ONNX models in `{catalog}.model`.
3. Operator runs `databricks bundle run deploy_serving_endpoint`.
4. Job resolves parameters (overrides or bundle defaults).
5. Notebook queries MLflow for the latest version of each model.
6. Notebook creates or updates the serving endpoint with three served entities, traffic
   routing, and inference table config.
7. Job waits until endpoint is `READY`.
8. After deployment, prediction requests to the endpoint are automatically logged to the
   inference table in `{inference_catalog}.{inference_schema}`.

On re-run (e.g. after retraining), the job picks up new latest versions and updates the
endpoint in place.

## Error handling

- **Blank parameters:** notebook raises `ValueError` before any API calls.
- **Missing model versions:** fail fast with `ValueError` naming which UC model has no
  registered versions. Operator must run the corresponding training job first.
- **Endpoint provisioning failure:** job fails after SDK timeout (~30 min) with endpoint
  state in the error output.
- **Inference table creation failure:** if a previous endpoint had a failed inference
  table (`auto_capture_config` state `FAILED`), Databricks requires creating a new
  endpoint. The job does not auto-delete broken endpoints; operator handles manually.
- **Permission errors:** fail fast with standard UC / Model Serving API exceptions. The
  job runner needs `CREATE SERVING ENDPOINT` and `CREATE TABLE` on the inference schema.
- **Re-runs:** idempotent update; each run redeploys latest versions and resets traffic
  to 33/33/34.

## Testing

Out of scope for v1 (notebook-only, no local unit tests).

Manual verification:

1. `databricks bundle validate && databricks bundle deploy --target dev`
2. Ensure all three training jobs have run successfully.
3. `databricks bundle run deploy_serving_endpoint`
4. Confirm endpoint state is `READY`:

   ```bash
   databricks serving-endpoints get flip_flopper_serving
   ```

   Adjust endpoint name for dev-mode resource prefixing if applicable.

5. Send a test prediction with a `feature_000`…`feature_032` payload (REST API or UI).
6. Verify inference table `{catalog}.model.{endpoint_name}_payload` appears in Catalog
   Explorer (payload rows may take a few minutes after the first query).

Optional follow-up: add a second notebook cell or job that sends N requests and verifies
traffic distribution across served entities via inference table `served_model_name` column.

## Out of scope

- GPU endpoints or provisioned throughput
- Per-model version overrides via job parameters
- Dynamic traffic split configuration (fixed 33/33/34)
- AI Gateway configuration (usage tracking, guardrails, rate limits)
- Automated traffic distribution verification
- Endpoint deletion or teardown job
- Scheduled or triggered job runs (manual/bundle-run only for v1)
- Blog post publication (this spec covers the job only)
- Changes to existing training jobs
