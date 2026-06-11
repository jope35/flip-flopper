# Deploy Serving Endpoint Job — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Databricks bundle job that deploys all three registered ONNX models to a single CPU Model Serving endpoint with a 33/33/34 traffic split and inference tables enabled.

**Architecture:** Mirror the existing training jobs — one bundle variable, one job YAML, one multi-cell notebook. The notebook resolves the latest UC model version for each model at runtime, then creates or updates the endpoint via the Databricks SDK. No local unit tests; manual smoke-test on Databricks only (blog-post demo scope).

**Tech Stack:** Databricks Asset Bundles, serverless job environments (client `"5"`), Unity Catalog, MLflow (`MlflowClient`), Databricks SDK (`WorkspaceClient`, `serving_endpoints` API)

**Design reference:** [2026-06-11-deploy-serving-endpoint-design.md](../specs/2026-06-11-deploy-serving-endpoint-design.md)

**Prerequisite:** All three training jobs must have run at least once so `{catalog}.{model_schema}` contains registered versions of `logistic_regression_onnx`, `lightgbm_onnx`, and `pytorch_mlp_onnx`. The job runner must have `EXECUTE` on each registered model (training-job owners typically already do).

**Simplicity note:** This is a blog-post demo, not production serving. Keep the notebook linear and readable — no shared Python modules, no retry wrappers, no dynamic traffic config. Straight SDK calls with clear `print` statements for blog excerpts.

**Validated caveats (acceptable for blog scope):**
- `auto_capture_config` is the **legacy** inference-table API (deprecated in favor of AI Gateway inference tables). It still works for custom-model endpoints and is simpler to show in a notebook — call this out in the blog as “legacy path, fine for demos.”
- Serving endpoints created by this notebook are **not** bundle-prefixed in dev mode (only jobs/schemas get `[dev user]` / `dev_user_` prefixes). Default name `flip_flopper_serving` is fine for a single-author workspace; shared workspaces need a per-user endpoint name override.
- Scale-to-zero means the **first query after idle** can take minutes while containers wake up — retry the smoke-test prediction if you get 503/timeouts.
- Inference table rows can take **up to an hour** to appear (best-effort delivery), not seconds.

---

## File map

| File | Responsibility |
|---|---|
| `databricks.yml` | Add `serving_endpoint_name` bundle variable (default `flip_flopper_serving`) |
| `resources/deploy_serving_endpoint.yml` | Job definition (8 parameters, serverless env, notebook task) |
| `src/jobs/deploy_serving_endpoint.ipynb` | Resolve latest versions, build config, create-or-update endpoint, verify READY |

**Untouched:** All three training jobs, `create_dummy_data`, UC schema resources.

---

### Task 1: Bundle variable

**Files:**
- Modify: `databricks.yml`

- [ ] **Step 1: Add `serving_endpoint_name` variable**

Add under the existing `variables:` block, after `pytorch_mlp_model_name`:

```yaml
  serving_endpoint_name:
    description: Model Serving endpoint name for the 3-model traffic split
    default: flip_flopper_serving
```

Do not change `dev` / `prod` target overrides — the default is sufficient.

**Dev collision note:** If multiple developers share one workspace, override at run time:

```bash
databricks bundle run deploy_serving_endpoint --target dev \
  --var="serving_endpoint_name=flip_flopper_serving_${USER}"
```

…and pass the same value as the `endpoint_name` job parameter (or add a dev-target variable override later). For the blog, the static default is fine.

- [ ] **Step 2: Validate bundle syntax**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add databricks.yml
git commit -m "Add serving_endpoint_name bundle variable for deploy job."
```

---

### Task 2: Job resource

**Files:**
- Create: `resources/deploy_serving_endpoint.yml`

- [ ] **Step 1: Create the job YAML**

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
        - name: pytorch_model_name
          default: ${var.pytorch_mlp_model_name}
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
              pytorch_model_name: "{{job.parameters.pytorch_model_name}}"
              endpoint_name: "{{job.parameters.endpoint_name}}"
              inference_catalog: "{{job.parameters.inference_catalog}}"
              inference_schema: "{{job.parameters.inference_schema}}"
```

- [ ] **Step 2: Validate bundle picks up the new job**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds; job `deploy_serving_endpoint` appears in output.

- [ ] **Step 3: Commit**

```bash
git add resources/deploy_serving_endpoint.yml
git commit -m "Add deploy_serving_endpoint bundle job resource."
```

---

### Task 3: Deployment notebook

**Files:**
- Create: `src/jobs/deploy_serving_endpoint.ipynb`

Five markdown section headers + five Python code cells. Keep each cell self-contained and short — easy to screenshot for the blog.

- [ ] **Step 1: Create notebook — Cell 1 (Parameters)**

Markdown header: `## Parameters`

```python
dbutils.widgets.text("model_catalog", "")
dbutils.widgets.text("model_schema", "")
dbutils.widgets.text("logistic_model_name", "")
dbutils.widgets.text("lightgbm_model_name", "")
dbutils.widgets.text("pytorch_model_name", "")
dbutils.widgets.text("endpoint_name", "")
dbutils.widgets.text("inference_catalog", "")
dbutils.widgets.text("inference_schema", "")

model_catalog = dbutils.widgets.get("model_catalog").strip()
model_schema = dbutils.widgets.get("model_schema").strip()
logistic_model_name = dbutils.widgets.get("logistic_model_name").strip()
lightgbm_model_name = dbutils.widgets.get("lightgbm_model_name").strip()
pytorch_model_name = dbutils.widgets.get("pytorch_model_name").strip()
endpoint_name = dbutils.widgets.get("endpoint_name").strip()
inference_catalog = dbutils.widgets.get("inference_catalog").strip()
inference_schema = dbutils.widgets.get("inference_schema").strip()

params = {
    "model_catalog": model_catalog,
    "model_schema": model_schema,
    "logistic_model_name": logistic_model_name,
    "lightgbm_model_name": lightgbm_model_name,
    "pytorch_model_name": pytorch_model_name,
    "endpoint_name": endpoint_name,
    "inference_catalog": inference_catalog,
    "inference_schema": inference_schema,
}
if not all(params.values()):
    missing = ", ".join(name for name, value in params.items() if not value)
    raise ValueError(f"All parameters must be non-empty; missing: {missing}")

logistic_path = f"{model_catalog}.{model_schema}.{logistic_model_name}"
lightgbm_path = f"{model_catalog}.{model_schema}.{lightgbm_model_name}"
pytorch_path = f"{model_catalog}.{model_schema}.{pytorch_model_name}"

print(f"Endpoint: {endpoint_name}")
print(f"Logistic: {logistic_path}")
print(f"LightGBM: {lightgbm_path}")
print(f"PyTorch:  {pytorch_path}")
```

- [ ] **Step 2: Create notebook — Cell 2 (Resolve latest model versions)**

Markdown header: `## Resolve latest model versions`

```python
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()


def latest_version(model_name: str) -> int:
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for {model_name}")
    return max(int(v.version) for v in versions)


versions = {
    "logistic_regression": latest_version(logistic_path),
    "lightgbm": latest_version(lightgbm_path),
    "pytorch_mlp": latest_version(pytorch_path),
}

for name, version in versions.items():
    print(f"{name}: version {version}")
```

- [ ] **Step 3: Create notebook — Cell 3 (Build endpoint config)**

Markdown header: `## Build endpoint config`

```python
from databricks.sdk.service.serving import (
    AutoCaptureConfigInput,
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    ServingModelWorkloadType,
    TrafficConfig,
)

served_entities = [
    ServedEntityInput(
        name="logistic_regression",
        entity_name=logistic_path,
        entity_version=str(versions["logistic_regression"]),
        workload_type=ServingModelWorkloadType.CPU,
        workload_size="Small",
        scale_to_zero_enabled=True,
    ),
    ServedEntityInput(
        name="lightgbm",
        entity_name=lightgbm_path,
        entity_version=str(versions["lightgbm"]),
        workload_type=ServingModelWorkloadType.CPU,
        workload_size="Small",
        scale_to_zero_enabled=True,
    ),
    ServedEntityInput(
        name="pytorch_mlp",
        entity_name=pytorch_path,
        entity_version=str(versions["pytorch_mlp"]),
        workload_type=ServingModelWorkloadType.CPU,
        workload_size="Small",
        scale_to_zero_enabled=True,
    ),
]

traffic_config = TrafficConfig(
    routes=[
        Route(served_model_name="logistic_regression", traffic_percentage=33),
        Route(served_model_name="lightgbm", traffic_percentage=33),
        Route(served_model_name="pytorch_mlp", traffic_percentage=34),
    ]
)

auto_capture_config = AutoCaptureConfigInput(
    enabled=True,
    catalog_name=inference_catalog,
    schema_name=inference_schema,
    table_name_prefix=endpoint_name,
)

config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_entities=served_entities,
    traffic_config=traffic_config,
    auto_capture_config=auto_capture_config,
)

inference_table = f"{inference_catalog}.{inference_schema}.{endpoint_name}_payload"
print(f"Inference table: {inference_table}")
print("Traffic split: 33% / 33% / 34%")
```

- [ ] **Step 4: Create notebook — Cell 4 (Create or update endpoint)**

Markdown header: `## Create or update endpoint`

```python
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import ResourceDoesNotExist

w = WorkspaceClient()

try:
    w.serving_endpoints.get(name=endpoint_name)
    print(f"Updating existing endpoint: {endpoint_name}")
    endpoint = w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=served_entities,
        traffic_config=traffic_config,
        auto_capture_config=auto_capture_config,
    ).result()
except ResourceDoesNotExist:
    print(f"Creating new endpoint: {endpoint_name}")
    endpoint = w.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=config,
        timeout=timedelta(minutes=30),
    )

print(f"Endpoint {endpoint.name} ready state: {endpoint.state.ready}")
```

- [ ] **Step 5: Create notebook — Cell 5 (Verify READY)**

Markdown header: `## Verify READY`

```python
from databricks.sdk.service.serving import EndpointStateReady

endpoint = w.serving_endpoints.get(name=endpoint_name)

if endpoint.state is None or endpoint.state.ready != EndpointStateReady.READY:
    raise RuntimeError(f"Endpoint not READY: {endpoint.state}")

print(f"Endpoint {endpoint.name} is READY")
print("Served entities:")
for entity in endpoint.config.served_entities:
    print(f"  {entity.name}: {entity.entity_name} v{entity.entity_version}")

print("Traffic:")
for route in endpoint.config.traffic_config.routes:
    print(f"  {route.served_model_name}: {route.traffic_percentage}%")

print(f"Inference table: {inference_table}")
```

- [ ] **Step 6: Validate notebook path resolves**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds with no notebook path errors.

- [ ] **Step 7: Commit**

```bash
git add src/jobs/deploy_serving_endpoint.ipynb
git commit -m "Add deploy_serving_endpoint notebook with 3-model traffic split."
```

---

### Task 4: Deploy and smoke-test

No automated unit tests (per spec). Manual verification only.

- [ ] **Step 1: Deploy bundle**

```bash
databricks bundle validate --target dev && databricks bundle deploy --target dev
```

Expected: job `deploy_serving_endpoint` deployed alongside existing jobs.

**Dev-mode note:** With `mode: development`, UC schemas may be prefixed (e.g. `dev_<username>_model` instead of `model`). Job parameters resolve via `${resources.schemas.model.name}` at deploy time — use the resolved values from the deployed job, not hardcoded `workspace.model`.

- [ ] **Step 2: Ensure all three models are registered**

If not already done:

```bash
databricks bundle run create_dummy_data --target dev
databricks bundle run train_logistic_regression --target dev
databricks bundle run train_lightgbm --target dev
databricks bundle run train_pytorch_mlp --target dev
```

Expected: each training job succeeds and prints a registered model version.

- [ ] **Step 3: Run the deploy job**

```bash
databricks bundle run deploy_serving_endpoint --target dev
```

Expected: job succeeds (~2–10 min). Notebook output includes:
- Resolved versions for all three models
- `Creating new endpoint: flip_flopper_serving` (first run) or `Updating existing endpoint: ...` (re-run)
- `Endpoint flip_flopper_serving is READY`
- Traffic split 33/33/34 and inference table location

- [ ] **Step 4: Confirm endpoint state via CLI**

```bash
databricks serving-endpoints get flip_flopper_serving
```

Expected: endpoint state is `READY` with three served entities.

- [ ] **Step 5: Send a test prediction**

Build a payload with `feature_000` through `feature_032` (all zeros is fine for a smoke test). Query via REST or the Serving UI. Example REST shape:

```json
{
  "dataframe_records": [
    {
      "feature_000": 0.0, "feature_001": 0.0, "feature_002": 0.0,
      "feature_003": 0.0, "feature_004": 0.0, "feature_005": 0.0,
      "feature_006": 0.0, "feature_007": 0.0, "feature_008": 0.0,
      "feature_009": 0.0, "feature_010": 0.0, "feature_011": 0.0,
      "feature_012": 0.0, "feature_013": 0.0, "feature_014": 0.0,
      "feature_015": 0.0, "feature_016": 0.0, "feature_017": 0.0,
      "feature_018": 0.0, "feature_019": 0.0, "feature_020": 0.0,
      "feature_021": 0.0, "feature_022": 0.0, "feature_023": 0.0,
      "feature_024": 0.0, "feature_025": 0.0, "feature_026": 0.0,
      "feature_027": 0.0, "feature_028": 0.0, "feature_029": 0.0,
      "feature_030": 0.0, "feature_031": 0.0, "feature_032": 0.0
    }
  ]
}
```

Expected: 200 response with a prediction (probability or class). If the endpoint scaled to zero, the first call may time out or return 503 — wait 2–5 minutes and retry.

- [ ] **Step 6: Verify inference table exists**

In Catalog Explorer, look for `{inference_catalog}.{inference_schema}.{endpoint_name}_payload` (e.g. `workspace.dev_<user>_model.flip_flopper_serving_payload` in dev mode). The table is created when the endpoint is created; **payload rows may take up to an hour** to appear after the first query (Databricks best-effort log delivery).

- [ ] **Step 7: Confirm re-run is idempotent**

Re-run one training job, then:

```bash
databricks bundle run deploy_serving_endpoint --target dev
```

Expected: job updates the endpoint to the new latest version without error; traffic split resets to 33/33/34.

---

## Blog-post talking points

Use this job as the capstone after the three training jobs:

1. **Same bundle pattern** — variable + YAML + notebook, just like training; only the notebook logic differs.
2. **Runtime version resolution** — no pinned model versions in the bundle; the job always picks up the latest UC registration after retraining.
3. **One endpoint, three models** — traffic split lets you A/B/C compare logistic regression, LightGBM, and PyTorch MLP on identical inputs.
4. **Inference tables (legacy API)** — `auto_capture_config` logs requests to `{catalog}.{schema}.{endpoint}_payload` in UC. Databricks now recommends AI Gateway inference tables for new work; legacy tables are simpler to configure in a short demo notebook.
5. **Minimal demo scope** — CPU Small, scale-to-zero, fixed 33/33/34 split; no AI Gateway, no GPU, no dynamic routing. Expect ~2–10 min deploy time and possible cold-start latency on first query.

## Explicitly out of scope

- Unit tests, wheels, shared Python modules
- GPU endpoints or provisioned throughput
- Per-model version overrides via job parameters
- Dynamic traffic split configuration
- AI Gateway (usage tracking, guardrails, rate limits)
- Automated traffic distribution verification
- Endpoint deletion or teardown job
- Scheduled or triggered job runs
- Blog post publication (this plan covers the job only)
- Changes to existing training jobs

## Spec coverage

| Design requirement | Task | Notes |
|---|---|---|
| `serving_endpoint_name` bundle variable (default `flip_flopper_serving`) | Task 1 | |
| `resources/deploy_serving_endpoint.yml` with client `"5"` | Task 2 | |
| Eight job parameters with bundle defaults | Task 2 | Reuses existing model name variables |
| Dependencies: `databricks-sdk`, `mlflow` | Task 2 | |
| Multi-cell notebook with section headers | Task 3 | Five cells |
| Parameter validation (`ValueError` on blank) | Task 3, Cell 1 | |
| Resolve latest UC model version per model | Task 3, Cell 2 | `MlflowClient.search_model_versions` |
| Fail fast if any model has zero versions | Task 3, Cell 2 | `ValueError` names missing model |
| Three CPU served entities (Small, scale-to-zero) | Task 3, Cell 3 | |
| Traffic split 33/33/34 | Task 3, Cell 3 | |
| Inference tables in same UC schema as models | Task 3, Cell 3 | `auto_capture_config` |
| Idempotent create-or-update | Task 3, Cell 4 | `ResourceDoesNotExist` → create, else update |
| Block until READY | Task 3, Cells 4–5 | `create_and_wait` / `update_config().result()` + explicit verify |
| Manual verification on Databricks | Task 4 | CLI + test prediction + inference table |
| Training jobs untouched | All tasks | |

## Validation checkpoints (go/no-go)

| Gate | Pass criteria | Fail action |
|---|---|---|
| G0 — Platform prereqs | Workspace has Model Serving + serverless compute enabled (account admin accepted terms) | Enable in account console |
| G1 — Bundle validate | `databricks bundle validate --target dev` exits 0 | Fix YAML before deploy |
| G2 — Models registered | All three UC models have at least one version | Run missing training job(s) |
| G3 — Permissions | Job runner has `EXECUTE` on all 3 UC models; `USE CATALOG` + `USE SCHEMA` + `CREATE TABLE` on inference schema; workspace entitlement to create serving endpoints | `GRANT EXECUTE ON FUNCTION …` per model; see [inference table requirements](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/inference-tables#requirements) |
| G4 — Job run | `deploy_serving_endpoint` completes; prints READY | Check notebook output / serving logs |
| G5 — Endpoint queryable | Test prediction returns 200 (retry after cold start) | Wait for READY or check served entity errors |
| G6 — Inference table | Table exists in Catalog Explorer; rows appear within ~1 hr of first query | Check `auto_capture_config` state; recreate endpoint if payload table is FAILED |
