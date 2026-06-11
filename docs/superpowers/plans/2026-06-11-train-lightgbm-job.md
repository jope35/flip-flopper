# Train LightGBM Job — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel Databricks bundle job that trains a basic `LGBMClassifier` on the dummy UC table, converts it to ONNX, and registers it in the UC Model Registry.

**Architecture:** Mirror `train_logistic_regression` exactly — one new bundle variable, one job YAML, one multi-cell notebook. Same six job parameters, same UC schemas, same MLflow ONNX registration flow. LightGBM ONNX conversion uses `onnxmltools.convert_lightgbm` (not LightGBM native export — that API does not exist in the Python package).

**Tech Stack:** Databricks Asset Bundles, serverless job environments (client `"5"`), Unity Catalog, pandas, lightgbm, onnxmltools, onnx, onnxruntime, MLflow

**Design reference:** [2026-06-11-train-lightgbm-job-design.md](../specs/2026-06-11-train-lightgbm-job-design.md)

**ONNX export:** Uses `onnxmltools.convert_lightgbm` — LightGBM Python has no native ONNX
`save_model` format ([docs](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.save_model)). Spec and plan are aligned.

**Prerequisite:** `create_dummy_data` job must have run so `{catalog}.{schema}.generated_data` exists.

---

## File map

| File | Responsibility |
|---|---|
| `databricks.yml` | Add `lightgbm_model_name` bundle variable (default `lightgbm_onnx`) |
| `resources/train_lightgbm.yml` | Job definition (parameters, serverless env, notebook task) |
| `src/jobs/train_lightgbm.ipynb` | Read UC table, train LightGBM, convert ONNX, register model |

**Untouched:** `train_logistic_regression` job, `create_dummy_data` job, UC schema resources.

---

### Task 1: Bundle variable

**Files:**
- Modify: `databricks.yml`

- [ ] **Step 1: Add `lightgbm_model_name` variable**

Add under the existing `variables:` block, after `model_name`:

```yaml
  lightgbm_model_name:
    description: Registered model name for LightGBM ONNX artifact
    default: lightgbm_onnx
```

Do not change `dev` / `prod` target overrides — the default is sufficient.

- [ ] **Step 2: Validate bundle syntax**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add databricks.yml
git commit -m "Add lightgbm_model_name bundle variable for train_lightgbm job."
```

---

### Task 2: Job resource

**Files:**
- Create: `resources/train_lightgbm.yml`

- [ ] **Step 1: Create the job YAML**

```yaml
resources:
  jobs:
    train_lightgbm:
      name: train_lightgbm
      parameters:
        - name: catalog
          default: ${var.catalog}
        - name: schema
          default: ${resources.schemas.data.name}
        - name: table_name
          default: ${var.table_name}
        - name: model_catalog
          default: ${var.catalog}
        - name: model_schema
          default: ${resources.schemas.model.name}
        - name: model_name
          default: ${var.lightgbm_model_name}
      environments:
        - environment_key: train_env
          spec:
            client: "5"
            dependencies:
              - pandas
              - lightgbm
              - onnxmltools
              - onnx
              - onnxruntime
      tasks:
        - task_key: train_lightgbm
          environment_key: train_env
          notebook_task:
            notebook_path: ../src/jobs/train_lightgbm.ipynb
            base_parameters:
              catalog: "{{job.parameters.catalog}}"
              schema: "{{job.parameters.schema}}"
              table_name: "{{job.parameters.table_name}}"
              model_catalog: "{{job.parameters.model_catalog}}"
              model_schema: "{{job.parameters.model_schema}}"
              model_name: "{{job.parameters.model_name}}"
```

- [ ] **Step 2: Validate bundle picks up the new job**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds; job `train_lightgbm` appears in output.

- [ ] **Step 3: Commit**

```bash
git add resources/train_lightgbm.yml
git commit -m "Add train_lightgbm bundle job resource."
```

---

### Task 3: Training notebook

**Files:**
- Create: `src/jobs/train_lightgbm.ipynb`

Four markdown section headers + four Python code cells. Mirror `src/jobs/train_logistic_regression.ipynb` structure; only the train/convert cell differs.

- [ ] **Step 1: Create notebook — Cell 1 (Parameters)**

Markdown header: `## Parameters`

```python
dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_name", "")
dbutils.widgets.text("model_catalog", "")
dbutils.widgets.text("model_schema", "")
dbutils.widgets.text("model_name", "")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
table_name = dbutils.widgets.get("table_name").strip()
model_catalog = dbutils.widgets.get("model_catalog").strip()
model_schema = dbutils.widgets.get("model_schema").strip()
model_name = dbutils.widgets.get("model_name").strip()

params = {
    "catalog": catalog,
    "schema": schema,
    "table_name": table_name,
    "model_catalog": model_catalog,
    "model_schema": model_schema,
    "model_name": model_name,
}
if not all(params.values()):
    missing = ", ".join(name for name, value in params.items() if not value)
    raise ValueError(f"All parameters must be non-empty; missing: {missing}")

full_table = f"{catalog}.{schema}.{table_name}"
full_model_name = f"{model_catalog}.{model_schema}.{model_name}"
```

- [ ] **Step 2: Create notebook — Cell 2 (Load data)**

Markdown header: `## Load data`

```python
pdf = spark.table(full_table).toPandas()

y = pdf["label"]
X = pdf[[c for c in pdf.columns if c.startswith("feature_")]]

print(f"Loaded {len(pdf)} rows with {X.shape[1]} features from {full_table}")
```

- [ ] **Step 3: Create notebook — Cell 3 (Train & convert to ONNX)**

Markdown header: `## Train & convert to ONNX`

Use `onnxmltools.convert_lightgbm` with `zipmap=False` so probability outputs are tensors (compatible with `infer_signature` and ONNX Runtime, per [onnxmltools docs](https://context7.com/onnx/onnxmltools/llms.txt)).

```python
import lightgbm as lgb
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

model = lgb.LGBMClassifier(random_state=42)
model.fit(X, y)

initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_lightgbm(model, initial_types=initial_type, zipmap=False)

print(f"Trained LGBMClassifier on {len(X)} rows and converted to ONNX")
```

- [ ] **Step 4: Create notebook — Cell 4 (Register in UC Model Registry)**

Markdown header: `## Register in UC Model Registry`

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

signature = infer_signature(X, model.predict_proba(X))

with mlflow.start_run():
    model_info = mlflow.onnx.log_model(
        onnx_model,
        artifact_path="model",
        signature=signature,
    )
    registered = mlflow.register_model(model_info.model_uri, full_model_name)

print(f"Registered {registered.name} version {registered.version}")
```

- [ ] **Step 5: Validate notebook path resolves**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds with no notebook path errors.

- [ ] **Step 6: Commit**

```bash
git add src/jobs/train_lightgbm.ipynb
git commit -m "Add train_lightgbm notebook with onnxmltools ONNX export."
```

---

### Task 4: Deploy and smoke-test

No automated unit tests (per spec). Manual verification only.

- [ ] **Step 1: Deploy bundle**

```bash
databricks bundle deploy --target dev
```

Expected: job `train_lightgbm` deployed alongside existing jobs.

**Dev-mode note:** With `mode: development`, UC schemas may be prefixed (e.g. `dev_<username>_data` instead of `data`). Job parameters resolve via `${resources.schemas.data.name}` at deploy time — use the resolved values from the deployed job, not hardcoded `workspace.data` ([bundle deployment modes](https://docs.databricks.com/aws/en/dev-tools/bundles/deployment-modes)).

- [ ] **Step 2: Ensure training data exists**

If not already done:

```bash
databricks bundle run create_dummy_data --target dev
```

Expected: training table exists at the catalog/schema/table resolved by job parameters.

- [ ] **Step 3: Run the training job**

```bash
databricks bundle run train_lightgbm --target dev
```

Expected: job succeeds. Notebook output includes:
- `Loaded 100000 rows with 33 features from ...`
- `Trained LGBMClassifier on 100000 rows and converted to ONNX`
- `Registered <catalog>.<model_schema>.lightgbm_onnx version N`

- [ ] **Step 4: Confirm model in UC Model Registry**

Use the UC CLI (not legacy `model-registry` commands):

```bash
databricks registered-models get workspace.model.lightgbm_onnx --include-aliases
databricks model-versions list workspace.model.lightgbm_onnx
```

Expected: registered model exists with at least one version. Adjust catalog/schema if dev-mode prefixing applies.

- [ ] **Step 5: Confirm logistic regression job is unaffected**

```bash
databricks bundle validate --target dev
```

Expected: `train_logistic_regression` still validates; no changes to its files.

---

## Blog-post talking points

Use this flow alongside the logistic regression job in the post:

1. **Same infrastructure, different algorithm** — identical job parameters and UC registration; only the train cell changes.
2. **ONNX for tree models** — LightGBM has no built-in Python ONNX exporter; use `onnxmltools.convert_lightgbm` (contrast with sklearn + `skl2onnx` for logistic regression).
3. **Side-by-side comparison** — linear model vs gradient-boosted trees, both registered as ONNX in UC.
4. **Minimal demo scope** — fit on all rows, no CV, no tuning; focus on the train → ONNX → register pipeline.

## Explicitly out of scope

- Unit tests, wheels, shared Python modules
- Train/test split, cross-validation, hyperparameter tuning
- Accuracy or evaluation metrics
- Model serving endpoint
- Changes to `train_logistic_regression`
- Refactoring common notebook code between the two training jobs

## Spec coverage

| Design requirement | Task | Notes |
|---|---|---|
| `lightgbm_model_name` bundle variable (default `lightgbm_onnx`) | Task 1 | |
| `resources/train_lightgbm.yml` with client `"5"` | Task 2 | |
| Six job parameters mirroring logistic regression | Task 2 | |
| Dependencies for LightGBM ONNX pipeline | Task 2 | `onnxmltools` + `onnx` + `onnxruntime` |
| Multi-cell notebook with section headers | Task 3 | |
| Parameter validation (`ValueError` on blank) | Task 3, Cell 1 | |
| Read UC table, split `label` / `feature_*` | Task 3, Cell 2 | |
| `LGBMClassifier(random_state=42)`, fit all rows | Task 3, Cell 3 | |
| ONNX export | Task 3, Cell 3 | `convert_lightgbm(..., zipmap=False)` |
| MLflow ONNX log + UC registration with signature | Task 3, Cell 4 | |
| Manual verification on Databricks | Task 4 | UC `registered-models` CLI |
| Logistic regression job untouched | All tasks | |

## Validation checkpoints (go/no-go)

| Gate | Pass criteria | Fail action |
|---|---|---|
| G1 — ONNX API | `convert_lightgbm` succeeds on 100-row sample locally or in notebook | Do not deploy; fix deps/versions |
| G2 — Bundle validate | `databricks bundle validate --target dev` exits 0 | Fix YAML before deploy |
| G3 — UC permissions | Caller has `CREATE_MODEL` on model schema + `SELECT` on data table | Grant privileges or use service principal |
| G4 — Job run | `train_lightgbm` completes; prints registered model version | Check cluster logs for pip install / conversion errors |
| G5 — Registry | `registered-models get` returns model metadata | Verify `full_model_name` matches dev-mode schema names |
