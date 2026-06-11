# Train PyTorch MLP Job — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parallel Databricks bundle job that trains a tiny 6-layer PyTorch MLP on the dummy UC table, exports it to ONNX via `torch.onnx.export`, and registers it in the UC Model Registry.

**Architecture:** Mirror `train_logistic_regression` and `train_lightgbm` — one new bundle variable, one job YAML, one multi-cell notebook. Same six job parameters, same UC schemas, same MLflow ONNX registration flow. The network is a fixed-architecture MLP: six `nn.Linear` layers (five hidden layers with 15 units, one output layer with 1 unit), ReLU activations between hidden layers, sigmoid on output for binary classification.

**Tech Stack:** Databricks Asset Bundles, serverless job environments (client `"5"`), Unity Catalog, pandas, PyTorch, onnx, onnxruntime, MLflow

**Design reference:** None yet — plan derived from [train-lightgbm-job plan](./2026-06-11-train-lightgbm-job.md) and existing training jobs.

**Prerequisite:** `create_dummy_data` job must have run so `{catalog}.{schema}.generated_data` exists.

---

## File map

| File | Responsibility |
|---|---|
| `databricks.yml` | Add `pytorch_mlp_model_name` bundle variable (default `pytorch_mlp_onnx`) |
| `resources/train_pytorch_mlp.yml` | Job definition (parameters, serverless env, notebook task) |
| `src/jobs/train_pytorch_mlp.ipynb` | Read UC table, train MLP, export ONNX, register model |

**Untouched:** `train_logistic_regression`, `train_lightgbm`, `create_dummy_data`, UC schema resources.

---

## Network definition (locked for this plan)

| Property | Value |
|---|---|
| Input size | 33 (`feature_000`…`feature_032` from dummy data) |
| Hidden layers | 5 × `Linear(15 → 15)` with `ReLU` |
| Output layer | 1 × `Linear(15 → 1)` with `Sigmoid` |
| Total `nn.Linear` layers | **6** |
| Neurons per hidden layer | **15** |
| Loss | `BCELoss` |
| Optimizer | `Adam(lr=0.001)` |
| Epochs | 10 (fixed, no tuning) |
| Batch size | 4096 |
| Random seed | 42 (`torch.manual_seed`) |
| Training scope | All rows; no train/test split, CV, or hyperparameter tuning |

---

### Task 1: Bundle variable

**Files:**
- Modify: `databricks.yml`

- [ ] **Step 1: Add `pytorch_mlp_model_name` variable**

Add under the existing `variables:` block, after `lightgbm_model_name`:

```yaml
  pytorch_mlp_model_name:
    description: Registered model name for PyTorch MLP ONNX artifact
    default: pytorch_mlp_onnx
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
git commit -m "Add pytorch_mlp_model_name bundle variable for train_pytorch_mlp job."
```

---

### Task 2: Job resource

**Files:**
- Create: `resources/train_pytorch_mlp.yml`

- [ ] **Step 1: Create the job YAML**

```yaml
resources:
  jobs:
    train_pytorch_mlp:
      name: train_pytorch_mlp
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
          default: ${var.pytorch_mlp_model_name}
      environments:
        - environment_key: train_env
          spec:
            client: "5"
            dependencies:
              - pandas
              - torch
              - onnx
              - onnxruntime
      tasks:
        - task_key: train_pytorch_mlp
          environment_key: train_env
          notebook_task:
            notebook_path: ../src/jobs/train_pytorch_mlp.ipynb
            base_parameters:
              catalog: "{{job.parameters.catalog}}"
              schema: "{{job.parameters.schema}}"
              table_name: "{{job.parameters.table_name}}"
              model_catalog: "{{job.parameters.model_catalog}}"
              model_schema: "{{job.parameters.model_schema}}"
              model_name: "{{job.parameters.model_name}}"
```

**Note:** Serverless environment 5 standard base does not pre-install PyTorch on CPU job tasks — `torch` is declared as a pip dependency. First run may take longer while PyTorch installs ([serverless environment 5](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/five)).

- [ ] **Step 2: Validate bundle picks up the new job**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds; job `train_pytorch_mlp` appears in output.

- [ ] **Step 3: Commit**

```bash
git add resources/train_pytorch_mlp.yml
git commit -m "Add train_pytorch_mlp bundle job resource."
```

---

### Task 3: Training notebook

**Files:**
- Create: `src/jobs/train_pytorch_mlp.ipynb`

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

Export with `torch.onnx.export` using dynamic batch axis ([PyTorch ONNX docs](https://github.com/pytorch/pytorch/blob/main/docs/source/onnx.md)).

```python
import os
import tempfile

import numpy as np
import onnx
import torch
import torch.nn as nn

HIDDEN = 15
EPOCHS = 10
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)


class SimpleMLP(nn.Module):
    """Six Linear layers: five hidden (15 units) + one output (1 unit)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

model = SimpleMLP(input_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss()

model.train()
for _ in range(EPOCHS):
    permutation = torch.randperm(X_tensor.size(0))
    for start in range(0, X_tensor.size(0), BATCH_SIZE):
        idx = permutation[start : start + BATCH_SIZE]
        batch_x = X_tensor[idx]
        batch_y = y_tensor[idx]
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
    onnx_path = f.name

dummy_input = torch.randn(1, X.shape[1], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["float_input"],
    output_names=["output"],
    dynamic_axes={
        "float_input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    opset_version=17,
)
onnx_model = onnx.load(onnx_path)
os.unlink(onnx_path)

print(
    f"Trained SimpleMLP (6 Linear layers, {HIDDEN} hidden units) "
    f"on {len(X)} rows and converted to ONNX"
)
```

- [ ] **Step 4: Create notebook — Cell 4 (Register in UC Model Registry)**

Markdown header: `## Register in UC Model Registry`

Build a two-column probability matrix so `infer_signature` matches the logistic regression / LightGBM pattern.

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

with torch.no_grad():
    pos_proba = model(X_tensor).numpy()
proba = np.hstack([1.0 - pos_proba, pos_proba])

signature = infer_signature(X, proba)

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
git add src/jobs/train_pytorch_mlp.ipynb
git commit -m "Add train_pytorch_mlp notebook with torch ONNX export."
```

---

### Task 4: Deploy and smoke-test

No automated unit tests (blog demo scope). Manual verification only.

- [ ] **Step 1: Deploy bundle**

```bash
databricks bundle deploy --target dev
```

Expected: job `train_pytorch_mlp` deployed alongside existing jobs.

**Dev-mode note:** With `mode: development`, UC schemas may be prefixed (e.g. `dev_<username>_data`). Job parameters resolve via `${resources.schemas.data.name}` at deploy time — use resolved values from the deployed job, not hardcoded FQNs ([bundle deployment modes](https://docs.databricks.com/aws/en/dev-tools/bundles/deployment-modes)).

- [ ] **Step 2: Ensure training data exists**

If not already done:

```bash
databricks bundle run create_dummy_data --target dev
```

Expected: training table exists at the catalog/schema/table resolved by job parameters.

- [ ] **Step 3: Run the training job**

```bash
databricks bundle run train_pytorch_mlp --target dev
```

Expected: job succeeds. Notebook output includes:
- `Loaded 100000 rows with 33 features from ...`
- `Trained SimpleMLP (6 Linear layers, 15 hidden units) on 100000 rows and converted to ONNX`
- `Registered <catalog>.<model_schema>.pytorch_mlp_onnx version N`

- [ ] **Step 4: Confirm model in UC Model Registry**

```bash
databricks registered-models get workspace.model.pytorch_mlp_onnx --include-aliases
databricks model-versions list workspace.model.pytorch_mlp_onnx
```

Expected: registered model exists with at least one version. Adjust catalog/schema if dev-mode prefixing applies.

- [ ] **Step 5: Confirm sibling training jobs are unaffected**

```bash
databricks bundle validate --target dev
```

Expected: `train_logistic_regression` and `train_lightgbm` still validate; no changes to their files.

---

## Blog-post talking points

1. **Same pipeline, different algorithm** — identical job parameters and UC registration; only the train cell changes.
2. **PyTorch → ONNX in one step** — `torch.onnx.export` with dynamic batch axis (contrast with `skl2onnx` for sklearn and `onnxmltools` for LightGBM).
3. **Tiny MLP on tabular data** — six layers, 15 hidden units, enough to show deep learning without obscuring the ONNX/registry story.
4. **Minimal demo scope** — fixed epochs, no GPU, no tuning; focus on train → ONNX → register.

## Explicitly out of scope

- Unit tests, wheels, shared Python modules
- Train/test split, cross-validation, hyperparameter tuning
- Accuracy or evaluation metrics
- GPU / AI Runtime compute (CPU serverless only)
- Model serving endpoint
- Changes to existing training jobs
- Refactoring common notebook code across training jobs
- Design spec document (optional follow-up)

## Requirements coverage

| Requirement | Task |
|---|---|
| `pytorch_mlp_model_name` bundle variable (default `pytorch_mlp_onnx`) | Task 1 |
| `resources/train_pytorch_mlp.yml` with client `"5"` | Task 2 |
| Six job parameters mirroring sibling jobs | Task 2 |
| Dependencies: pandas, torch, onnx, onnxruntime | Task 2 |
| Multi-cell notebook with section headers | Task 3 |
| Parameter validation (`ValueError` on blank) | Task 3, Cell 1 |
| Read UC table, split `label` / `feature_*` | Task 3, Cell 2 |
| 6-layer MLP, 15 hidden units, train all rows | Task 3, Cell 3 |
| ONNX export via `torch.onnx.export` | Task 3, Cell 3 |
| MLflow ONNX log + UC registration with signature | Task 3, Cell 4 |
| Manual verification on Databricks | Task 4 |
| Sibling jobs untouched | All tasks |

## Validation checkpoints (go/no-go)

| Gate | Pass criteria | Fail action |
|---|---|---|
| G1 — PyTorch install | Job environment installs `torch` without error | Pin `torch==2.9.0` or reduce deps |
| G2 — ONNX export | `torch.onnx.export` completes; `onnx.load` succeeds | Lower opset or simplify model |
| G3 — Bundle validate | `databricks bundle validate --target dev` exits 0 | Fix YAML before deploy |
| G4 — UC permissions | Caller has `CREATE_MODEL` on model schema + `SELECT` on data table | Grant privileges |
| G5 — Job run | `train_pytorch_mlp` completes; prints registered model version | Check pip install / training logs |
| G6 — Registry | `registered-models get` returns model metadata | Verify FQN matches dev-mode schema names |
