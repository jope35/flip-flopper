# Train LightGBM Job — Design

**Date:** 2026-06-11  
**Status:** Approved

## Summary

Add a Databricks job that reads a binary classification table from Unity Catalog, trains a
basic `LGBMClassifier` model on all rows, exports it to ONNX via LightGBM's built-in
exporter, and registers it in the Unity Catalog Model Registry. Input and output UC
locations are controlled by six job parameters. The UC `data` and `model` schemas are
bundle-managed. The notebook uses a multi-cell layout with section headers for readability
in Databricks and for blog excerpts.

This job is a parallel sibling to `train_logistic_regression` — both can coexist for a
side-by-side blog comparison. Prerequisite: run `create_dummy_data` first to populate
`{catalog}.data.generated_data`.

## Requirements

| Requirement | Decision |
|---|---|
| Algorithm | `lightgbm.LGBMClassifier` |
| Training scope | Fit on all rows; no train/test split, CV, or hyperparameter tuning |
| Input table shape | Fixed to dummy-data schema: `feature_000`…`feature_032` + `label` |
| Input UC location | Three job parameters: `catalog`, `schema`, `table_name` |
| Model registry location | Three job parameters: `model_catalog`, `model_schema`, `model_name` |
| ONNX export | Native `booster.save_model(path, format="onnx")` → `onnx.load` |
| Model format in registry | ONNX via `mlflow.onnx.log_model` |
| Task type | Notebook (`src/jobs/train_lightgbm.ipynb`) |
| Compute | Serverless environment with pip deps, client `"5"` |
| Schema management | Bundle-managed UC schemas: `data` (tables) and `model` (models) |
| Notebook layout | Multi-cell with section headers (same as logistic regression) |
| Default model name | `lightgbm_onnx` (new bundle variable `lightgbm_model_name`) |

## Architecture

Two bundle-managed UC schemas are created/updated during `databricks bundle deploy`. The
training job reads from the data schema and registers to the model schema at runtime.

```
bundle deploy
  ├── resources.schemas.data   →  {catalog}.data
  └── resources.schemas.model  →  {catalog}.model

bundle run create_dummy_data          (prerequisite)
  └── writes {catalog}.data.generated_data

bundle run train_lightgbm             (parallel to train_logistic_regression)
  └── notebook (serverless)
        ├── read table → pandas
        ├── LGBMClassifier.fit (all rows)
        ├── booster.save_model(format="onnx") → onnx.load
        └── mlflow.onnx.log_model → register {model_catalog}.{model_schema}.{model_name}
```

## Alternatives considered

| Approach | Why not chosen |
|---|---|
| `onnxmltools.convert_lightgbm` | Extra dependency and version quirks; native LightGBM export is simpler for a blog demo |
| Single parameterized notebook (`algorithm` widget) | Harder to excerpt for blog; unnecessary branching |
| Hardcode `lightgbm_onnx` in job YAML only (no bundle variable) | Inconsistent with how logistic regression uses `${var.model_name}` |
| Shared Python module for both training jobs | Over-engineered for a blog demo |

**Chosen approach:** Pure mirror of `train_logistic_regression` with LightGBM-specific
training and ONNX export.

## Components

### `databricks.yml`

Add one bundle variable:

```yaml
lightgbm_model_name:
  description: Registered model name for LightGBM ONNX artifact
  default: lightgbm_onnx
```

No other bundle changes. Existing `data` and `model` schema resources and target overrides
remain unchanged.

### `resources/train_lightgbm.yml`

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

### `src/jobs/train_lightgbm.ipynb`

Four code cells with markdown section headers above each:

#### Cell 1 — Parameters

1. Define widgets for all six parameters (populated by job `base_parameters`).
2. Validate all parameters are non-empty; raise `ValueError` otherwise.
3. Build `full_table = f"{catalog}.{schema}.{table_name}"`.
4. Build `full_model_name = f"{model_catalog}.{model_schema}.{model_name}"`.

#### Cell 2 — Load data

1. Read `spark.table(full_table).toPandas()`.
2. Set `y = pdf["label"]`.
3. Set `X = pdf[[c for c in pdf.columns if c.startswith("feature_")]]` (33 columns).
4. Print row count and feature count.

#### Cell 3 — Train & convert to ONNX

1. Instantiate `lgb.LGBMClassifier(random_state=42)` with default hyperparameters.
2. Call `model.fit(X, y)` on all rows.
3. Write ONNX to a temp file via `model.booster_.save_model(path, format="onnx")`.
4. Load with `onnx.load(path)` and clean up the temp file.
5. Print confirmation (no accuracy metrics or evaluation).

#### Cell 4 — Register in UC Model Registry

1. Call `mlflow.set_registry_uri("databricks-uc")`.
2. Build signature with `infer_signature(X, model.predict_proba(X))`.
3. Start an MLflow run.
4. Log the ONNX model with `mlflow.onnx.log_model(onnx_model, artifact_path="model", signature=signature)`.
5. Register with `mlflow.register_model(model_info.model_uri, full_model_name)`.
6. Print registered model name and version.

The notebook does **not** run `CREATE SCHEMA`. Both schemas must exist from a prior
`databricks bundle deploy`.

## Data flow

1. Operator runs `databricks bundle deploy --target dev` (or `prod`) to create/update UC
   schemas `data` and `model`.
2. Operator runs `databricks bundle run create_dummy_data` to populate the training table.
3. Operator runs `databricks bundle run train_lightgbm`.
4. Job resolves parameters (overrides or bundle defaults).
5. Notebook reads the UC table into pandas on serverless compute.
6. LightGBM trains on all 100k rows in driver memory.
7. Model exports to ONNX and registers at `{model_catalog}.{model_schema}.{model_name}`.

The logistic regression job is independent — either training job can run in any order
after dummy data exists.

## Error handling

- **Blank parameters:** notebook raises `ValueError` before reading data.
- **Schema missing:** job fails with a UC/Spark error; operator must deploy the bundle
  first. No runtime schema auto-creation.
- **Table missing or wrong shape:** fail fast with Spark/pandas errors (e.g. missing
  `label` or `feature_*` columns).
- **Permission errors:** fail fast with standard UC/Spark/MLflow exceptions.
- **Re-runs:** each run creates a new model version in the registry (expected for demo).

## Testing

Out of scope for v1 (notebook-only, no local unit tests).

Manual verification:

1. `databricks bundle deploy --target dev`
2. `databricks bundle run create_dummy_data`
3. `databricks bundle run train_lightgbm`
4. Confirm model appears in Unity Catalog at `{catalog}.model.lightgbm_onnx`
   with version 1 (or incremented on re-run).

Optional follow-up: notebook smoke test via existing pytest + Databricks Connect pattern
in `tests/conftest.py`.

## Out of scope

- Train/test split, cross-validation, or hyperparameter tuning
- Accuracy or other evaluation metrics
- Model serving endpoint deployment
- Generic column-name parameters (fixed dummy-data schema only)
- Shared Python module / wheel for testable training logic
- Refactoring common code between logistic regression and LightGBM notebooks
- Scheduled or triggered job runs (manual/bundle-run only for v1)
- Blog post publication (this spec covers the job only)
- Changes to the existing `train_logistic_regression` job
