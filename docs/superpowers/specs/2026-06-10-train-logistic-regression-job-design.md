# Train Logistic Regression Job — Design

**Date:** 2026-06-10  
**Status:** Approved

## Summary

Add a Databricks job that reads a binary classification table from Unity Catalog, trains a
basic `LogisticRegression` model on all rows, converts it to ONNX, and registers it in the
Unity Catalog Model Registry. Input and output UC locations are controlled by six job
parameters. The UC `data` and `model` schemas are bundle-managed. The notebook uses a
multi-cell layout with section headers for readability in Databricks and for blog excerpts.

**Prerequisite:** run `create_dummy_data` first to populate `{catalog}.data.generated_data`.

## Requirements

| Requirement | Decision |
|---|---|
| Algorithm | `sklearn.linear_model.LogisticRegression` |
| Training scope | Fit on all rows; no train/test split, CV, or hyperparameter tuning |
| Input table shape | Fixed to dummy-data schema: `feature_000`…`feature_032` + `label` |
| Input UC location | Three job parameters: `catalog`, `schema`, `table_name` |
| Model registry location | Three job parameters: `model_catalog`, `model_schema`, `model_name` |
| Model format | ONNX via `skl2onnx` |
| Task type | Notebook (`src/jobs/train_logistic_regression.ipynb`) |
| Compute | Serverless environment with pip deps |
| Schema management | Bundle-managed UC schemas: `data` (tables) and `model` (models) |
| Notebook layout | Multi-cell with section headers (Approach 2) |

## Architecture

Two bundle-managed UC schemas are created/updated during `databricks bundle deploy`. The
training job reads from the data schema and registers to the model schema at runtime.

```
bundle deploy
  ├── resources.schemas.data   →  {catalog}.data
  └── resources.schemas.model  →  {catalog}.model

bundle run create_dummy_data          (prerequisite)
  └── writes {catalog}.data.generated_data

bundle run train_logistic_regression
  └── notebook (serverless)
        ├── read table → pandas
        ├── LogisticRegression.fit (all rows)
        ├── skl2onnx convert
        └── mlflow.onnx.log_model → register {model_catalog}.{model_schema}.{model_name}
```

## Components

### `databricks.yml`

Add `model_schema` and `model_name` bundle variables. Rename the existing schema resource
to `data` and add a `model` schema resource. Update target defaults to use literal schema
names `data` and `model` (same in dev and prod).

```yaml
variables:
  catalog:
    description: The catalog to use
  schema:
    description: UC schema for data tables
    default: data
  model_schema:
    description: UC schema for registered models
    default: model
  table_name:
    description: Table name for generated dummy data
    default: generated_data
  model_name:
    description: Registered model name for logistic regression ONNX artifact
    default: logistic_regression_onnx

resources:
  schemas:
    data:
      name: ${var.schema}
      catalog_name: ${var.catalog}
    model:
      name: ${var.model_schema}
      catalog_name: ${var.catalog}
```

Target overrides (both dev and prod):

```yaml
variables:
  catalog: workspace
  schema: data
  model_schema: model
```

Update `resources/create_dummy_data.yml` to reference `resources.schemas.data` instead of
`resources.schemas.default`.

### `resources/train_logistic_regression.yml`

```yaml
resources:
  jobs:
    train_logistic_regression:
      name: train_logistic_regression
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
          default: ${var.model_name}
      environments:
        - environment_key: train_env
          spec:
            client: "4"
            dependencies:
              - pandas
              - scikit-learn
              - skl2onnx
              - onnx
      tasks:
        - task_key: train_logistic_regression
          environment_key: train_env
          notebook_task:
            notebook_path: ../src/jobs/train_logistic_regression.ipynb
            base_parameters:
              catalog: "{{job.parameters.catalog}}"
              schema: "{{job.parameters.schema}}"
              table_name: "{{job.parameters.table_name}}"
              model_catalog: "{{job.parameters.model_catalog}}"
              model_schema: "{{job.parameters.model_schema}}"
              model_name: "{{job.parameters.model_name}}"
```

### `src/jobs/train_logistic_regression.ipynb`

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

1. Instantiate `LogisticRegression(max_iter=1000, random_state=42)`.
2. Call `model.fit(X, y)` on all rows.
3. Build ONNX input type: `FloatTensorType([None, X.shape[1]])`.
4. Convert with `skl2onnx.convert_sklearn(model, initial_types=initial_type)`.
5. Print confirmation (no accuracy metrics or evaluation).

#### Cell 4 — Register in UC Model Registry

1. Call `mlflow.set_registry_uri("databricks-uc")`.
2. Start an MLflow run.
3. Log the ONNX model with `mlflow.onnx.log_model(onnx_model, artifact_path="model")`.
4. Register with `mlflow.register_model(model_uri, full_model_name)`.
5. Print registered model name and version.

The notebook does **not** run `CREATE SCHEMA`. Both schemas must exist from a prior
`databricks bundle deploy`.

## Data flow

1. Operator runs `databricks bundle deploy --target dev` (or `prod`) to create/update UC
   schemas `data` and `model`.
2. Operator runs `databricks bundle run create_dummy_data` to populate the training table.
3. Operator runs `databricks bundle run train_logistic_regression`.
4. Job resolves parameters (overrides or bundle defaults).
5. Notebook reads the UC table into pandas on serverless compute.
6. Logistic regression trains on all 100k rows in driver memory.
7. Model converts to ONNX and registers at `{model_catalog}.{model_schema}.{model_name}`.

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
3. `databricks bundle run train_logistic_regression`
4. Confirm model appears in Unity Catalog at `{catalog}.model.logistic_regression_onnx`
   with version 1 (or incremented on re-run).

Optional follow-up: notebook smoke test via existing pytest + Databricks Connect pattern
in `tests/conftest.py`.

## Out of scope

- Train/test split, cross-validation, or hyperparameter tuning
- Accuracy or other evaluation metrics
- Model serving endpoint deployment
- Generic column-name parameters (fixed dummy-data schema only)
- Shared Python module / wheel for testable training logic
- Scheduled or triggered job runs (manual/bundle-run only for v1)
- Blog post publication (this spec covers the job only)

## Alternatives considered

| Approach | Why not chosen |
|---|---|
| Single-cell notebook (Approach 1) | User chose multi-cell with section headers (Approach 2) |
| MLflow sklearn autolog + separate ONNX export | Two artifacts; harder to explain in a blog post |
| Same catalog/schema for data and model with `model_name` only | User chose three independent registration parameters |
| Environment-prefixed schemas (`dev_data`, `prod_model`) | User chose literal `data` and `model` in both targets |
| Train/test split with accuracy metric | User chose fit-on-all-rows for minimum code |
