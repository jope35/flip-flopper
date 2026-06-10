# Create Dummy Data Job — Design

**Date:** 2026-06-10  
**Status:** Approved

## Summary

Add a Databricks job that generates a 100,000-row binary classification dataset using
`sklearn.datasets.make_classification` and writes it to a Unity Catalog table. The UC
destination is controlled by three job parameters (`catalog`, `schema`, `table_name`) with
defaults from bundle configuration. The UC schema is bundle-managed (deployed via
`resources.schemas.default`), not created at runtime by the notebook.

## Requirements

| Requirement | Decision |
|---|---|
| Row count | Fixed at 100,000 in notebook code |
| Data generator | `sklearn.datasets.make_classification` |
| UC destination | Three job parameters: `catalog`, `schema`, `table_name` |
| Write mode | Overwrite table on each run |
| Task type | Notebook (`src/jobs/create_dummy_data.ipynb`) |
| Compute | Serverless environment with `pandas` and `scikit-learn` pip deps |
| Schema management | Bundle-managed UC schema resource (Approach 2) |

### Dataset shape

Reuse settings from the previous implementation for downstream classifier compatibility:

- `n_samples=100_000`
- `n_features=33`, `n_informative=26`, `n_redundant=0`, `n_repeated=0`
- `n_classes=2`
- `random_state=42`

Output columns: `feature_000` … `feature_032` (double), `label` (int64).

## Architecture

Single-task Databricks job deployed via Asset Bundle. The UC schema is created/updated
during `databricks bundle deploy`. The job writes only the table at runtime.

```
bundle deploy
  └── resources.schemas.default  →  {catalog}.{schema}

bundle run create_dummy_data
  └── notebook (serverless)
        └── make_classification → pandas → Spark → saveAsTable (overwrite)
              └── {catalog}.{schema}.{table_name}
```

## Components

### `databricks.yml`

Add a `table_name` bundle variable and a UC schema resource:

```yaml
variables:
  catalog:
    description: The catalog to use
  schema:
    description: The schema to use
  table_name:
    description: Table name for generated dummy data
    default: generated_data

resources:
  schemas:
    default:
      name: ${var.schema}
      catalog_name: ${var.catalog}
```

Existing target overrides remain unchanged (`dev`: catalog `workspace`, schema `dev`;
`prod`: catalog `workspace`, schema `prod`).

### `resources/create_dummy_data.yml`

```yaml
resources:
  jobs:
    create_dummy_data:
      name: create_dummy_data
      parameters:
        - name: catalog
          default: ${var.catalog}
        - name: schema
          default: ${resources.schemas.default.name}
        - name: table_name
          default: ${var.table_name}
      environments:
        - environment_key: dummy_data_env
          spec:
            client: "4"
            dependencies:
              - pandas
              - scikit-learn
      tasks:
        - task_key: create_dummy_data
          environment_key: dummy_data_env
          notebook_task:
            notebook_path: ../src/jobs/create_dummy_data.ipynb
            base_parameters:
              catalog: "{{job.parameters.catalog}}"
              schema: "{{job.parameters.schema}}"
              table_name: "{{job.parameters.table_name}}"
```

### `src/jobs/create_dummy_data.ipynb`

Single code cell that:

1. Defines widgets for `catalog`, `schema`, `table_name` (populated by job `base_parameters`).
2. Validates all three parameters are non-empty; raises `ValueError` otherwise.
3. Calls `make_classification` with the fixed parameters above.
4. Builds a pandas DataFrame with named feature columns and an int64 `label` column.
5. Converts to a Spark DataFrame via `spark.createDataFrame`.
6. Writes with `spark_df.write.mode("overwrite").saveAsTable(full_table)`.
7. Prints row count and full table name.

The notebook does **not** run `CREATE SCHEMA`. The schema must exist from a prior
`databricks bundle deploy`.

## Data flow

1. Operator runs `databricks bundle deploy --target dev` (or `prod`) to create/update
   the UC schema.
2. Operator runs `databricks bundle run create_dummy_data` (optionally overriding
   parameters).
3. Job resolves parameters (overrides or bundle defaults).
4. Notebook runs on serverless compute with `pandas` and `scikit-learn` available.
5. Pandas generates 100k rows in driver memory (~25 MB).
6. Spark overwrites the managed UC Delta table at `{catalog}.{schema}.{table_name}`.

## Error handling

- **Blank parameters:** notebook raises `ValueError` before generating data.
- **Schema missing:** job fails with a UC/Spark error; operator must deploy the bundle
  first. No runtime schema auto-creation.
- **Permission errors:** fail fast with standard UC/Spark exceptions.
- **Re-runs:** overwrite replaces table contents (idempotent destination, fresh data).

## Testing

Out of scope for v1 (notebook-only, no local unit tests).

Manual verification:

1. `databricks bundle deploy --target dev`
2. `databricks bundle run create_dummy_data`
3. `SELECT COUNT(*) FROM {catalog}.{schema}.{table_name}` → 100000

Optional follow-up: notebook smoke test via existing pytest + Databricks Connect pattern
in `tests/conftest.py`.

## Out of scope

- Parameterized row count (`n_samples` job parameter)
- Append or fail-if-exists write modes
- Shared Python module / wheel for testable generation logic
- Scheduled or triggered job runs (manual/bundle-run only for v1)

## Alternatives considered

| Approach | Why not chosen |
|---|---|
| Minimal notebook job with runtime `CREATE SCHEMA IF NOT EXISTS` | User chose bundle-managed schema (Approach 2) |
| Python wheel task with testable module | User chose notebook-only implementation |
| Single fully-qualified table parameter | User chose three separate parameters |
