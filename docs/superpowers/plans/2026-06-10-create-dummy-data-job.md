# Create Dummy Data Job — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal Databricks Asset Bundle job that generates 100k sklearn rows and writes them to a Unity Catalog table — simple enough to explain in a blog post.

**Architecture:** Bundle deploy creates the UC schema; a single serverless notebook task generates data with `make_classification`, converts pandas → Spark, and overwrites one managed table. Three job parameters (`catalog`, `schema`, `table_name`) control the destination. No wheels, no shared modules, no tests — just the happy path.

**Tech Stack:** Databricks Asset Bundles, serverless job environments, Unity Catalog, pandas, scikit-learn, PySpark

**Design reference:** [2026-06-10-create-dummy-data-job-design.md](../specs/2026-06-10-create-dummy-data-job-design.md)

---

## File map

| File | Responsibility |
|---|---|
| `databricks.yml` | Add `table_name` variable + bundle-managed UC schema |
| `resources/create_dummy_data.yml` | Job definition (parameters, serverless env, notebook task) |
| `src/jobs/create_dummy_data.ipynb` | Generate data and write table |

---

### Task 1: Bundle variables and UC schema

**Files:**
- Modify: `databricks.yml`

- [ ] **Step 1: Add `table_name` variable**

Add under the existing `variables:` block:

```yaml
  table_name:
    description: Table name for generated dummy data
    default: generated_data
```

- [ ] **Step 2: Add bundle-managed schema resource**

Add a top-level `resources:` block (after `variables:`, before `targets:`):

```yaml
resources:
  schemas:
    default:
      name: ${var.schema}
      catalog_name: ${var.catalog}
```

Existing `dev` / `prod` target overrides stay unchanged.

- [ ] **Step 3: Validate bundle syntax**

Run:

```bash
databricks bundle validate --target dev
```

Expected: validation succeeds with no schema/job errors.

---

### Task 2: Job resource

**Files:**
- Create: `resources/create_dummy_data.yml`

- [ ] **Step 1: Create the job YAML**

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

- [ ] **Step 2: Validate again**

Run:

```bash
databricks bundle validate --target dev
```

Expected: job `create_dummy_data` appears in validation output.

---

### Task 3: Notebook

**Files:**
- Create: `src/jobs/create_dummy_data.ipynb` (single Python code cell)

- [ ] **Step 1: Create notebook with one cell**

```python
import pandas as pd
from sklearn.datasets import make_classification

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_name", "")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
table_name = dbutils.widgets.get("table_name").strip()

if not all([catalog, schema, table_name]):
    raise ValueError("catalog, schema, and table_name must all be non-empty")

full_table = f"{catalog}.{schema}.{table_name}"

X, y = make_classification(
    n_samples=100_000,
    n_features=33,
    n_informative=26,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=42,
)

feature_cols = [f"feature_{i:03d}" for i in range(33)]
pdf = pd.DataFrame(X, columns=feature_cols)
pdf["label"] = y.astype("int64")

spark_df = spark.createDataFrame(pdf)
spark_df.write.mode("overwrite").saveAsTable(full_table)

print(f"Wrote {spark_df.count()} rows to {full_table}")
```

Keep it as one cell — easy to screenshot for the blog.

---

### Task 4: Deploy and smoke-test

No automated tests. Manual verification only.

- [ ] **Step 1: Deploy bundle**

```bash
databricks bundle deploy --target dev
```

Expected: UC schema `workspace.dev` created/updated; job deployed.

- [ ] **Step 2: Run the job**

```bash
databricks bundle run create_dummy_data --target dev
```

Expected: job succeeds; notebook prints `Wrote 100000 rows to workspace.dev.generated_data`.

- [ ] **Step 3: Confirm row count**

In a SQL warehouse or notebook:

```sql
SELECT COUNT(*) FROM workspace.dev.generated_data;
```

Expected: `100000`.

---

## Blog-post talking points

Use this flow in the post:

1. **Schema as infrastructure** — `resources.schemas` in the bundle, not `CREATE SCHEMA` in the notebook.
2. **Serverless + pip deps** — job-level `environments` instead of `%pip install` or a cluster.
3. **Parameterized destination** — three widgets/parameters, not one FQN string.
4. **Idempotent writes** — `overwrite` so re-runs are safe for demos.

## Explicitly out of scope

- Unit tests, wheels, shared Python modules
- Parameterized row count
- Schedules, alerts, monitoring
- Production error handling beyond blank-parameter check

## Spec coverage

| Design requirement | Task |
|---|---|
| 100k rows, fixed sklearn params | Task 3 |
| Three job parameters | Tasks 1–2 |
| Overwrite write mode | Task 3 |
| Serverless + pandas/sklearn | Task 2 |
| Bundle-managed UC schema | Task 1 |
| Manual verification | Task 4 |
