# CatBoost demo job (MLflow + UC, serverless) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a serverless Databricks job that trains a CatBoost model on synthetic sklearn data, logs to MLflow Experiment Tracking, and registers the model in Unity Catalog as `{catalog}.{schema}.flip_flopper_catboost_demo`.

**Architecture:** One Python entry script under `src/jobs/` uses lazy imports so local pytest can exercise CLI parsing without CatBoost/MLflow installed. The bundle declares a job-level serverless `environments` block (`client: "4"`, pip deps) and a `spark_python_task` with `environment_key` and no cluster attachment. Bundle variables supply catalog, schema, and experiment path.

**Tech Stack:** Databricks Declarative Automation Bundles, Lakeflow serverless workflows, CatBoost, scikit-learn, MLflow (Databricks tracking + UC registry)

---

## Planned file structure

- `databricks.yml` — add variable `catboost_experiment_name` (default `/Shared/flip-flopper/catboost-demo`).
- `resources/jobs.yml` — append job `train_catboost_demo` with `environments` + serverless task. If `resources/jobs.yml` does not exist, create it with a `resources:` → `jobs:` root containing at least this job (and any other jobs already required by the bundle).
- `src/jobs/train_catboost_demo.py` — argparse + `main()` with lazy imports; MLflow + CatBoost train/register.
- `tests/unit/test_train_catboost_demo_cli.py` — load script by path with `importlib`; test `parse_args` without heavy deps.
- `tests/unit/test_bundle_shape.py` — create if missing, or extend: assert `src/jobs/train_catboost_demo.py` exists (and keep any existing path assertions).

**Docs:** Spec is `docs/superpowers/specs/2026-04-09-catboost-mlflow-uc-job-design.md`. Optional: add a short bullet to `docs/runbooks/databricks-serving-ab-test.md` or a one-line README pointer for “train catboost demo job” — only if the repo already documents jobs there.

---

### Task 1: Failing test for `parse_args` (no CatBoost/MLflow import)

**Files:**

- Create: `tests/unit/test_train_catboost_demo_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_train_catboost_demo_cli.py
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_train_catboost_demo_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "src" / "jobs" / "train_catboost_demo.py"
    spec = importlib.util.spec_from_file_location("train_catboost_demo", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_args_required_catalog_schema() -> None:
    mod = _load_train_catboost_demo_module()
    ns = mod.parse_args(["--catalog", "main", "--schema", "serving"])
    assert ns.catalog == "main"
    assert ns.schema == "serving"
    assert ns.experiment_name == "/Shared/flip-flopper/catboost-demo"
    assert ns.registered_model_suffix == "flip_flopper_catboost_demo"


def test_parse_args_custom_experiment() -> None:
    mod = _load_train_catboost_demo_module()
    ns = mod.parse_args(
        [
            "--catalog",
            "c",
            "--schema",
            "s",
            "--experiment-name",
            "/Users/me@example.com/catboost-exp",
        ]
    )
    assert ns.experiment_name == "/Users/me@example.com/catboost-exp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_train_catboost_demo_cli.py -v`

Expected: **FAIL** — `FileNotFoundError` or `ModuleNotFoundError` / loader error because `src/jobs/train_catboost_demo.py` does not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_train_catboost_demo_cli.py
git commit -m "test: add train_catboost_demo CLI parse_args tests"
```

---

### Task 2: Minimal `train_catboost_demo.py` (stdlib + `parse_args` + `main` stub)

**Files:**

- Create: `src/jobs/train_catboost_demo.py`

- [ ] **Step 1: Implement module with argparse only at top level**

```python
# src/jobs/train_catboost_demo.py
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost on demo data; log to MLflow; register model in Unity Catalog.",
    )
    parser.add_argument("--catalog", required=True, help="Unity Catalog catalog name")
    parser.add_argument("--schema", required=True, help="Unity Catalog schema name")
    parser.add_argument(
        "--experiment-name",
        default="/Shared/flip-flopper/catboost-demo",
        help="MLflow experiment path",
    )
    parser.add_argument(
        "--registered-model-suffix",
        default="flip_flopper_catboost_demo",
        help="UC registered model name = {catalog}.{schema}.{suffix}",
    )
    return parser.parse_args()


def main() -> None:
    raise NotImplementedError("training pipeline — Task 3")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run CLI tests**

Run: `uv run pytest tests/unit/test_train_catboost_demo_cli.py -v`

Expected: **PASS**

- [ ] **Step 3: Commit**

```bash
git add src/jobs/train_catboost_demo.py
git commit -m "feat: scaffold train_catboost_demo job script CLI"
```

---

### Task 3: Bundle shape test for new script path

**Files:**

- Create or modify: `tests/unit/test_bundle_shape.py`

- [ ] **Step 1: Add or create assertion**

If the file does not exist:

```python
# tests/unit/test_bundle_shape.py
from pathlib import Path


def test_train_catboost_demo_script_exists() -> None:
    assert Path("src/jobs/train_catboost_demo.py").exists()
```

If the file already exists with other assertions, add:

```python
def test_train_catboost_demo_script_exists() -> None:
    assert Path("src/jobs/train_catboost_demo.py").exists()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/test_bundle_shape.py tests/unit/test_train_catboost_demo_cli.py -v`

Expected: **PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_bundle_shape.py
git commit -m "test: assert train_catboost_demo script path in bundle shape"
```

---

### Task 4: Full training, MLflow logging, and UC registration in `main()`

**Files:**

- Modify: `src/jobs/train_catboost_demo.py` (replace `main` body; keep `parse_args` signature compatible with tests)

- [ ] **Step 1: Replace `main` with lazy-import implementation**

```python
# src/jobs/train_catboost_demo.py
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost on demo data; log to MLflow; register model in Unity Catalog.",
    )
    parser.add_argument("--catalog", required=True, help="Unity Catalog catalog name")
    parser.add_argument("--schema", required=True, help="Unity Catalog schema name")
    parser.add_argument(
        "--experiment-name",
        default="/Shared/flip-flopper/catboost-demo",
        help="MLflow experiment path",
    )
    parser.add_argument(
        "--registered-model-suffix",
        default="flip_flopper_catboost_demo",
        help="UC registered model name = {catalog}.{schema}.{suffix}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from catboost import CatBoostClassifier
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    registered_name = f"{args.catalog}.{args.schema}.{args.registered_model_suffix}"

    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()
    if client.get_experiment_by_name(args.experiment_name) is None:
        client.create_experiment(args.experiment_name)
    mlflow.set_experiment(args.experiment_name)

    iterations = 80
    depth = 4
    learning_rate = 0.1

    with mlflow.start_run():
        mlflow.log_params(
            {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": learning_rate,
                "catalog": args.catalog,
                "schema": args.schema,
            }
        )

        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        mlflow.log_metric("val_accuracy", acc)

        model_info = mlflow.catboost.log_model(model, artifact_path="model")

    mv = mlflow.register_model(model_info.model_uri, registered_name)
    print(f"Registered model: {registered_name} version {mv.version}")
    print(f"Model URI: models:/{registered_name}/{mv.version}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run unit tests (still no heavy imports at module load)**

Run: `uv run pytest tests/unit/test_train_catboost_demo_cli.py tests/unit/test_bundle_shape.py -v`

Expected: **PASS**

- [ ] **Step 3: Commit**

```bash
git add src/jobs/train_catboost_demo.py
git commit -m "feat: CatBoost demo train, MLflow log, UC register_model"
```

---

### Task 5: Bundle variable for experiment path

**Files:**

- Modify: `databricks.yml`

- [ ] **Step 1: Add variable under `variables:`**

After existing variables (e.g. after `controller_cron`), add:

```yaml
  catboost_experiment_name:
    default: /Shared/flip-flopper/catboost-demo
```

- [ ] **Step 2: Commit**

```bash
git add databricks.yml
git commit -m "chore: add catboost_experiment_name bundle variable"
```

---

### Task 6: Serverless job definition

**Files:**

- Modify or create: `resources/jobs.yml`

- [ ] **Step 1: Append job** (preserve existing `resources:` / `jobs:` keys if the file already contains other jobs)

If `resources/jobs.yml` is **empty or missing**, create it with this full content (add sibling jobs later as needed):

```yaml
resources:
  jobs:
    train_catboost_demo:
      name: "[${bundle.target}] train catboost demo"
      environments:
        - environment_key: catboost_train_env
          spec:
            client: "4"
            dependencies:
              - catboost>=1.2
              - scikit-learn>=1.4.0
      tasks:
        - task_key: train_catboost_demo
          environment_key: catboost_train_env
          spark_python_task:
            python_file: ../src/jobs/train_catboost_demo.py
            parameters:
              - --catalog
              - ${var.catalog}
              - --schema
              - ${var.schema}
              - --experiment-name
              - ${var.catboost_experiment_name}
```

If the file **already** has `resources: jobs:` and other jobs, add only the `train_catboost_demo:` block at the same indentation as those job keys (siblings under `jobs:`).

**Important:** Do **not** add `new_cluster`, `job_cluster_key`, or `existing_cluster_id` on this task so it stays on serverless compute.

- [ ] **Step 2: Validate bundle (optional but recommended)**

Run: `databricks bundle validate -t dev`

Expected: **Success** (requires Databricks CLI and auth configured). If other resources are missing in a sparse checkout, fix bundle layout first.

- [ ] **Step 3: Commit**

```bash
git add resources/jobs.yml
git commit -m "feat: serverless job for CatBoost MLflow UC demo training"
```

---

### Task 7: Manual workspace validation

**Files:**

- None (operator check)

- [ ] **Step 1: Deploy**

Run: `databricks bundle deploy -t dev`

- [ ] **Step 2: Run job**

Run: `databricks bundle run train_catboost_demo -t dev`  
(or run the job from the Databricks UI)

Expected: Task succeeds; job logs show `Registered model: …` lines; MLflow UI shows a new run under the configured experiment; UC shows a new model version under `{catalog}.{schema}.flip_flopper_catboost_demo`.

- [ ] **Step 3: If pip / MLflow errors**

Add `mlflow` to `spec.dependencies` with a version compatible with the workspace, redeploy, and re-run. Remove redundant pin if the base serverless image already satisfies CatBoost logging.

---

### Task 8: Permissions and experiment path checklist (documentation only)

**Files:**

- Optional modify: `docs/runbooks/databricks-serving-ab-test.md` (or README)

- [ ] **Step 1: Document**

Add a short subsection listing:

- Job run-as identity needs UC `USE CATALOG` / `USE SCHEMA` and model registry create/update on the target schema.
- MLflow experiment create under `/Shared/...` may require elevated permissions; use a user-scoped experiment path or pre-create the experiment if `create_experiment` fails.

- [ ] **Step 2: Commit** (skip commit if no doc edits)

```bash
git add docs/runbooks/databricks-serving-ab-test.md
git commit -m "docs: CatBoost demo job permissions and experiment notes"
```

---

## Self-review (spec coverage)

| Spec section | Tasks |
|--------------|-------|
| Purpose (CatBoost, demo data, MLflow, UC, serverless) | 4, 6 |
| Non-goals | Acknowledged; not implemented |
| Data / train / val split / metrics | Task 4 |
| MLflow experiment + `catboost.log_model` + UC `register_model` | Task 4 |
| Serverless `environments`, `client: "4"`, no cluster | Task 6 |
| Bundle variables catalog / schema / experiment | 5, 6 |
| Error handling (fail on errors) | Task 4 — exceptions propagate; no swallow |
| Tests (CLI + bundle shape) | 1–3 |
| Manual validation | Task 7 |

**Placeholder scan:** None intentional.

**Type/signature consistency:** `parse_args()` and CLI flags match job `parameters` and tests.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-09-catboost-mlflow-uc-job-implementation.md`. Two execution options:**

**1. Subagent-driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline execution** — Run tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
