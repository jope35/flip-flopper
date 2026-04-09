# Design: CatBoost demo training job (MLflow + Unity Catalog, serverless)

**Status:** Approved for implementation planning  
**Date:** 2026-04-09

## Purpose

Add a **Databricks Lakeflow job** (Declarative Automation Bundle) that **trains a CatBoost classifier** on a **small in-process demo dataset**, **logs the run to MLflow Experiment Tracking**, and **registers the model in Unity Catalog** under a **fixed three-level name** derived from bundle variables. Compute uses **serverless workflows** (no classic job cluster for this task).

This job is **standalone**: it does not configure the serving A/B endpoint. It can be used as a template for future production training jobs that read from UC tables.

## Non-goals

- Reading training data from Unity Catalog tables or volumes (demo data only).
- Hyperparameter search, distributed training, or GPU tuning.
- Automating deployment to a model serving endpoint or wiring into the traffic controller.
- Pinning a specific Databricks Runtime version for the training task (serverless uses a platform-managed runtime).

## Architecture

| Piece | Role |
|-------|------|
| **`src/jobs/train_catboost_demo.py`** | CLI entry: build synthetic data, fit `CatBoostClassifier`, MLflow log + UC register, exit non-zero on failure. |
| **Bundle job resource** | Declares serverless task (`spark_python_task` + `environment_key`), job-level `environments` with pip dependencies, parameters `${var.catalog}` / `${var.schema}` (and optional experiment variable). |
| **MLflow** | `mlflow.set_tracking_uri("databricks")`, `mlflow.set_experiment(...)`, `mlflow.catboost.log_model`, metrics/params on the run. |
| **Unity Catalog Model Registry** | `mlflow.set_registry_uri("databricks-uc")`, `mlflow.register_model` with name `{catalog}.{schema}.flip_flopper_catboost_demo`. |

## Data and training

- **Data:** `sklearn.datasets.make_classification` with a fixed `random_state` for reproducibility; modest `n_samples` / `n_features` suitable for a quick job.
- **Split:** Train/validation split (e.g. 80/20) for holdout metrics.
- **Model:** `catboost.CatBoostClassifier` with small, fast hyperparameters; log key params to MLflow (e.g. iterations, depth).
- **Metrics:** Log at least one validation metric (e.g. accuracy or log loss) to MLflow.

## MLflow and Unity Catalog

- **Experiment:** Use a dedicated experiment path (bundle variable with a default such as `/Shared/flip-flopper/catboost-demo`, or target-specific override in `databricks.yml`). Create the experiment if permitted; otherwise document required pre-provisioning.
- **Run:** Single `mlflow.start_run()` context per job execution.
- **Model artifact:** `mlflow.catboost.log_model` for the native CatBoost flavor.
- **Registration:** Full UC name `f"{catalog}.{schema}.flip_flopper_catboost_demo"` (suffix fixed; catalog/schema from bundle variables passed as task parameters).
- **Success visibility:** Print registered model URI/version (or equivalent) to stdout for job logs.

## Compute (serverless)

- **Task:** `spark_python_task` with `python_file` pointing at the training script (relative path from the resource file, consistent with existing bundle jobs).
- **No classic compute** on this task: omit `new_cluster`, `job_cluster_key`, and `existing_cluster_id`.
- **Job `environments`:** One environment (e.g. `catboost_train_env`) with:
  - `spec.client: "4"` (required for serverless environment spec).
  - `spec.dependencies`: pip requirements including **`catboost`** and **`scikit-learn`**; include or pin **`mlflow`** only if the serverless base environment’s MLflow version is incompatible with `mlflow.catboost` in validation.
- **Task:** Set `environment_key` to that environment.

**Note:** Serverless workflow [requirements](https://docs.databricks.com/en/jobs/run-serverless-jobs.html) include Unity Catalog enabled and workloads compatible with standard access mode.

## Bundle integration

- **Placement:** New job under `resources/*.yml` (append to `jobs.yml` or add a dedicated file included by `databricks.yml`).
- **Naming:** Job display name prefixed with `[${bundle.target}]` to match existing resources.
- **Parameters:** Pass `--catalog` and `--schema` from `${var.catalog}` and `${var.schema}`; optional `--experiment-name` from a new variable if desired.

## Error handling

- Do not swallow failures: experiment creation, logging, or UC registration errors should **fail the task** with a clear message.
- Avoid silent fallbacks (e.g. skipping registration).

## Permissions

The job **run-as identity** (user or service principal) needs:

- **MLflow:** Permission to create/use the target experiment and write runs.
- **Unity Catalog:** `USE CATALOG` / `USE SCHEMA` on the target catalog and schema, and privilege to **create or update** the registered model (e.g. `CREATE MODEL` / registry permissions per org policy).

Document any workspace-specific setup in the implementation plan or runbook.

## Testing

- **Bundle shape:** Extend existing tests (e.g. assert new job script path exists) if the project already checks bundle artifacts.
- **Optional unit tests:** Argument parsing and/or deterministic synthetic data shape without calling MLflow (mock or pure functions).
- **Validation:** At least one manual or CI-deployed run in a dev workspace to confirm serverless task starts, pip dependencies resolve, and UC registration succeeds.

## References

- [Run serverless compute for workflows](https://docs.databricks.com/en/jobs/run-serverless-jobs.html)
- [Add tasks to jobs in Declarative Automation Bundles](https://docs.databricks.com/en/dev-tools/bundles/job-task-types.html)
- [MLflow Model Registry on Unity Catalog](https://docs.databricks.com/aws/en/mlflow/models-in-uc)
