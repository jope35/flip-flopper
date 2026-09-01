## Learned User Preferences

- Prefer Databricks serverless environment `client: "5"` in job YAML `environments.spec.client` fields.
- When asked to run a bundle job, use `databricks bundle validate && databricks bundle deploy && databricks bundle run "<job_name>"` in that order.
- New model training tasks should mirror the logistic regression pattern (notebook layout, UC registration flow); add them as tasks in `resources/run_pipeline.yml`.
- Training notebooks should use a multi-cell layout with markdown section headers (Parameters, Load data, Train & convert to ONNX, Register); keep `src/jobs/` as Databricks source `.py` format (`# Databricks notebook source`, `# COMMAND ----------`, `# MAGIC %md`), not `.ipynb`.
- Export tree models to ONNX with onnxmltools (not native LightGBM ONNX export); for XGBoost call `convert_xgboost` without `zipmap` (unsupported), and train on NumPy arrays (`X.to_numpy()`) so feature names match the `f%d` pattern onnxmltools expects; for SparkML Random Forest train on Spark (`VectorAssembler` + `RandomForestClassifier`), convert the fitted RF stage with `convert_sparkml` (`target_opset`), not the full Pipeline; on serverless patch `save_read_sparkml_model_data` and use UC Volume `onnx_scratch` (onnxmltools reads `spark.master`, which Spark Connect blocks; managed storage paths reject ad-hoc writes).
- Wrap each ONNX model in an MLflow pyfunc returning named columns `target` (predicted class), `proba`, and `model_name` so multi-model serving responses identify which model answered; disable scikit-learn ONNX ZipMap (`zipmap=False`) on the logistic regression export so the wrapper receives a probability matrix, not a dict; PyTorch MLP ONNX exports a single sigmoid output—use `.squeeze(-1)` for `proba`.
- For pyfunc model serving, declare full `mlflow` (not `mlflow-skinny`) in pip dependencies, and validate the pyfunc output shape locally (`mlflow.pyfunc.load_model().predict()`, requires `onnxruntime`) before UC registration since bundle validation can't prove runtime output shape.
- Use serverless compute for jobs rather than classic clusters.
- Keep training jobs and models simple; this repo is a blog-post demo, not production ML.
- Apply ponytail minimal-code principles to `src/jobs/` notebooks; mark intentional shortcuts with `ponytail:` comments.
- Model serving deploy jobs mirror the training job pattern: resolve latest UC versions at runtime, idempotent create-or-update, block until READY, set `scale_to_zero_enabled=True`, and do NOT configure inference tables (`auto_capture_config` or AI Gateway).
- Serving endpoint ops: never auto-delete non-READY endpoints (wait on IN_PROGRESS; fail on UPDATE_FAILED); compare models by UC `entity_name`+version; prefer bulk create with incremental 1→2→3 fallback on quota errors.

## Learned Workspace Facts

- Databricks Asset Bundle project named `flip_flopper`; default target is `dev` on workspace `https://dbc-b9d925fd-a82e.cloud.databricks.com`.
- Unity Catalog layout: catalog `workspace`; schemas `data` and `model` for tables and registered models (bundle-managed in `databricks.yml`). With the default `dev` target (`mode: development`), deploy prefixes schema names to `dev_<user>_<schema>`; prod uses unprefixed `workspace.data` and `workspace.model`.
- Registered models `logistic_regression_onnx`, `lightgbm_onnx`, `xgboost_onnx`, `pytorch_mlp_onnx`, and `sparkml_random_forest_onnx` are ONNX-backed pyfuncs sharing `feature_000`…`feature_032` (double) inputs and output `target` (long), `proba` (double), `model_name` (string). In UC they appear as both Models and Functions.
- Job definitions live in `resources/*.yml`; training and data notebooks live in `src/jobs/`.
- Current jobs: `run_pipeline` (single multi-task job: `create_dummy_data` → all `train_*` in parallel → `deploy_serving_endpoint`) and `query_serving_endpoint`.
- Bundle variable `serving_endpoint_name` (default `flip_flopper_serving`) configures the Model Serving endpoint name; endpoint serves `logistic_regression`, `lightgbm`, `xgboost`, `pytorch_mlp`, and `random_forest` with traffic split 20/20/20/20/20 (CPU Small, scale-to-zero). Callers can use traffic-split endpoint invocation or direct per-model served-entity calls.
- Full pipeline validation: `bundle destroy --target dev --auto-approve` → validate/deploy → `databricks bundle run "run_pipeline"`.
- Serverless job environments declare pip dependencies in `resources/run_pipeline.yml` under `environments.spec.dependencies` (one environment per task).
- Local Python dependencies are managed with uv (`uv sync --dev`); tests run with `uv run pytest`.

## OpenWiki

This repository has documentation located in the /openwiki directory.

Start here:
- [OpenWiki quickstart](openwiki/quickstart.md)

OpenWiki includes repository overview, architecture notes, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

When working in this repository, read the OpenWiki quickstart first, then follow its links to the relevant architecture, workflow, domain, operation, and testing notes.
