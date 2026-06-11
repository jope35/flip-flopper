## Learned User Preferences

- Prefer Databricks serverless environment `client: "5"` in job YAML `environments.spec.client` fields.
- When asked to run a bundle job, use `databricks bundle validate && databricks bundle deploy && databricks bundle run "<job_name>"` in that order.
- New model training jobs should mirror the logistic regression pattern (job YAML structure, notebook layout, UC registration flow).
- Training notebooks should use a multi-cell layout with markdown section headers (Parameters, Load data, Train & convert to ONNX, Register).
- Export tree models to ONNX with onnxmltools (not native LightGBM ONNX export).
- Export PyTorch models to ONNX with native `torch.onnx.export` (not onnxmltools).
- Use serverless compute for jobs rather than classic clusters.
- Keep training jobs and models simple; this repo is a blog-post demo, not production ML.
- Model serving deploy jobs mirror the training job pattern: resolve latest UC versions at runtime, idempotent create-or-update, block until READY, set `scale_to_zero_enabled=True`, and do NOT configure inference tables (`auto_capture_config` or AI Gateway).
- Serving endpoint ops: never auto-delete non-READY endpoints (wait on IN_PROGRESS; fail on UPDATE_FAILED); compare models by UC `entity_name`+version; prefer bulk create with incremental 1→2→3 fallback on quota errors.

## Learned Workspace Facts

- Databricks Asset Bundle project named `flip_flopper`; default target is `dev` on workspace `https://dbc-b9d925fd-a82e.cloud.databricks.com`.
- Unity Catalog layout: catalog `workspace`; schemas `data` and `model` for tables and registered models (bundle-managed in `databricks.yml`). With the default `dev` target (`mode: development`), deploy prefixes schema names to `dev_<user>_<schema>`; prod uses unprefixed `workspace.data` and `workspace.model`.
- Registered ONNX models: `logistic_regression_onnx`, `lightgbm_onnx`, `pytorch_mlp_onnx` (shared `feature_000`…`feature_032` input schema). In UC they appear as both Models and Functions.
- Job definitions live in `resources/*.yml`; training and data notebooks live in `src/jobs/`.
- Current jobs: `create_dummy_data`, `train_logistic_regression`, `train_lightgbm`, `train_pytorch_mlp`, and `deploy_serving_endpoint`.
- Bundle variable `serving_endpoint_name` (default `flip_flopper_serving`) configures the Model Serving endpoint name.
- Full pipeline validation: `bundle destroy --target dev --auto-approve` → validate/deploy → `create_dummy_data` → `train_*` → `deploy_serving_endpoint`.
- Serverless job environments declare pip dependencies directly in each job YAML under `environments.spec.dependencies`.
- Local Python dependencies are managed with uv (`uv sync --dev`); tests run with `uv run pytest`.
