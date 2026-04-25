## Learned User Preferences

- When asked to run a Databricks bundle job, use this sequence: `databricks bundle validate && databricks bundle deploy && databricks bundle run "<job_key>"` (replace `<job_key>` with the job resource name, for example `create_dummy_data`). Always validate, then deploy, then run—do not skip validate or deploy unless the user says otherwise.
- For `notebook_task` entrypoints that use `dbutils.widgets`, pass values through `notebook_task.base_parameters` keyed to widget names. For `spark_python_task`, `parameters` are command-line arguments to the Python file—read them with `sys.argv` or argparse, not `dbutils.widgets.get`.
- Databricks job notebooks and scripts here are meant to run only on Databricks; use `dbutils` directly for widgets instead of `globals()["dbutils"]` (use a targeted linter suppression for undefined `dbutils` if needed).
- When logging ONNX with `mlflow.onnx.log_model`, an extra `mlflow.log_artifact` of the raw `.onnx` is usually redundant unless something downstream needs a standalone file path.
- For Ruff, exclude `.cursor/` and `.claude/` from checks, and treat Databricks-injected notebook globals (`dbutils`, `spark`) as builtins to avoid false undefined-name lint errors.
- Prefer a `repo: local` pre-commit hook for Ruff formatting (use the project’s installed `ruff`) instead of pinning `ruff-pre-commit` by remote `rev`.

## Learned Workspace Facts

- This repo is meant to be open-sourced; keep changes simple and easy to understand (KISS).
- Classifier training notebooks default the `table_name` widget to `generated_data`; bundle jobs must point at a table that exists (for example run the `create_dummy_data` job first) or override `table_name` / job parameters accordingly.
- Sklearn RBF `SVC(probability=True)` is much more sensitive to training row count and runtime than tree models; when copying RF-style training notebooks, keep conservative `max_training_rows` defaults or add explicit size/runtime warnings.
- Each Databricks bundle job task must declare compute: `new_cluster`, `job_cluster_key`, `existing_cluster_id`, or `environment_key`; a bare task without one fails bundle validation.
- For classic single-node driver-only job clusters in this bundle, use `spark_version` 16.1.x-scala2.12, `node_type_id` i3.xlarge, `num_workers` 0, `spark.master` local[*], `spark.databricks.cluster.profile` singleNode, and `custom_tags` ResourceClass SingleNode.
- Serverless job tasks use job-level `environments` (for example `spec.environment_version` and `dependencies` for PyPI packages or `../dist/*.whl`) and the task `environment_key`; omit cluster fields on those tasks.
- In `resources/*.yml`, paths to bundle-root files use `../` (for example `notebook_path: ../src/jobs/...`).
- Unity Catalog schemas in Databricks Asset Bundles belong under **`resources.schemas`** in included YAML (for example `resources/schemas.yml`), not as a top-level `schemas:` key in `databricks.yml` (that key is invalid bundle config). Each schema is a named resource with at least `name` and `catalog_name`; optional fields include `comment`, `grants`, `lifecycle`, `properties`, and `storage_root` (see [bundle resources: schema](https://docs.databricks.com/aws/en/dev-tools/bundles/resources#schema-unity-catalog)).
- The bundle defines a wheel artifact (`artifacts.flip_flopper` with `build: uv build --wheel`); serverless job `environments` often list `../dist/*.whl` under `spec.dependencies` (path relative to the `resources/` file) so tasks install the built package.
- Unity Catalog requires MLflow models registered to UC to include full signature metadata (inputs and outputs). For `mlflow.onnx.log_model`, pass `signature=` (for example from `infer_signature` after an ONNX Runtime forward pass with representative tensors)—`input_example` alone may not populate what UC expects for ONNX.
- Jobs parameterize catalog and schema via bundle variables and schema resources (for example `${var.catalog}` and `${resources.schemas.default.name}`); notebook widget defaults may use project naming like `flip_flopper` where configured.



## user-specific guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.