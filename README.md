<br/>
<h1 style="font-size: 6em;"><p align="center"> 🩴🩴🩴 Flip-Flopper 🩴🩴🩴 </p></h1>
<h2><p align="center">Serve multiple classical ML models from one Databricks endpoint</p></h2>

<h3><p align="center">A small Databricks bundle for ONNX-backed multi-model serving</p></h3>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python 3.12.3" src="https://img.shields.io/badge/Python-3.12.3-blue.svg" />
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img alt="uv" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" />
  </a>
  <a href="https://pre-commit.com/">
    <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=yellow" />
  </a>
  <a href="https://docs.astral.sh/ruff/">
    <img alt="Ruff" src="https://img.shields.io/badge/Ruff-%3E%3D0.15.7-563D7C?logo=ruff&logoColor=white" />
  </a>
  <a href="https://docs.databricks.com/en/dev-tools/bundles/index.html">
    <img alt="Declarative Automation Bundles" src="https://img.shields.io/badge/Declarative%20Automation-Bundles-ff3621.svg" />
  </a>
  <a href="https://github.com/jope35/flip-flopper/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/jope35/flip-flopper/commits/main">
    <img alt="GitHub last commit (branch)" src="https://img.shields.io/github/last-commit/jope35/flip-flopper/main" />
  </a>
</p>
<br/>

- [What is Flip-Flopper?](#what-is-flip-flopper)
- [Highlights](#highlights)
- [Prerequisites](#prerequisites)
- [Repository structure](#repository-structure)
- [Configuration](#configuration)
- [Quickstart](#quickstart)
- [How the pipeline works](#how-the-pipeline-works)
- [Why ONNX-backed pyfunc models?](#why-onnx-backed-pyfunc-models)
- [Querying the endpoint](#querying-the-endpoint)
- [Troubleshooting](#troubleshooting)

# What is Flip-Flopper?

Flip-Flopper is a tiny, opinionated example of serving several classical ML models from **one Databricks Model Serving endpoint**.

The main concept is that each model is converted to ONNX, wrapped as an MLflow pyfunc with the same input and output contract, registered in Unity Catalog, and deployed behind a single endpoint.

The endpoint can be called in two ways:
- hit the endpoint normally and let Databricks route traffic across all served models
- call an individual served model directly when you want `logistic_regression`, `lightgbm`, `xgboost`, `pytorch_mlp`, or `random_forest`

# Highlights

- 🩴 **One endpoint, five models**: `logistic_regression`, `lightgbm`, `xgboost`, `pytorch_mlp`, and `random_forest` are deployed as served entities behind the same Model Serving endpoint.
- 🎯 **Direct model calls**: use Databricks' individual served-model invocation path to bypass the endpoint traffic split when you want one specific model.
- 📦 **ONNX-backed pyfunc contract**: all five models return the same columns: `target`, `proba`, and `model_name`.
- 🧭 **Unity Catalog first**: generated data lands in a UC table, and trained models are registered as UC models.
- ↔️ **Traffic split between models**: by default, the serving endpoint automatically splits incoming traffic across all five served models (20% each), letting you test multi-model serving or target a specific model directly.

# Prerequisites

- Databricks CLI installed and authenticated against the workspace you want to use.
- Model Serving enabled in the Databricks workspace.
- Unity Catalog enabled, with permissions to create schemas, tables, registered models, and serving endpoints.
- `uv` for local development.

This project is a good fit for **[Databricks Free Edition](https://www.databricks.com/learn/free-edition)**—you are encouraged to sign up for a free workspace and deploy the full pipeline there.

Useful references:

- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/)
- [Serve multiple models to a model serving endpoint](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint)
- [Query serving endpoints for custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints)
- [Declarative Automation Bundles](https://docs.databricks.com/dev-tools/bundles/)

# Repository structure

```text
.
├── databricks.yml                         # bundle variables, schemas, targets
├── resources/
│   ├── run_pipeline.yml                   # full demo pipeline (data → train → deploy)
│   ├── query_serving_endpoint.yml         # sends a test request to the endpoint
│   └── onnx_scratch_volume.yml            # UC Volume scratch space for the SparkML ONNX export
├── src/jobs/
│   ├── create_dummy_data.py
│   ├── train_logistic_regression.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   ├── train_pytorch_mlp.py
│   ├── train_sparkml_random_forest.py
│   ├── deploy_serving_endpoint.py
│   └── query_serving_endpoint.py
└── tests/
```

# Configuration

The important bundle settings live in `databricks.yml`.

| Variable                | Default                     | Meaning                              |
| ----------------------- | --------------------------- | ------------------------------------ |
| `catalog`               | `workspace`                 | Unity Catalog catalog                |
| `schema`                | `data`                      | schema for generated data            |
| `model_schema`          | `model`                     | schema for registered models         |
| `table_name`            | `generated_data`            | generated feature table              |
| `model_name`            | `logistic_regression_onnx`  | registered logistic regression model |
| `lightgbm_model_name`   | `lightgbm_onnx`             | registered LightGBM model            |
| `xgboost_model_name`    | `xgboost_onnx`              | registered XGBoost model             |
| `pytorch_mlp_model_name`| `pytorch_mlp_onnx`          | registered PyTorch MLP model         |
| `sparkml_rf_model_name` | `sparkml_random_forest_onnx`| registered SparkML Random Forest     |
| `serving_endpoint_name` | `flip_flopper_serving`      | Model Serving endpoint name          |

The single `dev` target uses Databricks bundle development mode, so deployed resources and schema names get a `dev_<your_user>` prefix. Point `workspace.host` in `databricks.yml` at your own workspace before deploying.

# Quickstart

From the repository root:

```bash
uv sync --dev
databricks bundle validate
databricks bundle deploy
databricks bundle run "run_pipeline"
```

The `run_pipeline` job runs the full demo:

1. create dummy data
2. train all five models in parallel (five concurrent job tasks on Free Edition — avoid other jobs while the pipeline runs)
3. deploy or update the serving endpoint

After the pipeline completes, query the endpoint:

```bash
databricks bundle run "query_serving_endpoint"
```

# How the pipeline works

`run_pipeline` is a single multi-task job:

```text
create_dummy_data
        │
        ├── train_logistic_regression
        ├── train_lightgbm
        ├── train_xgboost
        ├── train_pytorch_mlp
        └── train_sparkml_random_forest
                    │
                    ▼
          deploy_serving_endpoint
```

The training jobs read the same generated feature table, convert their fitted model to ONNX, wrap the ONNX artifact in an MLflow pyfunc, validate the prediction output shape, and register in Unity Catalog.

SparkML Random Forest uses `onnxmltools.convert_sparkml` (experimental); see the [Databricks KB on target_opset](https://kb.databricks.com/machine-learning/spark-ml-to-onnx-model-conversion-does-not-produce-the-same-model-predictions-differ).

The deployment job resolves the latest UC model versions at runtime and configures one endpoint with five served entities:

| Served entity         | UC model variable          | Traffic |
| --------------------- | -------------------------- | ------- |
| `logistic_regression` | `model_name`               | 20%     |
| `lightgbm`            | `lightgbm_model_name`      | 20%     |
| `xgboost`             | `xgboost_model_name`       | 20%     |
| `pytorch_mlp`         | `pytorch_mlp_model_name`   | 20%     |
| `random_forest`       | `sparkml_rf_model_name`    | 20%     |

Because Free Edition workspaces limit Model Serving capacity, the deployment job may add served entities incrementally when creating all five at once exceeds quota.

# Why ONNX-backed pyfunc models?

The trick in this repo is not that logistic regression, LightGBM, and XGBoost are fancy models. They are intentionally boring.

The useful part is that different model libraries are normalized into a common serving shape where possible:

- the same 33 feature columns: `feature_000` through `feature_032`
- pyfunc models return `target`, `proba`, `model_name`
- the same Databricks custom model serving request format

That makes it easy to put multiple model implementations behind one endpoint and still know which model produced each response. The `model_name` output is especially handy when calling the shared endpoint, because Databricks traffic routing decides which served entity receives the request.

# Querying the endpoint

The easiest way is the `query_serving_endpoint` job (`src/jobs/query_serving_endpoint.py`), which runs on serverless compute and authenticates as the workspace identity:

```bash
# traffic split across all served models
databricks bundle run "query_serving_endpoint"

# one specific served model
databricks bundle run "query_serving_endpoint" -- --served_model random_forest
```

You can also call the REST API directly from your machine. The only difference between the two invocation styles is the URL path — `/served-models/<name>/` targets one model, omitting it lets the endpoint's traffic split decide:

```bash
export DATABRICKS_HOST="https://<your-workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-token>"

# one request row with all 33 features set to 0.0
payload=$(python3 -c 'import json; print(json.dumps({"dataframe_records": [{f"feature_{i:03d}": 0.0 for i in range(33)}]}))')

# traffic split across all served models
curl -s -X POST "$DATABRICKS_HOST/serving-endpoints/flip_flopper_serving/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
  -d "$payload"

# one specific served model
curl -s -X POST "$DATABRICKS_HOST/serving-endpoints/flip_flopper_serving/served-models/random_forest/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
  -d "$payload"
```

The response contains a `predictions` payload with the pyfunc output. Each row includes the model's own `model_name`, so shared-endpoint calls identify which model answered.

# Troubleshooting

- **Endpoint is not ready yet**: Model Serving deployment can take several minutes. The deployment notebook waits for `READY` and fails on unrecoverable update failures.
- **Endpoint creation hits quota limits** (common on Free Edition): the deployment job first tries a bulk create, then falls back to adding served entities incrementally.
- **Permission errors**: the endpoint creator needs access to the UC catalog, schema, and registered models.
- **Unexpected prediction shape**: each pyfunc training notebook validates `mlflow.pyfunc.load_model(...).predict(...)` before registration.
- **SparkML ONNX parity**: `convert_sparkml` is experimental; ONNX probabilities may differ from Spark ML even when conversion succeeds. The training notebook smoke-tests output shape only.
- **SparkML ONNX on serverless**: `onnxmltools` reads `spark.conf.get("spark.master")` during tree export, which serverless Spark Connect blocks. The notebook patches `save_read_sparkml_model_data` and writes tree metadata to the bundle-managed UC Volume `onnx_scratch` (`/Volumes/{catalog}/{schema}/onnx_scratch`).
- **Wrong target objects**: remember that the `dev` target runs in bundle development mode, so deployed resource names and paths can be target-prefixed by Databricks.
