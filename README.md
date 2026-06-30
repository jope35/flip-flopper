<br />
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
  <a href="https://github.com/revodatanl/polly-pony/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <a href="https://github.com/revodatanl/polly-pony/commits/main">
    <img alt="GitHub last commit (branch)" src="https://img.shields.io/github/last-commit/revodatanl/polly-pony/main" />
  </a>
</p>
<br />

---

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

the main concept is that each model is converted to ONNX, wrapped as an MLflow pyfunc with the same input and output contract, registered in Unity Catalog, and deployed behind a single endpoint.

The endpoint can be called in two ways:
- hit the endpoint normally and let Databricks route traffic across all served models
- call an individual served model directly when you want `logistic_regression`, `lightgbm`, or `xgboost`

# Highlights

- 🩴 **One endpoint, three models**: `logistic_regression`, `lightgbm`, and `xgboost` are deployed as served entities behind the same Model Serving endpoint.
- 🎯 **Direct model calls**: use Databricks' individual served-model invocation path to bypass the endpoint traffic split when you want one specific model.
- 📦 **ONNX-backed pyfunc contract**: each model returns the same columns: `target`, `proba`, and `model_name`.
- 🧭 **Unity Catalog first**: generated data lands in a UC table, and trained models are registered as UC models.
- ↔️ **Traffic split between models**: by default, the serving endpoint automatically splits incoming traffic across `logistic_regression`, `lightgbm`, and `xgboost` models, with a configurable ratio (default 34/33/33), letting you test multi-model serving or target a specific model directly.

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
│   ├── create_dummy_data.yml              # creates the demo feature table
│   ├── train_logistic_regression.yml      # trains and registers logistic_regression_onnx
│   ├── train_lightgbm.yml                 # trains and registers lightgbm_onnx
│   ├── train_xgboost.yml                  # trains and registers xgboost_onnx
│   ├── deploy_serving_endpoint.yml        # creates or updates the serving endpoint
│   └── run_pipeline.yml                   # orchestrates the full workflow
├── src/jobs/
│   ├── create_dummy_data.ipynb
│   ├── train_logistic_regression.ipynb
│   ├── train_lightgbm.ipynb
│   ├── train_xgboost.ipynb
│   └── deploy_serving_endpoint.ipynb
└── tests/
```

# Configuration

The important bundle settings live in `databricks.yml`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `catalog` | `workspace` in both targets | Unity Catalog catalog |
| `schema` | `data` | schema for generated data |
| `model_schema` | `model` | schema for registered models |
| `table_name` | `generated_data` | generated feature table |
| `model_name` | `logistic_regression_onnx` | registered logistic regression model |
| `lightgbm_model_name` | `lightgbm_onnx` | registered LightGBM model |
| `xgboost_model_name` | `xgboost_onnx` | registered XGBoost model |
| `serving_endpoint_name` | `flip_flopper_serving` | Model Serving endpoint name |

The default `dev` target uses Databricks bundle development mode. The `prod` target uses production mode and an explicit workspace root path.

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
2. train all three models
3. deploy or update the serving endpoint

You can also run individual jobs:

```bash
databricks bundle run "create_dummy_data"
databricks bundle run "train_logistic_regression"
databricks bundle run "train_lightgbm"
databricks bundle run "train_xgboost"
databricks bundle run "deploy_serving_endpoint"
```

To deploy to production:

```bash
databricks bundle validate --target prod
databricks bundle deploy --target prod
databricks bundle run "run_pipeline" --target prod
```

# How the pipeline works

`run_pipeline` is the orchestrator job:

```text
create_dummy_data
        │
        ├── train_logistic_regression
        ├── train_lightgbm
        └── train_xgboost
                    │
                    ▼
          deploy_serving_endpoint
```

The three training jobs all read the same generated feature table, convert their fitted model to ONNX, wrap the ONNX artifact in an MLflow pyfunc, validate the pyfunc locally, and register it in Unity Catalog.

The deployment job resolves the latest UC model versions at runtime and configures one endpoint with three served entities:

| Served entity | UC model variable | Traffic |
| --- | --- | --- |
| `logistic_regression` | `model_name` | 34% |
| `lightgbm` | `lightgbm_model_name` | 33% |
| `xgboost` | `xgboost_model_name` | 33% |

Because Free Edition workspaces limit Model Serving capacity, the deployment job may add served entities incrementally (one model at a time) when creating all three at once exceeds quota.

# Why ONNX-backed pyfunc models?

The trick in this repo is not that logistic regression, LightGBM, and XGBoost are fancy models. They are intentionally boring.

The useful part is that different model libraries are normalized into the same serving shape:

- the same 33 feature columns: `feature_000` through `feature_032`
- the same output columns: `target`, `proba`, `model_name`
- the same Databricks custom model serving request format

That makes it easy to put multiple model implementations behind one endpoint and still know which model produced each response. The `model_name` output is especially handy when calling the shared endpoint, because Databricks traffic routing decides which served entity receives the request.

# Querying the endpoint

Set your workspace host and token first:

```bash
export DATABRICKS_HOST="https://dbc-b9d925fd-a82e.cloud.databricks.com"
export DATABRICKS_TOKEN="<your-token>"
```

Query the endpoint traffic split:

```python
import json
import os
import urllib.request

host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
endpoint = "flip_flopper_serving"

record = {f"feature_{i:03d}": 0.0 for i in range(33)}
payload = {"dataframe_records": [record]}

request = urllib.request.Request(
    f"{host}/serving-endpoints/{endpoint}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
)
request.add_header("Content-Type", "application/json")

with urllib.request.urlopen(request, timeout=60) as response:
    print(json.loads(response.read()))
```

Query one specific served model and ignore the traffic split:

```python
import json
import os
import urllib.request

host = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
endpoint = "flip_flopper_serving"
served_model = "xgboost"  # logistic_regression, lightgbm, or xgboost

record = {f"feature_{i:03d}": 0.0 for i in range(33)}
payload = {"dataframe_records": [record]}

request = urllib.request.Request(
    f"{host}/serving-endpoints/{endpoint}/served-models/{served_model}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    data=json.dumps(payload).encode("utf-8"),
    method="POST",
)
request.add_header("Content-Type", "application/json")

with urllib.request.urlopen(request, timeout=60) as response:
    print(json.loads(response.read()))
```

The response contains a `predictions` payload with the pyfunc output. Each row includes the model's own `model_name`, so shared-endpoint calls identify which model answered.

# Troubleshooting

- **Endpoint is not ready yet**: Model Serving deployment can take several minutes. The deployment notebook waits for `READY` and fails on unrecoverable update failures.
- **Endpoint creation hits quota limits** (common on Free Edition): the deployment job first tries a bulk create, then falls back to adding served entities incrementally.
- **Permission errors**: the endpoint creator needs access to the UC catalog, schema, and registered models.
- **Unexpected prediction shape**: each training notebook validates `mlflow.pyfunc.load_model(...).predict(...)` before registration.
- **Wrong target objects**: remember that the `dev` target runs in bundle development mode, so deployed resource names and paths can be target-prefixed by Databricks.
