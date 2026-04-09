# Design: A/B testing on a single Databricks model serving endpoint

**Status:** Approved for implementation planning  
**Date:** 2026-04-09

## Purpose

Run controlled A/B tests between two model variants behind **one** Mosaic AI Model Serving endpoint, using **Unity Catalog**–registered models, **inference logging** for request/response capture, a **binary feedback** table joined on `client_request_id`, and a **scheduled job** that updates **traffic split** based on a defined policy. **Initial and version-gated deployment** uses an **MLflow deployment job** as in [MLflow deployment jobs](https://docs.databricks.com/aws/en/mlflow/deployment-job). **Infrastructure and jobs** are deployable via a **Declarative Automation Bundle** (DAB).

## Non-goals

- Mixing incompatible model types on the same endpoint (e.g. custom PyFunc and external models together)—the platform forbids this.
- Sub-second or minute-level traffic reactions; inference log delivery is delayed (on the order of up to about an hour per platform docs).
- Defining the exact statistical or bandit algorithm in this document—only inputs, outputs, and safety constraints are fixed here; the algorithm is a pluggable module in the traffic controller.

## Architecture

| Component | Role |
|-----------|------|
| **UC registered model** | Source of truth for artifacts; versions promoted through governed workflow. |
| **Single serving endpoint** | Two `served_entities` (e.g. `control`, `challenger`) with `traffic_config.routes` summing to 100%. |
| **AI Gateway inference tables** | UC Delta table(s) logging requests/responses and metadata for monitoring and attribution. **Legacy** inference tables must not be used for new work (platform deprecation). |
| **Feedback table** | UC Delta table: `client_request_id`, binary `label` ∈ {0, 1}, timestamps as needed; idempotent upserts per `client_request_id`. |
| **Deployment job (MLflow)** | Triggered on new model versions; evaluation/approval/deploy per org policy; run-as **service principal** with least privilege; job parameters include `model_name` and `model_version`; max concurrent runs = 1. |
| **Traffic controller job** | Scheduled Lakeflow job (notebook or Python task): reads deduped inference + feedback, computes metrics per arm, applies policy, updates endpoint config via Workspace/Serving API. |
| **DAB** | Declares jobs, notebooks/scripts, variables per target (catalog, schema, endpoint name), and—where supported—`model_serving_endpoint` or equivalent; any gap filled by an idempotent bootstrap task. |

## Data flow

1. Clients call the endpoint with **`client_request_id`** at the top level of the request body (per model serving inference table documentation).
2. Gateway inference logging persists request/response rows (schema per [AI Gateway inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables), not legacy schema).
3. A separate pipeline (out of scope for this spec’s implementation detail) writes **binary feedback** keyed by `client_request_id`.
4. The traffic controller:
   - Restricts to a **lagged time window** so feedback and inference rows are likely complete.
   - **Dedupes** inference rows to one row per `client_request_id` (deterministic `ROW_NUMBER` or equivalent) because logging is at-least-once.
   - Joins to feedback on `client_request_id`.
   - Attributes each row to **control** or **challenger** using logged **model/served-model identity** (e.g. `request_metadata` / Gateway-equivalent fields), not inferred from traffic percentages.
5. Policy outputs new `traffic_percentage` values; controller applies **guardrails** then calls **PUT** serving endpoint config with the **full** `served_entities` and `traffic_config` payload.

## Feedback and metrics

- **Label:** binary 0/1 (e.g. bad/good or no-conversion/conversion).
- **Per-arm metrics:** support = count of joined rows with labels; rate = mean(label). Enforce **minimum support per arm** before allowing traffic changes.
- **Optional:** persist last applied split and policy decision audit in a small **controller state** Delta table.

## Safety and operations

- **Clamp** challenger share between configurable floor and ceiling; limit **maximum change per run** (step size).
- **Rollback:** if operational signals worsen beyond thresholds (e.g. error rate or latency from logged fields), revert to last-known-good split stored in state.
- **Permissions:** controller identity needs authority to update the endpoint; inference table and feedback table need appropriate `SELECT` for the job run-as identity.
- **Monitoring:** alert on inference capture `FAILED` (or Gateway equivalent) and on controller job failures.

## Bundle layout (reference)

- `databricks.yml`: targets, variables (`catalog`, `schema`, `endpoint_name`, table prefixes).
- `resources/*.yml`: deployment job, traffic controller job (schedule), notebooks or `spark_python_task` entrypoints, optional volume/schema resources as required by the workspace.
- `src/`: deployment notebooks (or shared modules) aligned with Databricks deployment-job templates; traffic policy code.

## Implementation note

Pin the **exact** AI Gateway inference table columns used for arm attribution and timestamps in the traffic controller code and tests; Gateway schemas can evolve independently of this spec.

## Testing

- **Unit tests** for dedupe, join, and policy logic (local or CI).
- **Integration:** dev endpoint with synthetic traffic and feedback; validate that applied config matches expected `traffic_config`.
- **Shadow mode** (optional): compute proposed split, log only, before enabling apply.

## References

- [Serve multiple models / traffic split](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint)
- [MLflow deployment jobs](https://docs.databricks.com/aws/en/mlflow/deployment-job)
- [Inference tables (legacy migration note)](https://docs.databricks.com/aws/en/machine-learning/model-serving/inference-tables)
- [AI Gateway inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables)
- [Declarative Automation Bundles resources](https://docs.databricks.com/aws/en/dev-tools/bundles/resources)
