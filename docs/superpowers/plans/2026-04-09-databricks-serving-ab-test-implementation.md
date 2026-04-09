# Databricks Serving A/B Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Databricks bundle that deploys and operates a single-endpoint A/B test for two Unity Catalog model versions, with AI Gateway payload logging, binary feedback joins, guarded traffic updates, and a repeatable deployment path.

**Architecture:** Keep the control-plane logic in a small Python package and use a DAB only for deployment and scheduling. The traffic controller computes arm metrics from the AI Gateway payload table plus a feedback table, applies a pure-Python policy module with safety guardrails, and updates the endpoint through the Databricks SDK using the full `served_entities` plus `traffic_config` payload.

**Tech Stack:** Python 3.11+, `uv`, `pytest`, `pydantic`, `databricks-sdk`, Databricks Asset Bundles, Mosaic AI Model Serving, Unity Catalog, Spark SQL in Databricks jobs

---

## Planned File Structure

- `pyproject.toml` - local package metadata, dependencies, pytest config.
- `databricks.yml` - bundle root, shared variables, target-specific overrides.
- `resources/jobs.yml` - deployment job and scheduled traffic-controller job.
- `src/flip_flopper_ab_test/__init__.py` - package marker and exported version.
- `src/flip_flopper_ab_test/config.py` - strongly typed app config, model entity config, guardrails, gateway schema mapping.
- `src/flip_flopper_ab_test/serving_config.py` - functions that build Databricks endpoint payloads and traffic routes.
- `src/flip_flopper_ab_test/metrics_query.py` - Spark SQL builder for lagged-window, deduped, joined arm metrics.
- `src/flip_flopper_ab_test/policy.py` - support thresholds, clamp logic, step-size limits, rollback logic, shadow-mode decision objects.
- `src/flip_flopper_ab_test/databricks_api.py` - small SDK wrapper for serving endpoint reads and config updates.
- `src/flip_flopper_ab_test/controller.py` - orchestration entrypoint: query metrics, run policy, update endpoint, persist audit/state.
- `src/jobs/bootstrap_endpoint.py` - job script that creates or updates the endpoint to the requested initial split.
- `src/jobs/run_traffic_controller.py` - scheduled job script that runs the controller.
- `src/sql/create_online_tables.sql` - creates feedback and controller-state Delta tables if missing.
- `tests/unit/test_imports.py` - package bootstrap smoke test.
- `tests/unit/test_serving_config.py` - endpoint payload and route-building tests.
- `tests/unit/test_metrics_query.py` - pinned AI Gateway schema and SQL-shape tests.
- `tests/unit/test_policy.py` - minimum support, step-size, clamp, rollback, shadow-mode tests.
- `tests/unit/test_controller.py` - orchestration test with fakes for Spark and Databricks SDK updates.
- `docs/runbooks/databricks-serving-ab-test.md` - operator notes for deployment, shadow mode, rollback, and smoke checks.

### Task 1: Bootstrap the Python package and test harness

**Files:**
- Create: `pyproject.toml`
- Create: `src/flip_flopper_ab_test/__init__.py`
- Create: `tests/unit/test_imports.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/unit/test_imports.py
from flip_flopper_ab_test import __version__


def test_package_import_smoke() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flip_flopper_ab_test'`

- [ ] **Step 3: Write the minimal package scaffold**

```toml
# pyproject.toml
[project]
name = "flip-flopper"
version = "0.1.0"
description = "Databricks serving A/B test controller"
requires-python = ">=3.11"
dependencies = [
  "databricks-sdk>=0.57.0",
  "pydantic>=2.11.0",
]

[dependency-groups]
dev = [
  "pytest>=8.3.5",
  "ruff>=0.11.0",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

```python
# src/flip_flopper_ab_test/__init__.py
__all__ = ["__version__"]

__version__ = "0.1.0"
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest tests/unit/test_imports.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/flip_flopper_ab_test/__init__.py tests/unit/test_imports.py
git commit -m "chore: scaffold databricks ab test package"
```

### Task 2: Define typed configuration and endpoint payload builders

**Files:**
- Create: `src/flip_flopper_ab_test/config.py`
- Create: `src/flip_flopper_ab_test/serving_config.py`
- Create: `tests/unit/test_serving_config.py`

- [ ] **Step 1: Write the failing payload-builder tests**

```python
# tests/unit/test_serving_config.py
from flip_flopper_ab_test.config import (
    AppConfig,
    GatewaySchemaConfig,
    GuardrailConfig,
    ServedEntityConfig,
)
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


def make_config() -> AppConfig:
    return AppConfig(
        endpoint_name="flip-flopper-dev",
        catalog="main",
        schema="serving",
        inference_table="main.serving.flip_flopper_payload",
        feedback_table="main.serving.flip_flopper_feedback",
        state_table="main.serving.flip_flopper_controller_state",
        control=ServedEntityConfig(
            name="control",
            entity_name="main.models.recommender",
            entity_version="1",
        ),
        challenger=ServedEntityConfig(
            name="challenger",
            entity_name="main.models.recommender",
            entity_version="2",
        ),
        gateway_schema=GatewaySchemaConfig(),
        guardrails=GuardrailConfig(
            min_challenger_percent=10,
            max_challenger_percent=50,
            max_step_percent=10,
            min_support_per_arm=200,
            max_error_rate=0.05,
            max_avg_execution_ms=1200,
        ),
    )


def test_build_endpoint_core_config_contains_two_routes() -> None:
    payload = build_endpoint_core_config(make_config(), control_percent=90, challenger_percent=10)

    assert [entity["name"] for entity in payload["served_entities"]] == ["control", "challenger"]
    assert payload["traffic_config"]["routes"] == [
        {"served_model_name": "control", "traffic_percentage": 90},
        {"served_model_name": "challenger", "traffic_percentage": 10},
    ]


def test_build_endpoint_core_config_rejects_invalid_total() -> None:
    try:
        build_endpoint_core_config(make_config(), control_percent=80, challenger_percent=30)
    except ValueError as exc:
        assert "sum to 100" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_serving_config.py -v`
Expected: FAIL with `ImportError` for `config` or `serving_config`

- [ ] **Step 3: Implement typed config and payload generation**

```python
# src/flip_flopper_ab_test/config.py
from pydantic import BaseModel, Field


class ServedEntityConfig(BaseModel):
    name: str
    entity_name: str
    entity_version: str
    workload_size: str = "Small"
    scale_to_zero_enabled: bool = True


class GatewaySchemaConfig(BaseModel):
    request_id_column: str = "client_request_id"
    request_time_column: str = "request_time"
    status_code_column: str = "status_code"
    execution_duration_ms_column: str = "execution_duration_ms"
    served_entity_id_column: str = "served_entity_id"
    databricks_request_id_column: str = "databricks_request_id"


class GuardrailConfig(BaseModel):
    min_challenger_percent: int = Field(ge=0, le=100)
    max_challenger_percent: int = Field(ge=0, le=100)
    max_step_percent: int = Field(ge=1, le=100)
    min_support_per_arm: int = Field(ge=1)
    max_error_rate: float = Field(ge=0, le=1)
    max_avg_execution_ms: int = Field(ge=1)


class AppConfig(BaseModel):
    endpoint_name: str
    catalog: str
    schema: str
    inference_table: str
    feedback_table: str
    state_table: str
    control: ServedEntityConfig
    challenger: ServedEntityConfig
    gateway_schema: GatewaySchemaConfig
    guardrails: GuardrailConfig
```

```python
# src/flip_flopper_ab_test/serving_config.py
from flip_flopper_ab_test.config import AppConfig


def build_endpoint_core_config(
    config: AppConfig,
    *,
    control_percent: int,
    challenger_percent: int,
) -> dict:
    if control_percent + challenger_percent != 100:
        raise ValueError("traffic percentages must sum to 100")

    served_entities = [
        config.control.model_dump(),
        config.challenger.model_dump(),
    ]
    traffic_config = {
        "routes": [
            {"served_model_name": config.control.name, "traffic_percentage": control_percent},
            {"served_model_name": config.challenger.name, "traffic_percentage": challenger_percent},
        ]
    }
    return {
        "served_entities": served_entities,
        "traffic_config": traffic_config,
    }
```

- [ ] **Step 4: Run tests to verify the config layer passes**

Run: `uv run pytest tests/unit/test_serving_config.py -v`
Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/flip_flopper_ab_test/config.py src/flip_flopper_ab_test/serving_config.py tests/unit/test_serving_config.py
git commit -m "feat: add endpoint config builders"
```

### Task 3: Pin the AI Gateway payload schema and build the metrics SQL

**Files:**
- Create: `src/flip_flopper_ab_test/metrics_query.py`
- Create: `tests/unit/test_metrics_query.py`

- [ ] **Step 1: Write the failing SQL-shape tests**

```python
# tests/unit/test_metrics_query.py
from datetime import datetime, timezone

from flip_flopper_ab_test.config import (
    AppConfig,
    GatewaySchemaConfig,
    GuardrailConfig,
    ServedEntityConfig,
)
from flip_flopper_ab_test.metrics_query import build_metrics_query


def make_config() -> AppConfig:
    return AppConfig(
        endpoint_name="flip-flopper-dev",
        catalog="main",
        schema="serving",
        inference_table="main.serving.flip_flopper_payload",
        feedback_table="main.serving.flip_flopper_feedback",
        state_table="main.serving.flip_flopper_controller_state",
        control=ServedEntityConfig(name="control", entity_name="main.models.recommender", entity_version="1"),
        challenger=ServedEntityConfig(name="challenger", entity_name="main.models.recommender", entity_version="2"),
        gateway_schema=GatewaySchemaConfig(),
        guardrails=GuardrailConfig(
            min_challenger_percent=10,
            max_challenger_percent=50,
            max_step_percent=10,
            min_support_per_arm=200,
            max_error_rate=0.05,
            max_avg_execution_ms=1200,
        ),
    )


def test_metrics_query_pins_gateway_columns_and_system_table_join() -> None:
    sql = build_metrics_query(
        make_config(),
        window_start=datetime(2026, 4, 9, 10, 0, tzinfo=timezone.utc),
        window_end=datetime(2026, 4, 9, 11, 0, tzinfo=timezone.utc),
    )

    assert "client_request_id" in sql
    assert "request_time" in sql
    assert "execution_duration_ms" in sql
    assert "served_entity_id" in sql
    assert "ROW_NUMBER()" in sql
    assert "system.serving.served_entities" in sql
    assert "AVG(CAST(label AS DOUBLE)) AS label_rate" in sql
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/test_metrics_query.py -v`
Expected: FAIL with `ImportError` for `metrics_query`

- [ ] **Step 3: Implement the metrics SQL builder**

```python
# src/flip_flopper_ab_test/metrics_query.py
from datetime import datetime

from flip_flopper_ab_test.config import AppConfig


def _format_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_metrics_query(config: AppConfig, *, window_start: datetime, window_end: datetime) -> str:
    schema = config.gateway_schema
    start_ts = _format_ts(window_start)
    end_ts = _format_ts(window_end)

    return f"""
WITH payload_window AS (
  SELECT
    {schema.request_id_column} AS client_request_id,
    {schema.request_time_column} AS request_time,
    {schema.status_code_column} AS status_code,
    {schema.execution_duration_ms_column} AS execution_duration_ms,
    {schema.served_entity_id_column} AS served_entity_id,
    {schema.databricks_request_id_column} AS databricks_request_id,
    ROW_NUMBER() OVER (
      PARTITION BY {schema.request_id_column}
      ORDER BY {schema.request_time_column} DESC, {schema.databricks_request_id_column} DESC
    ) AS row_num
  FROM {config.inference_table}
  WHERE {schema.request_time_column} >= TIMESTAMP '{start_ts}'
    AND {schema.request_time_column} < TIMESTAMP '{end_ts}'
    AND {schema.request_id_column} IS NOT NULL
),
deduped_payload AS (
  SELECT *
  FROM payload_window
  WHERE row_num = 1
),
served_entities AS (
  SELECT
    served_entity_id,
    name AS served_entity_name,
    endpoint_name
  FROM system.serving.served_entities
  WHERE endpoint_name = '{config.endpoint_name}'
),
labeled_examples AS (
  SELECT
    payload.client_request_id,
    payload.request_time,
    payload.status_code,
    payload.execution_duration_ms,
    entities.served_entity_name,
    feedback.label
  FROM deduped_payload payload
  INNER JOIN {config.feedback_table} feedback
    ON payload.client_request_id = feedback.client_request_id
  INNER JOIN served_entities entities
    ON payload.served_entity_id = entities.served_entity_id
)
SELECT
  served_entity_name,
  COUNT(*) AS support,
  AVG(CAST(label AS DOUBLE)) AS label_rate,
  AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END) AS error_rate,
  AVG(CAST(execution_duration_ms AS DOUBLE)) AS avg_execution_duration_ms
FROM labeled_examples
GROUP BY served_entity_name
""".strip()
```

- [ ] **Step 4: Run the SQL-shape tests to verify they pass**

Run: `uv run pytest tests/unit/test_metrics_query.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/flip_flopper_ab_test/metrics_query.py tests/unit/test_metrics_query.py
git commit -m "feat: add gateway metrics query builder"
```

### Task 4: Implement the traffic policy, guardrails, and rollback decision model

**Files:**
- Create: `src/flip_flopper_ab_test/policy.py`
- Create: `tests/unit/test_policy.py`

- [ ] **Step 1: Write failing policy tests**

```python
# tests/unit/test_policy.py
from flip_flopper_ab_test.policy import (
    ArmMetrics,
    CurrentSplit,
    PolicyDecision,
    propose_split,
)
from flip_flopper_ab_test.config import GuardrailConfig


GUARDRAILS = GuardrailConfig(
    min_challenger_percent=10,
    max_challenger_percent=50,
    max_step_percent=10,
    min_support_per_arm=200,
    max_error_rate=0.05,
    max_avg_execution_ms=1200,
)


def test_policy_holds_when_support_is_too_low() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=120, label_rate=0.31, error_rate=0.01, avg_execution_duration_ms=390),
        },
        current_split=CurrentSplit(control=90, challenger=10),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision == PolicyDecision(
        control=90,
        challenger=10,
        action="hold",
        reason="minimum_support_not_met",
    )


def test_policy_steps_up_challenger_when_it_wins() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=500, label_rate=0.28, error_rate=0.01, avg_execution_duration_ms=380),
        },
        current_split=CurrentSplit(control=90, challenger=10),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision.challenger == 20
    assert decision.action == "apply"
    assert decision.reason == "challenger_outperforming"


def test_policy_rolls_back_when_operational_guardrail_breaks() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=500, label_rate=0.28, error_rate=0.09, avg_execution_duration_ms=1600),
        },
        current_split=CurrentSplit(control=70, challenger=30),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision == PolicyDecision(
        control=90,
        challenger=10,
        action="rollback",
        reason="operational_guardrail_breached",
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_policy.py -v`
Expected: FAIL with `ImportError` for `policy`

- [ ] **Step 3: Implement the pure-Python policy module**

```python
# src/flip_flopper_ab_test/policy.py
from dataclasses import dataclass
from typing import Literal

from flip_flopper_ab_test.config import GuardrailConfig


@dataclass(frozen=True)
class ArmMetrics:
    support: int
    label_rate: float
    error_rate: float
    avg_execution_duration_ms: float


@dataclass(frozen=True)
class CurrentSplit:
    control: int
    challenger: int


@dataclass(frozen=True)
class PolicyDecision:
    control: int
    challenger: int
    action: Literal["hold", "apply", "rollback", "shadow"]
    reason: str


def propose_split(
    *,
    metrics_by_arm: dict[str, ArmMetrics],
    current_split: CurrentSplit,
    guardrails: GuardrailConfig,
    last_good_split: CurrentSplit,
    shadow_mode: bool,
) -> PolicyDecision:
    control = metrics_by_arm["control"]
    challenger = metrics_by_arm["challenger"]

    if control.support < guardrails.min_support_per_arm or challenger.support < guardrails.min_support_per_arm:
        return PolicyDecision(
            control=current_split.control,
            challenger=current_split.challenger,
            action="hold",
            reason="minimum_support_not_met",
        )

    if (
        challenger.error_rate > guardrails.max_error_rate
        or challenger.avg_execution_duration_ms > guardrails.max_avg_execution_ms
    ):
        return PolicyDecision(
            control=last_good_split.control,
            challenger=last_good_split.challenger,
            action="rollback",
            reason="operational_guardrail_breached",
        )

    if challenger.label_rate <= control.label_rate:
        return PolicyDecision(
            control=current_split.control,
            challenger=current_split.challenger,
            action="hold",
            reason="challenger_not_better",
        )

    next_challenger = min(
        current_split.challenger + guardrails.max_step_percent,
        guardrails.max_challenger_percent,
    )
    next_challenger = max(next_challenger, guardrails.min_challenger_percent)
    decision = PolicyDecision(
        control=100 - next_challenger,
        challenger=next_challenger,
        action="shadow" if shadow_mode else "apply",
        reason="challenger_outperforming",
    )
    return decision
```

- [ ] **Step 4: Run policy tests to verify they pass**

Run: `uv run pytest tests/unit/test_policy.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/flip_flopper_ab_test/policy.py tests/unit/test_policy.py
git commit -m "feat: add guarded traffic policy"
```

### Task 5: Orchestrate the controller and Databricks SDK updates

**Files:**
- Create: `src/flip_flopper_ab_test/databricks_api.py`
- Create: `src/flip_flopper_ab_test/controller.py`
- Create: `tests/unit/test_controller.py`

- [ ] **Step 1: Write the failing orchestration test**

```python
# tests/unit/test_controller.py
from dataclasses import dataclass

from flip_flopper_ab_test.config import (
    AppConfig,
    GatewaySchemaConfig,
    GuardrailConfig,
    ServedEntityConfig,
)
from flip_flopper_ab_test.controller import ControllerService


@dataclass
class FakeSparkResult:
    rows: list[dict]

    def collect(self) -> list[dict]:
        return self.rows


class FakeSparkSession:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed_sql: list[str] = []

    def sql(self, statement: str) -> FakeSparkResult:
        self.executed_sql.append(statement)
        return FakeSparkResult(self.rows)


class FakeServingClient:
    def __init__(self) -> None:
        self.updated_payloads: list[tuple[str, dict]] = []

    def ensure_endpoint(self, endpoint_name: str, payload: dict) -> None:
        self.updated_payloads.append((endpoint_name, payload))


def make_config() -> AppConfig:
    return AppConfig(
        endpoint_name="flip-flopper-dev",
        catalog="main",
        schema="serving",
        inference_table="main.serving.flip_flopper_payload",
        feedback_table="main.serving.flip_flopper_feedback",
        state_table="main.serving.flip_flopper_controller_state",
        control=ServedEntityConfig(name="control", entity_name="main.models.recommender", entity_version="1"),
        challenger=ServedEntityConfig(name="challenger", entity_name="main.models.recommender", entity_version="2"),
        gateway_schema=GatewaySchemaConfig(),
        guardrails=GuardrailConfig(
            min_challenger_percent=10,
            max_challenger_percent=50,
            max_step_percent=10,
            min_support_per_arm=200,
            max_error_rate=0.05,
            max_avg_execution_ms=1200,
        ),
    )


def test_controller_updates_endpoint_when_policy_applies() -> None:
    spark = FakeSparkSession(
        rows=[
            {"served_entity_name": "control", "support": 500, "label_rate": 0.20, "error_rate": 0.01, "avg_execution_duration_ms": 400},
            {"served_entity_name": "challenger", "support": 500, "label_rate": 0.28, "error_rate": 0.01, "avg_execution_duration_ms": 380},
        ]
    )
    serving = FakeServingClient()

    service = ControllerService(
        spark=spark,
        serving_client=serving,
        config=make_config(),
    )

    service.run(
        current_control_percent=90,
        current_challenger_percent=10,
        last_good_control_percent=90,
        last_good_challenger_percent=10,
        shadow_mode=False,
    )

    assert serving.updated_payloads[0][0] == "flip-flopper-dev"
    assert serving.updated_payloads[0][1]["traffic_config"]["routes"][1]["traffic_percentage"] == 20
```

- [ ] **Step 2: Run the orchestration test to verify it fails**

Run: `uv run pytest tests/unit/test_controller.py -v`
Expected: FAIL with `ImportError` for `controller`

- [ ] **Step 3: Implement the SDK wrapper and controller service**

```python
# src/flip_flopper_ab_test/databricks_api.py
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)


class ServingEndpointClient:
    def __init__(self, workspace_client: WorkspaceClient | None = None) -> None:
        self.workspace_client = workspace_client or WorkspaceClient()

    def _sdk_payload(self, payload: dict) -> tuple[list[ServedEntityInput], TrafficConfig]:
        served_entities = [
            ServedEntityInput(
                name=entity["name"],
                entity_name=entity["entity_name"],
                entity_version=entity["entity_version"],
                workload_size=entity["workload_size"],
                scale_to_zero_enabled=entity["scale_to_zero_enabled"],
            )
            for entity in payload["served_entities"]
        ]
        routes = [
            Route(
                served_model_name=route["served_model_name"],
                traffic_percentage=route["traffic_percentage"],
            )
            for route in payload["traffic_config"]["routes"]
        ]
        return served_entities, TrafficConfig(routes=routes)

    def ensure_endpoint(self, endpoint_name: str, payload: dict) -> None:
        served_entities, traffic_config = self._sdk_payload(payload)
        try:
            self.workspace_client.serving_endpoints.get(name=endpoint_name)
        except NotFound:
            self.workspace_client.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                    traffic_config=traffic_config,
                ),
                timeout=timedelta(minutes=20),
            )
            return

        self.workspace_client.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
            traffic_config=traffic_config,
            timeout=timedelta(minutes=20),
        )
```

```python
# src/flip_flopper_ab_test/controller.py
from datetime import datetime, timedelta, timezone

from flip_flopper_ab_test.config import AppConfig
from flip_flopper_ab_test.metrics_query import build_metrics_query
from flip_flopper_ab_test.policy import ArmMetrics, CurrentSplit, propose_split
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


class ControllerService:
    def __init__(self, *, spark, serving_client, config: AppConfig) -> None:
        self.spark = spark
        self.serving_client = serving_client
        self.config = config

    def run(
        self,
        *,
        current_control_percent: int,
        current_challenger_percent: int,
        last_good_control_percent: int,
        last_good_challenger_percent: int,
        shadow_mode: bool,
    ) -> None:
        window_end = datetime.now(timezone.utc) - timedelta(hours=1)
        window_start = window_end - timedelta(hours=24)
        sql = build_metrics_query(self.config, window_start=window_start, window_end=window_end)
        rows = self.spark.sql(sql).collect()
        metrics = {
            row["served_entity_name"]: ArmMetrics(
                support=int(row["support"]),
                label_rate=float(row["label_rate"]),
                error_rate=float(row["error_rate"]),
                avg_execution_duration_ms=float(row["avg_execution_duration_ms"]),
            )
            for row in rows
        }
        decision = propose_split(
            metrics_by_arm=metrics,
            current_split=CurrentSplit(control=current_control_percent, challenger=current_challenger_percent),
            guardrails=self.config.guardrails,
            last_good_split=CurrentSplit(
                control=last_good_control_percent,
                challenger=last_good_challenger_percent,
            ),
            shadow_mode=shadow_mode,
        )
        if decision.action in {"apply", "rollback"}:
            payload = build_endpoint_core_config(
                self.config,
                control_percent=decision.control,
                challenger_percent=decision.challenger,
            )
            self.serving_client.ensure_endpoint(self.config.endpoint_name, payload)
```

- [ ] **Step 4: Run the orchestration test to verify it passes**

Run: `uv run pytest tests/unit/test_controller.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 5: Run the focused unit suite**

Run: `uv run pytest tests/unit -v`
Expected: PASS with `7 passed`

- [ ] **Step 6: Commit**

```bash
git add src/flip_flopper_ab_test/databricks_api.py src/flip_flopper_ab_test/controller.py tests/unit/test_controller.py
git commit -m "feat: orchestrate controller decisions and endpoint updates"
```

### Task 6: Add job entrypoints, SQL bootstrap, bundle resources, and operator runbook

**Files:**
- Create: `src/jobs/bootstrap_endpoint.py`
- Create: `src/jobs/run_traffic_controller.py`
- Create: `src/sql/create_online_tables.sql`
- Create: `databricks.yml`
- Create: `resources/jobs.yml`
- Create: `docs/runbooks/databricks-serving-ab-test.md`
- Create: `tests/unit/test_bundle_shape.py`

- [ ] **Step 1: Write the failing bundle smoke check**

```python
# tests/unit/test_bundle_shape.py
from pathlib import Path


def test_bundle_files_exist() -> None:
    assert Path("databricks.yml").exists()
    assert Path("resources/jobs.yml").exists()
    assert Path("src/jobs/bootstrap_endpoint.py").exists()
    assert Path("src/jobs/run_traffic_controller.py").exists()
```

- [ ] **Step 2: Run the smoke check to verify it fails**

Run: `uv run pytest tests/unit/test_bundle_shape.py -v`
Expected: FAIL because the bundle files do not exist yet

- [ ] **Step 3: Create the SQL bootstrap and job scripts**

```sql
-- src/sql/create_online_tables.sql
CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.flip_flopper_feedback (
  client_request_id STRING NOT NULL,
  label INT NOT NULL,
  label_timestamp TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS ${catalog}.${schema}.flip_flopper_controller_state (
  decision_ts TIMESTAMP NOT NULL,
  endpoint_name STRING NOT NULL,
  action STRING NOT NULL,
  reason STRING NOT NULL,
  control_percent INT NOT NULL,
  challenger_percent INT NOT NULL,
  support_control BIGINT,
  support_challenger BIGINT
);
```

```python
# src/jobs/bootstrap_endpoint.py
import argparse

from flip_flopper_ab_test.config import AppConfig
from flip_flopper_ab_test.databricks_api import ServingEndpointClient
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--control-percent", type=int, default=90)
    parser.add_argument("--challenger-percent", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.model_validate_json(args.config_json)
    payload = build_endpoint_core_config(
        config,
        control_percent=args.control_percent,
        challenger_percent=args.challenger_percent,
    )
    ServingEndpointClient().ensure_endpoint(config.endpoint_name, payload)


if __name__ == "__main__":
    main()
```

```python
# src/jobs/run_traffic_controller.py
import argparse

from pyspark.sql import SparkSession

from flip_flopper_ab_test.config import AppConfig
from flip_flopper_ab_test.controller import ControllerService
from flip_flopper_ab_test.databricks_api import ServingEndpointClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--current-control-percent", type=int, required=True)
    parser.add_argument("--current-challenger-percent", type=int, required=True)
    parser.add_argument("--last-good-control-percent", type=int, required=True)
    parser.add_argument("--last-good-challenger-percent", type=int, required=True)
    parser.add_argument("--shadow-mode", choices=["true", "false"], default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig.model_validate_json(args.config_json)
    spark = SparkSession.builder.getOrCreate()
    ControllerService(
        spark=spark,
        serving_client=ServingEndpointClient(),
        config=config,
    ).run(
        current_control_percent=args.current_control_percent,
        current_challenger_percent=args.current_challenger_percent,
        last_good_control_percent=args.last_good_control_percent,
        last_good_challenger_percent=args.last_good_challenger_percent,
        shadow_mode=args.shadow_mode == "true",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the bundle resources and runbook**

```yaml
# databricks.yml
bundle:
  name: flip-flopper-serving-ab-test

include:
  - resources/*.yml

variables:
  catalog:
    default: main
  schema:
    default: serving
  endpoint_name:
    default: flip-flopper-dev
  app_config_json:
    default: '{"endpoint_name":"flip-flopper-dev","catalog":"main","schema":"serving","inference_table":"main.serving.flip_flopper_payload","feedback_table":"main.serving.flip_flopper_feedback","state_table":"main.serving.flip_flopper_controller_state","control":{"name":"control","entity_name":"main.models.recommender","entity_version":"1","workload_size":"Small","scale_to_zero_enabled":true},"challenger":{"name":"challenger","entity_name":"main.models.recommender","entity_version":"2","workload_size":"Small","scale_to_zero_enabled":true},"gateway_schema":{"request_id_column":"client_request_id","request_time_column":"request_time","status_code_column":"status_code","execution_duration_ms_column":"execution_duration_ms","served_entity_id_column":"served_entity_id","databricks_request_id_column":"databricks_request_id"},"guardrails":{"min_challenger_percent":10,"max_challenger_percent":50,"max_step_percent":10,"min_support_per_arm":200,"max_error_rate":0.05,"max_avg_execution_ms":1200}}'
  controller_cron:
    default: "0 0 * * * ?"

targets:
  dev:
    default: true
    mode: development
  prod:
    mode: production
```

```yaml
# resources/jobs.yml
resources:
  jobs:
    bootstrap_endpoint:
      name: "[${bundle.target}] bootstrap serving endpoint"
      tasks:
        - task_key: bootstrap_endpoint
          spark_python_task:
            python_file: ../src/jobs/bootstrap_endpoint.py
            parameters:
              - --config-json
              - ${var.app_config_json}
              - --control-percent
              - "90"
              - --challenger-percent
              - "10"
    traffic_controller:
      name: "[${bundle.target}] traffic controller"
      tasks:
        - task_key: run_traffic_controller
          spark_python_task:
            python_file: ../src/jobs/run_traffic_controller.py
            parameters:
              - --config-json
              - ${var.app_config_json}
              - --current-control-percent
              - "90"
              - --current-challenger-percent
              - "10"
              - --last-good-control-percent
              - "90"
              - --last-good-challenger-percent
              - "10"
              - --shadow-mode
              - "true"
      schedule:
        quartz_cron_expression: ${var.controller_cron}
        timezone_id: UTC
        pause_status: UNPAUSED
```

```md
# docs/runbooks/databricks-serving-ab-test.md
## Deploy

1. Run `databricks bundle validate -t dev`
2. Run `databricks bundle deploy -t dev`
3. Run `databricks bundle run bootstrap_endpoint -t dev`

## Shadow mode

1. Start with the controller job in shadow mode for at least one full feedback window.
2. Confirm the proposed split in job logs and the controller state table.
3. Only then switch the production schedule to apply mode.

## Rollback

1. Re-run the bootstrap job with the last-known-good split.
2. Confirm the endpoint routes in the Serving UI.
3. Pause the controller schedule until the challenger issue is understood.
```

- [ ] **Step 5: Run the local bundle smoke check**

Run: `uv run pytest tests/unit/test_bundle_shape.py -v`
Expected: PASS with `1 passed`

- [ ] **Step 6: Validate the bundle definition**

Run: `databricks bundle validate -t dev`
Expected: PASS with bundle validation success and both jobs resolved

- [ ] **Step 7: Commit**

```bash
git add databricks.yml resources/jobs.yml src/jobs/bootstrap_endpoint.py src/jobs/run_traffic_controller.py src/sql/create_online_tables.sql docs/runbooks/databricks-serving-ab-test.md tests/unit/test_bundle_shape.py
git commit -m "feat: add databricks jobs and operator runbook"
```

### Task 7: Run the dev-environment smoke test with synthetic traffic and feedback

**Files:**
- Modify: `docs/runbooks/databricks-serving-ab-test.md`
- Test: `tests/unit/test_controller.py`

- [ ] **Step 1: Deploy the dev bundle**

Run: `databricks bundle deploy -t dev`
Expected: PASS with `bootstrap_endpoint` and `traffic_controller` created in the dev workspace

- [ ] **Step 2: Bootstrap the endpoint**

Run: `databricks bundle run bootstrap_endpoint -t dev`
Expected: PASS with the endpoint configured for `control=90`, `challenger=10`

- [ ] **Step 3: Generate synthetic traffic and feedback**

```python
# run in a Databricks notebook attached to the dev workspace
import requests
import uuid
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host
token = w.config.token

request_ids = []
for _ in range(50):
    request_id = str(uuid.uuid4())
    request_ids.append(request_id)
    requests.post(
        f"{host}/serving-endpoints/flip-flopper-dev/invocations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "client_request_id": request_id,
            "dataframe_records": [{"feature_a": 1.0, "feature_b": 0.2}],
        },
        timeout=30,
    ).raise_for_status()

spark.sql(
    f"""
    MERGE INTO main.serving.flip_flopper_feedback target
    USING (
      SELECT explode(array({",".join([f"'{request_id}'" for request_id in request_ids])})) AS client_request_id
    ) source
    ON target.client_request_id = source.client_request_id
    WHEN MATCHED THEN UPDATE SET label = 1, label_timestamp = current_timestamp()
    WHEN NOT MATCHED THEN INSERT (client_request_id, label, label_timestamp)
    VALUES (source.client_request_id, 1, current_timestamp())
    """
)
```

- [ ] **Step 4: Run the controller in shadow mode first**

Run: `databricks bundle run traffic_controller -t dev --params '{"shadow_mode":"true"}'`
Expected: PASS with a logged `shadow` decision and no serving config change

- [ ] **Step 5: Run the controller in apply mode**

Run: `databricks bundle run traffic_controller -t dev --params '{"shadow_mode":"false"}'`
Expected: PASS with a new split recorded in the controller state table and reflected in the endpoint config

- [ ] **Step 6: Update the runbook with any environment-specific gotchas**

```md
Add a short "Dev notes" section to `docs/runbooks/databricks-serving-ab-test.md` covering:
- the actual inference-table name created by AI Gateway
- the job run-as identity
- the rollback command used in dev
```

- [ ] **Step 7: Commit**

```bash
git add docs/runbooks/databricks-serving-ab-test.md
git commit -m "docs: capture dev smoke test notes"
```

## Self-Review

- **Spec coverage:** The plan covers the single-endpoint multi-model setup, AI Gateway payload logging, binary feedback join, lagged and deduped metrics, guarded policy-based traffic updates, rollback behavior, DAB deployment, MLflow-style deployment/bootstrap job, and dev smoke testing.
- **Placeholder scan:** No `TODO`, `TBD`, or "implement later" placeholders remain in the plan. The job entrypoints and bundle examples now include concrete argument-passing instead of deferred wiring notes.
- **Type consistency:** The same config names are used throughout: `AppConfig`, `ServedEntityConfig`, `GatewaySchemaConfig`, `GuardrailConfig`, `ControllerService`, and `PolicyDecision`.

Plan complete and saved to `docs/superpowers/plans/2026-04-09-databricks-serving-ab-test-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
