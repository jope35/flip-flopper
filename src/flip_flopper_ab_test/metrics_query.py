from datetime import datetime
from typing import Any


def _format_ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_metrics_query(config: dict[str, Any], *, window_start: datetime, window_end: datetime) -> str:
    schema = config["gateway_schema"]
    start_ts = _format_ts(window_start)
    end_ts = _format_ts(window_end)
    endpoint_name = str(config["endpoint_name"]).replace("'", "''")

    return f"""
WITH payload_window AS (
  SELECT
    {schema["request_id_column"]} AS client_request_id,
    {schema["request_time_column"]} AS request_time,
    {schema["status_code_column"]} AS status_code,
    {schema["execution_duration_ms_column"]} AS execution_duration_ms,
    {schema["served_entity_id_column"]} AS served_entity_id,
    {schema["databricks_request_id_column"]} AS databricks_request_id,
    ROW_NUMBER() OVER (
      PARTITION BY {schema["request_id_column"]}
      ORDER BY {schema["request_time_column"]} DESC, {schema["databricks_request_id_column"]} DESC
    ) AS row_num
  FROM {config["inference_table"]}
  WHERE {schema["request_time_column"]} >= TIMESTAMP '{start_ts}'
    AND {schema["request_time_column"]} < TIMESTAMP '{end_ts}'
    AND {schema["request_id_column"]} IS NOT NULL
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
  WHERE endpoint_name = '{endpoint_name}'
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
  INNER JOIN {config["feedback_table"]} feedback
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
