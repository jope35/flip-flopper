from datetime import datetime, timezone
from pathlib import Path

from flip_flopper_ab_test.config import load_config
from flip_flopper_ab_test.metrics_query import build_metrics_query


def make_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "config" / "app.yaml")


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
