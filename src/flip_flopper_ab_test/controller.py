from datetime import datetime, timedelta, timezone
from typing import Any

from flip_flopper_ab_test.metrics_query import build_metrics_query
from flip_flopper_ab_test.policy import ArmMetrics, CurrentSplit, GuardrailConfig, propose_split
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


def _guardrails_from_config(config: dict[str, Any]) -> GuardrailConfig:
    g = config["guardrails"]
    return GuardrailConfig(
        min_challenger_percent=int(g["min_challenger_percent"]),
        max_challenger_percent=int(g["max_challenger_percent"]),
        max_step_percent=int(g["max_step_percent"]),
        min_support_per_arm=int(g["min_support_per_arm"]),
        max_error_rate=float(g["max_error_rate"]),
        max_avg_execution_ms=int(g["max_avg_execution_ms"]),
    )


class ControllerService:
    def __init__(self, *, spark, serving_client, config: dict[str, Any]) -> None:
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
        window_end: datetime | None = None,
        window_start: datetime | None = None,
    ) -> None:
        if window_end is None:
            window_end = datetime.now(timezone.utc) - timedelta(hours=1)
        if window_start is None:
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
            guardrails=_guardrails_from_config(self.config),
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
            self.serving_client.ensure_endpoint(self.config["endpoint_name"], payload)
