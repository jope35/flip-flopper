from dataclasses import dataclass
from pathlib import Path

from flip_flopper_ab_test.config import load_config
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


def make_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "config" / "app.yaml")


def test_controller_updates_endpoint_when_policy_applies() -> None:
    spark = FakeSparkSession(
        rows=[
            {
                "served_entity_name": "control",
                "support": 500,
                "label_rate": 0.20,
                "error_rate": 0.01,
                "avg_execution_duration_ms": 400,
            },
            {
                "served_entity_name": "challenger",
                "support": 500,
                "label_rate": 0.28,
                "error_rate": 0.01,
                "avg_execution_duration_ms": 380,
            },
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
