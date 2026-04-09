from typing import Any


def build_endpoint_core_config(
    config: dict[str, Any],
    *,
    control_percent: int,
    challenger_percent: int,
) -> dict[str, Any]:
    if control_percent + challenger_percent != 100:
        raise ValueError("traffic percentages must sum to 100")

    control = config["control"]
    challenger = config["challenger"]

    def entity_payload(entity: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": entity["name"],
            "entity_name": entity["entity_name"],
            "entity_version": entity["entity_version"],
            "workload_size": entity.get("workload_size", "Small"),
            "scale_to_zero_enabled": entity.get("scale_to_zero_enabled", True),
        }

    served_entities = [entity_payload(control), entity_payload(challenger)]
    traffic_config = {
        "routes": [
            {"served_model_name": control["name"], "traffic_percentage": control_percent},
            {"served_model_name": challenger["name"], "traffic_percentage": challenger_percent},
        ]
    }
    return {
        "served_entities": served_entities,
        "traffic_config": traffic_config,
    }
