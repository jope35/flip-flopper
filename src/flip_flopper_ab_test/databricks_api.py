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
