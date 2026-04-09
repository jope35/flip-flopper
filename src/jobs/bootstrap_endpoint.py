import argparse

from flip_flopper_ab_test.config import load_config
from flip_flopper_ab_test.databricks_api import ServingEndpointClient
from flip_flopper_ab_test.serving_config import build_endpoint_core_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True, help="Path to app config YAML")
    parser.add_argument("--control-percent", type=int, default=90)
    parser.add_argument("--challenger-percent", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_yaml)
    payload = build_endpoint_core_config(
        config,
        control_percent=args.control_percent,
        challenger_percent=args.challenger_percent,
    )
    ServingEndpointClient().ensure_endpoint(config["endpoint_name"], payload)


if __name__ == "__main__":
    main()
