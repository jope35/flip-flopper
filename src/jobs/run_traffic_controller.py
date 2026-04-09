import argparse

from flip_flopper_ab_test.config import load_config
from flip_flopper_ab_test.controller import ControllerService
from flip_flopper_ab_test.databricks_api import ServingEndpointClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", required=True, help="Path to app config YAML")
    parser.add_argument("--current-control-percent", type=int, required=True)
    parser.add_argument("--current-challenger-percent", type=int, required=True)
    parser.add_argument("--last-good-control-percent", type=int, required=True)
    parser.add_argument("--last-good-challenger-percent", type=int, required=True)
    parser.add_argument("--shadow-mode", choices=["true", "false"], default="true")
    return parser.parse_args()


def main() -> None:
    from pyspark.sql import SparkSession

    args = parse_args()
    config = load_config(args.config_yaml)
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
