"""Scheduled traffic controller entrypoint (orchestration to be wired to ControllerService)."""

import argparse

from flip_flopper_ab_test.config import load_config


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
    _ = SparkSession.builder.getOrCreate()
    # ControllerService integration pending (metrics, policy, endpoint updates).
    _ = (
        args.current_control_percent,
        args.current_challenger_percent,
        args.last_good_control_percent,
        args.last_good_challenger_percent,
        args.shadow_mode,
    )
    print(f"Loaded config for endpoint: {config.get('endpoint_name')}")


if __name__ == "__main__":
    main()
