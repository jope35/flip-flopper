# src/jobs/train_catboost_demo.py
from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost on demo data; log to MLflow; register model in Unity Catalog.",
    )
    parser.add_argument("--catalog", required=True, help="Unity Catalog catalog name")
    parser.add_argument("--schema", required=True, help="Unity Catalog schema name")
    parser.add_argument(
        "--experiment-name",
        default="/Shared/flip-flopper/catboost-demo",
        help="MLflow experiment path",
    )
    parser.add_argument(
        "--registered-model-suffix",
        default="flip_flopper_catboost_demo",
        help="UC registered model name = {catalog}.{schema}.{suffix}",
    )
    return parser.parse_args(argv)


def main() -> None:
    raise NotImplementedError("training pipeline — Task 3")


if __name__ == "__main__":
    main()
