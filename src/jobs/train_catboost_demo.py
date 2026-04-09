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
    args = parse_args()

    from catboost import CatBoostClassifier
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    registered_name = f"{args.catalog}.{args.schema}.{args.registered_model_suffix}"

    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()
    if client.get_experiment_by_name(args.experiment_name) is None:
        client.create_experiment(args.experiment_name)
    mlflow.set_experiment(args.experiment_name)

    iterations = 80
    depth = 4
    learning_rate = 0.1

    with mlflow.start_run():
        mlflow.log_params(
            {
                "iterations": iterations,
                "depth": depth,
                "learning_rate": learning_rate,
                "catalog": args.catalog,
                "schema": args.schema,
            }
        )

        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=False,
            random_seed=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        mlflow.log_metric("val_accuracy", acc)

        model_info = mlflow.catboost.log_model(model, artifact_path="model")

    mv = mlflow.register_model(model_info.model_uri, registered_name)
    print(f"Registered model: {registered_name} version {mv.version}")
    print(f"Model URI: models:/{registered_name}/{mv.version}")


if __name__ == "__main__":
    main()
