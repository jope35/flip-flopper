# Databricks notebook source
import pandas as pd
from sklearn.datasets import make_classification

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")
dbutils.widgets.text("table_name", "")

catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
table_name = dbutils.widgets.get("table_name").strip()

if not all([catalog, schema, table_name]):
    raise ValueError("catalog, schema, and table_name must all be non-empty")

full_table = f"{catalog}.{schema}.{table_name}"

X, y = make_classification(
    n_samples=100_000,
    n_features=33,
    n_informative=26,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=42,
)

feature_cols = [f"feature_{i:03d}" for i in range(33)]
pdf = pd.DataFrame(X, columns=feature_cols)
pdf["label"] = y.astype("int64")

spark_df = spark.createDataFrame(pdf)
spark_df.write.mode("overwrite").saveAsTable(full_table)

# ponytail: len(pdf) not spark_df.count() — avoids a second Spark action; upgrade if write can partially fail
print(f"Wrote {len(pdf)} rows to {full_table}")
